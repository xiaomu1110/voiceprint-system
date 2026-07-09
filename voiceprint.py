"""
Voiceprint Processing Core

Uses SpeechBrain ECAPA-TDNN for 192-dim voiceprint embeddings.
Supports multiple audio formats, auto-transcoding, resampling, bandpass filtering.
Provides voiceprint registration (single/batch), identification (1:N), and verification (1:1).

Device policy: NVIDIA CUDA is required. CPU execution is intentionally unsupported.
"""
import torch
import numpy as np
import os
import tempfile
import subprocess
import uuid
import soundfile as sf
from scipy.signal import butter, filtfilt
from config import (
    MODEL_NAME, SAMPLE_RATE, SIMILARITY_THRESHOLD,
    CUDA_DEVICE_INDEX, CONFIDENCE_SCALE,
    VERIFY_THRESHOLD, IDENTIFY_ABS_FLOOR, IDENTIFY_MARGIN,
    CONSISTENCY_THRESHOLD, FUSION_TRIM_RATIO,
    USE_ASNORM, AS_NORM_K, USE_SILERO_VAD,
    ENCODER_NAME, ERES2NETV2_MODEL_ID,
)
from scoring import CohortManager
import vad


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "N/A"
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024


def get_gpu_diagnostics() -> dict:
    """Return CUDA diagnostics without loading the SpeechBrain model."""
    info = {
        "requires_gpu": True,
        "device_policy": "gpu_required",
        "cuda_device_index": CUDA_DEVICE_INDEX,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            smi_gpus = []
            for line in r.stdout.strip().splitlines():
                name, driver, memory = [part.strip() for part in line.split(",", 2)]
                smi_gpus.append({"name": name, "driver_version": driver, "memory_total": memory})
            info["nvidia_smi_available"] = True
            info["nvidia_smi_gpus"] = smi_gpus
        else:
            info["nvidia_smi_available"] = False
            info["nvidia_smi_error"] = r.stderr.strip()[:300]
    except Exception as e:
        info["nvidia_smi_available"] = False
        info["nvidia_smi_error"] = str(e)

    if info["cuda_available"]:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["gpus"].append({
                "index": idx,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": props.total_memory,
                "total_memory": _format_bytes(props.total_memory),
            })
        if CUDA_DEVICE_INDEX < info["gpu_count"]:
            props = torch.cuda.get_device_properties(CUDA_DEVICE_INDEX)
            info.update({
                "device": f"cuda:{CUDA_DEVICE_INDEX}",
                "device_name": props.name,
                "gpu_name": props.name,
                "gpu_memory_total": _format_bytes(props.total_memory),
                "gpu_compute_capability": f"{props.major}.{props.minor}",
            })

    return info


def normalize_confidence_scores(raw_scores, scale=CONFIDENCE_SCALE):
    """Convert raw similarity scores into a normalized confidence distribution."""
    if not raw_scores:
        return []

    values = np.asarray([score for _, score in raw_scores], dtype=np.float64)
    values = np.nan_to_num(values, nan=-1.0, posinf=1.0, neginf=-1.0)
    scale = scale if scale and scale > 0 else 1.0
    shifted = (values - np.max(values)) * scale
    shifted = np.clip(shifted, -60.0, 60.0)
    exp_values = np.exp(shifted)
    total = float(np.sum(exp_values))

    if not np.isfinite(total) or total <= 0:
        confidences = np.full(len(raw_scores), 1.0 / len(raw_scores), dtype=np.float64)
    else:
        confidences = exp_values / total

    return [
        {
            "user_id": user_id,
            "raw_similarity": float(score),
            "normalized_confidence": float(confidence),
        }
        for (user_id, score), confidence in zip(raw_scores, confidences)
    ]


class VoiceprintExtractor:
    """Voiceprint feature extractor that requires CUDA GPU execution."""

    SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.mp3', '.m4a', '.aac', '.wma', '.opus', '.webm'}
    NEED_CONVERT = {'.mp3', '.m4a', '.aac', '.wma', '.opus', '.webm'}

    def __init__(self, model_name=MODEL_NAME, force_encoder=None):
        self.model_name = model_name
        self.model = None
        self._sv_pipeline = None
        self.encoder_name = "ecapa"
        self.ffmpeg_path = self._find_ffmpeg()
        self.device, self.device_name = self._select_device()
        want = force_encoder or ENCODER_NAME
        if want == "erres2netv2" and self._load_eres2netv2():
            self.encoder_name = "erres2netv2"
            self.model_name = "ERes2NetV2"
        else:
            self._load_model()

    def _select_device(self):
        """Select and validate the required CUDA device."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU acceleration is required, but CUDA is not available. "
                "Install an NVIDIA driver and a CUDA-enabled PyTorch build."
            )

        count = torch.cuda.device_count()
        if CUDA_DEVICE_INDEX < 0 or CUDA_DEVICE_INDEX >= count:
            raise RuntimeError(
                f"VOICEPRINT_CUDA_DEVICE_INDEX={CUDA_DEVICE_INDEX} is invalid; "
                f"{count} CUDA device(s) detected."
            )

        try:
            torch.cuda.set_device(CUDA_DEVICE_INDEX)
            _ = torch.tensor([1.0], device=f"cuda:{CUDA_DEVICE_INDEX}") * 1.0
            torch.cuda.synchronize(CUDA_DEVICE_INDEX)
        except Exception as e:
            raise RuntimeError(f"CUDA device initialization failed: {e}") from e

        return f"cuda:{CUDA_DEVICE_INDEX}", torch.cuda.get_device_name(CUDA_DEVICE_INDEX)

    def _find_ffmpeg(self):
        possible = ['ffmpeg',
                     r'C:\ffmpeg\bin\ffmpeg.exe',
                     r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
                     os.path.expanduser(r'~\ffmpeg\bin\ffmpeg.exe')]
        for p in possible:
            try:
                r = subprocess.run([p, '-version'], capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    return p
            except Exception:
                continue
        return None

    @staticmethod
    def _bandpass_filter(audio, sr, low=80, high=8000):
        nyq = sr / 2
        high = min(high, nyq - 100)
        low_n = max(low / nyq, 0.001)
        high_n = min(high / nyq, 0.999)
        b, a = butter(4, [low_n, high_n], btype='band')
        return filtfilt(b, a, audio).astype(np.float32)

    def _load_model(self):
        """Load pre-trained model on the required CUDA device."""
        from speechbrain.inference.speaker import EncoderClassifier
        savedir = os.path.join("data", "models", "ecapa_tdnn")

        self.model = EncoderClassifier.from_hparams(
            source=self.model_name,
            savedir=savedir,
            run_opts={"device": self.device},
        )
        print(f"Model loaded on GPU: {self.device} ({self.device_name})")

    def _load_eres2netv2(self):
        """Load the ModelScope ERes2NetV2 speaker-verification pipeline (Chinese-strong).
        Returns True on success; False to fall back to ECAPA."""
        try:
            from modelscope.pipelines import pipeline  # type: ignore
            self._sv_pipeline = pipeline("speaker-verification", model=ERES2NETV2_MODEL_ID)
            print(f"Model loaded: ERes2NetV2 ({ERES2NETV2_MODEL_ID}) on {self.device} ({self.device_name})")
            return True
        except Exception as e:
            print(f"ERes2NetV2 load failed, falling back to ECAPA: {e}")
            self._sv_pipeline = None
            return False

    def _convert_audio_ffmpeg(self, audio_path):
        if not self.ffmpeg_path:
            raise ValueError("ffmpeg not found, install ffmpeg or use WAV/FLAC/OGG")
        temp_path = os.path.join(tempfile.gettempdir(), f"vp_temp_{uuid.uuid4().hex}.wav")
        try:
            cmd = [self.ffmpeg_path, '-i', audio_path, '-ar', str(SAMPLE_RATE),
                   '-ac', '1', '-acodec', 'pcm_s16le', '-f', 'wav', '-y', temp_path]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                raise ValueError(f"ffmpeg error: {r.stderr[:200]}")
            return temp_path
        except subprocess.TimeoutExpired:
            raise ValueError("Audio conversion timeout")

    def _convert_audio_soundfile(self, audio_path):
        temp_path = os.path.join(tempfile.gettempdir(), f"vp_temp_{uuid.uuid4().hex}.wav")
        try:
            audio, sr = sf.read(audio_path, dtype='float32')
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            sf.write(temp_path, audio, sr)
            return temp_path
        except Exception as e:
            raise ValueError(f"Audio conversion failed: {e}")

    def _convert_audio(self, audio_path):
        ext = os.path.splitext(audio_path)[1].lower()
        if ext in {'.wav', '.flac', '.ogg'}:
            return audio_path, False
        if self.ffmpeg_path:
            try:
                return self._convert_audio_ffmpeg(audio_path), True
            except Exception:
                pass
        try:
            return self._convert_audio_soundfile(audio_path), True
        except Exception as e:
            raise ValueError(f"Audio format conversion failed: {e}")

    def load_audio(self, audio_path, target_sr=SAMPLE_RATE):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}")

        temp_path = None
        try:
            wav_path, is_temp = self._convert_audio(audio_path)
            if is_temp:
                temp_path = wav_path
            audio, sr = sf.read(wav_path, dtype='float32')
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            if sr != target_sr:
                from scipy import signal
                num_samples = int(len(audio) * target_sr / sr)
                audio = signal.resample(audio, num_samples)
                sr = target_sr
            audio = self._bandpass_filter(audio, sr)
            waveform = torch.FloatTensor(audio).unsqueeze(0)
            return waveform, sr
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    @staticmethod
    def evaluate_audio_quality(audio, sr):
        """Compute lightweight enrollment quality metrics for processed mono audio."""
        duration = len(audio) / sr if sr else 0.0
        if len(audio) == 0:
            return {
                "score": 0,
                "quality_text": "无效",
                "passed": False,
                "duration": 0,
                "effective_duration": 0,
                "silence_ratio": 1,
                "rms": 0,
                "peak": 0,
                "clipping_ratio": 0,
                "snr_db": 0,
                "message": "音频为空",
            }

        abs_audio = np.abs(audio)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(abs_audio))
        threshold = max(0.01, rms * 0.35)
        active = abs_audio > threshold
        if USE_SILERO_VAD:
            try:
                effective_duration = vad.effective_speech_duration(audio, sr, use_silero=True)
            except Exception:
                effective_duration = float(np.sum(active) / sr)
        else:
            effective_duration = float(np.sum(active) / sr)
        silence_ratio = 1.0 - (effective_duration / duration if duration else 0.0)
        clipping_ratio = float(np.mean(abs_audio >= 0.98))

        signal_rms = float(np.sqrt(np.mean(np.square(audio[active])))) if np.any(active) else rms
        noise = audio[~active]
        noise_rms = float(np.sqrt(np.mean(np.square(noise)))) if len(noise) else max(rms * 0.1, 1e-5)
        snr_db = float(20 * np.log10(max(signal_rms, 1e-6) / max(noise_rms, 1e-6)))

        score = 100.0
        messages = []
        if duration < 2.0:
            score -= 25
            messages.append("总时长偏短")
        elif duration > 15.0:
            score -= 10
            messages.append("总时长偏长")
        if effective_duration < 1.5:
            score -= 30
            messages.append("有效语音不足")
        if rms < 0.012:
            score -= 20
            messages.append("音量偏低")
        if peak < 0.05:
            score -= 15
            messages.append("峰值偏低")
        if clipping_ratio > 0.01:
            score -= 25
            messages.append("存在削波失真")
        if silence_ratio > 0.55:
            score -= 15
            messages.append("静音比例偏高")
        if snr_db < 10:
            score -= 15
            messages.append("噪声偏高")

        score = int(max(0, min(100, round(score))))
        if score >= 85:
            quality_text = "优秀"
        elif score >= 70:
            quality_text = "良好"
        elif score >= 60:
            quality_text = "可用"
        else:
            quality_text = "需重录"

        return {
            "score": score,
            "quality_text": quality_text,
            "passed": score >= 60,
            "duration": round(duration, 2),
            "effective_duration": round(effective_duration, 2),
            "silence_ratio": round(silence_ratio, 3),
            "rms": round(rms, 5),
            "peak": round(peak, 5),
            "clipping_ratio": round(clipping_ratio, 5),
            "snr_db": round(snr_db, 2),
            "message": "；".join(messages) if messages else "质量合格",
        }

    def prepare_enrollment_audio(self, input_path, output_path):
        """Convert, resample, filter, score, and write a playable WAV for enrollment."""
        waveform, sr = self.load_audio(input_path)
        audio = waveform.squeeze(0).cpu().numpy().astype(np.float32)
        quality = self.evaluate_audio_quality(audio, sr)
        sf.write(output_path, audio, sr, subtype="PCM_16")
        return quality

    def extract_embedding(self, audio_path):
        try:
            if self._sv_pipeline is not None:
                with torch.no_grad():
                    res = self._sv_pipeline([audio_path], output_emb=True)
                emb = np.asarray(res["embs"], dtype=np.float32)[0]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._normalize(emb)
            waveform, _ = self.load_audio(audio_path)
            waveform = waveform.to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_batch(waveform)
            return self._normalize(embedding.squeeze().cpu().numpy())
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            raise

    def extract_embedding_from_tensor(self, waveform):
        try:
            waveform = waveform.to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_batch(waveform)
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            raise

    def compute_similarity(self, embedding1, embedding2):
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        return float(np.dot(embedding1, embedding2))

    @staticmethod
    def _normalize(embedding):
        norm = float(np.linalg.norm(embedding))
        if norm < 1e-9:
            return embedding.astype(np.float32)
        return (embedding / norm).astype(np.float32)

    def _fuse_embeddings(self, embeddings):
        """Fuse segment embeddings in normalized space, with optional outlier trimming."""
        embs = [self._normalize(e) for e in embeddings]
        if len(embs) >= 4 and FUSION_TRIM_RATIO > 0:
            centroid = self._normalize(np.mean(embs, axis=0))
            sims = [self.compute_similarity(e, centroid) for e in embs]
            keep = max(1, int(np.ceil(len(embs) * (1.0 - FUSION_TRIM_RATIO))))
            idx = np.argsort(sims)[-keep:]
            embs = [embs[i] for i in idx]
        return self._normalize(np.mean(embs, axis=0))

    def verify_speaker(self, audio_path, reference_embeddings, threshold=SIMILARITY_THRESHOLD):
        test_embedding = self.extract_embedding(audio_path)
        scores = []
        for user_id, ref_embedding in reference_embeddings:
            similarity = self.compute_similarity(test_embedding, ref_embedding)
            scores.append((user_id, similarity))
        scores.sort(key=lambda x: x[1], reverse=True)
        if scores:
            best_user_id, best_similarity = scores[0]
            if best_similarity >= threshold:
                return {"matched": True, "user_id": best_user_id,
                        "similarity": best_similarity, "all_scores": scores}
        return {"matched": False, "user_id": None,
                "similarity": scores[0][1] if scores else 0.0, "all_scores": scores}

    def verify_speaker_by_id(self, audio_path, user_id, user_embeddings, threshold=SIMILARITY_THRESHOLD):
        test_embedding = self.extract_embedding(audio_path)
        similarities = [self.compute_similarity(test_embedding, ref) for ref in user_embeddings]
        max_sim = max(similarities) if similarities else 0.0
        return {"matched": max_sim >= threshold, "similarity": max_sim}

    def get_device_info(self) -> dict:
        info = get_gpu_diagnostics()
        info.update({
            "device": self.device,
            "device_name": self.device_name,
            "model_name": self.model_name,
            "model_runtime": "gpu",
            "gpu_memory_allocated": _format_bytes(torch.cuda.memory_allocated(CUDA_DEVICE_INDEX)),
            "gpu_memory_reserved": _format_bytes(torch.cuda.memory_reserved(CUDA_DEVICE_INDEX)),
        })
        return info


class VoiceprintDatabase:
    """Voiceprint database management."""

    def __init__(self, db_manager, extractor):
        self.db = db_manager
        self.extractor = extractor
        self.cohort = CohortManager(k=AS_NORM_K)
        self._cohort_stamp = -1

    def _refresh_cohort(self):
        """Reload the impostor cohort when AS-norm is enabled and the pool is stale."""
        if not USE_ASNORM:
            return
        try:
            n = len(self.db.get_all_voiceprints())
        except Exception:
            n = -1
        if not self.cohort.is_ready() or n != self._cohort_stamp:
            self.cohort.load_from_db(self.db, self.extractor)
            self._cohort_stamp = n

    def register_voiceprint(self, user_id, audio_path):
        embedding = self.extractor.extract_embedding(audio_path)
        return self.db.add_voiceprint(user_id, embedding, audio_path)

    def batch_register_voiceprint(
        self,
        user_id,
        audio_items,
        min_quality_score=60,
        consistency_threshold=CONSISTENCY_THRESHOLD,
    ):
        candidates = []
        rejected = []
        for item in audio_items:
            if isinstance(item, dict):
                path = item["path"]
                metadata = item.get("metadata", {})
            else:
                path = item
                metadata = {}
            try:
                quality_score = metadata.get("quality_score")
                if min_quality_score is not None and quality_score is not None and quality_score < min_quality_score:
                    rejected.append({
                        "path": path,
                        "reason": "音频质量不足",
                        "quality_score": quality_score,
                    })
                    continue
                embedding = self.extractor.extract_embedding(path)
                candidates.append({"path": path, "embedding": embedding, "metadata": metadata})
            except Exception as e:
                rejected.append({"path": path, "reason": str(e)})

        if not candidates:
            if rejected:
                first_reason = rejected[0].get("reason", "No valid audio data")
                raise ValueError(f"No valid audio data: {first_reason}")
            raise ValueError("No valid audio data")

        existing_segments = self.db.get_enrollment_segments(user_id)
        legacy_voiceprints = [] if existing_segments else self.db.get_voiceprints(user_id)
        reference_embeddings = [self.extractor._normalize(seg["embedding"]) for seg in existing_segments]
        reference_embeddings.extend(self.extractor._normalize(vp["embedding"]) for vp in legacy_voiceprints)

        accepted = []
        if consistency_threshold is None:
            accepted = candidates
        elif reference_embeddings:
            reference = self.extractor._normalize(np.mean(reference_embeddings, axis=0))
            for item in candidates:
                score = self.extractor.compute_similarity(item["embedding"], reference)
                item["consistency_score"] = score
                if score < consistency_threshold:
                    rejected.append({
                        "path": item["path"],
                        "reason": "与该用户已有声纹差异过大",
                        "consistency_score": score,
                    })
                else:
                    accepted.append(item)
        elif len(candidates) >= 3:
            centroid = self.extractor._normalize(np.mean([item["embedding"] for item in candidates], axis=0))
            for item in candidates:
                score = self.extractor.compute_similarity(item["embedding"], centroid)
                item["consistency_score"] = score
                if score < consistency_threshold:
                    rejected.append({
                        "path": item["path"],
                        "reason": "与本次其他片段差异过大",
                        "consistency_score": score,
                    })
                else:
                    accepted.append(item)
        else:
            accepted = candidates

        if not accepted:
            raise ValueError("All audio segments were rejected by quality or consistency checks")

        all_embeddings = reference_embeddings + [item["embedding"] for item in accepted]
        fused = self.extractor._fuse_embeddings(all_embeddings)

        segment_ids = []
        for item in accepted:
            segment_ids.append(
                self.db.add_enrollment_segment(
                    user_id,
                    item["embedding"],
                    item["path"],
                    item["metadata"],
                )
            )

        vp_id = self.db.replace_voiceprint(user_id, fused, sample_count=len(all_embeddings))
        return {
            "vp_id": vp_id,
            "segment_ids": segment_ids,
            "added_segments": len(accepted),
            "total_segments": len(all_embeddings),
            "rejected": rejected,
        }

    def verify_speaker(self, audio_path, threshold=SIMILARITY_THRESHOLD):
        all_voiceprints = self.db.get_all_voiceprints()
        if not all_voiceprints:
            self.db.add_verify_log(
                None,
                0.0,
                "no_match",
                audio_path,
                threshold=threshold,
                threshold_type="normalized_confidence",
                mode="identify",
                candidate_count=0,
                candidates=[],
            )
            return {
                "matched": False,
                "user_id": None,
                "user_name": None,
                "department": "",
                "similarity": 0.0,
                "normalized_confidence": None,
                "candidate_count": 0,
                "all_scores": [],
                "message": "未识别到已注册人员",
            }

        ref_embeddings = [(vp["user_id"], vp["embedding"]) for vp in all_voiceprints]
        raw_result = self.extractor.verify_speaker(audio_path, ref_embeddings, threshold)

        user_meta = {
            vp["user_id"]: {
                "user_id": vp["user_id"],
                "user_name": vp.get("name") or "未知",
                "department": vp.get("department") or "",
            }
            for vp in all_voiceprints
        }

        user_scores = {}
        for uid, score in raw_result.get("all_scores", []):
            if uid not in user_scores or score > user_scores[uid]:
                user_scores[uid] = score

        sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        if not sorted_users:
            self.db.add_verify_log(
                None,
                0.0,
                "no_match",
                audio_path,
                threshold=threshold,
                threshold_type="normalized_confidence",
                mode="identify",
                candidate_count=0,
                candidates=[],
            )
            return {
                "matched": False,
                "user_id": None,
                "user_name": None,
                "department": "",
                "similarity": 0.0,
                "normalized_confidence": None,
                "candidate_count": 0,
                "all_scores": [],
                "message": "未识别到已注册人员",
            }

        normalized = normalize_confidence_scores(sorted_users)
        candidates = []
        for item in normalized:
            meta = user_meta.get(item["user_id"], {})
            candidates.append({
                **meta,
                "similarity": item["raw_similarity"],
                "raw_similarity": item["raw_similarity"],
                "normalized_confidence": item["normalized_confidence"],
            })

        best = candidates[0]
        best_uid = best["user_id"]
        best_score = best["raw_similarity"]
        best_confidence = best["normalized_confidence"]
        second_score = candidates[1]["raw_similarity"] if len(candidates) >= 2 else 0.0
        top1_margin = (best_score - second_score) if len(candidates) >= 2 else None
        # Joint decision: softmax confidence (relative rank) AND absolute cosine floor AND
        # top-1 advantage margin. The absolute floor prevents N=1 / low-score false accepts.
        is_matched = (
            best_confidence >= threshold
            and best_score >= IDENTIFY_ABS_FLOOR
            and (top1_margin is None or top1_margin >= IDENTIFY_MARGIN)
        )

        result = {
            "matched": is_matched,
            "user_id": best_uid if is_matched else None,
            "user_name": best["user_name"] if is_matched else None,
            "department": best["department"] if is_matched else "",
            "similarity": best_score,
            "normalized_confidence": best_confidence,
            "absolute_floor": IDENTIFY_ABS_FLOOR,
            "top1_margin": top1_margin,
            "margin_threshold": IDENTIFY_MARGIN,
            "threshold": threshold,
            "threshold_type": "normalized_confidence",
            "candidate_count": len(candidates),
            "all_scores": candidates,
            "message": "识别成功" if is_matched else "未识别到已注册人员",
        }
        self.db.add_verify_log(
            best_uid,
            best_score,
            "match" if is_matched else "no_match",
            audio_path,
            normalized_confidence=best_confidence,
            threshold=threshold,
            threshold_type="normalized_confidence",
            mode="identify",
            candidate_count=len(candidates),
            candidates=candidates,
        )
        return result

    def verify_speaker_by_id(self, audio_path, user_id, threshold=VERIFY_THRESHOLD):
        user_voiceprints = self.db.get_voiceprints(user_id)
        user = self.db.get_user(user_id)
        if not user_voiceprints:
            self.db.add_verify_log(
                user_id,
                0.0,
                "no_match",
                audio_path,
                threshold=threshold,
                threshold_type="raw_similarity",
                mode="verify",
                candidate_count=0,
                candidates=[],
            )
            return {
                "matched": False,
                "similarity": 0.0,
                "user_id": user_id,
                "user_name": user["name"] if user else "未知",
                "department": user.get("department", "") if user else "",
                "message": "该用户尚未注册声纹",
            }
        segments = self.db.get_enrollment_segments(user_id)
        user_embeddings = (
            [seg["embedding"] for seg in segments] if segments
            else [vp["embedding"] for vp in user_voiceprints]
        )
        test_embedding = self.extractor.extract_embedding(audio_path)
        similarities = [self.extractor.compute_similarity(test_embedding, ref) for ref in user_embeddings]
        max_sim = max(similarities) if similarities else 0.0
        best_idx = int(np.argmax(similarities)) if similarities else 0
        result = {"matched": max_sim >= threshold, "similarity": max_sim}
        if USE_ASNORM:
            self._refresh_cohort()
            if self.cohort.is_ready():
                best_ref = user_embeddings[best_idx]
                result["score_asnorm"] = self.cohort.as_norm(
                    test_embedding, best_ref, max_sim, exclude_user_id=user_id
                )
                result["asnorm_enabled"] = True
        result.update({
            "user_id": user_id,
            "user_name": user["name"] if user else "未知",
            "department": user.get("department", "") if user else "",
            "message": "验证通过" if result["matched"] else "验证失败",
        })
        self.db.add_verify_log(
            user_id,
            result["similarity"],
            "match" if result["matched"] else "no_match",
            audio_path,
            threshold=threshold,
            threshold_type="raw_similarity",
            mode="verify",
            candidate_count=len(user_embeddings),
            candidates=None,
        )
        return result
