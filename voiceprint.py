"""
Voiceprint Processing Core

Uses SpeechBrain ECAPA-TDNN for 192-dim voiceprint embeddings.
Supports multiple audio formats, auto-transcoding, resampling, bandpass filtering.
Provides voiceprint registration (single/batch), identification (1:N), and verification (1:1).

Device selection: auto / cuda / cpu with automatic fallback.
"""
import torch
import numpy as np
import os
import tempfile
import subprocess
import soundfile as sf
from scipy.signal import butter, filtfilt
from config import (
    MODEL_NAME, SAMPLE_RATE, SIMILARITY_THRESHOLD,
    DEVICE_PREFERENCE, ALLOW_CPU_FALLBACK, CUDA_DEVICE_INDEX,
)


class VoiceprintExtractor:
    """Voiceprint feature extractor with GPU/CPU auto-selection."""

    SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.mp3', '.m4a', '.aac', '.wma', '.opus', '.webm'}
    NEED_CONVERT = {'.mp3', '.m4a', '.aac', '.wma', '.opus', '.webm'}

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.ffmpeg_path = self._find_ffmpeg()
        self.device, self.device_name, self.fallback_reason = self._select_device()
        self._load_model()

    def _select_device(self):
        """Select compute device based on config and availability.

        Returns:
            (device_str, device_name, fallback_reason)
        """
        pref = DEVICE_PREFERENCE

        # Forced CPU
        if pref == "cpu":
            return "cpu", "CPU", "forced by config"

        # Try CUDA
        if torch.cuda.is_available():
            try:
                idx = CUDA_DEVICE_INDEX
                name = torch.cuda.get_device_name(idx)
                # Quick sanity check
                torch.cuda.set_device(idx)
                _ = torch.tensor([1.0], device=f"cuda:{idx}")
                return f"cuda:{idx}", name, ""
            except Exception as e:
                if pref == "gpu":
                    raise RuntimeError(f"GPU requested but initialization failed: {e}")
                if ALLOW_CPU_FALLBACK:
                    return "cpu", "CPU", f"CUDA init failed: {e}"
                raise

        # CUDA not available
        if pref == "gpu":
            raise RuntimeError("GPU requested but CUDA is not available (torch.cuda.is_available()=False)")

        if pref == "auto":
            reason = "CUDA not available"
            if not torch.backends.cudnn.enabled:
                reason += ", cuDNN not enabled"
            return "cpu", "CPU", reason

        return "cpu", "CPU", "unknown"

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
        """Load pre-trained model with GPU → CPU fallback."""
        from speechbrain.inference.speaker import EncoderClassifier
        savedir = os.path.join("data", "models", "ecapa_tdnn")

        try:
            self.model = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=savedir,
                run_opts={"device": self.device},
            )
            print(f"Model loaded on: {self.device} ({self.device_name})")
            if self.fallback_reason:
                print(f"  Fallback reason: {self.fallback_reason}")
        except Exception as e:
            if self.device.startswith("cuda") and ALLOW_CPU_FALLBACK:
                print(f"GPU model load failed ({e}), falling back to CPU...")
                self.device = "cpu"
                self.device_name = "CPU"
                self.fallback_reason = str(e)
                self.model = EncoderClassifier.from_hparams(
                    source=self.model_name,
                    savedir=savedir,
                    run_opts={"device": "cpu"},
                )
                print("Model loaded on: CPU (fallback)")
            else:
                raise

    def _convert_audio_ffmpeg(self, audio_path):
        if not self.ffmpeg_path:
            raise ValueError("ffmpeg not found, install ffmpeg or use WAV/FLAC/OGG")
        temp_path = os.path.join(tempfile.gettempdir(), f"vp_temp_{os.getpid()}.wav")
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
        temp_path = os.path.join(tempfile.gettempdir(), f"vp_temp_{os.getpid()}.wav")
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

    def extract_embedding(self, audio_path):
        try:
            waveform, _ = self.load_audio(audio_path)
            waveform = waveform.to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_batch(waveform)
            return embedding.squeeze().cpu().numpy()
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
        info = {
            "device": self.device,
            "device_name": self.device_name,
            "fallback": bool(self.fallback_reason),
            "fallback_reason": self.fallback_reason,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
        return info


class VoiceprintDatabase:
    """Voiceprint database management."""

    def __init__(self, db_manager, extractor):
        self.db = db_manager
        self.extractor = extractor

    def register_voiceprint(self, user_id, audio_path):
        embedding = self.extractor.extract_embedding(audio_path)
        return self.db.add_voiceprint(user_id, embedding, audio_path)

    def batch_register_voiceprint(self, user_id, audio_paths):
        embeddings = []
        for path in audio_paths:
            try:
                embeddings.append(self.extractor.extract_embedding(path))
            except Exception as e:
                print(f"Skipping {path}: {e}")
        if not embeddings:
            raise ValueError("No valid audio data")
        avg = np.mean(embeddings, axis=0)
        avg = avg / np.linalg.norm(avg)
        return self.db.replace_voiceprint(user_id, avg)

    def verify_speaker(self, audio_path, threshold=SIMILARITY_THRESHOLD):
        all_voiceprints = self.db.get_all_voiceprints()
        if not all_voiceprints:
            return {"matched": False, "user_id": None, "user_name": None,
                    "similarity": 0.0, "message": "No voiceprints registered"}

        ref_embeddings = [(vp['user_id'], vp['embedding']) for vp in all_voiceprints]
        result = self.extractor.verify_speaker(audio_path, ref_embeddings, threshold)

        user_scores = {}
        for uid, score in result.get('all_scores', []):
            if uid not in user_scores or score > user_scores[uid]:
                user_scores[uid] = score

        sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        if not sorted_users:
            return {"matched": False, "user_id": None, "user_name": None,
                    "similarity": 0.0, "message": "No voiceprints", "all_scores": []}

        best_uid, best_score = sorted_users[0]
        is_matched = best_score >= threshold
        user = self.db.get_user(best_uid)
        result['user_id'] = best_uid
        result['user_name'] = user['name'] if user else "Unknown"
        result['department'] = user.get('department', "") if user else ""
        result['similarity'] = best_score
        result['matched'] = is_matched
        result['all_scores'] = [(uid, score) for uid, score in sorted_users]
        self.db.add_verify_log(best_uid, best_score, "Match" if is_matched else "No match", audio_path)
        return result

    def verify_speaker_by_id(self, audio_path, user_id, threshold=SIMILARITY_THRESHOLD):
        user_voiceprints = self.db.get_voiceprints(user_id)
        if not user_voiceprints:
            return {"matched": False, "similarity": 0.0, "message": "No voiceprints for this user"}
        user_embeddings = [vp['embedding'] for vp in user_voiceprints]
        result = self.extractor.verify_speaker_by_id(audio_path, user_id, user_embeddings, threshold)
        user = self.db.get_user(user_id)
        self.db.add_verify_log(user_id, result['similarity'],
                               "Match" if result['matched'] else "No match", audio_path)
        return result
