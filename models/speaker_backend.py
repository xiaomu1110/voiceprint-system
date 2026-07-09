"""
Swappable speaker-encoder backends.

Provides a SpeakerEncoder abstraction so the system can move off the default
SpeechBrain ECAPA-TDNN to stronger Chinese/multilingual models (ERes2NetV2,
WavLM-SPK) without changing the verification/scoring layer.

Heavy dependencies (speechbrain, modelscope, transformers) are imported lazily
inside __init__/methods, so `import models.speaker_backend` triggers no model
loads. Each embed() returns an L2-normalized 1-D numpy vector.
"""
from __future__ import annotations

import numpy as np
import torch

from config import ERES2NETV2_MODEL_ID


def _normalize(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    if norm < 1e-9:
        return embedding.astype(np.float32)
    return (embedding / norm).astype(np.float32)


class SpeakerEncoder:
    """Abstract speaker encoder."""

    name = "base"
    dim = 192
    sample_rate = 16000

    @property
    def device(self) -> str:
        return getattr(self, "_device", "cpu")

    def embed(self, wav_path: str) -> np.ndarray:
        raise NotImplementedError

    def embed_waveform(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        raise NotImplementedError

    def is_ready(self) -> bool:
        return True


class ECAPAEncoder(SpeakerEncoder):
    """SpeechBrain ECAPA-TDNN, reusing the project's VoiceprintExtractor
    (audio preprocessing + GPU model load) so there is a single source of truth."""

    name = "ecapa"
    dim = 192

    def __init__(self, device: str | None = None):
        from voiceprint import VoiceprintExtractor  # local import; loads the GPU model
        self._vpe = VoiceprintExtractor(force_encoder="ecapa")
        self._device = self._vpe.device
        if device and device != self._device:
            print(f"[ecapa] requested device {device} but VoiceprintExtractor uses {self._device}")

    def embed_waveform(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        wav = waveform.to(self._device) if isinstance(waveform, torch.Tensor) \
            else torch.FloatTensor(waveform).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._vpe.model.encode_batch(wav)
        return _normalize(emb.squeeze().cpu().numpy())

    def embed(self, wav_path: str) -> np.ndarray:
        waveform, _ = self._vpe.load_audio(wav_path)
        return self.embed_waveform(waveform, self.sample_rate)

    def is_ready(self) -> bool:
        return getattr(self._vpe, "model", None) is not None


class _LazyHEncoder(SpeakerEncoder):
    """Base for encoders whose backend dependency may not be installed.
    Subclasses implement _load() and _embed_from_model()."""

    _install_hint = "pip install <backend>"

    def __init__(self, device: str | None = None):
        self._device = device or "cuda:0"
        self._model = None
        self._error: str | None = None
        try:
            self._model = self._load()
        except Exception as e:  # noqa: BLE001 - surface a clear message, do not crash import
            self._error = f"{type(e).__name__}: {e}"

    def _load(self):
        raise NotImplementedError

    def _embed_from_model(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        raise NotImplementedError

    def is_ready(self) -> bool:
        return self._model is not None

    def _fail(self):
        raise RuntimeError(
            f"{self.name} backend is not available. Install with: {self._install_hint}"
            + (f"\nDetail: {self._error}" if self._error else "")
        )

    def embed_waveform(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        if self._model is None:
            self._fail()
        wav = waveform.to(self._device) if isinstance(waveform, torch.Tensor) \
            else torch.FloatTensor(waveform).unsqueeze(0).to(self._device)
        emb = self._embed_from_model(wav, sr)
        return _normalize(np.asarray(emb, dtype=np.float32).reshape(-1))

    def embed(self, wav_path: str) -> np.ndarray:
        if self._model is None:
            self._fail()
        import soundfile as sf
        from scipy.signal import resample as scipy_resample
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != self.sample_rate:
            data = scipy_resample(data, int(len(data) * self.sample_rate / sr))
            sr = self.sample_rate
        waveform = torch.FloatTensor(data).unsqueeze(0)
        return self.embed_waveform(waveform, sr)


class ERes2NetV2Encoder(_LazyHEncoder):
    """ERes2NetV2 (3D-Speaker/ModelScope) — strong on Chinese + multilingual.

    Requires `modelscope` and a downloaded ERes2NetV2 checkpoint. The exact
    embedding-extraction call depends on the installed speakerlab/modelscope
    version; wire _embed_from_model to your checkpoint's forward when enabling.
    """

    name = "erres2netv2"
    dim = 192
    _install_hint = "pip install modelscope sortedcontainers addict datasets Pillow"

    def __init__(self, device: str | None = None, model_id: str = ERES2NETV2_MODEL_ID):
        self.model_id = model_id
        super().__init__(device=device)

    def _load(self):
        from modelscope.pipelines import pipeline  # type: ignore
        return pipeline("speaker-verification", model=self.model_id)

    def embed(self, wav_path: str) -> np.ndarray:
        # The ModelScope SV pipeline is path-based and does its own preprocessing.
        # However, it may not support all container formats (e.g. WebM from browser
        # recordings). Convert to WAV first for reliability.
        if self._model is None:
            self._fail()
        import os, tempfile, uuid, soundfile as sf
        from scipy.signal import resample as scipy_resample
        ext = os.path.splitext(wav_path)[1].lower()
        temp_wav = None
        if ext not in {'.wav', '.flac', '.ogg'}:
            # Try soundfile conversion first (handles mp3, ogg, etc.)
            try:
                data, sr = sf.read(wav_path, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != self.sample_rate:
                    data = scipy_resample(data, int(len(data) * self.sample_rate / sr))
                    sr = self.sample_rate
                temp_wav = os.path.join(tempfile.gettempdir(), f"vp_enc_{uuid.uuid4().hex}.wav")
                sf.write(temp_wav, data, sr, subtype="PCM_16")
                wav_path = temp_wav
            except Exception:
                # If soundfile can't read it, try ffmpeg via VoiceprintExtractor
                try:
                    from voiceprint import VoiceprintExtractor
                    _tmp_ext = VoiceprintExtractor()
                    wav_path, _ = _tmp_ext._convert_audio(wav_path)
                    temp_wav = wav_path
                except Exception:
                    pass  # best-effort: pass original path
        try:
            with torch.no_grad():
                res = self._model([wav_path], output_emb=True)
            emb = np.asarray(res["embs"], dtype=np.float32)[0]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return _normalize(emb)
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

    def _embed_from_model(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        raise NotImplementedError(
            "ERes2NetV2 is path-based; use embed(wav_path) instead of embed_waveform()"
        )


class WavLMSpkEncoder(_LazyHEncoder):
    """WavLM self-supervised speaker encoder — best generalization to unseen
    dialects/English. Requires `transformers` and a speaker head checkpoint."""

    name = "wavlm"
    dim = 256
    _install_hint = "pip install transformers torch"

    def __init__(self, device: str | None = None, model_id: str = "microsoft/wavlm-base-plus"):
        self.model_id = model_id
        super().__init__(device=device)

    def _load(self):
        from transformers import AutoModel, AutoProcessor  # type: ignore
        self._proc = AutoProcessor.from_pretrained(self.model_id)
        return AutoModel.from_pretrained(self.model_id).to(self._device).eval()

    def _embed_from_model(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        wav_np = waveform.squeeze(0).cpu().numpy()
        inputs = self._proc(wav_np, sampling_rate=sr, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        hidden = out.last_hidden_state  # (1, T, H)
        emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()  # mean-pooled utterance embedding
        return emb


class EnsembleEncoder(SpeakerEncoder):
    """Combines multiple encoders. With mode='concat' the embeddings are
    concatenated (then L2-normalized); with mode='score' each backend scores
    independently and the caller fuses via scoring.score_fusion."""

    def __init__(self, encoders: list[SpeakerEncoder], weights: list[float] | None = None,
                 mode: str = "concat"):
        if not encoders:
            raise ValueError("EnsembleEncoder requires at least one encoder")
        self.encoders = encoders
        self.weights = weights
        self.mode = mode
        self.name = "ensemble"
        self.dim = sum(e.dim for e in encoders) if mode == "concat" else encoders[0].dim
        self._device = encoders[0].device

    def is_ready(self) -> bool:
        return all(e.is_ready() for e in self.encoders)

    def embed_waveform(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        if self.mode != "concat":
            raise RuntimeError("EnsembleEncoder in 'score' mode scores per-backend; use embed() per encoder + scoring.score_fusion")
        parts = [e.embed_waveform(waveform, sr) for e in self.encoders]
        return _normalize(np.concatenate(parts))

    def embed(self, wav_path: str) -> np.ndarray:
        if self.mode != "concat":
            raise RuntimeError("EnsembleEncoder in 'score' mode: call each encoder.embed() and fuse with scoring.score_fusion")
        parts = [e.embed(wav_path) for e in self.encoders]
        return _normalize(np.concatenate(parts))


def make_encoder(name: str, device: str | None = None, **kw) -> SpeakerEncoder:
    """Factory. name ∈ {ecapa, erres2netv2, wavlm, ensemble}."""
    name = (name or "").strip().lower()
    if name == "ecapa":
        return ECAPAEncoder(device=device)
    if name == "erres2netv2":
        return ERes2NetV2Encoder(device=device, **kw)
    if name == "wavlm":
        return WavLMSpkEncoder(device=device, **kw)
    if name == "ensemble":
        return EnsembleEncoder(**kw)
    raise ValueError(f"unknown encoder '{name}'; expected ecpa|erres2netv2|wavlm|ensemble")
