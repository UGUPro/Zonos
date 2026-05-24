import io
import os
from typing import Any, Dict, Optional

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

# -------- Config --------
DEFAULT_DEVICE = os.environ.get("ZONOS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
VOICES_DIR = os.environ.get("ZONOS_VOICES_DIR", "./voices")

# A short silence prefix often improves stability; common practice is ~100ms silence.
# (Used in community Zonos UIs as default prefix audio.)
DEFAULT_PREFIX_AUDIO = os.environ.get("ZONOS_PREFIX_AUDIO", "assets/silence_100ms.wav")

# Cache models by HF id (e.g., "Zyphra/Zonos-v0.1-transformer")
_MODEL_CACHE: Dict[str, Zonos] = {}

app = FastAPI(title="Zonos OpenAI-Compatible TTS")

# -------- OpenAI-like request model --------
class SpeechRequest(BaseModel):
    model: str = Field(default="Zyphra/Zonos-v0.1-transformer")
    input: str
    voice: Optional[str] = Field(default=None, description="Voice name mapping to ./voices/<voice>.(wav|mp3)")
    response_format: str = Field(default="wav", description="wav or mp3")
    speed: float = Field(default=1.0, description="Mapped to Zonos speaking_rate")
    language: str = Field(default="en-us")

    # OpenAI doesn't define emotion, but we allow it for convenience:
    # {"happiness":1.0, "sadness":0.0, ...}
    emotion: Optional[Dict[str, float]] = None


def _load_model(model_id: str) -> Zonos:
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    m = Zonos.from_pretrained(model_id, device=DEFAULT_DEVICE)
    m.to(DEFAULT_DEVICE)
    # Many examples run bfloat16 for speed on modern GPUs
    if DEFAULT_DEVICE.startswith("cuda"):
        m.bfloat16()
    m.eval()
    _MODEL_CACHE[model_id] = m
    return m


def _load_voice_audio(voice: str) -> Optional[str]:
    """
    Returns a filepath to the voice reference audio if found.
    """
    if not voice:
        return None

    # Allow passing an explicit path if you want:
    if os.path.exists(voice):
        return voice

    # Otherwise resolve as voices/<voice>.(wav|mp3|flac)
    for ext in (".wav", ".mp3", ".flac", ".ogg"):
        p = os.path.join(VOICES_DIR, voice + ext)
        if os.path.exists(p):
            return p

    return None


def _emotion_vector(emotion: Optional[Dict[str, float]], device: str) -> torch.Tensor:
    """
    Zonos community UIs commonly use an 8-dim vector with labels:
    Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral. :contentReference[oaicite:3]{index=3}
    """
    e = emotion or {}
    vec = [
        float(e.get("happiness", 0.05)),
        float(e.get("sadness", 0.05)),
        float(e.get("disgust", 0.05)),
        float(e.get("fear", 0.05)),
        float(e.get("surprise", 0.05)),
        float(e.get("anger", 0.05)),
        float(e.get("other", 0.05)),
        float(e.get("neutral", 0.60)),
    ]
    return torch.tensor([vec], device=device)


def _encode_wav_bytes(wav: torch.Tensor, sr: int) -> bytes:
    # torchaudio.save wants (channels, time) on CPU
    buf = io.BytesIO()
    torchaudio.save(buf, wav.cpu(), sample_rate=sr, format="wav")
    return buf.getvalue()


@app.post("/v1/audio/speech")
def openai_audio_speech(req: SpeechRequest) -> Response:
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="`input` must be non-empty")

    model = _load_model(req.model)

    # Speaker embedding (optional)
    speaker_embedding = None
    voice_path = _load_voice_audio(req.voice) if req.voice else None
    if voice_path:
        wav_ref, sr_ref = torchaudio.load(voice_path)
        speaker_embedding = model.make_speaker_embedding(wav_ref, sr_ref)
        # match dtype/device patterns used in common Zonos scripts :contentReference[oaicite:4]{index=4}
        speaker_embedding = speaker_embedding.to(DEFAULT_DEVICE, dtype=torch.bfloat16 if DEFAULT_DEVICE.startswith("cuda") else torch.float32)

    # Prefix audio codes (optional, but recommended)
    audio_prefix_codes = None
    if DEFAULT_PREFIX_AUDIO and os.path.exists(DEFAULT_PREFIX_AUDIO):
        wav_prefix, sr_prefix = torchaudio.load(DEFAULT_PREFIX_AUDIO)
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        wav_prefix = torchaudio.functional.resample(wav_prefix, sr_prefix, model.autoencoder.sampling_rate)
        wav_prefix = wav_prefix.to(DEFAULT_DEVICE, dtype=torch.float32)
        with torch.autocast(DEFAULT_DEVICE if DEFAULT_DEVICE.startswith("cuda") else "cpu", dtype=torch.float32):
            audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))

    # Map OpenAI "speed" to Zonos "speaking_rate".
    # Community examples commonly expose speaking_rate roughly in [0..40]. :contentReference[oaicite:5]{index=5}
    # We'll treat speed=1.0 as 15.0 (typical default), and scale linearly.
    speaking_rate = max(0.0, min(40.0, 15.0 * float(req.speed)))

    cond = make_cond_dict(
        text=req.input,
        language=req.language,
        speaker=speaker_embedding,
        emotion=_emotion_vector(req.emotion, DEFAULT_DEVICE),
        speaking_rate=speaking_rate,
        device=DEFAULT_DEVICE,
    )

    conditioning = model.prepare_conditioning(cond)

    # Conservative token budget: ~30 seconds at common settings in example UIs :contentReference[oaicite:6]{index=6}
    max_new_tokens = 86 * 30

    with torch.inference_mode():
        codes = model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=audio_prefix_codes,
            max_new_tokens=max_new_tokens,
            cfg_scale=2.0,
            batch_size=1,
            sampling_params=dict(min_p=0.10),
        )
        wav_out = model.autoencoder.decode(codes).cpu().detach()
        sr_out = model.autoencoder.sampling_rate

    # Ensure shape (channels, time)
    if wav_out.dim() == 2 and wav_out.size(0) > 1:
        wav_out = wav_out[0:1, :]

    fmt = (req.response_format or "wav").lower()
    if fmt == "wav":
        data = _encode_wav_bytes(wav_out, sr_out)
        return Response(content=data, media_type="audio/wav")

    if fmt == "mp3":
        try:
            from pydub import AudioSegment
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"mp3 requested but pydub not installed: {e}")

        wav_bytes = _encode_wav_bytes(wav_out, sr_out)
        seg = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        out = io.BytesIO()
        seg.export(out, format="mp3")
        return Response(content=out.getvalue(), media_type="audio/mpeg")

    raise HTTPException(status_code=400, detail="response_format must be 'wav' or 'mp3'")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "device": DEFAULT_DEVICE}
