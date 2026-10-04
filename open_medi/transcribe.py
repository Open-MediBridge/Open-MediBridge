"""Speech-to-text with faster-whisper (CTranslate2 reimplementation of Whisper).

The model (default: large-v3-turbo) is loaded lazily on first use so the API
starts instantly; device/compute_type default to CTranslate2's auto-detection
(int8 on CPU, float16 on CUDA).
"""

import os
import site
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from open_medi.config import Settings, get_settings


def _add_nvidia_dll_dirs() -> None:
    """Make cuBLAS/cuDNN wheels loadable on Windows.

    The nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels install their DLLs
    under site-packages\\nvidia\\<pkg>\\bin, which CTranslate2 does not
    search on Windows. CTranslate2 loads CUDA libraries with a plain
    LoadLibrary call, which ignores os.add_dll_directory registrations,
    so the directories are prepended to PATH as well.
    """
    if sys.platform != "win32":
        return
    dirs: list[str] = []
    for package_dir in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_dir = Path(package_dir) / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for bin_dir in nvidia_dir.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            dirs.append(str(bin_dir))
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")


@dataclass
class Transcript:
    text: str
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None
    segments: list[dict] = field(default_factory=list)


@lru_cache(maxsize=1)
def _get_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    _add_nvidia_dll_dirs()
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def model_loaded() -> bool:
    """True if the Whisper weights are already in memory."""
    info = _get_model.cache_info()
    return bool(info and info.currsize)


def transcribe_file(
    path: str | Path,
    language: str | None = None,
    settings: Settings | None = None,
) -> Transcript:
    s = settings or get_settings()
    model = _get_model(s.whisper_model, s.whisper_device, s.whisper_compute_type)
    segments, info = model.transcribe(
        str(path),
        language=language,
        vad_filter=True,          # skip silence, big speedup on recordings
        beam_size=5,
    )
    segs = [
        {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()}
        for seg in segments           # iterating here runs the actual inference
    ]
    text = " ".join(seg["text"] for seg in segs if seg["text"]).strip()
    return Transcript(
        text=text,
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        segments=segs,
    )


def transcribe_upload(
    data: bytes,
    filename: str = "audio.webm",
    language: str | None = None,
    settings: Settings | None = None,
) -> Transcript:
    """Transcribe raw uploaded bytes, preserving the original file extension."""
    suffix = Path(filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return transcribe_file(tmp_path, language=language, settings=settings)
    finally:
        tmp_path.unlink(missing_ok=True)
