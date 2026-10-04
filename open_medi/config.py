"""Central configuration, loaded from environment variables / .env."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root at import time so every consumer sees it.
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # --- Whisper (speech-to-text) ---
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str

    # --- Embeddings ---
    embed_model: str

    # --- Knowledge base / retrieval ---
    db_path: Path
    top_k: int

    # --- LLM (OpenAI-compatible) ---
    llm_base_url: str | None
    llm_api_key: str
    llm_model: str
    llm_temperature: float

    # --- WHO ICD-11 API (https://icd.who.int/icdapi) ---
    icd_client_id: str | None
    icd_client_secret: str | None
    icd_api_version: str

    # --- Prompting ---
    max_context_chars: int


def _load() -> Settings:
    db_path = Path(os.getenv("DB_PATH", "data/open_medi.db"))
    return Settings(
        whisper_model=os.getenv("WHISPER_MODEL", "large-v3-turbo"),
        whisper_device=os.getenv("WHISPER_DEVICE", "auto"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "auto"),
        embed_model=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        db_path=db_path if db_path.is_absolute() else PROJECT_ROOT / db_path,
        top_k=int(os.getenv("TOP_K", "6")),
        llm_base_url=os.getenv("LLM_BASE_URL") or None,
        llm_api_key=os.getenv("LLM_API_KEY", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "llama3.2"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        icd_client_id=os.getenv("ICD_CLIENT_ID") or None,
        icd_client_secret=os.getenv("ICD_CLIENT_SECRET") or None,
        icd_api_version=os.getenv("ICD_API_VERSION", "v2"),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "9000")),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _load()


# DISCLAIMER is attached to every diagnosis, rendered in the UI, and stated in
# the README: this is an educational tool, not a medical device.
DISCLAIMER = (
    "open-medi is an educational demonstration, not a medical device. "
    "Suggestions are statistical guesses over a small public dataset and an "
    "AI language model. Always consult a qualified healthcare professional."
)
