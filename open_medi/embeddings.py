"""Local text embeddings via fastembed (ONNX runtime, no torch required).

The default model (all-MiniLM-L6-v2) produces 384-dim vectors and runs on
CPU; weights are downloaded once on first use and cached locally.
"""

from functools import lru_cache
from typing import Iterable

from open_medi.config import get_settings


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def embed_texts(texts: Iterable[str], model_name: str | None = None) -> list[list[float]]:
    """Embed a batch of texts with the configured (or named) model."""
    name = model_name or get_settings().embed_model
    model = _get_model(name)
    return [vec.tolist() for vec in model.embed(list(texts))]


def embed_query(text: str, model_name: str | None = None) -> list[float]:
    return embed_texts([text], model_name)[0]
