"""Retrieve the most relevant knowledge-base chunks for a symptom query.

Cosine similarity over the in-memory chunk matrix. The embedder is injected
as a callable (defaulting to the local fastembed model) so tests can swap in
a deterministic fake.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from open_medi import db
from open_medi.config import get_settings


@dataclass
class Hit:
    disease_id: int
    disease_name: str
    kind: str          # 'symptoms' | 'description' | 'precautions'
    text: str
    score: float


class Retriever:
    def __init__(
        self,
        conn,
        embed_fn: Callable[[str], list[float]] | None = None,
    ):
        self.conn = conn
        self.embed_fn = embed_fn or _default_embed
        self.index = db.ChunkIndex(conn)
        self._diseases: dict[int, str] = {
            row["id"]: row["name"]
            for row in conn.execute("SELECT id, name FROM diseases")
        }

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or get_settings().top_k
        if len(self.index) == 0:
            return []
        q = np.asarray(self.embed_fn(query), dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.index.matrix @ q
        order = np.argsort(-scores)[:top_k]
        hits = []
        for i in order:
            disease_id = int(self.index.disease_ids[i])
            hits.append(
                Hit(
                    disease_id=disease_id,
                    disease_name=self._diseases.get(disease_id, f"disease#{disease_id}"),
                    kind=self.index.kinds[i],
                    text=self.index.texts[i],
                    score=float(scores[i]),
                )
            )
        return hits


def _default_embed(text: str) -> list[float]:
    from open_medi import embeddings

    return embeddings.embed_query(text)
