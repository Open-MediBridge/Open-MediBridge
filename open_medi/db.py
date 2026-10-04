"""SQLite knowledge base: schema, connections, and chunk loading.

The knowledge base lives in a single SQLite file (default data/open_medi.db).
Embeddings are stored as little-endian float32 BLOBs and searched with numpy
cosine similarity (see retriever.py). At the starter dataset's scale (~150
chunks) a brute-force scan is instant; if the base grows past ~100k chunks,
swap in sqlite-vec (https://github.com/asg017/sqlite-vec) behind the same
Retriever interface.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS diseases (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,
    description  TEXT NOT NULL DEFAULT '',
    precautions  TEXT NOT NULL DEFAULT '[]',  -- JSON array of strings
    icd11_code   TEXT,                        -- WHO ICD-11 MMS code (may be NULL)
    icd11_title  TEXT,                        -- official ICD-11 title, when known
    source       TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS symptoms (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT NOT NULL UNIQUE,
    severity INTEGER NOT NULL DEFAULT 0      -- 1..7 from the dataset, 0 if unknown
);

CREATE TABLE IF NOT EXISTS disease_symptoms (
    disease_id INTEGER NOT NULL REFERENCES diseases(id) ON DELETE CASCADE,
    symptom_id INTEGER NOT NULL REFERENCES symptoms(id) ON DELETE CASCADE,
    PRIMARY KEY (disease_id, symptom_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    disease_id INTEGER NOT NULL REFERENCES diseases(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,               -- 'symptoms' | 'description' | 'precautions'
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL                -- little-endian float32 vector
);

CREATE INDEX IF NOT EXISTS idx_chunks_disease ON chunks(disease_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the knowledge base with sane pragmas.

    check_same_thread=False is required because FastAPI runs the lifespan and
    sync handlers in different threads; access stays serialized through this
    single short-lived-transaction connection.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def is_ready(conn: sqlite3.Connection) -> bool:
    """True once the base has been built (chunks + embedding metadata present)."""
    try:
        n = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    except sqlite3.OperationalError:
        return False
    return n > 0 and get_meta(conn, "embed_model") is not None


def get_disease(conn: sqlite3.Connection, disease_id: int) -> dict:
    row = conn.execute("SELECT * FROM diseases WHERE id = ?", (disease_id,)).fetchone()
    if row is None:
        return {}
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "precautions": json.loads(row["precautions"]),
        "icd11_code": row["icd11_code"],
        "icd11_title": row["icd11_title"],
        "source": row["source"],
    }


class ChunkIndex:
    """All chunk embeddings held in memory for cosine search."""

    def __init__(self, conn: sqlite3.Connection):
        rows = conn.execute(
            "SELECT id, disease_id, kind, text, embedding FROM chunks"
        ).fetchall()
        self.ids = np.array([r["id"] for r in rows], dtype=np.int64)
        self.disease_ids = np.array([r["disease_id"] for r in rows], dtype=np.int64)
        self.kinds = [r["kind"] for r in rows]
        self.texts = [r["text"] for r in rows]
        if rows:
            mat = np.stack([
                np.frombuffer(r["embedding"], dtype="<f4") for r in rows
            ])
        else:
            mat = np.zeros((0, 1), dtype=np.float32)
        self.matrix = _l2_normalize(mat)

    def __len__(self) -> int:
        return len(self.texts)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def pack_embedding(vec) -> bytes:
    """Serialize a vector for the embedding BLOB column."""
    return np.asarray(vec, dtype="<f4").tobytes()
