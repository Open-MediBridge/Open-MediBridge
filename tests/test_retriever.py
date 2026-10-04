"""Offline unit tests: knowledge base writes, cosine retrieval, and diagnosis
parsing — all with a deterministic fake embedder (no models downloaded)."""

import hashlib

import numpy as np
import pytest

from open_medi import db
from open_medi.diagnosis import build_context, build_user_prompt, parse_diagnosis
from open_medi.retriever import Hit, Retriever

DIM = 64


def fake_embed(text: str) -> list[float]:
    """Bag-of-words hashing embedding: shared words -> shared directions."""
    vec = np.zeros(DIM, dtype=np.float32)
    for word in text.lower().split():
        word = word.strip("—:,()." + "\"'")
        if word:
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = np.linalg.norm(vec)
    return (vec / norm if norm else vec).tolist()


@pytest.fixture
def kb(tmp_path):
    conn = db.connect(tmp_path / "kb.db")
    db.init_schema(conn)
    cur = conn.cursor()
    rows = [
        (
            "Common Cold",
            "A mild viral infection of the nose and throat.",
            "Common Cold — presenting symptoms: runny nose, sneezing, sore throat, mild cough.",
        ),
        (
            "Acne",
            "A skin condition with pimples and oily skin.",
            "Acne — presenting symptoms: pimples on face, oily skin, blackheads.",
        ),
    ]
    for name, description, chunk_text in rows:
        cur.execute(
            "INSERT INTO diseases(name, description, precautions, source) VALUES(?, ?, '[]', 'test')",
            (name, description),
        )
        disease_id = cur.lastrowid
        cur.execute(
            "INSERT INTO chunks(disease_id, kind, text, embedding) VALUES(?, 'symptoms', ?, ?)",
            (disease_id, chunk_text, db.pack_embedding(fake_embed(chunk_text))),
        )
    conn.commit()
    yield conn
    conn.close()


def test_retrieval_ranks_matching_disease_first(kb):
    retriever = Retriever(kb, embed_fn=fake_embed)
    hits = retriever.search("my nose is runny and I keep sneezing", top_k=2)
    assert len(hits) == 2
    assert hits[0].disease_name == "Common Cold"
    assert hits[0].score > hits[1].score


def test_retrieval_respects_top_k_and_empty_index(tmp_path):
    conn = db.connect(tmp_path / "empty.db")
    db.init_schema(conn)
    retriever = Retriever(conn, embed_fn=fake_embed)
    assert retriever.search("anything") == []


def test_build_context_deduplicates_disease_and_kind():
    hits = [
        Hit(1, "Migraine", "symptoms", "Migraine — symptoms: headache", 0.9),
        Hit(1, "Migraine", "symptoms", "Migraine — symptoms: headache (dup)", 0.8),
        Hit(1, "Migraine", "description", "Migraine: a headache disorder", 0.7),
    ]
    context = build_context(hits)
    assert context.count("[1]") == 1
    assert context.count("[2]") == 1
    assert "dup" not in context


def test_build_user_prompt_includes_symptoms_and_context(kb):
    retriever = Retriever(kb, embed_fn=fake_embed)
    prompt = build_user_prompt("runny nose and sneezing", retriever.search("runny nose and sneezing"))
    assert "runny nose and sneezing" in prompt
    assert "Common Cold" in prompt


def test_parse_diagnosis_normalizes_and_sorts():
    raw = """
    Sure! Here is my answer:
    ```json
    {"summary": "s", "conditions": [
        {"name": "A", "confidence": 150, "matched_symptoms": ["x"], "rationale": "r"},
        {"name": "B", "confidence": 40, "matched_symptoms": [], "rationale": ""},
        {"bad entry": true}
    ], "red_flags": ["chest pain"], "advice": "see a doctor"}
    ```
    """
    result = parse_diagnosis(raw)
    assert [c["name"] for c in result["conditions"]] == ["A", "B"]
    assert result["conditions"][0]["confidence"] == 100  # clamped
    assert result["red_flags"] == ["chest pain"]
    assert "not a medical device" in result["disclaimer"]


def test_parse_diagnosis_rejects_non_json():
    with pytest.raises(ValueError):
        parse_diagnosis("I cannot help with that.")
