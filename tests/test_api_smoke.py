"""API smoke test with a fake LLM and fake embedder — no models, no network."""

import json

import pytest
from fastapi.testclient import TestClient

import open_medi.main as main_module
from open_medi import db
from open_medi.config import get_settings
from test_retriever import fake_embed

FAKE_LLM_REPLY = json.dumps(
    {
        "summary": "Symptoms suggest a common cold.",
        "conditions": [
            {
                "name": "Common Cold",
                "confidence": 70,
                "matched_symptoms": ["runny nose", "sneezing"],
                "rationale": "Matches the common cold symptom excerpt.",
            }
        ],
        "red_flags": [],
        "advice": "Rest and hydrate; see a doctor if it worsens.",
    }
)


class FakeLLM:
    def __init__(self, *args, **kwargs):
        pass

    def complete(self, system: str, user: str) -> str:
        return FAKE_LLM_REPLY

    def reachable(self) -> bool:
        return True


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "kb.db"))
    get_settings.cache_clear()
    monkeypatch.setattr(main_module, "LLM", FakeLLM)
    monkeypatch.setattr("open_medi.retriever._default_embed", fake_embed)

    # Seed a minimal knowledge base into the settings' DB path.
    conn = db.connect(get_settings().db_path)
    db.init_schema(conn)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO diseases(name, description, precautions, icd11_code, source) "
        "VALUES('Common Cold', 'viral infection', '[]', 'CA00', 'test')"
    )
    cold_id = cur.lastrowid
    text = "Common Cold — presenting symptoms: runny nose, sneezing, sore throat."
    cur.execute(
        "INSERT INTO chunks(disease_id, kind, text, embedding) VALUES(?, 'symptoms', ?, ?)",
        (cold_id, text, db.pack_embedding(fake_embed(text))),
    )
    db.set_meta(conn, "embed_model", "fake-model")
    db.set_meta(conn, "dim", "64")
    conn.commit()
    conn.close()

    with TestClient(main_module.app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_health_reports_ready(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["knowledge_base"]["ready"] is True
    assert body["knowledge_base"]["diseases"] == 1
    assert body["whisper"]["model"]
    assert body["icd11"]["configured"] is False


def test_diagnose_attaches_icd_code(client):
    response = client.post(
        "/api/diagnose", json={"text": "I have a runny nose and sneezing"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["conditions"][0]["name"] == "Common Cold"
    assert body["conditions"][0]["icd11_code"] == "CA00"  # looked up from the DB
    assert "not a medical device" in body["disclaimer"]


def test_diagnose_rejects_short_text(client):
    assert client.post("/api/diagnose", json={"text": "h"}).status_code == 422


def test_icd_search_without_credentials_is_503(client):
    response = client.get("/api/icd/search", params={"q": "migraine"})
    assert response.status_code == 503
    assert "icd.who.int" in response.json()["detail"]
