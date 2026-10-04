"""FastAPI application: transcription, RAG diagnosis, ICD-11 coding, web UI."""

import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from open_medi import db
from open_medi.config import DISCLAIMER, get_settings
from open_medi.diagnosis import SYSTEM_PROMPT, build_user_prompt, parse_diagnosis
from open_medi.icd import ICDClient
from open_medi.llm import LLM, LLMError
from open_medi.retriever import Retriever
from open_medi.transcribe import model_loaded, transcribe_upload

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    conn = db.connect(settings.db_path)
    db.init_schema(conn)
    app.state.conn = conn
    app.state.settings = settings
    app.state.retriever = Retriever(conn) if db.is_ready(conn) else None
    app.state.llm = LLM(settings)
    app.state.icd = ICDClient(
        settings.icd_client_id, settings.icd_client_secret, settings.icd_api_version
    )
    yield
    conn.close()


app = FastAPI(
    title="open-medi",
    version="0.1.0",
    description=(
        "Voice-powered clinical decision *support* (educational demo, not a "
        "medical device): faster-whisper transcription + RAG over a "
        "symptom-disease knowledge base + OpenAI-compatible LLM, with WHO "
        "ICD-11 coding."
    ),
    lifespan=lifespan,
)

# Same-origin UI needs no CORS; open it up for local frontend development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DiagnoseRequest(BaseModel):
    text: str = Field(min_length=2, description="Symptoms, typed or transcribed")
    top_k: int | None = Field(default=None, ge=1, le=20)


class Condition(BaseModel):
    name: str
    icd11_code: str | None = None
    confidence: int
    matched_symptoms: list[str] = []
    rationale: str = ""


class Diagnosis(BaseModel):
    summary: str
    conditions: list[Condition] = []
    red_flags: list[str] = []
    advice: str
    disclaimer: str = DISCLAIMER


def _require_ready() -> Retriever:
    retriever = app.state.retriever
    if retriever is None or len(retriever.index) == 0:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is empty. Build it first: python scripts/build_db.py",
        )
    return retriever


def _icd_variants(name: str) -> set[str]:
    """Normalization variants for fuzzy disease-name -> ICD code matching,
    since LLMs sometimes reword retrieved condition names."""
    lowered = " ".join(name.strip().lower().split())
    no_parens = " ".join(re.sub(r"\([^)]*\)", " ", lowered).split())
    return {v for v in (lowered, no_parens) if v}


def _attach_icd_codes(conn, diagnosis: dict) -> dict:
    """Look up stored ICD-11 codes for each condition name (deterministic,
    so the LLM never invents codes)."""
    rows = conn.execute("SELECT name, icd11_code FROM diseases WHERE icd11_code IS NOT NULL").fetchall()
    by_variant: dict[str, str] = {}
    for row in rows:
        code = row["icd11_code"]
        for variant in _icd_variants(row["name"]):
            by_variant.setdefault(variant, code)
    for condition in diagnosis.get("conditions", []):
        variants = _icd_variants(condition["name"])
        code = next((by_variant[v] for v in variants if v in by_variant), None)
        if code is None:
            # last resort: containment either way (e.g. LLM wrote
            # "gastroesophageal reflux disease (gerd)" vs stored "GERD")
            code = next(
                (
                    stored
                    for stored in by_variant
                    if stored in next(iter(variants), "") or next(iter(variants), "") in stored
                ),
                None,
            )
        condition["icd11_code"] = code
    return diagnosis


def _diagnose(text: str, top_k: int | None) -> dict:
    retriever = _require_ready()
    hits = retriever.search(text, top_k=top_k)
    if not hits:
        raise HTTPException(status_code=503, detail="Retrieval returned no context.")
    prompt = build_user_prompt(text, hits)
    try:
        raw = app.state.llm.complete(SYSTEM_PROMPT, prompt)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    try:
        diagnosis = parse_diagnosis(raw)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Unparseable LLM reply: {exc}") from exc
    return _attach_icd_codes(app.state.conn, diagnosis)


# --- endpoints ---------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    conn = app.state.conn
    settings = app.state.settings
    ready = db.is_ready(conn)
    counts = (
        {
            "diseases": conn.execute("SELECT COUNT(*) c FROM diseases").fetchone()["c"],
            "chunks": conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"],
        }
        if ready
        else {}
    )
    return {
        "status": "ok" if ready else "needs-knowledge-base",
        "knowledge_base": {
            "path": str(settings.db_path),
            "ready": ready,
            **counts,
        },
        "whisper": {
            "model": settings.whisper_model,
            "device": settings.whisper_device,
            "compute_type": settings.whisper_compute_type,
            "loaded": model_loaded(),
        },
        "llm": {
            "base_url": settings.llm_base_url or "(OpenAI default)",
            "model": settings.llm_model,
        },
        "icd11": {"configured": app.state.icd.configured},
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Empty upload.")
    try:
        transcript = transcribe_upload(data, filename=file.filename or "audio.webm", language=language)
    except Exception as exc:  # noqa: BLE001 - decoding failures come from many libs
        raise HTTPException(status_code=415, detail=f"Could not decode audio: {exc}") from exc
    return transcript.__dict__


@app.post("/api/diagnose")
def diagnose(request: DiagnoseRequest) -> dict:
    return _diagnose(request.text, request.top_k)


@app.post("/api/pipeline")
async def pipeline(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Empty upload.")
    try:
        transcript = transcribe_upload(data, filename=file.filename or "audio.webm", language=language)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=415, detail=f"Could not decode audio: {exc}") from exc
    if not transcript.text:
        return {
            "transcript": transcript.__dict__,
            "diagnosis": None,
            "disclaimer": DISCLAIMER,
        }
    return {
        "transcript": transcript.__dict__,
        "diagnosis": _diagnose(transcript.text, top_k=None),
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/diseases")
def diseases() -> list[dict]:
    rows = app.state.conn.execute(
        "SELECT id, name, icd11_code FROM diseases ORDER BY name"
    ).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/icd/search")
def icd_search(q: str) -> dict:
    """Live search against the WHO ICD-11 API (requires credentials in .env)."""
    client: ICDClient = app.state.icd
    if not client.configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "ICD-11 API is not configured. Register free credentials at "
                "https://icd.who.int/icdapi and set ICD_CLIENT_ID / "
                "ICD_CLIENT_SECRET in .env."
            ),
        )
    try:
        return {"query": q, "results": client.search(q)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"ICD-11 API request failed: {exc}") from exc
