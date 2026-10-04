# ⚕ open-medi

**Voice-powered clinical decision support. Educational demo, not a medical device.**

Describe your symptoms out loud (or type them) and open-medi will:

1. **Transcribe** your speech with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   (`large-v3-turbo`, CTranslate2; int8 on CPU and float16 on CUDA, auto-detected).
2. **Retrieve** the most relevant condition knowledge from a local SQLite
   knowledge base (symptom → disease dataset + WHO ICD-11 codes) using
   local sentence embeddings (fastembed / ONNX, no torch).
3. **Reason** over the retrieved excerpts with any OpenAI-compatible LLM
   (local [Ollama](https://ollama.com) by default; OpenAI / Groq / Z.ai / vLLM
   work by editing `.env`) and return candidate conditions, matched symptoms,
   red flags, and general advice.
4. **Code** each candidate with its **WHO ICD-11** code, looked up from the
   knowledge base (enriched live from the [ICD-11 API](https://icd.who.int/icdapi)
   when credentials are configured, so the LLM never invents codes.

```
 mic ──▶ /api/pipeline ──▶ faster-whisper ──▶ "text" ──▶ embeddings ──▶ SQLite RAG (top-k)
                                                                        │
                                              JSON diagnosis ◀── LLM ◀──┘
                                                    │
                                       conditions + ICD-11 codes ◀── diseases table
```

---

## Quickstart (Windows)

```bat
cd E:\open-medi\open-medi

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

copy .env.example .env

:: build the knowledge base (downloads the public dataset + embedding model)
.venv\Scripts\python scripts\build_db.py

:: start the API + web UI
.venv\Scripts\uvicorn open_medi.main:app --reload
```

Open **http://127.0.0.1:8000**, tap the mic, and describe how you feel, or
type symptoms into the text box.

First-run downloads (cached afterwards):

| What | Size | When |
|---|---|---|
| Embedding model (`all-MiniLM-L6-v2`, ONNX) | ~90 MB | `build_db.py` |
| Dataset CSVs (41 diseases / 132 symptoms) | ~600 KB | `build_db.py` |
| Whisper `large-v3-turbo` weights | ~1.6 GB | first transcription |

> Tip: set `WHISPER_MODEL=small` in `.env` for a quick smoke test before
> committing to the full turbo download. A spoken sample,
> `sample_symptoms.wav`, is included for testing without a microphone:
> `curl -F "file=@sample_symptoms.wav" http://127.0.0.1:8000/api/pipeline`

#### GPU acceleration

`WHISPER_DEVICE=auto` uses the NVIDIA GPU when one is present, but on Windows
CTranslate2 also needs the cuBLAS/cuDNN runtime DLLs. Install them as wheels
and the app registers their locations automatically:

```bat
.venv\Scripts\pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

If you manage the environment with `uv` and still hit
`Library cublas64_12.dll is not found`, prepend the wheel bin folders to
`PATH` before starting the server:

```powershell
uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

$cudaLibs = @(
    "$(uv run python -c "import nvidia.cublas; print(nvidia.cublas.__path__[0])")\bin"
    "$(uv run python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")\bin"
) -join ';'
$env:PATH = "$cudaLibs;$env:PATH"
```

On a CPU-only machine, nothing to do: CTranslate2 falls back to int8 on CPU.

### LLM backend

The default `.env` expects a local [Ollama](https://ollama.com) server:

```bat
ollama pull llama3.2
ollama serve
```

To use a hosted provider instead, edit the `LLM_*` lines in `.env`
(ready-made blocks for OpenAI, Groq, and Z.ai are commented in
`.env.example`). **No code changes needed.**

### WHO ICD-11 API (optional but recommended)

The scaffold ships with best-effort ICD-11 codes (`data/icd11_hints.json`).
To enrich/override them with the official WHO data:

1. Register (free) at <https://icd.who.int/icdapi> to get client credentials.
2. Add to `.env`:
   ```
   ICD_CLIENT_ID=...
   ICD_CLIENT_SECRET=...
   ```
3. Re-run `python scripts\build_db.py`; every disease is looked up live
   (codes stored with source `icd11-api`, hints stay marked `hint`).

`GET /api/icd/search?q=...` provides live coding search from the UI.

> Verify hint codes against the API before any real-world use; they are
> assigned from the ICD-11 MMS classification and provided as a convenience.

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web UI |
| `/api/health` | GET | Knowledge base / model / LLM / ICD status |
| `/api/transcribe` | POST | multipart `file` (+ optional `language`) → transcript JSON |
| `/api/diagnose` | POST | `{"text": "...", "top_k": 6}` → grounded diagnosis JSON |
| `/api/pipeline` | POST | multipart `file` → `{transcript, diagnosis}` in one call |
| `/api/diseases` | GET | Browse the knowledge base |
| `/api/icd/search` | GET | Live WHO ICD-11 coding search |

```bash
# transcribe + diagnose in one call
curl -s -F "file=@question.webm" http://127.0.0.1:8000/api/pipeline

# text-only diagnosis
curl -s http://127.0.0.1:8000/api/diagnose \
  -H "Content-Type: application/json" \
  -d '{"text": "burning chest after meals and a sour taste"}'
```

Diagnosis response shape:

```json
{
  "summary": "…",
  "conditions": [
    {"name": "Gastroesophageal Reflux Disease", "icd11_code": null,
     "confidence": 62, "matched_symptoms": ["burning chest"], "rationale": "…"}
  ],
  "red_flags": ["…"],
  "advice": "…",
  "disclaimer": "…"
}
```

`confidence` is the LLM's estimate, not a calibrated probability.

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` | any faster-whisper model id |
| `WHISPER_DEVICE` | `auto` | `cpu` / `cuda` to force |
| `WHISPER_COMPUTE_TYPE` | `auto` | `int8` / `float16` / `float32` … |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | any fastembed model |
| `DB_PATH` | `data/open_medi.db` | knowledge base location |
| `TOP_K` | `6` | retrieved chunks per query |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama; any OpenAI-compatible |
| `LLM_API_KEY` | `ollama` | real key for hosted providers |
| `LLM_MODEL` | `llama3.2` | |
| `LLM_TEMPERATURE` | `0.2` | keep low for grounded answers |
| `ICD_CLIENT_ID` / `ICD_CLIENT_SECRET` | unset | WHO ICD-11 API credentials |
| `MAX_CONTEXT_CHARS` | `9000` | RAG prompt budget |

## Extending the knowledge base

```bash
# point at your own copies of the four CSVs (Kaggle schema)
python scripts/build_db.py --data-dir path\to\csvs

# or a custom JSON with the same schema as data/seed_fallback.json
python scripts/build_db.py --seed path\to\my.json

# rebuild fully offline
python scripts/build_db.py --offline --skip-icd-live
```

The SQLite schema (`open_medi/db.py`) stores diseases, symptoms with
severities, relations, and text chunks with float32 embedding BLOBs.
Retrieval is numpy cosine similarity in memory, instant at this scale;
for >~100k chunks swap in [sqlite-vec](https://github.com/asg017/sqlite-vec)
behind the same `Retriever` interface.

## Tests

```bash
.venv\Scripts\python -m pytest
```

The suite is fully offline: a deterministic hashing embedder stands in for
the ONNX model, and a fake LLM stands in for the backend.

## Attribution & safety

- Symptom/disease data: Kaggle *“Disease Symptom Description”* dataset
  ([itachi9604](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset),
  mirrored on [GitHub](https://github.com/TheBlueZayn/Disease-Symptom-Prediction)),
  used here for research and educational purposes.
- Disease classification: WHO **ICD-11** via the ICD API
  (<https://icd.who.int/icdapi>), © WHO. Verify codes before production use.
- **This project is not a medical device.** It does not diagnose, treat, or
  replace professional judgement, and its outputs are uncalibrated statistical
  guesses. Every API response and UI screen carries this disclaimer.

## Project structure

```
open-medi/
├── open_medi/
│   ├── config.py        # env-driven settings (dotenv)
│   ├── db.py            # SQLite schema + chunk index
│   ├── embeddings.py    # fastembed (ONNX) wrapper
│   ├── retriever.py     # cosine top-k retrieval
│   ├── llm.py           # OpenAI-compatible client (Ollama/OpenAI/Groq/Z.ai)
│   ├── transcribe.py    # faster-whisper large-v3-turbo wrapper
│   ├── diagnosis.py     # RAG prompt + strict JSON parsing
│   ├── icd.py           # WHO ICD-11 API client (OAuth2 client credentials)
│   └── main.py          # FastAPI app + static UI
├── scripts/build_db.py  # dataset → SQLite (+ embeddings, + ICD-11)
├── data/                # seed_fallback.json, icd11_hints.json, open_medi.db (built)
├── web/                 # index.html / app.js / style.css (no build step)
└── tests/
```
