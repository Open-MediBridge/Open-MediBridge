"""RAG diagnosis: ground the LLM in retrieved knowledge-base chunks and
parse its answer into a stable JSON shape."""

import json
import re

from open_medi.config import DISCLAIMER, Settings, get_settings
from open_medi.retriever import Hit

SYSTEM_PROMPT = """You are open-medi, a clinical decision SUPPORT assistant for educational use.
You receive a person's reported symptoms (typed, or transcribed from speech, possibly
imperfectly) and numbered excerpts retrieved from a symptom-disease knowledge base.

Strict rules:
- Base candidate conditions ONLY on the retrieved excerpts. Never add conditions from
  your own knowledge. If nothing fits well, return an empty "conditions" list and say so
  in the summary.
- Use each condition's name EXACTLY as written in the excerpts (no rewording, no
  expansions in parentheses beyond what the excerpt shows).
- "confidence" is a rough 0-100 estimate of how well the retrieved excerpt matches the
  reported symptoms: 60+ when the core reported symptoms appear in the excerpt,
  30-59 for a partial match, 1-29 for a weak one, 0 only when nothing fits.
- Mention which reported symptoms matched (use the person's own words where possible).
- Always list "red_flags": symptoms from the report (or excerpts) that would warrant
  urgent or emergency care. Use [] if none apply.
- Always give cautious, general "advice" and recommend seeing a healthcare professional.
- Reply with ONE JSON object and nothing else (no markdown fences), shaped exactly:
{
  "summary": "one or two sentences",
  "conditions": [
    {"name": "...", "confidence": 0, "matched_symptoms": ["..."], "rationale": "..."}
  ],
  "red_flags": ["..."],
  "advice": "..."
}"""

MAX_CONDITIONS = 5


def build_context(hits: list[Hit]) -> str:
    """Render retrieved hits as a numbered context block, deduplicated."""
    blocks, seen = [], set()
    for hit in hits:
        key = (hit.disease_id, hit.kind)
        if key in seen:
            continue
        seen.add(key)
        blocks.append(f"[{len(blocks) + 1}] {hit.text}")
    return "\n\n".join(blocks)


def build_user_prompt(symptom_text: str, hits: list[Hit], settings: Settings | None = None) -> str:
    s = settings or get_settings()
    context = build_context(hits)
    budget = s.max_context_chars
    if len(context) > budget:
        context = context[:budget] + "\n[context truncated]"
    return (
        f"Reported symptoms / transcript:\n\"\"\"\n{symptom_text.strip()}\n\"\"\"\n\n"
        f"Retrieved knowledge-base excerpts:\n{context}\n\n"
        "Respond with the JSON object described in the system prompt."
    )


def parse_diagnosis(raw: str) -> dict:
    """Extract and normalize the LLM's JSON answer; always attach the disclaimer."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("LLM reply contained no JSON object.")
    payload = json.loads(match.group(0))

    conditions = []
    for cond in (payload.get("conditions") or [])[:MAX_CONDITIONS]:
        if not isinstance(cond, dict) or not cond.get("name"):
            continue
        try:
            confidence = int(round(float(cond.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0
        matched = cond.get("matched_symptoms") or []
        conditions.append(
            {
                "name": str(cond["name"]).strip(),
                "confidence": max(0, min(100, confidence)),
                "matched_symptoms": [str(m) for m in matched][:12],
                "rationale": str(cond.get("rationale", "")).strip(),
            }
        )
    conditions.sort(key=lambda c: c["confidence"], reverse=True)

    red_flags = payload.get("red_flags") or []
    return {
        "summary": str(payload.get("summary", "")).strip(),
        "conditions": conditions,
        "red_flags": [str(f) for f in red_flags][:10],
        "advice": str(payload.get("advice", "")).strip(),
        "disclaimer": DISCLAIMER,
    }
