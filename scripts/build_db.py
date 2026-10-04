#!/usr/bin/env python
"""Build the open-medi SQLite knowledge base.

Sources, in order of preference:
  1. CSVs in --data-dir (dataset.csv, symptom_Description.csv,
     symptom_precaution.csv, Symptom-severity.csv)
  2. The public mirror at github.com/TheBlueZayn/Disease-Symptom-Prediction
     (same data as the Kaggle "Disease Symptom Description" dataset:
     41 diseases, 132 symptoms, descriptions, precautions, severities)
  3. The bundled data/seed_fallback.json (offline, ~12 conditions)

ICD-11 codes: taken from data/icd11_hints.json (best-effort, marked 'hint'),
then overridden by live WHO ICD-11 API lookups when ICD_CLIENT_ID /
ICD_CLIENT_SECRET are set (marked 'icd11-api').

Usage:
    python scripts/build_db.py [--data-dir DIR] [--seed FILE] [--offline]
                               [--skip-icd-live] [--db PATH]
"""

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from open_medi import db  # noqa: E402
from open_medi.config import get_settings  # noqa: E402
from open_medi.embeddings import embed_texts  # noqa: E402

DATASET_BASES = [
    "https://raw.githubusercontent.com/TheBlueZayn/Disease-Symptom-Prediction/main",
    "https://raw.githubusercontent.com/TheBlueZayn/Disease-Symptom-Prediction/master",
]
DATASET_FILES = [
    "dataset.csv",
    "symptom_Description.csv",
    "symptom_precaution.csv",
    "Symptom-severity.csv",
]
SEED_PATH = PROJECT_ROOT / "data" / "seed_fallback.json"
HINTS_PATH = PROJECT_ROOT / "data" / "icd11_hints.json"

# Clean up known typos / casing quirks in the public dataset.
NAME_FIXES = {
    "peptic ulcer diseae": "Peptic ulcer disease",
    "dimorphic hemmorhoids(piles)": "Dimorphic hemorrhoids (piles)",
    "osteoarthristis": "Osteoarthritis",
    "(vertigo) paroymsal positional vertigo": "Paroxysmal positional vertigo",
    "diabetes": "Diabetes mellitus",
    "hypoglycemia": "Hypoglycemia",
    "tb": "Tuberculosis",
}


def normalize_disease_name(raw: str) -> str:
    name = " ".join(str(raw).split()).strip()
    return NAME_FIXES.get(name.lower(), name)


def normalize_symptom(raw) -> str | None:
    text = " ".join(str(raw).strip().lower().replace("_", " ").split())
    if not text or text in {"nan", "none"}:
        return None
    return text


def fetch_csv(name: str) -> str | None:
    for base in DATASET_BASES:
        url = f"{base}/{name.replace(' ', '%20')}"
        try:
            print(f"  downloading {url} ...")
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
                return response.read().decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"    failed: {exc}")
    return None


def parse_dataset_csvs(csv_texts: dict[str, str]) -> dict:
    records: dict[str, dict] = {}

    def entry(name: str) -> dict:
        return records.setdefault(
            name,
            {"symptoms": set(), "description": "", "precautions": [],
             "icd11_code": None, "icd11_title": None, "source": "kaggle disease-symptom-description"},
        )

    reader = csv.DictReader(io.StringIO(csv_texts["dataset.csv"]))
    for row in reader:
        name = normalize_disease_name(row.get("Disease", ""))
        if not name:
            continue
        rec = entry(name)
        for i in range(1, 18):
            symptom = normalize_symptom(row.get(f"Symptom_{i}"))
            if symptom:
                rec["symptoms"].add(symptom)

    reader = csv.DictReader(io.StringIO(csv_texts["symptom_Description.csv"]))
    for row in reader:
        name = normalize_disease_name(row.get("Disease", ""))
        description = (row.get("Description") or "").strip()
        if name and description:
            entry(name)["description"] = description

    reader = csv.DictReader(io.StringIO(csv_texts["symptom_precaution.csv"]))
    for row in reader:
        name = normalize_disease_name(row.get("Disease", ""))
        if not name:
            continue
        rec = entry(name)
        for i in range(1, 5):
            step = (row.get(f"Precaution_{i}") or "").strip()
            if step and step.lower() != "nan":
                rec["precautions"].append(step)

    severity = {}
    reader = csv.DictReader(io.StringIO(csv_texts["Symptom-severity.csv"]))
    for row in reader:
        symptom = normalize_symptom(row.get("Symptom"))
        if symptom and row.get("weight"):
            severity[symptom] = int(row["weight"])
    return records, severity


def parse_seed_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    for cond in payload.get("conditions", []):
        name = normalize_disease_name(cond["name"])
        records[name] = {
            "symptoms": {normalize_symptom(s) for s in cond.get("symptoms", [])} - {None},
            "description": cond.get("description", ""),
            "precautions": cond.get("precautions", []),
            "icd11_code": cond.get("icd11_code"),
            "icd11_title": cond.get("icd11_title"),
            "source": cond.get("source", "seed"),
        }
    return records


def apply_icd(records: dict, skip_live: bool) -> None:
    hints = json.loads(HINTS_PATH.read_text(encoding="utf-8")).get("codes", {})
    client = None
    settings = get_settings()
    if not skip_live and settings.icd_client_id and settings.icd_client_secret:
        from open_medi.icd import ICDClient

        client = ICDClient(settings.icd_client_id, settings.icd_client_secret,
                           settings.icd_api_version)
        print("ICD-11: live WHO API credentials found; enriching ...")
    else:
        print("ICD-11: using bundled hints only "
              "(set ICD_CLIENT_ID / ICD_CLIENT_SECRET for live WHO lookups).")

    for name, rec in sorted(records.items()):
        if hint := hints.get(name):
            rec["icd11_code"], rec["icd11_title"] = hint["code"], hint["title"]
        if client is None:
            continue
        live = client.lookup(name)
        time.sleep(0.4)  # be polite to the WHO endpoint
        if live and live.get("code"):
            rec["icd11_code"] = live["code"]
            rec["icd11_title"] = live["title"] or rec["icd11_title"]


def build_chunks(records: dict, severity: dict[str, int]) -> list[dict]:
    chunks = []
    for name, rec in records.items():
        symptom_bits = [
            f"{s} (severity {severity[s]}/7)" if severity.get(s) else s
            for s in sorted(rec["symptoms"])
        ]
        icd_bit = f" ICD-11 code {rec['icd11_code']}." if rec.get("icd11_code") else ""
        if symptom_bits:
            chunks.append({
                "disease": name, "kind": "symptoms",
                "text": f"{name} — presenting symptoms: " + ", ".join(symptom_bits) + ".",
            })
        if rec["description"]:
            chunks.append({
                "disease": name, "kind": "description",
                "text": f"{name}: {rec['description']}{icd_bit}",
            })
        if rec["precautions"]:
            chunks.append({
                "disease": name, "kind": "precautions",
                "text": f"Management and precautions for {name}: "
                        + "; ".join(rec["precautions"]) + ".",
            })
    return chunks


def write_db(conn, records: dict, chunks: list[dict], vectors: list[list[float]],
             embed_model: str, source_label: str) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks")
    cur.execute("DELETE FROM disease_symptoms")
    cur.execute("DELETE FROM symptoms")
    cur.execute("DELETE FROM diseases")

    disease_ids: dict[str, int] = {}
    symptom_ids: dict[str, int] = {}
    chunk_index = 0
    for name, rec in records.items():
        cur.execute(
            "INSERT INTO diseases(name, description, precautions, icd11_code, icd11_title, source) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (name, rec["description"], json.dumps(rec["precautions"]),
             rec.get("icd11_code"), rec.get("icd11_title"), rec["source"]),
        )
        disease_ids[name] = cur.lastrowid
        for symptom in rec["symptoms"]:
            if symptom not in symptom_ids:
                cur.execute("INSERT INTO symptoms(name, severity) VALUES(?, 0)", (symptom,))
                symptom_ids[symptom] = cur.lastrowid
            cur.execute(
                "INSERT OR IGNORE INTO disease_symptoms(disease_id, symptom_id) VALUES(?, ?)",
                (disease_ids[name], symptom_ids[symptom]),
            )

    for chunk, vector in zip(chunks, vectors):
        cur.execute(
            "INSERT INTO chunks(disease_id, kind, text, embedding) VALUES(?, ?, ?, ?)",
            (disease_ids[chunk["disease"]], chunk["kind"], chunk["text"],
             db.pack_embedding(vector)),
        )
        chunk_index += 1

    db.set_meta(conn, "embed_model", embed_model)
    db.set_meta(conn, "dim", str(len(vectors[0])))
    db.set_meta(conn, "built_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    db.set_meta(conn, "dataset_source", source_label)
    conn.commit()
    print(f"  wrote {len(records)} diseases, {len(symptom_ids)} symptoms, "
          f"{chunk_index} chunks -> {conn.execute('PRAGMA database_list').fetchone()[2]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", help="output SQLite path (default from settings)")
    parser.add_argument("--data-dir", help="directory holding the four dataset CSVs")
    parser.add_argument("--seed", help="custom seed JSON with the same schema as seed_fallback.json")
    parser.add_argument("--offline", action="store_true", help="skip downloads, use the seed")
    parser.add_argument("--skip-icd-live", action="store_true",
                        help="do not call the WHO ICD-11 API even if credentials exist")
    args = parser.parse_args()

    settings = get_settings()
    db_path = Path(args.db) if args.db else settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    records: dict = {}
    severity: dict[str, int] = {}
    source_label = "seed_fallback"

    csv_texts: dict[str, str] | None = None
    if args.data_dir:
        data_dir = Path(args.data_dir)
        csv_texts = {
            f: (data_dir / f).read_text(encoding="utf-8-sig")
            for f in DATASET_FILES if (data_dir / f).exists()
        }
        if len(csv_texts) != len(DATASET_FILES):
            print(f"data-dir {data_dir} is missing files: "
                  f"{sorted(set(DATASET_FILES) - set(csv_texts))}")
            csv_texts = None
    elif not args.offline:
        print("Downloading disease-symptom dataset (Kaggle mirror) ...")
        csv_texts = {f: fetch_csv(f) for f in DATASET_FILES}
        if any(text is None for text in csv_texts.values()):
            csv_texts = None

    if csv_texts:
        records, severity = parse_dataset_csvs(csv_texts)
        source_label = "kaggle disease-symptom-description (41 diseases)"
    else:
        reason = "--offline was set" if args.offline else "dataset download unavailable"
        print(f"{reason}; falling back to bundled seed "
              f"({args.seed or SEED_PATH}).")
        records = parse_seed_json(Path(args.seed) if args.seed else SEED_PATH)

    if not records:
        print("No knowledge records could be loaded.", file=sys.stderr)
        return 1

    apply_icd(records, skip_live=args.skip_icd_live)

    chunks = build_chunks(records, severity)
    print(f"Embedding {len(chunks)} chunks with {settings.embed_model} "
          "(first run downloads model weights) ...")
    vectors = embed_texts([c["text"] for c in chunks])
    if len({len(v) for v in vectors}) != 1:
        print("Inconsistent embedding dimensions returned.", file=sys.stderr)
        return 1

    conn = db.connect(db_path)
    try:
        db.init_schema(conn)
        write_db(conn, records, chunks, vectors, settings.embed_model, source_label)
    finally:
        conn.close()

    print("Done. Start the app with: uvicorn open_medi.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
