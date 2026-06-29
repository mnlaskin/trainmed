#!/usr/bin/env python3
"""Build the cross-company `competitive_insights` collection.

This is the ONE place cross-company content lives. Each insight is a curated,
sourced comparison (e.g. "all-suture vs PEEK vs biocomposite anchors") used only for
explicit competitive comparison and competitive roleplay — it is NEVER mixed into a
single-company product answer (load_chunks(company) does not read this directory).

Input : data/competitive/insights.json  — a list of insight objects, e.g.
    {
      "topic": "Anchor materials: all-suture vs PEEK vs biocomposite",
      "category": "anchor",
      "companies": ["Arthrex", "Stryker", "SmithNephew"],
      "summary": "...",
      "arthrex_position": "...",
      "stryker_position": "...",
      "smithnephew_position": "...",
      "rep_talk_track": "...",
      "evidence": ["..."],
      "sources": ["https://..."],
      "confidence": "high",
      "notes": "verifier caveat, if any"
    }
Output: data/kb/competitive_insights/ci_NNN_<slug>.json  (chunk-shaped dicts)

Stdlib only. Idempotent: rewrites the collection from the input each run.

Usage:
    python scripts/ingest_competitive_insights.py
    python scripts/ingest_competitive_insights.py --input data/competitive/insights.json --clean
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "competitive" / "insights.json"
OUT_DIR = ROOT / "data" / "kb" / "competitive_insights"

# Sentinel company value. It is NOT a real company key, so load_chunks("Stryker")
# (which filters by exact company equality) can never sweep these in.
COMPETITIVE_COMPANY = "_competitive"

DISPLAY = {"Arthrex": "Arthrex", "Stryker": "Stryker", "SmithNephew": "Smith & Nephew"}


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "insight"


_POSITION_LABELS = [
    ("arthrex_position", "Arthrex"),
    ("stryker_position", "Stryker"),
    ("smithnephew_position", "Smith & Nephew"),
]


def _compose_text(ins: dict) -> str:
    parts = [ins["topic"].strip(), "", ins.get("summary", "").strip(), ""]
    for key, label in _POSITION_LABELS:
        val = (ins.get(key) or "").strip()
        if val:
            parts.append(f"{label}: {val}")
    if ins.get("rep_talk_track"):
        parts += ["", f"Rep positioning: {ins['rep_talk_track'].strip()}"]
    if ins.get("evidence"):
        parts += ["", "Evidence: " + " ".join(f"({i+1}) {e.strip()}" for i, e in enumerate(ins["evidence"]))]
    if ins.get("notes"):
        parts += ["", f"Note: {ins['notes'].strip()}"]
    return "\n".join(p for p in parts if p is not None)


def build(input_path: Path, clean: bool) -> dict:
    insights = json.loads(input_path.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if clean:
        for f in OUT_DIR.glob("*.json"):
            f.unlink()

    written = 0
    for idx, ins in enumerate(insights):
        cid = f"ci_{idx:03d}_{_slug(ins['topic'])}"
        text = _compose_text(ins)
        record = {
            "chunk_id": cid,
            "collection": "competitive",          # marker: cross-company, not a product corpus
            "company": COMPETITIVE_COMPANY,        # sentinel → never matches a real company filter
            "companies": ins.get("companies", []),
            "source_id": cid,
            "source_type": "competitive_insight",
            "source_title": ins["topic"],
            "source_url": (ins.get("sources") or [""])[0],
            "category": "competitive_insight",
            "product_line": ins.get("category", "Rotator Cuff"),
            "topic": ins["topic"],
            "summary": ins.get("summary", ""),
            "arthrex_position": ins.get("arthrex_position", ""),
            "stryker_position": ins.get("stryker_position", ""),
            "smithnephew_position": ins.get("smithnephew_position", ""),
            "rep_talk_track": ins.get("rep_talk_track", ""),
            "evidence": ins.get("evidence", []),
            "sources": ins.get("sources", []),
            "confidence": ins.get("confidence", "medium"),
            "notes": ins.get("notes", ""),
            "word_count": len(text.split()),
            "text": text,
        }
        (OUT_DIR / f"{cid}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written += 1
    return {"insights": len(insights), "written": written, "out_dir": OUT_DIR}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the competitive_insights collection.")
    p.add_argument("--input", default=str(DEFAULT_INPUT), help="insights JSON (list of objects)")
    p.add_argument("--clean", action="store_true", help="delete existing insight chunks first")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"No insights file at {input_path}. Create it first (see script docstring).")
        return 1

    result = build(input_path, args.clean)
    print(f"competitive_insights built: {result['written']} insight chunk(s) -> "
          f"{result['out_dir'].relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
