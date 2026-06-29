#!/usr/bin/env python3
"""Backward-compat migration: stamp the multi-company schema onto existing chunks.

The original KB (109 chunks in data/kb/chunks/) predates multi-company support — none
of those chunks carry a `company` field, and the firewall in trainmed.rag REFUSES to
guess. This script backfills, IN PLACE and idempotently:

  - company            (default: Arthrex — the only brand those chunks ever were)
  - product_line / product_family / product_name / category / technique
        derived deterministically via trainmed.companies.infer_product_fields, the
        SAME inference the live ingester uses, so a migrated chunk and a freshly
        ingested one are identical.
  - advantages / disadvantages / clinical_references   (empty lists if absent;
        existing values are preserved — never clobbered)

Idempotent: by default it only FILLS fields that are missing/empty, so re-running is a
no-op. Pass --reinfer to re-derive product fields after improving the taxonomy.

Usage:
    python scripts/migrate_add_company.py --dry-run        # preview, write nothing
    python scripts/migrate_add_company.py                  # backfill Arthrex (default)
    python scripts/migrate_add_company.py --company Arthrex --reinfer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trainmed import companies as co  # noqa: E402

KB_DIR = ROOT / "data" / "kb"
LEGACY_CHUNKS_DIR = KB_DIR / "chunks"

# Canonical field order so migrated files look exactly like freshly-ingested ones.
FIELD_ORDER = [
    "chunk_id", "company", "source_id", "source_type", "source_title", "source_url",
    "channel", "upload_date", "page_count", "language", "procedure_family",
    "product_line", "product_family", "product_name", "category", "technique",
    "advantages", "disadvantages", "clinical_references",
    "chunk_index", "chunk_count", "word_count", "timestamp_range", "topics", "text",
]


def chunks_dir_for(company: str) -> Path:
    company = co.canonical_company(company)
    return LEGACY_CHUNKS_DIR if company == co.DEFAULT_COMPANY else KB_DIR / company / "chunks"


def _reorder(record: dict) -> dict:
    """Canonical field order; any unexpected keys are appended at the end."""
    ordered = {k: record[k] for k in FIELD_ORDER if k in record}
    for k, v in record.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def migrate_chunk(record: dict, company: str, reinfer: bool) -> tuple[dict, list[str]]:
    """Return (migrated_record, list_of_changed_field_names)."""
    changed: list[str] = []

    if record.get("company") != company:
        record["company"] = company
        changed.append("company")

    prod = co.infer_product_fields(
        text=record.get("text", ""),
        title=record.get("source_title", "") or record.get("source_id", ""),
        company=company,
        source_type=record.get("source_type", "pdf"),
    )
    for key in ("product_line", "product_family", "product_name", "category", "technique"):
        missing = key not in record or record.get(key) in (None, "", [])
        if missing or reinfer:
            if record.get(key) != prod[key]:
                record[key] = prod[key]
                changed.append(key)

    for key in ("advantages", "disadvantages", "clinical_references"):
        if key not in record or record.get(key) is None:
            record[key] = []
            changed.append(key)

    return _reorder(record), changed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backfill the multi-company schema onto existing chunks.")
    p.add_argument("--company", default=co.DEFAULT_COMPANY,
                   help=f"company to stamp ({', '.join(co.known_companies())}); default {co.DEFAULT_COMPANY}")
    p.add_argument("--reinfer", action="store_true",
                   help="re-derive product fields even if already present (after a taxonomy change)")
    p.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = p.parse_args(argv)

    company = co.canonical_company(args.company)
    chunks_dir = chunks_dir_for(company)
    if not chunks_dir.exists():
        print(f"No chunks dir for {company} at {chunks_dir.relative_to(ROOT)}", flush=True)
        return 1

    files = sorted(chunks_dir.glob("*.json"))
    if not files:
        print(f"No chunk files in {chunks_dir.relative_to(ROOT)}", flush=True)
        return 1

    touched = unchanged = 0
    field_tally: dict[str, int] = {}
    for f in files:
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ✗ skip {f.name}: {exc}", file=sys.stderr)
            continue
        migrated, changed = migrate_chunk(record, company, args.reinfer)
        if changed:
            touched += 1
            for c in changed:
                field_tally[c] = field_tally.get(c, 0) + 1
            if not args.dry_run:
                f.write_text(json.dumps(migrated, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            unchanged += 1

    verb = "would update" if args.dry_run else "updated"
    print(f"[{company}] {verb} {touched} chunk(s); {unchanged} already current "
          f"({len(files)} total in {chunks_dir.relative_to(ROOT)}).")
    if field_tally:
        print("Fields backfilled:")
        for k in FIELD_ORDER:
            if k in field_tally:
                print(f"  - {k}: {field_tally[k]}")
    if args.dry_run:
        print("Dry run — no files written. Re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
