"""Structured per-product spec registry — clean, queryable product specifications.

One JSON file per company at ``data/kb/specs/<Company>.json``: a list of product records with
normalized numeric / material / fixation fields plus provenance (`sources`). Every numeric value
was validated to appear verbatim in that product's ingested chunks (zero-hallucination), so the
Knowledge Assistant can answer point-blank spec questions from clean structured data instead of
re-deriving them from prose. Company-scoped throughout — the contamination firewall is preserved
(a lookup only ever reads the selected company's registry).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import companies as co

SPECS_DIR = Path(__file__).resolve().parents[2] / "data" / "kb" / "specs"

# numeric fields + human label used in the prompt block
NUM_LABEL = {
    "anchor_diameter_mm": "anchor Ø",
    "anchor_length_mm": "anchor length",
    "drill_size_mm": "drill/punch",
    "suture_tape_width_mm": "suture/tape width",
}
LIST_LABEL = {
    "material": "material",
    "fixation_type": "fixation",
    "key_features": "key features",
    "advantages": "advantages",
    "disadvantages": "disadvantages",
    "clinical_references": "clinical references",
}

_SPEC_QUERY = re.compile(
    r"\b(\d+(?:\.\d+)?\s*mm|diameter|diameters|\bsize\b|\bsizes\b|length|drill|punch|reamer|"
    r"\bwidth\b|gauge|spec|specs|specification|material|peek|all[- ]?suture|biocomposite|titanium|"
    r"how big|what size|which size|compare|comparison|\bvs\.?\b|versus)\b",
    re.I,
)


def load_specs(company: str) -> list[dict]:
    company = co.canonical_company(company)
    f = SPECS_DIR / f"{company}.json"
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def is_spec_query(q: str | None) -> bool:
    """True when a question is asking for exact specs / a size comparison."""
    return bool(q and _SPEC_QUERY.search(q))


def match_specs(company: str, query: str, limit: int = 4) -> list[dict]:
    """Registry records whose product family / name is named in the query (company-scoped)."""
    recs = load_specs(company)
    if not recs or not query:
        return []
    ql = " " + re.sub(r"[^a-z0-9]+", " ", query.lower()).strip() + " "

    def hit(r: dict) -> bool:
        for key in (r.get("product_family", ""), r.get("product_name", "")):
            t = re.sub(r"[^a-z0-9]+", " ", str(key).lower()).strip()
            if t and (" " + t + " ") in ql:
                return True
        # distinctive single tokens (>=4 chars) of the family name, e.g. "swivelock"
        for tok in re.findall(r"[a-z0-9]{4,}", str(r.get("product_family", "")).lower()):
            if (" " + tok + " ") in ql:
                return True
        return False

    return [r for r in recs if hit(r)][:limit]


def _fmt_nums(values) -> str:
    return ", ".join(f"{float(x):g} mm" for x in (values or []))


def format_specs_block(records: list[dict]) -> str:
    """Authoritative structured-spec text block to prepend to the model context."""
    if not records:
        return ""
    out = [
        "STRUCTURED PRODUCT SPECS (authoritative — extracted and validated against the cited "
        "sources; use these EXACT figures and still cite the underlying source [n]):"
    ]
    for r in records:
        out.append(f"• {r.get('product_name') or r.get('product_family')} ({r.get('company')}):")
        for field, label in NUM_LABEL.items():
            s = _fmt_nums(r.get(field))
            if s:
                out.append(f"    - {label}: {s}")
        for field, label in LIST_LABEL.items():
            v = r.get(field) or []
            if v:
                out.append(f"    - {label}: " + "; ".join(str(x) for x in v))
    return "\n".join(out)
