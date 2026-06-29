"""Per-company configuration + product taxonomy for TrainMed's multi-company KB.

This is the single place that knows about each company we support. It drives:
  - the company *firewall* (canonical keys + display names),
  - per-company prompt/persona/scorer parameterization in `rag.py`,
  - product-field *inference* used by both ingestion (`scripts/ingest_to_kb.py`)
    and the backward-compat backfill (`scripts/migrate_add_company.py`).

Adding a company = adding one entry to COMPANIES + (optionally) one to TAXONOMY.
Nothing else in the codebase hard-codes a brand.

Design rule: companies are kept STRICTLY separate. The raw technique/spec corpora
never mix; only the curated `competitive_insights` collection is cross-company.
"""

from __future__ import annotations

import re

# Canonical company keys used everywhere (chunk `company` field, dir names,
# retriever namespaces, cache filenames). Keep these filesystem-safe.
ARTHREX = "Arthrex"
STRYKER = "Stryker"
SMITH_NEPHEW = "SmithNephew"

# The company every legacy chunk (and any chunk missing a `company`) belongs to.
# The existing KB was 100% Arthrex, so this is the safe backward-compatible default.
DEFAULT_COMPANY = ARTHREX


# ── company registry ──────────────────────────────────────────────────────────
#
# `product_terms` is the lowercase, phrase-normalized vocab the roleplay coach's
# heuristic scorer rewards (see rag._heuristic_score). `portfolio` is a short
# clause spliced into the system prompt / roleplay persona so the assistant
# speaks as the right brand without any per-company prompt forks.

COMPANIES: dict[str, dict] = {
    ARTHREX: {
        "display_name": "Arthrex",
        "portfolio": (
            "the Arthrex rotator cuff and shoulder portfolio — SpeedBridge, SutureBridge, "
            "FiberTak, SwiveLock, FiberTape, knotless double-row constructs and SCR"
        ),
        "product_terms": {
            "speedbridge", "suturebridge", "speedfix", "fibertak", "swivelock", "fibertape",
            "fiberwire", "corkscrew", "pushlock", "suturetak", "biocomposite", "knotless",
            "double-row", "doublerow", "footprint", "anchor", "all-suture", "allsuture",
            "arthroflex", "cuffmend", "tensionable", "ripstop", "scr",
        },
    },
    STRYKER: {
        "display_name": "Stryker",
        "portfolio": (
            "the Stryker Sports Medicine shoulder and rotator cuff portfolio — ReelX STT, "
            "Iconix all-suture anchors, AlphaVent, Omega knotless and knotless double-row constructs"
        ),
        "product_terms": {
            "reelx", "iconix", "alphavent", "omega", "citrelock", "citrenak", "twinfix",
            "knotilus", "nanotack", "inspace", "knotless", "double-row", "doublerow",
            "all-suture", "allsuture", "anchor", "footprint", "scr", "tensionable",
            "instability", "bankart", "transosseous",
        },
    },
    SMITH_NEPHEW: {
        "display_name": "Smith & Nephew",
        "portfolio": (
            "the Smith & Nephew shoulder and rotator cuff portfolio — HEALICOIL, Q-FIX "
            "all-suture anchors, FOOTPRINT Ultra, REGENESORB and knotless double-row constructs"
        ),
        "product_terms": {
            "healicoil", "qfix", "q-fix", "footprint", "regenesorb", "regeneten", "multifix",
            "bioraptor", "ultrabraid", "ultratape", "knotless", "double-row", "doublerow",
            "all-suture", "allsuture", "anchor", "bioinductive", "scr", "tensionable",
        },
    },
}


def canonical_company(value: str | None) -> str:
    """Map a loose user/string input to a canonical company key (or DEFAULT)."""
    if not value:
        return DEFAULT_COMPANY
    v = re.sub(r"[^a-z0-9]+", "", value.lower())
    table = {
        "arthrex": ARTHREX,
        "stryker": STRYKER,
        "smithnephew": SMITH_NEPHEW,
        "smith": SMITH_NEPHEW,
        "sn": SMITH_NEPHEW,
        "smithandnephew": SMITH_NEPHEW,
    }
    return table.get(v, value if value in COMPANIES else DEFAULT_COMPANY)


def display_name(company: str) -> str:
    return COMPANIES.get(company, {}).get("display_name", company)


def portfolio(company: str) -> str:
    return COMPANIES.get(company, {}).get(
        "portfolio", f"the {display_name(company)} rotator cuff and shoulder portfolio"
    )


def product_terms(company: str) -> set[str]:
    return COMPANIES.get(company, COMPANIES[DEFAULT_COMPANY])["product_terms"]


def known_companies() -> list[str]:
    return list(COMPANIES.keys())


# ── product taxonomy (for inference) ──────────────────────────────────────────
#
# Ordered, per-company list of (regex, fields). First match (by list order =
# priority) wins for product_family/product_name/product_line; `technique` tags
# accumulate across every match so a chunk can be both "double_row" and "knotless".
# Used by ingestion AND backfill so a chunk's product metadata is derived the same
# way no matter how it entered the KB.

# Each entry: (compiled_regex, product_family, product_name, product_line, technique[list])
_TAXONOMY_RAW: dict[str, list[tuple[str, str, str, str, list[str]]]] = {
    ARTHREX: [
        (r"speed\s*bridge",       "SpeedBridge",  "SpeedBridge",  "Rotator Cuff", ["double_row", "knotless", "transosseous_equivalent"]),
        (r"suture\s*bridge",      "SutureBridge", "SutureBridge", "Rotator Cuff", ["double_row", "transosseous_equivalent"]),
        (r"speed\s*fix",          "SpeedFix",     "SpeedFix",     "Rotator Cuff", ["knotless", "single_row"]),
        (r"fiber\s*ta(?:k|ck)",   "FiberTak",     "FiberTak",     "Rotator Cuff", ["knotless", "all_suture"]),
        (r"swivel\s*lock",        "SwiveLock",    "SwiveLock",    "Rotator Cuff", ["knotless"]),
        (r"push\s*lock",          "PushLock",     "PushLock",     "Rotator Cuff", ["knotless"]),
        (r"suture\s*ta(?:k|ck)",  "SutureTak",    "SutureTak",    "Shoulder",     []),
        (r"cork\s*screw",         "Corkscrew",    "Corkscrew",    "Rotator Cuff", []),
        (r"fiber\s*tape",         "FiberTape",    "FiberTape",    "Rotator Cuff", []),
        (r"arthro\s*flex",        "ArthroFLEX",   "ArthroFLEX",   "Rotator Cuff", ["augmentation"]),
        (r"cuff\s*mend",          "CuffMend",     "CuffMend",     "Rotator Cuff", ["augmentation"]),
        (r"superior capsular|capsular reconstruction|\bscr\b", "SCR", "Superior Capsular Reconstruction", "Shoulder", ["SCR"]),
        (r"bankart|remplissage|labral|labrum|instability",     "Instability", "Bankart Repair", "Shoulder", ["instability"]),
        (r"tenodesis|biceps",     "Biceps Tenodesis", "Biceps Tenodesis", "Shoulder", []),
    ],
    STRYKER: [
        # Instability first: an instability-titled guide stays Instability/Shoulder even
        # though it also names the Iconix/NanoTack anchors it uses.
        (r"instability|bankart|labral|labrum", "Instability", "Shoulder Instability Repair", "Shoulder", ["instability"]),
        (r"reel\s*x",   "ReelX STT", "ReelX STT", "Rotator Cuff", ["knotless", "tensionable"]),
        (r"iconix",     "Iconix",    "Iconix",    "Rotator Cuff", ["all_suture", "knotless"]),
        (r"knotilus",   "Knotilus+", "Knotilus+ Knotless Anchor", "Rotator Cuff", ["knotless"]),
        (r"alpha\s*vent", "AlphaVent", "AlphaVent", "Rotator Cuff", ["knotless"]),
        (r"omega",      "Omega Knotless", "Omega Knotless", "Rotator Cuff", ["knotless", "double_row"]),
        (r"in\s*space", "InSpace", "InSpace Balloon Spacer", "Shoulder", ["balloon_spacer", "irreparable"]),
        (r"nano\s*tack", "NanoTack", "NanoTack", "Shoulder", ["instability", "all_suture"]),
        (r"citre(?:lock|nak)", "Citrelock", "Citrelock", "Shoulder", []),
        (r"twin\s*fix", "TwinFix",   "TwinFix",   "Rotator Cuff", []),
        (r"superior capsular|capsular reconstruction|\bscr\b", "SCR", "Superior Capsular Reconstruction", "Shoulder", ["SCR"]),
    ],
    SMITH_NEPHEW: [
        (r"heali\s*coil", "HEALICOIL", "HEALICOIL", "Rotator Cuff", ["double_row", "knotless"]),
        (r"q[-\s]*fix",   "Q-FIX",     "Q-FIX",     "Rotator Cuff", ["all_suture", "knotless"]),
        (r"multi\s*fix", "MULTIFIX", "MULTIFIX S ULTRA", "Rotator Cuff", ["knotless"]),
        (r"foot\s*print", "FOOTPRINT Ultra", "FOOTPRINT Ultra", "Rotator Cuff", ["knotless"]),
        (r"regene\s*ten", "REGENETEN", "REGENETEN Bioinductive Implant", "Rotator Cuff", ["bioinductive", "augmentation"]),
        (r"regenesorb",   "REGENESORB", "REGENESORB", "Rotator Cuff", []),
        (r"bio\s*raptor", "BIORAPTOR", "BIORAPTOR", "Shoulder", []),
        (r"ultra\s*(?:braid|tape)", "ULTRABRAID", "ULTRABRAID", "Rotator Cuff", []),
        (r"superior capsular|capsular reconstruction|\bscr\b", "SCR", "Superior Capsular Reconstruction", "Shoulder", ["SCR"]),
    ],
}

_TAXONOMY: dict[str, list[tuple[re.Pattern, str, str, str, list[str]]]] = {
    company: [(re.compile(pat, re.IGNORECASE), fam, name, line, tech) for pat, fam, name, line, tech in rules]
    for company, rules in _TAXONOMY_RAW.items()
}

# Title/text signals that a source is a peer-reviewed study or clinical-evidence
# summary rather than a guide.
_STUDY_SIGNALS = re.compile(
    r"\b(rationale|results|outcomes|scientific update|biomechanic|cadaver|meta[- ]?analysis|"
    r"systematic review|randomi[sz]ed|rct|et al\.?|journal|arthroscopy techniques|clinical study|"
    r"clinical evidence|clinical summary|clinical outcomes|clinical practice guideline|"
    r"evidence matters|evidence in focus|evidence summary|evidence collection|"
    r"literature summary|literature review|key literature)\b",
    re.IGNORECASE,
)
# Title signals that a source is a dedicated product/anchor brochure.
_ANCHOR_SIGNALS = re.compile(r"\b(anchor|all[- ]suture)\b", re.IGNORECASE)


def infer_product_fields(
    *, text: str, title: str, company: str, source_type: str
) -> dict:
    """Derive product_line / product_family / product_name / category / technique.

    Deterministic and shared by ingestion + backfill. Never invents specs — it only
    classifies by known product keywords, so it is safe to run on any company corpus.
    """
    rules = _TAXONOMY.get(company, [])
    family = name = line = ""
    technique: list[str] = []
    # The title is the strongest product signal: a match there sets the primary product,
    # overriding higher-priority families that only appear in the body. Technique tags
    # still accumulate across both title and body.
    for rx, fam, pname, pline, tech in rules:
        if family:
            break
        if rx.search(title):
            family, name, line = fam, pname, pline
    for rx, fam, pname, pline, tech in rules:
        if rx.search(f"{title}\n{text}"):
            if not family:  # body fallback when nothing matched the title
                family, name, line = fam, pname, pline
            for t in tech:
                if t not in technique:
                    technique.append(t)

    # Category precedence: study > anchor brochure > technique guide (default).
    if _STUDY_SIGNALS.search(title) or (source_type == "pdf" and _STUDY_SIGNALS.search(title)):
        category = "clinical_study"
    elif _ANCHOR_SIGNALS.search(title) and family:
        category = "anchor"
    else:
        category = "technique_guide"

    return {
        "product_line": line or "Rotator Cuff",  # whole KB is rotator-cuff/shoulder
        "product_family": family,
        "product_name": name or family,
        "category": category,
        "technique": technique,
    }


# Canonical, ordered metadata keys every chunk should carry (existing + new).
# Used to normalize records so the schema is stable across companies.
NEW_METADATA_FIELDS = (
    "company",
    "product_line",
    "product_family",
    "product_name",
    "category",
    "technique",
    "advantages",
    "disadvantages",
    "clinical_references",
)
