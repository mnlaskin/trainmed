#!/usr/bin/env python3
"""Chunk extracted transcripts into a retrieval-ready knowledge base.

Pipeline position:
    extract (trainmed.cli) -> data/transcripts/<id>.{json,md}
    THIS SCRIPT          -> data/kb/chunks/<id>_NNN.json  +  data/kb/<family>_index.md

What it does:
  1. Reads every .md file in data/transcripts/ (the human/LLM-readable transcripts).
  2. For each, loads the sibling <id>.json when present to get *timed segments*,
     so chunks get an accurate timestamp_range. Falls back to word-windowing the
     markdown body (timestamp_range = null) when no segments are available.
  3. Breaks each transcript into clean chunks of ~500-1000 words with ~20% overlap.
     Short transcripts (<= max) become a single chunk; trivially short ones
     (< MIN_CHUNK_WORDS) are skipped.
  4. Writes one JSON file per chunk with rich metadata (video_title, video_url,
     procedure_family, timestamp_range, word_count, chunk_index/chunk_count, ...).
  5. Writes a summary index markdown listing ingested videos and the main topics
     covered (by scanning for known technique/product keywords).

Stdlib only. Idempotent: rewrites the chunks for whatever transcripts exist now.

Usage:
    python scripts/ingest_to_kb.py
    python scripts/ingest_to_kb.py --family rotator_cuff
    python scripts/ingest_to_kb.py --target 800 --overlap 0.2 --clean
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root = parent of scripts/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so `trainmed` imports when run as a script
from trainmed import companies as co  # noqa: E402

TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
PDFS_DIR = ROOT / "data" / "pdfs"
KB_DIR = ROOT / "data" / "kb"
LEGACY_CHUNKS_DIR = KB_DIR / "chunks"  # Arthrex stays here for backward-compat


def chunks_dir_for(company: str) -> Path:
    """Where a company's chunks live. Arthrex keeps the legacy flat dir (so the
    committed KB is undisturbed); every other company gets data/kb/<Company>/chunks/."""
    company = co.canonical_company(company)
    return LEGACY_CHUNKS_DIR if company == co.DEFAULT_COMPANY else KB_DIR / company / "chunks"


MIN_CHUNK_WORDS = 20  # below this a transcript is treated as empty (music/intro only)

# Known shoulder / rotator-cuff techniques, products, and anatomy. Used to derive
# "main topics covered" for the index and per-chunk topic tags. (label, regex)
TOPIC_PATTERNS: list[tuple[str, str]] = [
    ("SpeedBridge", r"speed\s*bridge"),
    ("SutureBridge", r"suture\s*bridge"),
    ("SpeedFix", r"speed\s*fix"),
    ("FiberTak", r"fiber\s*tak"),
    ("SwiveLock", r"swivel\s*lock|swivelock"),
    ("PushLock", r"push\s*lock"),
    ("SutureTak", r"suture\s*tak"),
    ("InternalBrace", r"internal\s*brace"),
    ("CuffMend", r"cuff\s*mend"),
    ("Superior Capsular Reconstruction", r"superior capsular|capsular reconstruction|\bSCR\b"),
    ("Knotless fixation", r"knotless"),
    ("Double-row repair", r"double[-\s]?row"),
    ("Bioinductive implant", r"bioinductive|regeneten"),
    ("Suture anchors", r"\banchor"),
    ("Subscapularis", r"subscapularis"),
    ("Supraspinatus", r"supraspinatus"),
    ("Infraspinatus", r"infraspinatus"),
    ("Biceps tenodesis", r"tenodesis|biceps"),
    ("Shoulder instability", r"instability|bankart|remplissage|labral|labrum"),
    ("Rotator cuff repair", r"rotator cuff"),
    ("Shoulder arthroplasty", r"arthroplasty|shoulder replacement"),
]


# ── transcript loading ────────────────────────────────────────────────────────


def parse_front_matter(md_text: str) -> tuple[dict, str]:
    """Split a markdown file into (front_matter_dict, body_text)."""
    meta: dict = {}
    if not md_text.startswith("---"):
        return meta, md_text.strip()
    parts = md_text.split("---", 2)
    if len(parts) < 3:
        return meta, md_text.strip()
    _, fm, body = parts
    for line in fm.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        meta[key.strip()] = val.strip()
    # Drop the leading "# Title" heading line from the body if present.
    body = body.strip()
    body = re.sub(r"^#\s+.*\n+", "", body, count=1)
    return meta, body.strip()


def load_segments(video_id: str) -> list[dict]:
    """Load timed segments from the sibling JSON, or [] if unavailable."""
    json_path = TRANSCRIPTS_DIR / f"{video_id}.json"
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data.get("segments", []) or []


# ── chunking ──────────────────────────────────────────────────────────────────


def _wc(text: str) -> int:
    return len(text.split())


def _split_list(value: str | None) -> list[str]:
    """Parse an optional front-matter list field (';' or ',' separated) into a list."""
    if not value:
        return []
    parts = re.split(r"[;,]", value)
    return [p.strip() for p in parts if p.strip()]


def fmt_ts(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def chunk_segments(
    segments: list[dict], target: int, overlap_words: int
) -> list[dict]:
    """Chunk timed segments into ~target-word groups with word overlap.

    Returns dicts with text + start/end seconds. Never splits a caption line.
    """
    chunks: list[dict] = []
    n = len(segments)
    i = 0
    while i < n:
        j = i
        wc = 0
        while j < n and wc < target:
            wc += _wc(segments[j].get("text", ""))
            j += 1
        group = segments[i:j]
        text = " ".join(" ".join(s.get("text", "").split()) for s in group).strip()
        start = float(group[0].get("start", 0.0))
        last = group[-1]
        end = float(last.get("start", 0.0)) + float(last.get("duration", 0.0))
        chunks.append({"text": text, "start": start, "end": end})
        if j >= n:
            break
        # Step back over ~overlap_words worth of trailing segments.
        back = 0
        k = j - 1
        while k > i and back < overlap_words:
            back += _wc(segments[k].get("text", ""))
            k -= 1
        i = max(k + 1, i + 1)
    return chunks


def chunk_text(text: str, target: int, overlap_words: int) -> list[dict]:
    """Word-window fallback when no timed segments are available."""
    words = text.split()
    n = len(words)
    if n <= target:
        return [{"text": " ".join(words), "start": None, "end": None}]
    step = max(1, target - overlap_words)
    chunks: list[dict] = []
    for start in range(0, n, step):
        window = words[start : start + target]
        if not window:
            break
        chunks.append({"text": " ".join(window), "start": None, "end": None})
        if start + target >= n:
            break
    return chunks


def detect_topics(text: str) -> list[str]:
    low = text.lower()
    found = []
    for label, pat in TOPIC_PATTERNS:
        if re.search(pat, low, flags=re.IGNORECASE):
            found.append(label)
    return found


# ── main ──────────────────────────────────────────────────────────────────────


def _source_dirs(company: str, kind: str) -> list[Path]:
    """Source markdown dirs for a company. Arthrex uses the legacy flat dirs
    (data/transcripts, data/pdfs); other companies use data/<kind>/<Company>/.
    This keeps an Arthrex PDF from ever being re-labeled as a competitor during ingest."""
    base = ROOT / "data" / kind
    dirs: list[Path] = []
    if co.canonical_company(company) == co.DEFAULT_COMPANY and base.exists():
        dirs.append(base)
    sub = base / co.canonical_company(company)
    if sub.exists():
        dirs.append(sub)
    return dirs


def _collect_docs(company: str) -> list[tuple[Path, str]]:
    """That company's transcript/PDF markdown files paired with their source_type."""
    docs: list[tuple[Path, str]] = []
    for d in _source_dirs(company, "transcripts"):
        for p in sorted(d.glob("*.md")):
            docs.append((p, "youtube"))
    for d in _source_dirs(company, "pdfs"):
        for p in sorted(d.glob("*.md")):
            docs.append((p, "pdf"))
    return docs


def build(company: str, family: str, target: int, overlap_ratio: float, clean: bool) -> dict:
    company = co.canonical_company(company)
    out_dir = chunks_dir_for(company)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlap_words = int(target * overlap_ratio)

    if clean:  # company-scoped: only this company's chunks
        for f in out_dir.glob("*.json"):
            f.unlink()

    docs = _collect_docs(company)
    sources: list[dict] = []
    total_chunks = 0
    total_words = 0
    skipped: list[dict] = []

    for md_path, source_type in docs:
        meta, body = parse_front_matter(md_path.read_text(encoding="utf-8"))
        source_id = meta.get("video_id") or meta.get("doc_id") or md_path.stem
        title = meta.get("title") or source_id
        if source_type == "youtube":
            url = meta.get("url") or f"https://www.youtube.com/watch?v={source_id}"
            segments = load_segments(source_id)  # PDFs have no timed segments
        else:
            url = meta.get("source_url") or meta.get("url") or ""
            segments = []

        body_words = _wc(body)
        if body_words < MIN_CHUNK_WORDS:
            skipped.append(
                {"source_id": source_id, "title": title, "words": body_words, "source_type": source_type}
            )
            continue

        if segments:
            raw_chunks = chunk_segments(segments, target, overlap_words)
        else:
            raw_chunks = chunk_text(body, target, overlap_words)

        page_count = int(meta["page_count"]) if meta.get("page_count", "").isdigit() else None

        # Write one JSON per chunk with rich metadata.
        chunk_count = len(raw_chunks)
        src_topics: set[str] = set()
        for idx, rc in enumerate(raw_chunks):
            ts_range = None
            if rc["start"] is not None and rc["end"] is not None:
                ts_range = {
                    "start_seconds": round(rc["start"], 2),
                    "end_seconds": round(rc["end"], 2),
                    "label": f"{fmt_ts(rc['start'])}-{fmt_ts(rc['end'])}",
                }
            topics = detect_topics(rc["text"])
            src_topics.update(topics)
            chunk_id = f"{source_id}_{idx:03d}"
            # Multi-company metadata. Product fields are inferred from the text/title
            # against this company's taxonomy; advantages/disadvantages/clinical_refs
            # default empty (filled by structured product/competitive ingestion).
            prod = co.infer_product_fields(
                text=rc["text"], title=title, company=company, source_type=source_type
            )
            meta_advantages = _split_list(meta.get("advantages"))
            meta_disadvantages = _split_list(meta.get("disadvantages"))
            meta_clinical = _split_list(meta.get("clinical_references"))
            record = {
                "chunk_id": chunk_id,
                "company": company,                       # ← the firewall key
                "source_id": source_id,
                "source_type": source_type,  # "youtube" | "pdf"
                "source_title": title,
                "source_url": url,
                "channel": meta.get("channel") or None,
                "upload_date": meta.get("upload_date") or None,
                "page_count": page_count,
                "language": meta.get("language") or None,
                "procedure_family": family,
                "product_line": prod["product_line"],
                "product_family": prod["product_family"],
                "product_name": prod["product_name"],
                "category": prod["category"],
                "technique": prod["technique"],
                "advantages": meta_advantages,
                "disadvantages": meta_disadvantages,
                "clinical_references": meta_clinical,
                "chunk_index": idx,
                "chunk_count": chunk_count,
                "word_count": _wc(rc["text"]),
                "timestamp_range": ts_range,  # null for PDFs
                "topics": topics,
                "text": rc["text"],
            }
            (out_dir / f"{chunk_id}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            total_chunks += 1
            total_words += record["word_count"]

        sources.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "title": title,
                "url": url,
                "channel": meta.get("channel") or "",
                "word_count": body_words,
                "chunk_count": chunk_count,
                "topics": sorted(src_topics),
            }
        )

    index_path = write_index(company, family, sources, total_chunks, total_words, skipped)
    return {
        "company": company,
        "sources": len(sources),
        "videos": sum(1 for s in sources if s["source_type"] == "youtube"),
        "pdfs": sum(1 for s in sources if s["source_type"] == "pdf"),
        "chunks": total_chunks,
        "words": total_words,
        "skipped": skipped,
        "index_path": index_path,
        "out_dir": out_dir,
    }


def write_index(
    company: str,
    family: str,
    sources: list[dict],
    total_chunks: int,
    total_words: int,
    skipped: list[dict],
) -> Path:
    KB_DIR.mkdir(parents=True, exist_ok=True)
    family_label = family.replace("_", " ").title()
    company_label = co.display_name(company)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Tally topics across all sources.
    topic_sources: dict[str, list[str]] = {}
    for s in sources:
        for t in s["topics"]:
            topic_sources.setdefault(t, []).append(s["title"])

    n_video = sum(1 for s in sources if s["source_type"] == "youtube")
    n_pdf = sum(1 for s in sources if s["source_type"] == "pdf")

    lines = [
        f"# {company_label} — {family_label} Knowledge Base — Index",
        "",
        f"_Generated {now}_",
        "",
        f"- **Company:** `{company}`",
        f"- **Sources ingested:** {len(sources)}  ({n_video} YouTube, {n_pdf} PDF)",
        f"- **Chunks created:** {total_chunks}",
        f"- **Total words:** {total_words:,}",
        f"- **Procedure family:** `{family}`",
        "",
        "## Ingested sources",
        "",
        "| Source | Type | Channel / Host | Words | Chunks |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for s in sorted(sources, key=lambda x: (x["source_type"], x["title"].lower())):
        title = s["title"].replace("|", "\\|")
        typ = "YouTube" if s["source_type"] == "youtube" else "PDF"
        host = s["channel"] or ("—" if s["source_type"] == "pdf" else "")
        lines.append(
            f"| [{title}]({s['url']}) | {typ} | {host} | {s['word_count']:,} | {s['chunk_count']} |"
        )

    lines += ["", "## Main topics covered", ""]
    for topic in sorted(topic_sources, key=lambda t: (-len(topic_sources[t]), t)):
        count = len(topic_sources[topic])
        lines.append(f"- **{topic}** — {count} source{'s' if count != 1 else ''}")

    if skipped:
        lines += ["", "## Skipped (no usable text)", ""]
        for s in skipped:
            lines.append(f"- {s['title']} (`{s['source_id']}`) — {s['words']} words")

    rel_dir = chunks_dir_for(company).relative_to(ROOT)
    lines += [
        "",
        "---",
        f"_Built by `scripts/ingest_to_kb.py --company {company}`. Chunks live in `{rel_dir}/`._",
        "",
    ]
    index_path = KB_DIR / f"{company.lower()}_{family}_index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Chunk transcripts/PDFs into the KB (company-scoped).")
    p.add_argument("--company", default=co.DEFAULT_COMPANY,
                   help=f"company corpus to build ({', '.join(co.known_companies())}); default {co.DEFAULT_COMPANY}")
    p.add_argument("--family", default="rotator_cuff", help="procedure_family tag")
    p.add_argument("--target", type=int, default=800, help="target words/chunk (500-1000)")
    p.add_argument("--overlap", type=float, default=0.2, help="overlap ratio (0-0.5)")
    p.add_argument("--clean", action="store_true", help="delete this company's existing chunks first")
    args = p.parse_args(argv)

    company = co.canonical_company(args.company)
    docs = _collect_docs(company)
    if not docs:
        searched = ", ".join(str(d.relative_to(ROOT)) for d in
                             _source_dirs(company, "transcripts") + _source_dirs(company, "pdfs")) or "(none found)"
        print(f"No source markdown for {company}. Looked in: {searched}", flush=True)
        print(f"  → Ingest PDFs first, e.g.:  python -m trainmed.pdf_ingest --company {company} --from-file data/urls/<file>.txt")
        return 1

    result = build(company, args.family, args.target, args.overlap, args.clean)
    print(
        f"[{result['company']}] KB built: {result['sources']} sources "
        f"({result['videos']} YouTube + {result['pdfs']} PDF) -> {result['chunks']} chunks "
        f"({result['words']:,} words). Chunks: {result['out_dir'].relative_to(ROOT)}/  Index: {result['index_path'].relative_to(ROOT)}"
    )
    if result["skipped"]:
        print(f"Skipped {len(result['skipped'])} empty source(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
