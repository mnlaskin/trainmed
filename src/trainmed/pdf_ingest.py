"""Download PDF surgical-technique / product guides and extract clean text.

Pipeline position (parallel to the YouTube path):
    PDF URLs (or a page to scrape) -> download -> pypdf text -> data/pdfs/<slug>.md
    then scripts/ingest_to_kb.py chunks data/pdfs/*.md alongside the transcripts.

Design notes:
  - Uses **pypdf** (pure-Python) — deliberately NOT PyMuPDF, which has no wheels for
    Python 3.14 yet. pypdf is slower but installs everywhere.
  - Downloads are idempotent (skip if the file already exists unless --overwrite).
  - Robust: a failing URL is logged and skipped; the batch continues.
  - Output .md mirrors the transcript front-matter shape so the existing chunker
    can read it with no special-casing, plus `source_type: pdf`.

Usage:
    python -m trainmed.pdf_ingest "https://host/guide.pdf" "https://host/guide2.pdf"
    python -m trainmed.pdf_ingest --from-file data/urls/arthrex_rotator_cuff_pdfs.txt
    python -m trainmed.pdf_ingest --scrape "https://www.arthrex.com/...publications-page"
    python -m trainmed.pdf_ingest --from-file urls.txt --family rotator_cuff
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from . import companies as co

ROOT = Path(__file__).resolve().parents[2]
PDFS_MD_DIR = ROOT / "data" / "pdfs"
PDFS_RAW_DIR = ROOT / "data" / "raw" / "pdfs"

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) TrainMed-ingest/0.1"


def md_dir_for(company: str) -> Path:
    """Extracted-markdown dir for a company. Arthrex keeps the legacy flat dir;
    others use data/pdfs/<Company>/ so ingest stays company-isolated."""
    company = co.canonical_company(company)
    return PDFS_MD_DIR if company == co.DEFAULT_COMPANY else PDFS_MD_DIR / company


def raw_dir_for(company: str) -> Path:
    company = co.canonical_company(company)
    return PDFS_RAW_DIR if company == co.DEFAULT_COMPANY else PDFS_RAW_DIR / company


# ── helpers ───────────────────────────────────────────────────────────────────


def slugify(value: str, max_len: int = 80) -> str:
    """Filesystem-safe slug from a URL tail or title."""
    value = value.strip().lower()
    value = re.sub(r"\.pdf$", "", value)
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:max_len] or "document"


def _read_url_file(path: str) -> list[tuple[str, str | None]]:
    """One URL per line, returning (url, title_override). Full-line `#` comments
    are skipped; an inline `<url>   # Nice Title` trailing comment becomes the
    title override (so PDFs get clean citation titles instead of filename junk)."""
    out: list[tuple[str, str | None]] = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        url, _, comment = ln.partition("#")
        url = url.strip()
        if not url:
            continue
        title = comment.strip() or None
        out.append((url, title))
    return out


def _fetch_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def scrape_pdf_links(page_url: str, timeout: int = 45) -> list[str]:
    """Find .pdf links on an HTML page, resolved to absolute URLs."""
    req = urllib.request.Request(page_url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    hrefs = re.findall(r'href=["\']([^"\']+?\.pdf[^"\']*)["\']', html, flags=re.IGNORECASE)
    seen, out = set(), []
    for h in hrefs:
        absolute = urllib.parse.urljoin(page_url, h)
        if absolute not in seen:
            seen.add(absolute)
            out.append(absolute)
    return out


def download_pdf(url: str, overwrite: bool = False, company: str = co.DEFAULT_COMPANY) -> Path | None:
    """Download a PDF to data/raw/pdfs/<company>/. Returns the path, or None on failure."""
    raw_dir = raw_dir_for(company)
    raw_dir.mkdir(parents=True, exist_ok=True)
    name = Path(urllib.parse.urlparse(url).path).name or slugify(url)
    if not name.lower().endswith(".pdf"):
        name = slugify(url) + ".pdf"
    dest = raw_dir / name
    if dest.exists() and not overwrite:
        return dest
    try:
        data = _fetch_bytes(url)
    except Exception as exc:
        print(f"  ✗ download failed: {url} ({exc})", file=sys.stderr)
        return None
    if not data[:5].startswith(b"%PDF"):
        print(f"  ✗ not a PDF (no %PDF header): {url}", file=sys.stderr)
        return None
    dest.write_bytes(data)
    return dest


def extract_text(pdf_path: Path) -> tuple[str, int, str | None]:
    """Extract (clean_text, page_count, embedded_title) from a PDF via pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    raw = "\n\n".join(pages)
    # Collapse runaway whitespace while preserving paragraph breaks.
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    text = raw.strip()
    title = None
    try:
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title.strip()
    except Exception:
        title = None
    return text, len(reader.pages), title


def write_md(
    *,
    title: str,
    source_url: str,
    family: str,
    page_count: int,
    text: str,
    slug: str,
    company: str = co.DEFAULT_COMPANY,
) -> Path:
    md_dir = md_dir_for(company)
    md_dir.mkdir(parents=True, exist_ok=True)
    front = [
        "---",
        f"title: {title}",
        f"doc_id: {slug}",
        f"company: {co.canonical_company(company)}",
        f"source_url: {source_url}",
        "source_type: pdf",
        f"procedure_family: {family}",
        f"page_count: {page_count}",
        "source: pdf",
        "---",
        "",
        f"# {title}",
        "",
        text,
        "",
    ]
    md_path = md_dir / f"{slug}.md"
    md_path.write_text("\n".join(front), encoding="utf-8")
    return md_path


# ── orchestration ─────────────────────────────────────────────────────────────


def ingest_url(url: str, family: str, overwrite: bool, title_override: str | None = None,
               company: str = co.DEFAULT_COMPANY) -> dict | None:
    pdf_path = download_pdf(url, overwrite=overwrite, company=company)
    if pdf_path is None:
        return None
    try:
        text, pages, embedded_title = extract_text(pdf_path)
    except Exception as exc:
        print(f"  ✗ extract failed: {url} ({exc})", file=sys.stderr)
        return None
    if len(text.split()) < 20:
        print(f"  ✗ extracted text too short (scanned/image PDF?): {url}", file=sys.stderr)
        return None
    slug = slugify(Path(urllib.parse.urlparse(url).path).stem or url)
    # Title precedence: explicit override (from the seed file) > PDF metadata > filename.
    title = title_override or embedded_title or Path(urllib.parse.urlparse(url).path).stem.replace("-", " ").title()
    md_path = write_md(
        title=title,
        source_url=url,
        family=family,
        page_count=pages,
        text=text,
        slug=slug,
        company=company,
    )
    return {"url": url, "slug": slug, "title": title, "pages": pages, "words": len(text.split()), "md": md_path}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download + extract PDFs into data/pdfs/<company>/.")
    p.add_argument("urls", nargs="*", help="direct PDF URLs")
    p.add_argument("--from-file", metavar="PATH", help="file of PDF URLs (one per line)")
    p.add_argument("--scrape", metavar="PAGE_URL", help="scrape .pdf links from this page")
    p.add_argument("--company", default=co.DEFAULT_COMPANY,
                   help=f"company this content belongs to ({', '.join(co.known_companies())}); default {co.DEFAULT_COMPANY}")
    p.add_argument("--family", default="rotator_cuff", help="procedure_family tag")
    p.add_argument("--overwrite", action="store_true", help="re-download + re-extract")
    args = p.parse_args(argv)
    company = co.canonical_company(args.company)

    # Normalize every input to (url, title_override).
    targets: list[tuple[str, str | None]] = [(u, None) for u in args.urls]
    if args.from_file:
        targets.extend(_read_url_file(args.from_file))
    if args.scrape:
        found = scrape_pdf_links(args.scrape)
        print(f"Scraped {len(found)} .pdf link(s) from {args.scrape}", file=sys.stderr)
        targets.extend((u, None) for u in found)
    # De-dup by URL, preserve order (first title override wins).
    seen: set[str] = set()
    deduped: list[tuple[str, str | None]] = []
    for url, title in targets:
        if url and url not in seen:
            seen.add(url)
            deduped.append((url, title))
    if not deduped:
        p.error("provide PDF URLs, --from-file, or --scrape")

    print(f"Ingesting {len(deduped)} PDF(s) for {company} -> {md_dir_for(company).relative_to(ROOT)}/ ...", file=sys.stderr)
    ok = failed = 0
    for i, (url, title) in enumerate(deduped, 1):
        print(f"[{i}/{len(deduped)}] {url}")
        result = ingest_url(url, args.family, args.overwrite, title_override=title, company=company)
        if result:
            print(f"  ✓ {result['words']:,} words, {result['pages']} pages  {result['title'][:60]}")
            ok += 1
        else:
            failed += 1
    print(f"\nDone. {ok} ingested, {failed} failed. Next: python scripts/ingest_to_kb.py --company {company}",
          file=sys.stderr)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
