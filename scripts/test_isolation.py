#!/usr/bin/env python3
"""Contamination / isolation test for the multi-company KB — the firewall's regression net.

Asserts the things that must hold for company separation to be safe, so a future ingest
can't silently regress it. Exits non-zero on any failure (wire into CI).

Checks:
  1. Every chunk in every corpus carries the full multi-company schema.
  2. load_chunks(company) returns ONLY that company's chunks.
  3. Company corpora are disjoint by chunk_id (no chunk shared across companies).
  4. competitive_insights never leak into any single-company corpus.
  5. A per-company retriever, queried with a COMPETITOR's product name, still returns
     only the selected company's chunks (the retrieval firewall holds end to end).
  6. The embeddings-cache fingerprint changes when the company changes (cache can't be
     reused across corpora).

Usage:  PYTHONPATH=src python scripts/test_isolation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trainmed import companies as co  # noqa: E402
from trainmed import rag  # noqa: E402

REQUIRED_FIELDS = [
    "chunk_id", "company", "source_type", "source_title",
    "product_line", "product_family", "product_name", "category",
    "technique", "advantages", "disadvantages", "clinical_references", "text",
]

# Cross-company probe queries: each names a competitor's flagship to try to provoke a leak.
PROBES = {
    "Arthrex": ["Omega knotless anchor", "HEALICOIL double row", "Iconix all-suture"],
    "Stryker": ["SpeedBridge FiberTape", "HEALICOIL REGENESORB", "SwiveLock anchor"],
    "SmithNephew": ["SpeedBridge knotless", "Omega AlphaVent", "FiberTak all-suture"],
}

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)
        print(f"  ✗ {msg}")
    else:
        print(f"  ✓ {msg}")


def main() -> int:
    companies = rag.available_companies()
    print(f"Companies with a KB: {companies}\n")
    if not companies:
        print("No company KBs found — nothing to test.")
        return 0

    corpora = {c: rag.load_chunks(c) for c in companies}
    all_ids_by_company = {c: {ch["chunk_id"] for ch in chunks} for c, chunks in corpora.items()}

    print("[1] schema completeness")
    for c, chunks in corpora.items():
        missing = [ch["chunk_id"] for ch in chunks
                   if any(f not in ch for f in REQUIRED_FIELDS)]
        check(not missing, f"{c}: all {len(chunks)} chunks have the full schema"
              + (f" (missing on {missing[:3]}...)" if missing else ""))

    print("\n[2] load_chunks(company) returns only that company")
    for c, chunks in corpora.items():
        bad = [ch["chunk_id"] for ch in chunks if ch.get("company") != c]
        check(not bad, f"{c}: every loaded chunk is tagged {c}"
              + (f" (offenders: {bad[:3]})" if bad else ""))

    print("\n[3] corpora are disjoint by chunk_id")
    pairs = [(a, b) for i, a in enumerate(companies) for b in companies[i + 1:]]
    for a, b in pairs:
        overlap = all_ids_by_company[a] & all_ids_by_company[b]
        check(not overlap, f"{a} ∩ {b} share no chunk_ids" + (f" (shared: {list(overlap)[:3]})" if overlap else ""))
    if not pairs:
        print("  (only one company — disjointness trivially holds)")

    print("\n[4] competitive_insights stay out of company corpora")
    ci = rag.load_competitive_insights()
    ci_ids = {x["chunk_id"] for x in ci}
    print(f"  competitive_insights loaded: {len(ci)}")
    for c, chunks in corpora.items():
        leaked = ci_ids & all_ids_by_company[c]
        check(not leaked, f"{c}: no competitive_insight chunk present"
              + (f" (leaked: {list(leaked)[:3]})" if leaked else ""))
    check(all(x.get("company") == "_competitive" for x in ci),
          "every competitive_insight is tagged the _competitive sentinel (never a real company)")

    print("\n[5] retrieval firewall under cross-company probes")
    backend, _ = rag.resolve_embed_backend("tfidf")  # deterministic, no API key needed
    for c, chunks in corpora.items():
        retr = rag.build_retriever(chunks, backend, namespace=c)
        for q in PROBES.get(c, []):
            hits = retr.retrieve(q, 5)
            leak = [ch["chunk_id"] for ch, _ in hits if ch.get("company") != c]
            check(not leak, f"{c}: query {q!r} returns only {c} chunks"
                  + (f" (LEAK: {leak})" if leak else ""))

    print("\n[6] embeddings-cache fingerprint is company-bound")
    if len(companies) >= 2:
        a, b = companies[0], companies[1]
        fa = rag._fingerprint(corpora[a], a)
        fb = rag._fingerprint(corpora[b], b)
        check(fa != fb, f"fingerprint({a}) != fingerprint({b})")
    else:
        # Single company: same chunks, different namespace must still differ.
        c = companies[0]
        check(rag._fingerprint(corpora[c], c) != rag._fingerprint(corpora[c], "competitive"),
              "fingerprint is namespace-bound (same chunks, different namespace → different hash)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} isolation check(s) failed.")
        return 1
    print("PASSED: all isolation checks green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
