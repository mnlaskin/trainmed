#!/usr/bin/env python3
"""CLI for the TrainMed RAG engine (the engine itself lives in trainmed.rag).

    data/kb/chunks/*.json  ->  embed (cached)  ->  cosine top-k  ->  cited answer

Backends auto-select and upgrade when an API key appears (no code change):
  embeddings (--backend): auto | openai | voyage | st | tfidf
  generation (--gen):     auto | anthropic | openai | extractive

Usage:
    python scripts/test_rag.py                              # auto backends, 5 default Qs
    ANTHROPIC_API_KEY=... OPENAI_API_KEY=... python scripts/test_rag.py
    python scripts/test_rag.py -k 5 --backend openai --gen anthropic
    python scripts/test_rag.py --question "How does SpeedBridge differ from SutureBridge?"

Run with the package importable (`pip install -e .` or `PYTHONPATH=src`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running straight from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trainmed import companies as co  # noqa: E402
from trainmed.rag import (  # noqa: E402
    DEFAULT_QUESTIONS,
    KB_DIR,
    build_retriever,
    format_sources,
    generate_answer,
    load_chunks,
    resolve_embed_backend,
    resolve_gen_backend,
)


def run(questions, k, embed_backend, gen_backend, embed_model, gen_model, use_cache, company) -> int:
    company = co.canonical_company(company)
    chunks = load_chunks(company)  # company-scoped: the retrieval firewall
    if not chunks:
        print(f"No chunks for {company}. Ingest it first (python scripts/ingest_to_kb.py --company {company}).")
        return 1

    embed_backend, e_reason = resolve_embed_backend(embed_backend)
    gen_backend, g_reason = resolve_gen_backend(gen_backend)
    # namespace=company → per-company embeddings cache (no cross-company collision).
    retriever = build_retriever(chunks, embed_backend, use_cache=use_cache, model=embed_model, namespace=company)

    n_pdf = sum(1 for c in chunks if c.get("source_type") == "pdf")
    n_vid = len(chunks) - n_pdf
    cache_note = " [cached]" if getattr(retriever, "cached", False) else ""
    header = (
        f"RAG test [{company}] — {len(chunks)} chunks ({n_vid} video, {n_pdf} pdf) | "
        f"embeddings: {retriever.name}{cache_note} ({e_reason}) | "
        f"generation: {gen_backend} ({g_reason}) | top-k={k}"
    )
    print(header + "\n")
    out = ["# TrainMed RAG test results\n", f"_{header}_\n"]

    for qi, q in enumerate(questions, 1):
        hits = retriever.retrieve(q, k)  # returns chunk dicts (no global indexing)
        retrieved = [c for c, _ in hits]
        scores = [s for _, s in hits]
        answer = generate_answer(q, retrieved, gen_backend, gen_model, company=company)
        sources = format_sources(retrieved, scores)
        out.append(f"## Q{qi}. {q}\n\n**Answer:**\n\n{answer}\n\n**Sources:**\n```\n{sources}\n```\n")
        print(f"{'='*80}\nQ{qi}. {q}\n{'-'*80}\nANSWER:\n{answer}\n\nSOURCES:\n{sources}\n")

    results_path = KB_DIR / ("rag_test_results.md" if company == co.DEFAULT_COMPANY
                             else f"rag_test_results_{company.lower()}.md")
    results_path.write_text("\n".join(out), encoding="utf-8")
    print(f"{'='*80}\nWrote results to {results_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local RAG test over a company's KB chunks.")
    p.add_argument("--company", default=co.DEFAULT_COMPANY,
                   help=f"company KB to test ({', '.join(co.known_companies())}); default {co.DEFAULT_COMPANY}")
    p.add_argument("--question", action="append", help="custom question (repeatable)")
    p.add_argument("-k", type=int, default=4, help="top-k chunks to retrieve")
    p.add_argument("--backend", default="auto", choices=["auto", "openai", "voyage", "st", "tfidf"])
    p.add_argument("--gen", default="auto", choices=["auto", "anthropic", "openai", "extractive"])
    p.add_argument("--embed-model", default=None, help="override embedding model id")
    p.add_argument("--gen-model", default=None, help="override generation model id")
    p.add_argument("--no-cache", action="store_true", help="ignore the embeddings cache")
    args = p.parse_args(argv)

    questions = args.question if args.question else DEFAULT_QUESTIONS
    return run(
        questions, args.k, args.backend, args.gen,
        args.embed_model, args.gen_model, not args.no_cache, args.company,
    )


if __name__ == "__main__":
    raise SystemExit(main())
