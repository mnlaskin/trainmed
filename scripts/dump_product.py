#!/usr/bin/env python3
"""Dump one company+product_family's chunks (text + provenance) as JSON.

Used by the structured-spec-extraction agents so each agent sees ONLY its product's
ingested text. Company-scoped (firewall-safe): never reads another company's chunks.

    python scripts/dump_product.py "Arthrex" "SwiveLock"
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trainmed import rag  # noqa: E402

company, family = sys.argv[1], sys.argv[2]
out = []
for c in rag.load_chunks(company):
    if (c.get("product_family") or "") == family:
        out.append({
            "chunk_id": c.get("chunk_id"),
            "title": c.get("source_title"),
            "url": rag.deep_link(c),
            "category": c.get("category"),
            "text": c.get("text", ""),
        })
print(json.dumps({"company": company, "product_family": family, "chunks": out}, ensure_ascii=False))
