# TrainMed

AI-powered training and knowledge platform for medical device and pharmaceutical
companies. TrainMed ingests procedural videos, surgical technique recordings,
product training webinars, and sales calls, then powers:

- An **intelligent chatbot** for product knowledge and procedure questions.
- An **interactive AI roleplay trainer** for sales reps, clinical specialists,
  and surgeons to practice scenarios, objection handling, and technique discussions.

## Multi-company (Arthrex · Stryker · Smith & Nephew)

TrainMed serves multiple companies from one app with a **strict contamination
firewall** — a product answer or roleplay for one company is never grounded in another's
content. Every chunk carries a `company` field plus product metadata
(`product_line`, `product_family`, `product_name`, `category`, `technique`,
`advantages`, `disadvantages`, `clinical_references`); each company has its own retriever;
and cross-company selling points live in a separate `competitive_insights` collection.

**See [`docs/MULTI_COMPANY.md`](docs/MULTI_COMPANY.md)** for the schema, the firewall, the
CLI, and the Stryker / Smith & Nephew ingestion plan.

```bash
# Ingest a company's shoulder PDFs end to end (Stryker shown):
python -m trainmed.pdf_ingest --company Stryker --from-file data/urls/stryker_shoulder_pdfs.txt
python scripts/ingest_to_kb.py --company Stryker
PYTHONPATH=src python scripts/test_isolation.py     # prove no cross-company leakage
```

## Current focus: transcript extraction → knowledge base

This repo currently contains the **ingestion pipeline**: tools to extract
transcripts from public YouTube videos (and, later, company video libraries) and
store them as structured local files that will feed the knowledge base.

### Architecture (today)

```
YouTube URL / playlist / channel / URL-list file
        │
        ▼
  extract.py  ──(captions via youtube-transcript-api)──► transcript segments
        │     ──(metadata via yt-dlp)──────────────────► title, channel, etc.
        ▼
  store.py    ──► data/transcripts/<video_id>.json   (structured)
              ──► data/transcripts/<video_id>.md     (human/LLM readable)
```

Storage is **local files first**. A vector DB / Supabase+pgvector layer comes
once extraction is solid.

## Setup

```bash
cd ~/Desktop/TrainMed
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# A single video
python -m trainmed.cli "https://www.youtube.com/watch?v=VIDEO_ID"

# Multiple videos / playlists / channels at once
python -m trainmed.cli "https://www.youtube.com/playlist?list=PLAYLIST_ID"

# From a file with one URL per line
python -m trainmed.cli --from-file urls.txt

# List available languages for a video without downloading
python -m trainmed.cli --list-langs "https://youtu.be/VIDEO_ID"
```

Output lands in `data/transcripts/`. Run with `PYTHONPATH=src` or install the
package (`pip install -e .`) so `trainmed` is importable.
