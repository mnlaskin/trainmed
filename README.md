# TrainMed

AI-powered training and knowledge platform for medical device and pharmaceutical
companies. TrainMed ingests procedural videos, surgical technique recordings,
product training webinars, and sales calls, then powers:

- An **intelligent chatbot** for product knowledge and procedure questions.
- An **interactive AI roleplay trainer** for sales reps, clinical specialists,
  and surgeons to practice scenarios, objection handling, and technique discussions.

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
