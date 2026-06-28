"""Local-file persistence for transcripts.

Writes two artifacts per video into `data/transcripts/`:
  - <video_id>.json : structured record (metadata + segments + full_text)
  - <video_id>.md   : human/LLM-readable markdown with a YAML-ish front matter

Local files first; a vector DB / Supabase layer can read these later.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import Transcript

# Repo root = three levels up from this file (src/trainmed/store.py).
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "transcripts"


def _format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def transcript_to_markdown(t: Transcript) -> str:
    lines = [
        "---",
        f"title: {t.title or ''}",
        f"video_id: {t.video_id}",
        f"url: {t.url}",
        f"channel: {t.channel or ''}",
        f"upload_date: {t.upload_date or ''}",
        f"language: {t.language or ''}",
        f"auto_generated: {t.is_generated}",
        f"source: {t.source}",
        "---",
        "",
        f"# {t.title or t.video_id}",
        "",
        t.text(),
        "",
    ]
    return "\n".join(lines)


def save(
    t: Transcript, data_dir: Path | None = None, *, write_markdown: bool = True
) -> dict[str, Path]:
    """Persist a transcript; returns the paths written."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    json_path = data_dir / f"{t.video_id}.json"
    json_path.write_text(
        json.dumps(t.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    written = {"json": json_path}

    if write_markdown:
        md_path = data_dir / f"{t.video_id}.md"
        md_path.write_text(transcript_to_markdown(t), encoding="utf-8")
        written["markdown"] = md_path

    return written


def already_saved(video_id: str, data_dir: Path | None = None) -> bool:
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return (data_dir / f"{video_id}.json").exists()
