"""Data models for extracted transcripts.

Kept as plain dataclasses so we can serialize to JSON without extra deps and
swap in a DB layer later without changing the extraction code.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class Segment:
    """A single timed caption segment."""

    text: str
    start: float  # seconds from start of video
    duration: float  # seconds

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Transcript:
    """A fully extracted transcript plus the video metadata around it."""

    video_id: str
    url: str
    title: str | None = None
    channel: str | None = None
    channel_id: str | None = None
    duration: float | None = None  # seconds
    upload_date: str | None = None  # YYYYMMDD as returned by yt-dlp
    description: str | None = None
    language: str | None = None
    is_generated: bool | None = None  # auto-generated captions vs human
    source: str = "youtube"
    segments: list[Segment] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["full_text"] = self.text()
        d["word_count"] = len(self.text().split())
        return d

    def text(self) -> str:
        """Continuous transcript text, whitespace-normalized."""
        parts = [" ".join(seg.text.split()) for seg in self.segments]
        return " ".join(p for p in parts if p)
