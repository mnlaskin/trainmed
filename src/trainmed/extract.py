"""Transcript extraction for a single YouTube video.

Captions come from youtube-transcript-api (fast, no audio download). Metadata
(title, channel, duration, ...) comes from yt-dlp. Both are best-effort: a video
with captions but a metadata hiccup still yields a usable transcript, and vice
versa.
"""

from __future__ import annotations

from .models import Segment, Transcript
from .youtube import watch_url


class TranscriptUnavailable(Exception):
    """Raised when no captions could be retrieved for a video."""


def fetch_metadata(video_id: str) -> dict:
    """Fetch video metadata via yt-dlp. Returns {} on failure."""
    from yt_dlp import YoutubeDL

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(watch_url(video_id), download=False)
    except Exception:
        return {}
    if not isinstance(info, dict):
        return {}
    return {
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "channel_id": info.get("channel_id"),
        "duration": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "description": info.get("description"),
    }


def _fetch_segments(
    video_id: str, languages: list[str] | None
) -> tuple[list[Segment], str | None, bool | None]:
    """Return (segments, language_code, is_generated) using youtube-transcript-api.

    Supports both the >=1.0 instance API and the older static API so the tool
    works across versions without pinning hard.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    prefs = languages or ["en"]

    # New API (>= 1.0): instance with .list()/.fetch()
    if hasattr(YouTubeTranscriptApi, "__init__") and hasattr(
        YouTubeTranscriptApi, "list"
    ):
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            chosen = _choose_transcript(transcript_list, prefs)
            fetched = chosen.fetch()
            segments = [
                Segment(text=s.text, start=s.start, duration=s.duration)
                for s in fetched
            ]
            return segments, chosen.language_code, chosen.is_generated
        except AttributeError:
            pass  # fall through to legacy path

    # Legacy API (< 1.0): static methods returning list[dict]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    chosen = _choose_transcript(transcript_list, prefs)
    raw = chosen.fetch()
    segments = [
        Segment(text=s["text"], start=s["start"], duration=s["duration"])
        for s in raw
    ]
    return segments, chosen.language_code, chosen.is_generated


def _choose_transcript(transcript_list, prefs: list[str]):
    """Prefer a manually created transcript in a preferred language, then any
    generated one, then translate to the first preference if possible."""
    try:
        return transcript_list.find_manually_created_transcript(prefs)
    except Exception:
        pass
    try:
        return transcript_list.find_generated_transcript(prefs)
    except Exception:
        pass
    # Any available transcript; translate to first preference if needed.
    for t in transcript_list:
        if t.language_code in prefs:
            return t
    for t in transcript_list:
        if t.is_translatable and prefs:
            try:
                return t.translate(prefs[0])
            except Exception:
                continue
        return t
    raise TranscriptUnavailable("no transcripts listed for this video")


def extract(
    video_id: str,
    *,
    languages: list[str] | None = None,
    with_metadata: bool = True,
) -> Transcript:
    """Extract a full Transcript for a single video id."""
    try:
        segments, lang, is_generated = _fetch_segments(video_id, languages)
    except TranscriptUnavailable:
        raise
    except Exception as exc:  # normalize library-specific errors
        raise TranscriptUnavailable(str(exc)) from exc

    if not segments:
        raise TranscriptUnavailable("transcript was empty")

    meta = fetch_metadata(video_id) if with_metadata else {}
    return Transcript(
        video_id=video_id,
        url=watch_url(video_id),
        title=meta.get("title"),
        channel=meta.get("channel"),
        channel_id=meta.get("channel_id"),
        duration=meta.get("duration"),
        upload_date=meta.get("upload_date"),
        description=meta.get("description"),
        language=lang,
        is_generated=is_generated,
        segments=segments,
    )


def list_languages(video_id: str) -> list[dict]:
    """List available caption tracks for a video (for --list-langs)."""
    from youtube_transcript_api import YouTubeTranscriptApi

    if hasattr(YouTubeTranscriptApi, "list"):
        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)
        except AttributeError:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    else:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    out = []
    for t in transcript_list:
        out.append(
            {
                "language": t.language,
                "language_code": t.language_code,
                "is_generated": t.is_generated,
                "is_translatable": t.is_translatable,
            }
        )
    return out
