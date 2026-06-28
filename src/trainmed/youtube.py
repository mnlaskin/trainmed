"""YouTube URL helpers and video enumeration.

`expand_targets` turns whatever the user passed (a single video URL, a playlist,
a channel, or a bare video id) into a flat list of (video_id, url) pairs using
yt-dlp's flat playlist extraction (metadata only, no download).
"""

from __future__ import annotations

import re

_VIDEO_ID_RE = re.compile(r"^[0-9A-Za-z_-]{11}$")


def parse_video_id(value: str) -> str | None:
    """Extract an 11-char video id from a URL or bare id, or None."""
    value = value.strip()
    if _VIDEO_ID_RE.match(value):
        return value

    # youtu.be/<id>
    m = re.search(r"youtu\.be/([0-9A-Za-z_-]{11})", value)
    if m:
        return m.group(1)

    # youtube.com/watch?v=<id>, /shorts/<id>, /embed/<id>, /live/<id>
    m = re.search(r"[?&]v=([0-9A-Za-z_-]{11})", value)
    if m:
        return m.group(1)
    m = re.search(r"youtube\.com/(?:shorts|embed|live)/([0-9A-Za-z_-]{11})", value)
    if m:
        return m.group(1)

    return None


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _is_collection(value: str) -> bool:
    """True if the value looks like a playlist or channel (multiple videos)."""
    return any(
        token in value
        for token in ("list=", "/playlist", "/channel/", "/@", "/c/", "/user/")
    )


def expand_targets(values: list[str]) -> list[tuple[str, str]]:
    """Resolve mixed inputs to an ordered, de-duplicated list of (id, url).

    Single videos are resolved locally without a network call. Collections
    (playlists/channels) are expanded via yt-dlp's flat extractor.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(vid: str) -> None:
        if vid and vid not in seen:
            seen.add(vid)
            out.append((vid, watch_url(vid)))

    for value in values:
        value = value.strip()
        if not value:
            continue

        vid = parse_video_id(value)
        if vid and not _is_collection(value):
            add(vid)
            continue

        for collected in _expand_collection(value):
            add(collected)

    return out


def _expand_collection(url: str) -> list[str]:
    """Flat-list a playlist or channel into video ids via yt-dlp."""
    from yt_dlp import YoutubeDL

    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,  # don't resolve each video, just list ids
        "skip_download": True,
    }
    ids: list[str] = []
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        entries = info.get("entries") if isinstance(info, dict) else None
        if entries is None:
            # A single video URL slipped through to here.
            vid = info.get("id") if isinstance(info, dict) else None
            if vid:
                ids.append(vid)
        else:
            for entry in entries:
                if not entry:
                    continue
                vid = entry.get("id")
                if vid:
                    ids.append(vid)
    return ids
