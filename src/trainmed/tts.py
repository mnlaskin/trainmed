"""Text-to-speech for the roleplay surgeon voice.

Server-side proxy to ElevenLabs (keeps the API key off the browser). If no
ELEVENLABS_API_KEY is set, `status()` reports the "browser" provider and the UI
falls back to the Web Speech API (speechSynthesis) — so voice still works with
zero setup, just lower quality.

Stdlib only (urllib) — no ElevenLabs SDK dependency.

Env vars:
  ELEVENLABS_API_KEY   required for ElevenLabs voice (else browser fallback)
  ELEVENLABS_VOICE_ID  override the default voice (default: Adam — deep, authoritative male)
  ELEVENLABS_MODEL     override the model (default: eleven_turbo_v2_5 — natural + responsive)
"""

from __future__ import annotations

import json
import os
import urllib.request

# Premade ElevenLabs voices well suited to an authoritative male "surgeon".
# NOTE: these premade voice IDs work on accounts created before ~March 2026 and are
# slated to expire after 2026-12-31. On a newer account, add a voice to your library
# and set ELEVENLABS_VOICE_ID to its account-scoped ID. If the ID is invalid the API
# returns an error, the /api/tts route 502s, and the UI falls back to browser speech.
VOICES = {
    "Custom": "APIhFhqDf7R3k5WtU5UI",  # user-selected default voice
    "Adam": "pNInz6obpgDQGcFmaJgB",   # deep, mature, authoritative
    "Josh": "TxGEqnHWrfWFTfGW9XjX",   # younger, confident male
    "Antoni": "ErXwobaYiN019PkySvjV",  # warm, professional male
    "Arnold": "VR6AewLTigWG4xSOukaG",  # crisp, assertive male
}
DEFAULT_VOICE_NAME = "Custom"
DEFAULT_MODEL = "eleven_turbo_v2_5"
MAX_CHARS = 1800  # cap per request to bound latency/cost

_API = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
# The /stream variant returns the SAME MP3 bytes, but ElevenLabs flushes them as they
# synthesize instead of buffering the whole clip — so the browser can start playback on
# the first chunk. Same auth, same body, same audio; just incremental delivery.
_STREAM_API = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"


def have_key() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def voice_id() -> str:
    return os.environ.get("ELEVENLABS_VOICE_ID") or VOICES[DEFAULT_VOICE_NAME]


def voice_name() -> str:
    vid = voice_id()
    for name, vd in VOICES.items():
        if vd == vid:
            return name
    return "custom"


def model_id() -> str:
    return os.environ.get("ELEVENLABS_MODEL") or DEFAULT_MODEL


def status() -> dict:
    """What the UI needs to decide how to speak."""
    if have_key():
        return {"provider": "elevenlabs", "voice": voice_name(), "model": model_id()}
    return {"provider": "browser", "voice": "browser default", "model": None}


def _build_request(text: str, url: str):
    """Shared ElevenLabs request (auth + body + voice settings). Raises if no key."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    payload = json.dumps(
        {
            "text": text[:MAX_CHARS],
            "model_id": model_id(),
            # Tuned for a calm, authoritative clinical cadence.
            "voice_settings": {
                "stability": 0.55,
                "similarity_boost": 0.75,
                "style": 0.1,
                "use_speaker_boost": True,
                "speed": 0.95,
            },
        }
    ).encode("utf-8")
    return urllib.request.Request(
        url.format(voice_id=voice_id()),
        data=payload,
        method="POST",
        headers={
            "xi-api-key": key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )


def synthesize(text: str) -> bytes:
    """Return the full MP3 bytes for `text` from ElevenLabs. Raises if no key / on HTTP error."""
    req = _build_request(text, _API)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def synthesize_stream(text: str, chunk_size: int = 4096):
    """Yield MP3 bytes from ElevenLabs' streaming endpoint as they arrive, so the browser can
    begin playback on the first chunk instead of waiting for the whole clip. The urlopen happens
    on the first `next()`, so an auth/quota/HTTP error is raised there — letting the route surface
    it as a 502 BEFORE committing to a 200 streaming response (client then falls back gracefully)."""
    req = _build_request(text, _STREAM_API)
    resp = urllib.request.urlopen(req, timeout=60)
    try:
        while True:
            # read1() returns bytes as soon as any arrive (one underlying read), instead of
            # blocking to fill chunk_size — that's what makes playback start on the first chunk.
            chunk = resp.read1(chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        resp.close()
