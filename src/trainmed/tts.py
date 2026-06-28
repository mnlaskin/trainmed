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


def synthesize(text: str) -> bytes:
    """Return MP3 bytes for `text` from ElevenLabs. Raises if no key / on HTTP error."""
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
    req = urllib.request.Request(
        _API.format(voice_id=voice_id()),
        data=payload,
        method="POST",
        headers={
            "xi-api-key": key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()
