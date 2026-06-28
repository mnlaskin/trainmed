"""Speech-to-text for the roleplay rep voice input (ElevenLabs Scribe).

Server-side proxy: the browser records audio (MediaRecorder) and POSTs the raw
bytes to /api/stt; this module forwards them to ElevenLabs Speech-to-Text as a
multipart request and returns the transcript. Keeps the API key off the client.

Stdlib only (urllib builds the multipart body) — no ElevenLabs SDK, and the
browser→server hop is raw bytes so the server needs no python-multipart dependency.

Env vars:
  ELEVENLABS_API_KEY    required (shared with TTS); without it the UI falls back to
                        the browser SpeechRecognition API, or to typing.
  ELEVENLABS_STT_MODEL  override the model (default: scribe_v2; scribe_v1 deprecated)
"""

from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_STT_MODEL = "scribe_v2"  # scribe_v1 is deprecated (removal 2026-07-09)
_API = "https://api.elevenlabs.io/v1/speech-to-text"


def have_key() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def model_id() -> str:
    return os.environ.get("ELEVENLABS_STT_MODEL") or DEFAULT_STT_MODEL


def status() -> dict:
    """What the UI needs to decide how to capture voice input."""
    if have_key():
        return {"provider": "elevenlabs", "model": model_id()}
    return {"provider": "none", "model": None}


def _multipart(fields: dict[str, str], file_bytes: bytes, filename: str, content_type: str) -> tuple[bytes, str]:
    """Build a multipart/form-data body. Returns (body, boundary)."""
    boundary = "----trainmed" + os.urandom(12).hex()
    out: list[bytes] = []
    for name, value in fields.items():
        out.append(f"--{boundary}\r\n".encode())
        out.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        out.append(f"{value}\r\n".encode())
    out.append(f"--{boundary}\r\n".encode())
    out.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    out.append(f"Content-Type: {content_type}\r\n\r\n".encode())
    out.append(file_bytes)
    out.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(out), boundary


def transcribe(audio: bytes, content_type: str = "audio/webm", filename: str = "audio.webm") -> str:
    """Transcribe audio bytes via ElevenLabs Scribe. Raises if no key / on HTTP error."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    body, boundary = _multipart(
        {"model_id": model_id()}, audio, filename, content_type or "audio/webm"
    )
    req = urllib.request.Request(
        _API,
        data=body,
        method="POST",
        headers={
            "xi-api-key": key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("text") or "").strip()
