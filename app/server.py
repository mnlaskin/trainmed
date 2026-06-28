"""TrainMed web chatbot — FastAPI + a single static page, streaming over SSE.

Reuses the shared engine in trainmed.rag (no duplicated logic). The retriever is
built once at startup; each request retrieves top-k chunks, streams the citations
first (so the panel renders instantly) then the answer tokens.

Run (from the repo root, with the package importable):
    pip install -e .                       # so `trainmed` imports
    pip install fastapi "uvicorn[standard]"
    uvicorn app.server:app --port 8000     # add ANTHROPIC_API_KEY / OPENAI_API_KEY to go neural
    # open http://localhost:8000

Backends resolve at startup, so set the API key BEFORE launching.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `trainmed` importable straight from a checkout (no `pip install -e .` needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, Response, StreamingResponse  # noqa: E402

from trainmed import stt, tts  # noqa: E402
from trainmed.rag import (  # noqa: E402
    DIFFICULTY,
    ROLES,
    SCENARIO_BY_ID,
    SCENARIOS,
    build_retriever,
    citation,
    load_chunks,
    resolve_embed_backend,
    resolve_gen_backend,
    score_exchange,
    stream_answer,
    stream_roleplay,
)

HERE = Path(__file__).resolve().parent
app = FastAPI(title="TrainMed — Rotator Cuff Assistant")

# Build everything once at startup (68 chunks → instant for TF-IDF).
CHUNKS = load_chunks()
EMBED_BACKEND, EMBED_REASON = resolve_embed_backend("auto")
GEN_BACKEND, GEN_REASON = resolve_gen_backend("auto")
RETRIEVER = build_retriever(CHUNKS, EMBED_BACKEND) if CHUNKS else None


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/meta")
def meta():
    n_pdf = sum(1 for c in CHUNKS if c.get("source_type") == "pdf")
    return {
        "chunks": len(CHUNKS),
        "videos": len(CHUNKS) - n_pdf,
        "pdfs": n_pdf,
        "embed_backend": getattr(RETRIEVER, "name", EMBED_BACKEND),
        "embed_reason": EMBED_REASON,
        "gen_backend": GEN_BACKEND,
        "gen_reason": GEN_REASON,
    }


@app.get("/api/ask")
def ask(q: str, k: int = 4):
    if not RETRIEVER:
        return {"error": "No KB chunks. Run scripts/ingest_to_kb.py first."}

    hits = RETRIEVER.search(q, k)
    retrieved = [CHUNKS[i] for i, _ in hits]
    cites = [citation(n, CHUNKS[i], s) for n, (i, s) in enumerate(hits, 1)]

    def sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def events():
        yield sse("sources", cites)  # render the citations panel immediately
        for delta in stream_answer(q, retrieved, GEN_BACKEND):
            yield sse("token", delta)
        yield sse("done", {"embed": EMBED_BACKEND, "gen": GEN_BACKEND})

    return StreamingResponse(events(), media_type="text/event-stream")


# ── roleplay trainer ──────────────────────────────────────────────────────────


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/roleplay/scenarios")
def roleplay_scenarios():
    return {
        "scenarios": [
            {"id": s["id"], "label": s["label"], "opening": s["opening"], "goal": s["goal"]}
            for s in SCENARIOS
        ],
        "roles": [{"id": k, "label": v} for k, v in ROLES.items()],
        "difficulties": list(DIFFICULTY.keys()),
        "gen_backend": GEN_BACKEND,
    }


@app.post("/api/roleplay/turn")
async def roleplay_turn(request: Request):
    body = await request.json()
    scenario = SCENARIO_BY_ID.get(body.get("scenario_id")) or SCENARIOS[0]
    role = body.get("role", "sales_rep")
    difficulty = body.get("difficulty", "medium")
    history = body.get("history", [])
    message = (body.get("message") or "").strip()

    # Ground the surgeon's pushback (and the coach's scoring) in real Arthrex content.
    query = message or scenario["goal"]
    hits = RETRIEVER.search(query, 4) if RETRIEVER else []
    retrieved = [CHUNKS[i] for i, _ in hits]

    def events():
        for delta in stream_roleplay(scenario, role, difficulty, history, message, retrieved, GEN_BACKEND):
            yield _sse("surgeon", delta)
        # Signal the reply is complete so the client can start TTS immediately,
        # in parallel with the coach scoring below (which no longer blocks the voice).
        yield _sse("surgeon_done", {})
        card = score_exchange(scenario, message, retrieved, GEN_BACKEND)
        yield _sse("feedback", card)
        yield _sse("done", {"gen": GEN_BACKEND})

    return StreamingResponse(events(), media_type="text/event-stream")


# ── voice (text-to-speech) ────────────────────────────────────────────────────


@app.get("/api/tts/status")
def tts_status():
    """Tells the UI whether to use the server (ElevenLabs) or browser speech."""
    return tts.status()


@app.get("/api/tts")
def tts_speak(text: str):
    """Stream `text` as MP3 via ElevenLabs so playback can begin on the first chunk.
    503 → the UI uses the browser fallback; 502 → client falls back gracefully."""
    if not tts.have_key():
        return Response(status_code=503, content="ELEVENLABS_API_KEY not set; use browser speech")
    try:
        stream = tts.synthesize_stream(text)
        first = next(stream)  # force the connection now → auth/quota errors become a 502, not a broken stream
    except StopIteration:
        first = b""
    except Exception as exc:  # surface as 502 BEFORE we commit to a 200 stream
        return Response(status_code=502, content=f"TTS error: {exc}")

    def body():
        if first:
            yield first
        yield from stream

    return StreamingResponse(
        body(),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )


# ── speech-to-text (microphone input) ─────────────────────────────────────────


@app.get("/api/stt/status")
def stt_status():
    """Tells the UI whether server STT (ElevenLabs) is available."""
    return stt.status()


@app.post("/api/stt")
async def stt_transcribe(request: Request):
    """Transcribe raw audio bytes (sent as the request body) via ElevenLabs Scribe.
    503 → UI falls back to browser SpeechRecognition or typing."""
    if not stt.have_key():
        return Response(status_code=503, content="ELEVENLABS_API_KEY not set; use browser speech input")
    audio = await request.body()
    if not audio:
        return Response(status_code=400, content="empty audio")
    ctype = request.headers.get("content-type", "audio/webm").split(";")[0]
    ext = "mp4" if "mp4" in ctype else "ogg" if "ogg" in ctype else "webm"
    try:
        text = stt.transcribe(audio, content_type=ctype, filename=f"audio.{ext}")
    except Exception as exc:
        return Response(status_code=502, content=f"STT error: {exc}")
    return {"text": text}
