"""TrainMed web chatbot — FastAPI + a single static page, streaming over SSE.

Reuses the shared engine in trainmed.rag (no duplicated logic). Multi-company: ONE
retriever PER company is built once at startup into an immutable RETRIEVERS dict, plus
a separate competitive-insights retriever. Each request resolves its company from an
allowlist (no defaulting to a wrong corpus) and retrieves ONLY from that company's
retriever — the contamination firewall. Retrieved chunks pass a fail-closed company
guard before they ever reach a prompt or citation.

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

from trainmed import companies as co  # noqa: E402
from trainmed import stt, tts  # noqa: E402
from trainmed import rag  # noqa: E402

HERE = Path(__file__).resolve().parent
app = FastAPI(title="TrainMed — Surgical Sales Assistant")

EMBED_BACKEND, EMBED_REASON = rag.resolve_embed_backend("auto")
GEN_BACKEND, GEN_REASON = rag.resolve_gen_backend("auto")

# Build ALL company retrievers once at startup into immutable dicts (no per-request
# global mutation → no cross-company race under uvicorn concurrency). Each retriever
# OWNS its (already company-filtered) chunks, so no caller indexes a shared list.
AVAILABLE_COMPANIES = rag.available_companies()
CHUNKS_BY_COMPANY: dict[str, list] = {}
RETRIEVERS: dict[str, object] = {}
for _company in AVAILABLE_COMPANIES:
    _chunks = rag.load_chunks(_company)
    if not _chunks:
        continue
    CHUNKS_BY_COMPANY[_company] = _chunks
    RETRIEVERS[_company] = rag.build_retriever(_chunks, EMBED_BACKEND, namespace=_company)

# Cross-company competitive insights — a SEPARATE retriever, only used for explicit
# comparison / competitive roleplay. Never mixed into a single-company product answer.
COMPETITIVE_CHUNKS = rag.load_competitive_insights()
COMPETITIVE_RETRIEVER = (
    rag.build_retriever(COMPETITIVE_CHUNKS, EMBED_BACKEND, namespace="competitive")
    if COMPETITIVE_CHUNKS else None
)

# The company the UI defaults to: Arthrex if present, else the first available.
DEFAULT_UI_COMPANY = (
    co.DEFAULT_COMPANY if co.DEFAULT_COMPANY in RETRIEVERS
    else (AVAILABLE_COMPANIES[0] if AVAILABLE_COMPANIES else co.DEFAULT_COMPANY)
)


def resolve_company(value: str | None) -> str:
    """Canonical company for a request, defaulting to the UI default when unset."""
    if not value:
        return DEFAULT_UI_COMPANY
    return co.canonical_company(value)


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/companies")
def companies():
    """Every known company + whether it has an ingested KB (for the UI selector)."""
    return {
        "companies": [
            {
                "id": c,
                "label": co.display_name(c),
                "available": c in RETRIEVERS,
                "chunks": len(CHUNKS_BY_COMPANY.get(c, [])),
            }
            for c in co.known_companies()
        ],
        "default": DEFAULT_UI_COMPANY,
        "competitive_insights": len(COMPETITIVE_CHUNKS),
    }


@app.get("/api/meta")
def meta(company: str | None = None):
    company = resolve_company(company)
    chunks = CHUNKS_BY_COMPANY.get(company, [])
    retriever = RETRIEVERS.get(company)
    n_pdf = sum(1 for c in chunks if c.get("source_type") == "pdf")
    return {
        "company": company,
        "company_label": co.display_name(company),
        "available": company in RETRIEVERS,
        "chunks": len(chunks),
        "videos": len(chunks) - n_pdf,
        "pdfs": n_pdf,
        "competitive_insights": len(COMPETITIVE_CHUNKS),
        "embed_backend": getattr(retriever, "name", EMBED_BACKEND),
        "embed_reason": EMBED_REASON,
        "gen_backend": GEN_BACKEND,
        "gen_reason": GEN_REASON,
    }


@app.get("/api/ask")
def ask(q: str, k: int = 4, company: str | None = None):
    company = resolve_company(company)
    retriever = RETRIEVERS.get(company)
    if retriever is None:
        return {
            "error": f"No knowledge base for {co.display_name(company)} yet. "
                     f"Ingest it first: python -m trainmed.pdf_ingest --company {company} "
                     f"--from-file data/urls/<file>.txt && python scripts/ingest_to_kb.py --company {company}.",
            "company": company,
        }

    # retrieve() returns chunk dicts directly (no positional indexing into a global
    # list); the guard then fail-closes on any chunk that isn't this company's.
    hits = retriever.retrieve(q, k)
    retrieved = rag.enforce_company([c for c, _ in hits], company)
    cites = [rag.citation(n, c, s) for n, (c, s) in enumerate(hits, 1)
             if c.get("company") == company]

    def events():
        yield _sse("sources", cites)  # render the citations panel immediately
        for delta in rag.stream_answer(q, retrieved, GEN_BACKEND, company=company):
            yield _sse("token", delta)
        yield _sse("done", {"company": company, "embed": EMBED_BACKEND, "gen": GEN_BACKEND})

    return StreamingResponse(events(), media_type="text/event-stream")


# ── roleplay trainer ──────────────────────────────────────────────────────────


@app.get("/api/roleplay/scenarios")
def roleplay_scenarios(company: str | None = None):
    company = resolve_company(company)
    scens = rag.scenarios_for(company)
    return {
        "company": company,
        "company_label": co.display_name(company),
        "scenarios": [
            {"id": s["id"], "label": s["label"], "opening": s["opening"],
             "goal": s["goal"], "competitive": bool(s.get("competitive")),
             "persona": s.get("persona", ""), "prospect": s.get("prospect", {})}
            for s in scens
        ],
        "roles": [{"id": k, "label": v} for k, v in rag.ROLES.items()],
        "difficulties": [
            {"id": d, "label": rag.DIFFICULTY_META.get(d, {}).get("label", d.capitalize()),
             "blurb": rag.DIFFICULTY_META.get(d, {}).get("blurb", "")}
            for d in rag.DIFFICULTY.keys()
        ],
        "gen_backend": GEN_BACKEND,
    }


@app.post("/api/roleplay/turn")
async def roleplay_turn(request: Request):
    body = await request.json()
    company = resolve_company(body.get("company"))
    scenario = rag.scenario_by_id(body.get("scenario_id"), company)
    role = body.get("role", "sales_rep")
    difficulty = body.get("difficulty", "medium")
    history = body.get("history", [])
    message = (body.get("message") or "").strip()

    # Ground the surgeon's pushback (and the coach's scoring) in THIS company's content
    # only. For a competitive scenario, ALSO add cross-company competitive insights —
    # the curated comparison notes, never the raw competitor product corpus.
    query = message or scenario.get("goal", "")
    retriever = RETRIEVERS.get(company)
    grounded = [c for c, _ in retriever.retrieve(query, 4)] if retriever else []
    grounded = rag.enforce_company(grounded, company)  # fail-closed
    if scenario.get("competitive") and COMPETITIVE_RETRIEVER is not None:
        grounded = grounded[:3] + [c for c, _ in COMPETITIVE_RETRIEVER.retrieve(query, 2)]

    def events():
        for delta in rag.stream_roleplay(
            scenario, role, difficulty, history, message, grounded, GEN_BACKEND, company=company
        ):
            yield _sse("surgeon", delta)
        # Signal the reply is complete so the client can start TTS immediately,
        # in parallel with the coach scoring below (which no longer blocks the voice).
        yield _sse("surgeon_done", {})
        card = rag.score_exchange(scenario, message, grounded, GEN_BACKEND, company=company)
        yield _sse("feedback", card)
        yield _sse("done", {"company": company, "gen": GEN_BACKEND})

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
