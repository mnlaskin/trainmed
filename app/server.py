"""TrainMed web chatbot — FastAPI + a single static page, streaming over SSE.

Reuses the shared engine in trainmed.rag (no duplicated logic). Multi-company: ONE
retriever PER company, built lazily and memoized on first use (and pre-warmed by a
background thread), plus a separate competitive-insights retriever. Building is deferred
off the import path so the server binds — and the platform health check passes — without
waiting on cold-disk KB reads. Each request resolves its company from an allowlist (no
defaulting to a wrong corpus) and retrieves ONLY from that company's retriever — the
contamination firewall. Retrieved chunks pass a fail-closed company guard before they
ever reach a prompt or citation.

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
import threading
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

# Companies that have a KB directory on disk (cheap — just a dir scan, no file reads).
AVAILABLE_COMPANIES = rag.available_companies()

# The company the UI defaults to: Arthrex if it has a KB, else the first available.
DEFAULT_UI_COMPANY = (
    co.DEFAULT_COMPANY if co.DEFAULT_COMPANY in AVAILABLE_COMPANIES
    else (AVAILABLE_COMPANIES[0] if AVAILABLE_COMPANIES else co.DEFAULT_COMPANY)
)

# Retrievers are built LAZILY (and memoized), NOT at import. Building one reads that
# company's whole KB from disk and constructs the TF-IDF / embedding index; doing all
# of that at module import blocks uvicorn from binding the port — and the Render deploy
# health check from ever getting a response — on a cold, slow free-tier instance. So we
# defer the work behind getters and warm it in a background thread once the server is
# already accepting connections. The lock keeps the "ONE retriever per company" guarantee
# (and the firewall) intact under concurrent first-touch requests.
_CHUNKS: dict[str, list] = {}
_RETRIEVERS: dict[str, object] = {}
_COMPETITIVE: dict[str, object] = {}      # "chunks" + "retriever" populated on first use
_build_lock = threading.RLock()


def get_chunks(company: str) -> list:
    """This company's chunks, loaded + memoized on first access (fail-closed: [] if none)."""
    if company not in _CHUNKS:
        if company not in AVAILABLE_COMPANIES:
            return []
        with _build_lock:
            if company not in _CHUNKS:
                _CHUNKS[company] = rag.load_chunks(company)
    return _CHUNKS.get(company, [])


def get_retriever(company: str):
    """This company's retriever, built + memoized on first access (None if no KB)."""
    if company in _RETRIEVERS:
        return _RETRIEVERS[company]
    chunks = get_chunks(company)
    if not chunks:
        return None
    with _build_lock:
        if company not in _RETRIEVERS:
            _RETRIEVERS[company] = rag.build_retriever(chunks, EMBED_BACKEND, namespace=company)
    return _RETRIEVERS[company]


def competitive_chunks() -> list:
    if "chunks" not in _COMPETITIVE:
        with _build_lock:
            if "chunks" not in _COMPETITIVE:
                _COMPETITIVE["chunks"] = rag.load_competitive_insights()
    return _COMPETITIVE["chunks"]


def get_competitive_retriever():
    """Cross-company competitive insights — a SEPARATE retriever, only for competitive
    roleplay. Never mixed into a single-company product answer."""
    if "retriever" not in _COMPETITIVE:
        chunks = competitive_chunks()
        with _build_lock:
            if "retriever" not in _COMPETITIVE:
                _COMPETITIVE["retriever"] = (
                    rag.build_retriever(chunks, EMBED_BACKEND, namespace="competitive")
                    if chunks else None
                )
    return _COMPETITIVE["retriever"]


def _warm() -> None:
    """Pre-build every retriever AFTER the server is already accepting connections, so the
    first real request is fast without blocking startup or the deploy health check."""
    for c in AVAILABLE_COMPANIES:
        try:
            get_retriever(c)
        except Exception:  # a single bad corpus must not kill warming for the others
            pass
    try:
        get_competitive_retriever()
    except Exception:
        pass


# Warm in the background; daemon so it never holds the process open on shutdown.
threading.Thread(target=_warm, name="trainmed-warm", daemon=True).start()


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


@app.get("/healthz")
def healthz():
    """Liveness probe for the platform health check. Intentionally does NO data access so
    it returns 200 the instant the server binds — even while retrievers are still warming."""
    return {"status": "ok"}


@app.get("/api/companies")
def companies():
    """Every known company + whether it has an ingested KB (for the UI selector)."""
    return {
        "companies": [
            {
                "id": c,
                "label": co.display_name(c),
                "available": bool(get_chunks(c)),
                "chunks": len(get_chunks(c)),
            }
            for c in co.known_companies()
        ],
        "default": DEFAULT_UI_COMPANY,
        "competitive_insights": len(competitive_chunks()),
    }


@app.get("/api/meta")
def meta(company: str | None = None):
    company = resolve_company(company)
    chunks = get_chunks(company)
    retriever = _RETRIEVERS.get(company)
    n_pdf = sum(1 for c in chunks if c.get("source_type") == "pdf")
    return {
        "company": company,
        "company_label": co.display_name(company),
        "available": bool(chunks),
        "chunks": len(chunks),
        "videos": len(chunks) - n_pdf,
        "pdfs": n_pdf,
        "competitive_insights": len(competitive_chunks()),
        "embed_backend": getattr(retriever, "name", EMBED_BACKEND),
        "embed_reason": EMBED_REASON,
        "gen_backend": GEN_BACKEND,
        "gen_reason": GEN_REASON,
    }


@app.get("/api/ask")
def ask(q: str, k: int = 4, company: str | None = None):
    company = resolve_company(company)
    retriever = get_retriever(company)
    if retriever is None:
        return {
            "error": f"No knowledge base for {co.display_name(company)} yet. "
                     f"Ingest it first: python -m trainmed.pdf_ingest --company {company} "
                     f"--from-file data/urls/<file>.txt && python scripts/ingest_to_kb.py --company {company}.",
            "company": company,
        }

    # retrieve() returns chunk dicts directly (no positional indexing into a global list).
    # Build the model context and the UI citations TOGETHER, sharing one [n] numbering, so an
    # inline [n] in the answer always maps to cites[n-1]. The company guard fail-closes on any
    # chunk that isn't this company's before it can be cited.
    retrieved: list[dict] = []
    cites: list[dict] = []
    for c, s in retriever.retrieve(q, k):
        if c.get("company") != company:
            continue
        retrieved.append(c)
        cites.append(rag.citation(len(retrieved), c, s))

    # Competitive question → ALSO add the curated cross-company comparison notes (sentinel
    # company="_competitive"). This is the ONLY cross-brand data the chatbot ever sees; the
    # other company's raw product corpus is never pulled in, so the firewall holds.
    competitive = rag.is_competitive_query(q, company)
    if competitive:
        comp = get_competitive_retriever()
        if comp is not None:
            for c, s in comp.retrieve(q, 2):
                retrieved.append(c)
                cites.append(rag.citation(len(retrieved), c, s))

    def events():
        yield _sse("sources", cites)  # render the citations panel immediately
        for delta in rag.stream_answer(q, retrieved, GEN_BACKEND, company=company):
            yield _sse("token", delta)
        yield _sse("done", {"company": company, "embed": EMBED_BACKEND, "gen": GEN_BACKEND,
                            "competitive": competitive})

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
    retriever = get_retriever(company)
    grounded = [c for c, _ in retriever.retrieve(query, 4)] if retriever else []
    grounded = rag.enforce_company(grounded, company)  # fail-closed
    comp = get_competitive_retriever() if scenario.get("competitive") else None
    if comp is not None:
        grounded = grounded[:3] + [c for c, _ in comp.retrieve(query, 2)]

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
