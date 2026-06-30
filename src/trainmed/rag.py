"""TrainMed RAG engine — the single source of truth for retrieval + generation.

Both the CLI (`scripts/test_rag.py`) and the web app (`app/server.py`) import from
here, so there is exactly one copy of the chunk loading, tokenization, retrievers,
backend selection, prompt, and answer generation.

Backend-flexible and auto-selecting (see resolve_embed_backend / resolve_gen_backend):
  embeddings: openai -> voyage -> sentence-transformers -> tfidf
  generation: anthropic -> openai -> extractive
Neural embeddings are cached to data/kb/embeddings_<namespace>_<backend>_<model>.npz,
namespaced per company so each company's corpus has its own cache.

Multi-company: every chunk carries a `company` field and corpora are kept strictly
separate. load_chunks(company) is the retrieval firewall — it returns ONLY that
company's chunks — so a per-company retriever can never surface a competitor's text.
The cross-company `competitive_insights` collection is loaded separately and only
used for explicit comparison / competitive roleplay.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from . import companies as co


def _warn(msg: str) -> None:
    """Loud, fail-closed warning to stderr (used by the contamination guards)."""
    print(f"[trainmed.rag] WARNING: {msg}", file=sys.stderr, flush=True)

# rag.py lives at src/trainmed/rag.py -> repo root is parents[2].
ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / "data" / "kb"
CHUNKS_DIR = KB_DIR / "chunks"  # legacy flat dir (Arthrex); kept for backward-compat
COMPETITIVE_DIR = KB_DIR / "competitive_insights"  # cross-company collection (kept separate)

DEFAULT_QUESTIONS = [
    "What is the Arthrex SpeedBridge and which anchors and sutures does it use to repair a rotator cuff tear?",
    "How does the SutureBridge double-row technique differ from the SpeedBridge for rotator cuff repair?",
    "What graft and fixation are used in superior capsular reconstruction, and when is it indicated?",
    "How is a subscapularis tendon tear repaired with a knotless technique?",
    "What biological or augmentation options can improve rotator cuff healing?",
]

GEN_MODEL_ANTHROPIC = "claude-sonnet-4-6"
GEN_MODEL_OPENAI = "gpt-4o-mini"
EMBED_MODEL_OPENAI = "text-embedding-3-small"
EMBED_MODEL_VOYAGE = "voyage-3"
EMBED_MODEL_ST = "sentence-transformers/all-MiniLM-L6-v2"

_STOPWORDS = set(
    "a an and are as at be by for from has have how in into is it its of on or that the "
    "this to was were what when where which who will with you your we our us can do does "
    "using use used".split()
)

# Auto-captions split/mistranscribe product names. Canonicalize so distinctive
# names match (helps TF-IDF; harmless for neural embeddings).
_PHRASE_NORMALIZE = [
    (re.compile(r"\bspeed\s*bridge\b"), "speedbridge"),
    (re.compile(r"\bsuture\s*bridge\b"), "suturebridge"),
    (re.compile(r"\bspeed\s*fix\b"), "speedfix"),
    (re.compile(r"\bfiber\s*ta(?:k|ck)\b"), "fibertak"),
    (re.compile(r"\bswivel\s*lock\b"), "swivelock"),
    (re.compile(r"\bpush\s*lock\b"), "pushlock"),
    (re.compile(r"\bsuture\s*ta(?:k|ck)\b"), "suturetak"),
    (re.compile(r"\binternal\s*brace\b"), "internalbrace"),
    (re.compile(r"\bcuff\s*men[d]?\b"), "cuffmend"),
    (re.compile(r"\barrix\b"), "arthrex"),
    # Competitor product names (Stryker / Smith & Nephew) so TF-IDF matches them too.
    (re.compile(r"\breel\s*x\b"), "reelx"),
    (re.compile(r"\balpha\s*vent\b"), "alphavent"),
    (re.compile(r"\bheali\s*coil\b"), "healicoil"),
    (re.compile(r"\bq[-\s]*fix\b"), "qfix"),
    (re.compile(r"\bfoot\s*print\b"), "footprint"),
    (re.compile(r"\btwin\s*fix\b"), "twinfix"),
]


# ── chunk loading + tokenization ──────────────────────────────────────────────
#
# The contamination firewall lives here: company corpora are physically + logically
# separate. Legacy Arthrex chunks stay in data/kb/chunks/ (flat). New companies write
# to data/kb/<Company>/chunks/. The cross-company competitive_insights collection is
# NEVER returned by load_chunks() — only by load_competitive_insights().


def _company_chunk_dirs() -> list[tuple[str | None, Path]]:
    """(company_hint, dir) for every directory that may hold company chunks.

    company_hint is the company inferred from the path (None for the legacy flat
    dir, where the company comes from the chunk's own `company` field instead).
    """
    dirs: list[tuple[str | None, Path]] = []
    if CHUNKS_DIR.exists():
        dirs.append((None, CHUNKS_DIR))  # legacy flat = Arthrex by field/default
    for child in sorted(KB_DIR.glob("*")):
        if not child.is_dir() or child.name in ("chunks", "competitive_insights"):
            continue
        sub = child / "chunks"
        if sub.exists():
            dirs.append((co.canonical_company(child.name), sub))
    return dirs


def _resolve_company(chunk: dict, path_hint: str | None, src: Path) -> str | None:
    """Resolve a chunk's company FAIL-CLOSED. Returns None (= skip the chunk) on any
    ambiguity so a mislabeled file can never silently land in the wrong corpus.

      - per-company dir (path_hint set): the directory is authoritative. A chunk
        whose own `company` disagrees is dropped + warned (contamination guard).
      - legacy flat dir (path_hint None): company comes from the field, defaulting to
        Arthrex (the only company that ever lived there). A non-Arthrex field here is
        a misfiled chunk and is dropped + warned.
    """
    field = chunk.get("company")
    if path_hint is not None:  # data/kb/<Company>/chunks/
        if field and co.canonical_company(field) != path_hint:
            _warn(f"dropping {src.name}: company={field!r} but it lives in {path_hint}/chunks/")
            return None
        return path_hint
    # legacy flat dir = Arthrex
    if field and co.canonical_company(field) != co.DEFAULT_COMPANY:
        _warn(f"dropping {src.name}: company={field!r} found in legacy flat dir (expected {co.DEFAULT_COMPANY})")
        return None
    return co.DEFAULT_COMPANY


def load_chunks(company: str | None = None) -> list[dict]:
    """Load KB chunks, optionally restricted to one company.

    When `company` is given, ONLY that company's chunks are returned — this is the
    retrieval firewall. `company=None` returns every company's chunks (used by tools
    that report KB-wide stats; never used to build a single answer's retriever).
    Every returned chunk is guaranteed to have a correct `company` field; ambiguous
    files are skipped, never misrouted.
    """
    want = co.canonical_company(company) if company else None
    out: list[dict] = []
    for hint, d in _company_chunk_dirs():
        for f in sorted(d.glob("*.json")):
            try:
                c = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                _warn(f"skipping unreadable chunk file {f}")
                continue
            resolved = _resolve_company(c, hint, f)
            if resolved is None:
                continue
            c["company"] = resolved
            if want is None or resolved == want:
                out.append(c)
    return out


def enforce_company(chunks: list[dict], company: str) -> list[dict]:
    """Belt-and-suspenders firewall: drop (and warn on) any chunk whose company does
    not match. Call this on retrieved chunks before they enter a prompt, citation,
    roleplay grounding, or scorer — so even a wrong-retriever bug fails closed."""
    want = co.canonical_company(company)
    safe = []
    for c in chunks:
        if c.get("company") == want:
            safe.append(c)
        else:
            _warn(f"contamination guard dropped chunk {c.get('chunk_id')!r} "
                  f"(company={c.get('company')!r}) from a {want} response")
    return safe


def available_companies() -> list[str]:
    """Distinct companies that actually have (valid) chunks on disk, sorted."""
    seen: set[str] = set()
    for hint, d in _company_chunk_dirs():
        for f in d.glob("*.json"):
            try:
                c = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            comp = _resolve_company(c, hint, f)
            if comp is not None:
                seen.add(comp)
    return sorted(seen)


def load_competitive_insights() -> list[dict]:
    """Load the cross-company competitive_insights collection (chunk-shaped dicts).

    Deliberately separate from load_chunks(): these are curated comparison notes, used
    only for explicit competitive comparison / competitive roleplay, and are never
    blended into a single-company product answer.
    """
    if not COMPETITIVE_DIR.exists():
        return []
    out: list[dict] = []
    for f in sorted(COMPETITIVE_DIR.glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _normalize_phrases(text: str) -> str:
    text = text.lower()
    for pat, repl in _PHRASE_NORMALIZE:
        text = pat.sub(repl, text)
    return text


def _tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", _normalize_phrases(text))
    return [t for t in toks if len(t) > 1 and t not in _STOPWORDS]


def _fingerprint(chunks: list[dict], namespace: str = "") -> str:
    """Hash the corpus. The namespace (company/collection) and each chunk's company
    are hashed FIRST so a cache file can never be reused across corpora even if two
    chunks share chunk_id+text — the cross-company cache-collision hole is closed."""
    h = hashlib.sha256()
    h.update(namespace.encode("utf-8"))
    h.update(b"\x00")
    for c in chunks:
        h.update((c.get("company") or "").encode("utf-8"))
        h.update(b"\x00")
        h.update(c["chunk_id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(c["text"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ── retrievers ────────────────────────────────────────────────────────────────


class _RetrieverBase:
    """Retrievers OWN their chunk list and return chunk dicts directly via retrieve(),
    so no caller ever indexes an external global list (the positional-index coupling
    that was the single largest silent cross-company leak vector)."""

    chunks: list[dict] = []

    def retrieve(self, query: str, k: int) -> list[tuple[dict, float]]:
        return [(self.chunks[i], s) for i, s in self.search(query, k)]


class TfidfRetriever(_RetrieverBase):
    name = "tfidf (pure-python)"
    cached = False

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        docs = [c["text"] for c in chunks]
        tokenized = [_tokenize(d) for d in docs]
        vocab: dict[str, int] = {}
        for toks in tokenized:
            for t in set(toks):
                vocab.setdefault(t, len(vocab))
        self.vocab = vocab
        n = len(docs)
        df = np.zeros(len(vocab))
        for toks in tokenized:
            for t in set(toks):
                df[vocab[t]] += 1
        self.idf = np.log((1 + n) / (1 + df)) + 1.0
        self.matrix = np.vstack([self._vec(toks) for toks in tokenized]) if docs else np.zeros((0, 0))

    def _vec(self, toks: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t in toks:
            if t in self.vocab:
                v[self.vocab[t]] += 1.0
        nz = v > 0
        v[nz] = (1.0 + np.log(v[nz])) * self.idf[nz]
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self._vec(_tokenize(query))
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]


class EmbeddingRetriever(_RetrieverBase):
    """Neural embeddings (OpenAI / Voyage / sentence-transformers) + cosine, with a
    persistent on-disk cache keyed by a fingerprint of the chunk texts."""

    def __init__(
        self,
        chunks: list[dict],
        backend: str,
        use_cache: bool = True,
        model: str | None = None,
        namespace: str = "all",
    ):
        self.chunks = chunks
        self.backend = backend
        if backend == "openai":
            self.model = model or EMBED_MODEL_OPENAI
            self._embed = self._make_openai()
        elif backend == "voyage":
            self.model = model or EMBED_MODEL_VOYAGE
            self._embed = self._make_voyage()
        elif backend == "st":
            self.model = model or EMBED_MODEL_ST
            self._embed = self._make_st()
        else:  # pragma: no cover
            raise ValueError(backend)
        self.name = f"{backend}:{self.model}"

        # Namespace the cache by corpus (company / collection) so per-company
        # retrievers don't share — and silently thrash — one embeddings_<backend>.npz.
        # The namespace is folded into BOTH the filename AND the fingerprint+contents,
        # so a cache built for one company can never be loaded for another (fail-closed).
        self.namespace = namespace
        fp = _fingerprint(chunks, namespace)
        slug = re.sub(r"[^a-z0-9]+", "-", f"{namespace}_{backend}_{self.model}".lower()).strip("-")
        cache_path = KB_DIR / f"embeddings_{slug}.npz"
        self.cached = False
        if use_cache and cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            if str(data["fingerprint"]) == fp and str(data.get("namespace", "")) == namespace:
                self.matrix = data["matrix"]
                self.cached = True
        if not self.cached:
            mat = self._embed([c["text"] for c in chunks], is_query=False)
            self.matrix = self._normalize(np.asarray(mat, dtype=float))
            KB_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, matrix=self.matrix, fingerprint=fp, namespace=namespace)

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    def _make_openai(self):
        from openai import OpenAI

        client = OpenAI()

        def embed(texts, is_query=False):
            resp = client.embeddings.create(model=self.model, input=list(texts))
            return [d.embedding for d in resp.data]

        return embed

    def _make_voyage(self):
        import voyageai

        client = voyageai.Client()

        def embed(texts, is_query=False):
            res = client.embed(list(texts), model=self.model, input_type="query" if is_query else "document")
            return res.embeddings

        return embed

    def _make_st(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model)

        def embed(texts, is_query=False):
            return model.encode(list(texts), show_progress_bar=False)

        return embed

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self._normalize(np.asarray(self._embed([query], is_query=True), dtype=float))[0]
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]


def _have(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def resolve_embed_backend(requested: str) -> tuple[str, str]:
    if requested != "auto":
        return requested, "explicitly requested"
    if os.environ.get("OPENAI_API_KEY") and _have("openai"):
        return "openai", "OPENAI_API_KEY set"
    if os.environ.get("VOYAGE_API_KEY") and _have("voyageai"):
        return "voyage", "VOYAGE_API_KEY set"
    if _have("sentence_transformers"):
        return "st", "sentence-transformers installed"
    return "tfidf", "no embedding API key — set OPENAI_API_KEY (or VOYAGE_API_KEY) for neural"


def resolve_gen_backend(requested: str) -> tuple[str, str]:
    if requested != "auto":
        return requested, "explicitly requested"
    if os.environ.get("ANTHROPIC_API_KEY") and _have("anthropic"):
        return "anthropic", "ANTHROPIC_API_KEY set"
    if os.environ.get("OPENAI_API_KEY") and _have("openai"):
        return "openai", "OPENAI_API_KEY set"
    return "extractive", "no LLM API key — set ANTHROPIC_API_KEY (or OPENAI_API_KEY) for generated answers"


def build_retriever(
    chunks: list[dict],
    backend: str,
    use_cache: bool = True,
    model: str | None = None,
    namespace: str = "all",
):
    if backend == "tfidf":
        return TfidfRetriever(chunks)
    return EmbeddingRetriever(chunks, backend, use_cache=use_cache, model=model, namespace=namespace)


# ── generation ────────────────────────────────────────────────────────────────


# Per-company examples of well-formatted, bolded product names/sizes. These only SHOW the
# model the bolding style for the selected brand — they never inject another brand's products
# (the firewall already guarantees the excerpts are single-company).
_PROMPT_EXAMPLES = {
    co.ARTHREX: "**2.6 mm FiberTak RC**, **4.75-mm SwiveLock**, **1.7-mm FiberTape**",
    co.STRYKER: "**ReelX STT**, **2.3 mm Iconix** all-suture, **AlphaVent** knotless anchors",
    co.SMITH_NEPHEW: "**HEALICOIL PRO**, **Q-FIX** all-suture anchor, **FOOTPRINT Ultra**",
}


def system_prompt(company: str = co.DEFAULT_COMPANY) -> str:
    """The product-knowledge chatbot's system prompt, parameterized by company.

    Same guidance for every brand — only the company name and the product-name examples
    change — so accuracy/spec-discipline never depends on which company is selected. The
    format is tuned for medical-device REPS: continuously numbered procedure steps, bolded
    products/specs, practical OR/selling tips, and a "why choose this" close — all kept
    strictly grounded in (and cited to) the source excerpts.
    """
    name = co.display_name(company)
    ex = _PROMPT_EXAMPLES.get(company, f"exact {name} product names and sizes")
    return (
        f"You are a senior {name} clinical sales specialist coaching a {name} rep on ROTATOR CUFF "
        "and SHOULDER REPAIR. Be practical, confident, and genuinely useful — write the kind of answer "
        "a rep would actually want to read and could show a surgeon. Answer ONLY from the numbered "
        "source excerpts and cite every factual claim inline with [n].\n\n"
        f"COMPANY SCOPE: You represent {name}. Every excerpt below is {name} material — speak only to "
        f"{name} products. Never name or compare a competitor's product unless an excerpt explicitly does.\n\n"
        "SCOPE: Stay on rotator cuff / shoulder repair. Do NOT wander into unrelated procedures (ankle, "
        "knee, arthroplasty, etc.) unless the question asks. If the excerpts don't cover what's asked, "
        "say so plainly instead of guessing.\n\n"
        "GROUNDING — zero hallucination: exact figures (drill/punch/reamer diameters, anchor sizes, "
        "suture/tape specs) and EVERY clinical claim appear ONLY if they are in the excerpts, each cited "
        "[n]. If a number isn't in the excerpts, tell the rep to confirm it in the technique guide — never "
        "invent one.\n\n"
        f"PRODUCT NAMING: **Bold** every implant, instrument, and spec, using exact {name} product names "
        f"and sizes (e.g. {ex}).\n\n"
        "CHOOSE THE FORMAT BY THE QUESTION:\n\n"
        "A) PROCEDURE / TECHNIQUE / 'walk me through' — structure the answer as:\n"
        "   - **Bottom line** — one sentence on what the construct achieves.\n"
        "   - **When to use it** — tear pattern / patient selection / indication, when the excerpts support it.\n"
        "   - **Steps** — ONE continuously numbered list: number 1, 2, 3, … straight through; NEVER "
        "restart at 1 and NEVER split the steps into two separate lists. One or two sentences per step, "
        "bolding the products and specs used in that step.\n"
        "   - **Rep tips** — 2-3 practical pointers a rep actually uses: an OR-time or workflow win, a "
        "tissue-handling reminder, a common first-timer pitfall to pre-empt, or a likely objection and how "
        "to answer it. Tie tips to the excerpts where you can; generic technique/selling reminders are okay "
        "but must NOT introduce new specs or clinical claims.\n"
        "   - **Why choose this technique** — 2-3 short selling points built from the advantages the "
        "excerpts state (cite them), including why a surgeon would pick it over the alternative they're "
        "using when the excerpts support that, then one concrete next step the rep can ask for (a trial "
        "case, an in-service).\n\n"
        "B) QUICK FACT (a single spec, a comparison, a 'what is X') — skip the step list: lead with the "
        "direct answer in one bolded, cited line; add 2-4 tight bullets only if they help; then one "
        "practical rep tip if a source supports it.\n\n"
        f"COMPETITIVE — when the excerpts include cross-company comparison notes, or the question names "
        f"another company or product: give a BALANCED, grounded comparison. State {name}'s documented "
        "advantages (cite them), represent the competitor ONLY as the notes describe — never invent a "
        "competitor spec, number, or claim — and acknowledge where they are legitimately strong. If the "
        f"data doesn't cover a head-to-head point, say so plainly and pivot to {name}'s documented "
        "strengths rather than guessing. End with the single strongest, evidence-backed reason to "
        f"choose {name}.\n\n"
        "STYLE: No preamble, no restating the question, no empty marketing adjectives. Specific over vague. "
        "Sound like a rep who has been in the OR a thousand times."
    )


# Backward-compatible module constant (Arthrex) for any caller importing SYSTEM_PROMPT.
SYSTEM_PROMPT = system_prompt(co.DEFAULT_COMPANY)


def _loc(c: dict) -> str:
    if c.get("timestamp_range"):
        return c["timestamp_range"]["label"]
    if c.get("source_type") == "pdf":
        return f"PDF, {c.get('page_count') or '?'} pp"
    if c.get("source_type") == "competitive_insight":
        return "cross-company note"
    return "n/a"


def _kind(c: dict) -> str:
    if c.get("source_type") == "youtube":
        return "video"
    if c.get("source_type") == "competitive_insight":
        return "cross-company comparison note"
    return "surgical guide"


def _context_block(retrieved: list[dict]) -> str:
    parts = []
    for n, c in enumerate(retrieved, 1):
        parts.append(f"[{n}] {c['source_title']} ({_kind(c)}, {_loc(c)})\n{c['text']}")
    return "\n\n".join(parts)


def _user_prompt(question: str, retrieved: list[dict]) -> str:
    return f"Source excerpts:\n{_context_block(retrieved)}\n\nQuestion: {question}"


def generate_answer(
    question: str, retrieved: list[dict], backend: str, model: str | None = None,
    company: str = co.DEFAULT_COMPANY,
) -> str:
    user = _user_prompt(question, retrieved)
    sys_prompt = system_prompt(company)
    if backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model or GEN_MODEL_ANTHROPIC,
            max_tokens=1000,
            system=sys_prompt,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()
    if backend == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model or GEN_MODEL_OPENAI,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
            max_tokens=1000,
        )
        return resp.choices[0].message.content.strip()
    return _extractive_answer(retrieved)


def stream_answer(
    question: str, retrieved: list[dict], backend: str, model: str | None = None,
    company: str = co.DEFAULT_COMPANY,
):
    """Yield answer text deltas. Mirrors generate_answer but streams (for the web UI)."""
    user = _user_prompt(question, retrieved)
    sys_prompt = system_prompt(company)
    if backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        with client.messages.stream(
            model=model or GEN_MODEL_ANTHROPIC,
            max_tokens=1000,
            system=sys_prompt,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                yield text
        return
    if backend == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model or GEN_MODEL_OPENAI,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
            max_tokens=1000,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        return
    yield _extractive_answer(retrieved)


def _extractive_answer(retrieved: list[dict]) -> str:
    lines = ["(extractive — set ANTHROPIC_API_KEY or OPENAI_API_KEY for a synthesized answer)"]
    for n, c in enumerate(retrieved[:2], 1):
        words = c["text"].split()
        excerpt = " ".join(words[:75]) + (" …" if len(words) > 75 else "")
        lines.append(f"\nFrom [{n}] {c['source_title']}:\n{excerpt}")
    return "\n".join(lines)


# ── citations / output helpers ────────────────────────────────────────────────


def _estimate_pdf_page(c: dict) -> int | None:
    """Best-effort page for a PDF chunk, used only as a navigation jump (#page=N).

    Ingestion stored the chunk's position (chunk_index / chunk_count) and the document's
    total page_count, but NOT exact per-chunk page spans — so this lands the reader NEAR the
    cited passage rather than on page 1, clamped into range. Returns None when there isn't
    enough info to estimate (so we just open the file). It is deliberately a silent nav aid:
    we never DISPLAY this as "page N", to avoid implying a precision the data doesn't have.
    """
    pages = c.get("page_count") or 0
    cnt = c.get("chunk_count") or 0
    idx = c.get("chunk_index")
    if pages < 2 or cnt < 2 or idx is None:
        return None
    return max(1, min(pages, int(idx / cnt * pages) + 1))


def deep_link(c: dict) -> str:
    """YouTube → exact-timestamp link; PDF → file URL (+ best-effort #page jump)."""
    url = c.get("source_url") or ""
    if c.get("source_type") == "youtube" and c.get("timestamp_range"):
        start = int(c["timestamp_range"]["start_seconds"])
        vid = c.get("source_id")
        if vid:
            return f"https://www.youtube.com/watch?v={vid}&t={start}s"
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}t={start}s"
    if c.get("source_type") == "pdf" and url and "#" not in url:
        page = _estimate_pdf_page(c)
        if page:
            return f"{url}#page={page}"  # honored by inline PDF viewers; ignored on download
    return url


def citation(n: int, c: dict, score: float) -> dict:
    """UI-ready citation dict."""
    return {
        "n": n,
        "type": c.get("source_type"),  # "youtube" | "pdf"
        "title": c["source_title"],
        "loc": _loc(c),
        "url": deep_link(c),
        "score": round(score, 3),
    }


def format_sources(retrieved: list[dict], scores: list[float]) -> str:
    lines = []
    for n, (c, s) in enumerate(zip(retrieved, scores), 1):
        typ = (c.get("source_type") or "?").upper()
        lines.append(
            f"  [{n}] ({typ}) {c['source_title']}  ·  {_loc(c)}  ·  score {s:.3f}\n      {deep_link(c)}"
        )
    return "\n".join(lines)


# ── competitive-question detection ────────────────────────────────────────────
#
# Distinctive (brand-specific) product names per company — used ONLY to detect when a
# query references a DIFFERENT company so the chatbot can pull the curated cross-company
# comparison notes. Deliberately excludes generic technique words shared across brands
# (knotless, double-row, anchor, all-suture, footprint, scr, tensionable, …) so a generic
# question never gets mis-flagged as competitive.
_BRAND_PRODUCTS: dict[str, set[str]] = {
    co.ARTHREX: {"speedbridge", "suturebridge", "speedfix", "fibertak", "swivelock", "fibertape",
                 "fiberwire", "pushlock", "suturetak", "corkscrew", "arthroflex", "cuffmend"},
    co.STRYKER: {"reelx", "iconix", "alphavent", "omega", "knotilus", "nanotack", "inspace",
                 "twinfix", "citrelock", "citrenak"},
    co.SMITH_NEPHEW: {"healicoil", "qfix", "footprintultra", "regenesorb", "regeneten", "multifix",
                      "bioraptor", "ultrabraid", "ultratape"},
}
_BRAND_NAMES: dict[str, set[str]] = {
    co.ARTHREX: {"arthrex"}, co.STRYKER: {"stryker"}, co.SMITH_NEPHEW: {"smithnephew"},
}
_COMP_WORDS = re.compile(r"\b(competitor|competitors|competition|rival|rivals|head[- ]?to[- ]?head)\b", re.I)


def is_competitive_query(q: str, company: str = co.DEFAULT_COMPANY) -> bool:
    """True when a Knowledge question is really asking for a cross-company comparison.

    Fires on explicit competitor words, or on a mention of ANY company (name or distinctive
    product) other than the one currently selected. Generic shared technique terms never
    trigger it — so single-brand questions stay single-brand.
    """
    if not q:
        return False
    if _COMP_WORDS.search(q):
        return True
    flat = re.sub(r"[^a-z0-9]+", "", q.lower())  # "Q-FIX" -> "qfix", "Smith & Nephew" -> "smithnephew"
    for c in co.COMPANIES:
        if c == company:
            continue
        for term in _BRAND_NAMES.get(c, set()) | _BRAND_PRODUCTS.get(c, set()):
            if term and term in flat:
                return True
    return False


# ── roleplay trainer ──────────────────────────────────────────────────────────
#
# The AI plays a surgeon/customer (grounded in the same RAG corpus so its technical
# pushback is accurate); the trainee plays the rep. After each rep message, a coach
# scores it on five dimensions. Both the persona and the coach use the resolved
# generation backend; with no API key they fall back to a canned-but-grounded
# persona line and a real deterministic heuristic scorecard.

ROLES = {
    "sales_rep": "field sales representative",
    "clinical_specialist": "clinical specialist (expected to go deeper technically)",
}

DIFFICULTY = {
    "easy": "You are receptive and curious. Push back lightly and accept good answers readily.",
    "medium": "You are busy and somewhat skeptical. Push back on vague or unsupported claims and expect specifics.",
    "hard": "You are demanding and time-pressured. Challenge every claim, ask for data and exact specs, and stay skeptical until the rep clearly earns it.",
}

# Rep-facing copy for the difficulty selector (the DIFFICULTY values above are the
# surgeon's in-character instructions; these are what the trainee reads).
DIFFICULTY_META = {
    "easy":   {"label": "Easy",   "blurb": "Receptive and curious. Accepts solid answers readily — a good warm-up."},
    "medium": {"label": "Medium", "blurb": "Busy and skeptical. Expects specifics and pushes back on vague claims."},
    "hard":   {"label": "Hard",   "blurb": "Demanding and time-pressured. Challenges every claim and wants data."},
}

SCENARIOS = [
    {
        "id": "skeptical_surgeon",
        "label": "Skeptical surgeon — why switch?",
        "persona": "Dr. Reyes, a high-volume sports-medicine surgeon ~15 years out who already does "
        "a transosseous-equivalent double-row and is satisfied with his results. Pleasant but "
        "guarded — he's been pitched a hundred times and assumes 'new' means marketing, not medicine.",
        "goal": "Reframe from 'rip out what works' to 'same gold-standard double-row, executed "
        "knotless,' and earn one trial case.",
        "opening": "Look, I do a transosseous-equivalent double-row already and my patients do great. "
        "My retear rate is fine. So before you open a tray — why would I change anything about a "
        "technique that's working for me?",
        "prospect": {
            "name": "Dr. Marcus Reyes", "title": "Sports Medicine Orthopedic Surgeon",
            "setting": "Private practice + hospital-affiliated ASC", "volume": "~300 rotator cuff repairs / yr",
            "current": "Transosseous-equivalent double-row", "style": "Pleasant but guarded; pitched constantly",
        },
    },
    {
        "id": "cost_objection",
        "label": "Cost / value-analysis objection",
        "persona": "Dr. Okafor, an efficient surgeon who also sits on the hospital value-analysis "
        "committee. Not hostile to Arthrex, but has to defend every SKU to administration and won't "
        "accept 'it's better' without a total-cost argument.",
        "goal": "Shift from implant unit price to total cost per case (OR time, reliability, revision "
        "avoidance) and give defensible numbers.",
        "opening": "I priced this out. Your SpeedBridge construct — two medial, two lateral, all that "
        "FiberTape — runs me more in implants per case than what I'm using now. Give me the business "
        "case, not the brochure.",
        "prospect": {
            "name": "Dr. Adaeze Okafor", "title": "Orthopedic Surgeon · Value-Analysis Committee",
            "setting": "Mid-size community hospital", "volume": "~220 rotator cuff repairs / yr",
            "current": "Cost-sensitive knotless double-row", "style": "Efficient, ROI-driven; defends every SKU",
        },
    },
    {
        "id": "new_product_launch",
        "label": "New product launch — FiberTak SpeedBridge",
        "persona": "Dr. Lindqvist, an early-adopter-leaning surgeon who already runs classic "
        "SpeedBridge. Open to the all-suture upgrade but pragmatic — wants to know what it actually "
        "buys her and whether it adds steps or risk.",
        "goal": "Position FiberTak SpeedBridge as the next evolution of a construct she trusts — more "
        "medial fixation and compression for larger tears — without added complexity.",
        "opening": "I keep hearing about the all-suture version — FiberTak SpeedBridge. I'm already "
        "happy with my SwiveLock medial row. What does going to soft anchors actually get me, and is "
        "it going to add a bunch of steps to my case?",
        "prospect": {
            "name": "Dr. Erika Lindqvist", "title": "Shoulder & Elbow Surgeon",
            "setting": "Academic-affiliated sports medicine", "volume": "~280 rotator cuff repairs / yr",
            "current": "Classic SpeedBridge (SwiveLock medial row)", "style": "Early adopter; pragmatic about added steps",
        },
    },
    {
        "id": "technique_walkthrough",
        "label": "First-time SpeedBridge walkthrough",
        "persona": "Dr. Tran, a competent arthroscopist doing her first SpeedBridge case with you at "
        "her shoulder. Confident in anatomy but new to this knotless construct and a little tense "
        "about the sequence and tensioning in front of her team.",
        "goal": "Guide her cleanly through the medial-then-lateral sequence, pre-empting first-timer "
        "errors, so she finishes a secure construct.",
        "opening": "Okay, I've got the tray open and the footprint's prepped. Walk me through it — "
        "medial row first, right? And tell me where people screw this up the first time, because I "
        "don't want to learn that the hard way mid-case.",
        "prospect": {
            "name": "Dr. Lily Tran", "title": "Orthopedic Surgeon (Arthroscopy)",
            "setting": "Regional orthopedic group", "volume": "~120 rotator cuff repairs / yr",
            "current": "Single-row; first knotless case", "style": "Competent but a little tense in front of her team",
        },
    },
    {
        "id": "clinical_data_grill",
        "label": "Clinical-data grill (hard)",
        "persona": "Dr. Halvorsen, an academic shoulder surgeon who publishes and reviews for "
        "journals. Polite but relentless — he distinguishes cadaveric from clinical and data-on-file "
        "from peer-reviewed, and will catch a rep who blurs them.",
        "goal": "Defend the knotless double-row evidence base accurately — right study for each claim, "
        "honest about biomechanical vs clinical — to earn an academic influencer's credibility.",
        "opening": "I'll save us both time: I don't care about brochures, I care about the literature. "
        "You're telling me knotless self-reinforcement is real and clinically meaningful — cite me the "
        "actual studies, and don't quote me a cadaver study as if it's a patient outcome.",
        "prospect": {
            "name": "Dr. Sven Halvorsen", "title": "Academic Shoulder Surgeon",
            "setting": "University medical center", "volume": "~200 repairs / yr + active research",
            "current": "Multiple systems; publishes outcomes", "style": "Relentless; holds reps to peer-review rigor",
        },
    },
    {
        "id": "competitor_switch",
        "label": "Competitive switch — surgeon on a rival system",
        "competitive": True,
        "persona": "Dr. Bauer, a surgeon who currently uses a competitor's knotless double-row system "
        "(Stryker / Smith & Nephew) and is happy enough with it. Loyal to his rep and skeptical that a "
        "switch is worth the disruption to his OR routine.",
        "goal": "Win a head-to-head: position the Arthrex construct against the competitor's on the "
        "dimensions that matter (fixation, footprint compression, evidence, OR efficiency) — accurately, "
        "without trashing the competitor — and earn one side-by-side trial case.",
        "opening": "Honest with you — I'm already doing knotless double-row with another company's "
        "anchors and I'm not unhappy. Why is yours actually better, and don't just tell me it's 'the "
        "gold standard'?",
        "prospect": {
            "name": "Dr. Klaus Bauer", "title": "Sports Medicine Orthopedic Surgeon",
            "setting": "High-volume orthopedic specialty hospital", "volume": "~340 rotator cuff repairs / yr",
            "current": "Competitor knotless double-row system", "style": "Loyal to his current rep; switch-averse",
        },
    },
]

SCENARIO_BY_ID = {s["id"]: s for s in SCENARIOS}
ARTHREX_SCENARIOS = SCENARIOS  # alias: the default company's curated scenario set


def _generic_scenarios(company: str) -> list[dict]:
    """Company-aware scenario set for brands without a hand-authored one.

    Product-agnostic on purpose — the surgeon's technical facts come from that
    company's KB at runtime (and competitive_insights for the competitive scenario),
    so we never bake in a guessed competitor spec.
    """
    name = co.display_name(company)
    return [
        {
            "id": "skeptical_surgeon",
            "label": "Skeptical surgeon — why switch?",
            "persona": f"Dr. Reyes, a high-volume sports-medicine surgeon who already does a "
            f"transosseous-equivalent double-row and is satisfied with his results. Guarded — he's been "
            f"pitched a hundred times and assumes 'new' means marketing, not medicine.",
            "goal": f"Reframe from 'rip out what works' to a {name} knotless double-row that protects his "
            f"outcomes, and earn one trial case.",
            "opening": "I do a transosseous-equivalent double-row already and my patients do great. So "
            "before you open a tray — why would I change anything that's working for me?",
            "prospect": {
                "name": "Dr. Marcus Reyes", "title": "Sports Medicine Orthopedic Surgeon",
                "setting": "Private practice + hospital-affiliated ASC", "volume": "~300 rotator cuff repairs / yr",
                "current": "Transosseous-equivalent double-row", "style": "Pleasant but guarded; pitched constantly",
            },
        },
        {
            "id": "cost_objection",
            "label": "Cost / value-analysis objection",
            "persona": f"Dr. Okafor, an efficient surgeon on the hospital value-analysis committee who has "
            f"to defend every SKU to administration and won't accept 'it's better' without a total-cost argument.",
            "goal": "Shift from implant unit price to total cost per case (OR time, reliability, revision avoidance).",
            "opening": f"I priced out your {name} construct and it runs me more in implants per case than "
            f"what I use now. Give me the business case, not the brochure.",
            "prospect": {
                "name": "Dr. Adaeze Okafor", "title": "Orthopedic Surgeon · Value-Analysis Committee",
                "setting": "Mid-size community hospital", "volume": "~220 rotator cuff repairs / yr",
                "current": "Cost-sensitive knotless double-row", "style": "Efficient, ROI-driven; defends every SKU",
            },
        },
        {
            "id": "technique_walkthrough",
            "label": "First-time technique walkthrough",
            "persona": f"Dr. Tran, a competent arthroscopist doing her first case on the {name} knotless "
            f"construct with you at her shoulder — confident in anatomy but new to this sequence.",
            "goal": "Guide her cleanly medial-then-lateral, pre-empting first-timer errors, to a secure construct.",
            "opening": "Okay, tray's open and the footprint's prepped. Walk me through it — medial row first, "
            "right? And tell me where people screw this up the first time.",
            "prospect": {
                "name": "Dr. Lily Tran", "title": "Orthopedic Surgeon (Arthroscopy)",
                "setting": "Regional orthopedic group", "volume": "~120 rotator cuff repairs / yr",
                "current": f"Single-row; first {name} knotless case", "style": "Competent but a little tense in front of her team",
            },
        },
        {
            "id": "clinical_data_grill",
            "label": "Clinical-data grill (hard)",
            "persona": f"Dr. Halvorsen, an academic shoulder surgeon who publishes and reviews for journals "
            f"and will catch a rep who blurs cadaveric data with clinical outcomes.",
            "goal": f"Defend the {name} knotless double-row evidence base accurately — right study for each "
            f"claim, honest about biomechanical vs clinical.",
            "opening": "I care about the literature, not brochures. Cite me the actual studies behind your "
            "construct, and don't quote a cadaver study as if it's a patient outcome.",
            "prospect": {
                "name": "Dr. Sven Halvorsen", "title": "Academic Shoulder Surgeon",
                "setting": "University medical center", "volume": "~200 repairs / yr + active research",
                "current": "Multiple systems; publishes outcomes", "style": "Relentless; holds reps to peer-review rigor",
            },
        },
        {
            "id": "competitor_switch",
            "label": "Competitive switch — surgeon on a rival system",
            "competitive": True,
            "persona": f"Dr. Bauer, a surgeon who currently uses a competitor's knotless double-row system "
            f"(e.g. Arthrex SpeedBridge) and is happy enough with it. Skeptical a switch to {name} is worth "
            f"disrupting his OR routine.",
            "goal": f"Win a head-to-head: position the {name} construct against the competitor's on fixation, "
            f"footprint compression, evidence, and OR efficiency — accurately, without trashing the rival — "
            f"and earn one side-by-side trial case.",
            "opening": f"Honest with you — I'm already doing knotless double-row with another company's "
            f"anchors and I'm not unhappy. Why is {name} actually better?",
            "prospect": {
                "name": "Dr. Klaus Bauer", "title": "Sports Medicine Orthopedic Surgeon",
                "setting": "High-volume orthopedic specialty hospital", "volume": "~340 rotator cuff repairs / yr",
                "current": "Competitor knotless double-row system", "style": "Loyal to his current rep; switch-averse",
            },
        },
    ]


SCENARIOS_BY_COMPANY: dict[str, list[dict]] = {
    co.ARTHREX: ARTHREX_SCENARIOS,
    co.STRYKER: _generic_scenarios(co.STRYKER),
    co.SMITH_NEPHEW: _generic_scenarios(co.SMITH_NEPHEW),
}


def scenarios_for(company: str = co.DEFAULT_COMPANY) -> list[dict]:
    return SCENARIOS_BY_COMPANY.get(co.canonical_company(company), ARTHREX_SCENARIOS)


def scenario_by_id(scenario_id: str, company: str = co.DEFAULT_COMPANY) -> dict:
    scens = scenarios_for(company)
    return next((s for s in scens if s["id"] == scenario_id), scens[0])

# Heuristic-scorer vocab (lowercased; matched after _normalize_phrases).
_PRODUCT_TERMS = {
    "speedbridge", "suturebridge", "speedfix", "fibertak", "swivelock", "fibertape",
    "fiberwire", "corkscrew", "pushlock", "suturetak", "biocomposite", "knotless",
    "double-row", "doublerow", "footprint", "anchor", "all-suture", "allsuture",
    "arthroflex", "cuffmend", "tensionable", "ripstop", "scr",
}
_TECH_TERMS = {
    "supraspinatus", "subscapularis", "infraspinatus", "tuberosity", "tendon", "humerus",
    "glenoid", "socket", "punch", "drill", "ream", "medial", "lateral", "row",
    "compression", "tension", "footprint", "suture", "tape", "mattress",
}
_EVIDENCE_TERMS = {
    "data", "study", "studies", "biomechanical", "load", "load-to-failure", "cyclic",
    "compared", "vs", "versus", "retear", "pull-out", "pullout", "contact", "evidence",
    "shown", "demonstrated", "cadaver", "footprint", "n)", "strength",
}
_ACK_TERMS = {
    "understand", "hear you", "fair", "good question", "makes sense", "appreciate",
    "i get", "valid", "agree", "right that",
}
_HEDGES = [
    "i think", "i guess", "maybe", "probably", "sort of", "kind of", "not sure",
    "i believe", "might be", "perhaps", "um", "uh ", "i feel like",
]
# Discovery — questions that surface the surgeon's world before pitching.
_DISCOVERY_PHRASES = [
    "how many", "how often", "what are you", "what do you", "what's your", "whats your",
    "which", "currently using", "right now", "today", "walk me through", "tell me",
    "in your hands", "your patients", "your cases", "your volume", "your routine",
    "what matters", "what would", "how do you", "where do you", "what's driving",
    "biggest challenge", "pain point", "what's important", "help me understand",
]
# Closing — advancing toward a concrete commitment.
_CLOSE_PHRASES = [
    "trial case", "one case", "next case", "side-by-side", "side by side", "in-service",
    "in service", "evaluation", "evaluate", "would you be open", "would you be willing",
    "can we", "let's", "lets ", "set up", "schedule", "bring in a tray", "bring a tray",
    "try it", "give it a try", "follow up", "follow-up", "demo", "book a", "put it on",
    "do a case together", "watch a case", "next step",
]


def roleplay_system(
    scenario: dict, role: str, difficulty: str, context: str, company: str = co.DEFAULT_COMPANY
) -> str:
    persona = scenario.get("persona", "a skeptical orthopedic surgeon")
    goal = scenario.get("goal", "")
    r = ROLES.get(role, ROLES["sales_rep"])
    d = DIFFICULTY.get(difficulty, DIFFICULTY["medium"])
    name = co.display_name(company)
    competitive = scenario.get("competitive")
    excerpt_label = (
        f"{name} reference excerpts + cross-company competitive notes"
        if competitive
        else f"{name} reference excerpts"
    )
    return (
        f"You are role-playing as {persona}, speaking out loud to a TrainMed user who is a {name} {r} "
        f"practicing this sales/clinical conversation. You already opened with: "
        f"\"{scenario.get('opening', '')}\"\n\n"
        f"SPOKEN DIALOGUE ONLY — your entire response is read aloud by a text-to-speech voice, so "
        f"output ONLY the words you actually say. NEVER include stage directions, action descriptions, "
        f"sound effects, emotes, or narration of any kind: no *laughs*, no (sighs), no [pause], no "
        f"asterisks, no parentheticals describing tone or actions, no 'he says'. Just the plain "
        f"spoken sentences, nothing else.\n\n"
        f"STAY FULLY IN CHARACTER as the surgeon/customer. React naturally, ask pointed questions, and "
        f"raise realistic objections. Only be convinced by specific, accurate, well-supported answers. "
        f"{d}\n\n"
        f"Keep replies SHORT and conversational — 1 to 3 sentences, like real OR/booth talk. When you "
        f"make or challenge a technical claim, rely on the {excerpt_label} below so your facts are "
        f"correct; do not invent specifications. Do NOT break character, do NOT coach the rep, do NOT "
        f"hand them the answer, and never use [n] citation markers — you are the customer, not a "
        f"document.\n\n"
        f"(The rep's objective, for your awareness: {goal})\n\n"
        f"{excerpt_label} (for your technical accuracy only):\n{context}"
    )


def _history_to_messages(history: list[dict]) -> list[dict]:
    """Map [{role: rep|surgeon, content}] → alternating user/assistant; merge repeats;
    drop a leading assistant (the opening lives in the system prompt)."""
    msgs = []
    for turn in history or []:
        role = "user" if turn.get("role") in ("rep", "user") else "assistant"
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n\n" + content
        else:
            msgs.append({"role": role, "content": content})
    while msgs and msgs[0]["role"] == "assistant":
        msgs.pop(0)
    return msgs


def stream_roleplay(scenario, role, difficulty, history, message, retrieved, backend, model=None,
                    company=co.DEFAULT_COMPANY):
    """Stream the AI surgeon's reply to the rep's latest message."""
    system = roleplay_system(scenario, role, difficulty, _context_block(retrieved), company)
    convo = _history_to_messages(history)
    convo.append({"role": "user", "content": message})

    if backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        with client.messages.stream(
            model=model or GEN_MODEL_ANTHROPIC, max_tokens=260, system=system, messages=convo
        ) as stream:
            for text in stream.text_stream:
                yield text
        return
    if backend == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model or GEN_MODEL_OPENAI,
            messages=[{"role": "system", "content": system}, *convo],
            max_tokens=260,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        return
    yield _extractive_roleplay(scenario, retrieved)


def _extractive_roleplay(scenario: dict, retrieved: list[dict]) -> str:
    topic = retrieved[0]["source_title"].split("—")[0].strip() if retrieved else "this construct"
    return (
        "[demo persona — set ANTHROPIC_API_KEY or OPENAI_API_KEY for live role-play] "
        f"Okay, but you haven't convinced me yet. Be specific about {topic}: what's the real "
        "advantage, and what exactly are the anchor sizes and the data behind it?"
    )


# ── scoring ───────────────────────────────────────────────────────────────────

SCORE_DIMENSIONS = ["rapport", "discovery", "objection_handling", "product_knowledge", "closing"]

def coach_system(company: str = co.DEFAULT_COMPANY) -> str:
    name = co.display_name(company)
    return (
        f"You are a tough but fair {name} medical-sales coach. Score the REP's latest message in a "
        f"roleplay with a surgeon, using the {name} source excerpts as the ground truth of what is "
        "accurate. Score these five dimensions, each 1-5:\n"
        "- rapport: acknowledged the surgeon, built trust, stayed composed and respectful (not defensive).\n"
        "- discovery: asked about the surgeon's current technique, case volume, priorities or pain points "
        "instead of just pitching.\n"
        "- objection_handling: directly addressed the surgeon's concern with concrete evidence or data "
        "rather than dodging it.\n"
        "- product_knowledge: named the correct products/specs, accurate and grounded in the excerpts "
        "(penalize wrong or unsupported specs).\n"
        "- closing: advanced the sale toward a concrete next step — a trial case, an in-service, or a "
        "side-by-side evaluation.\n"
        "Reward specific, correct, confident answers; penalize vagueness, wrong specs, hedging, dodging "
        "the objection, and failing to ask for a next step. Respond with ONLY a JSON object, no prose, of "
        "exactly this shape:\n"
        '{"scores":{"rapport":1-5,"discovery":1-5,"objection_handling":1-5,"product_knowledge":1-5,'
        '"closing":1-5},"feedback":"1-2 sentence critique","tips":["tip","tip"]}'
    )


# Backward-compatible module constant (Arthrex).
COACH_SYSTEM = coach_system(co.DEFAULT_COMPANY)


def score_exchange(scenario, message, retrieved, backend, model=None, company=co.DEFAULT_COMPANY) -> dict:
    """Score a rep message. LLM coach when a key is set; deterministic heuristic otherwise."""
    if backend in ("anthropic", "openai"):
        card = _llm_score(scenario, message, retrieved, backend, model, company)
        if card:
            card["method"] = backend
            return card
    card = _heuristic_score(message, company)
    card["method"] = "heuristic"
    return card


def _llm_score(scenario, message, retrieved, backend, model, company=co.DEFAULT_COMPANY):
    name = co.display_name(company)
    user = (
        f"Scenario: {scenario.get('label')} — surgeon persona: {scenario.get('persona')}\n\n"
        f"{name} source excerpts (ground truth):\n{_context_block(retrieved)}\n\n"
        f"REP's latest message to score:\n\"\"\"{message}\"\"\""
    )
    try:
        if backend == "anthropic":
            import anthropic

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=model or GEN_MODEL_ANTHROPIC, max_tokens=400,
                system=coach_system(company), messages=[{"role": "user", "content": user}],
            )
            raw = "".join(b.text for b in msg.content if b.type == "text")
        else:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model=model or GEN_MODEL_OPENAI,
                messages=[{"role": "system", "content": coach_system(company)},
                          {"role": "user", "content": user}],
                max_tokens=400,
            )
            raw = resp.choices[0].message.content
        return _parse_scorecard(raw)
    except Exception:
        return None


def _parse_scorecard(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    scores = {k: int(round(float(data.get("scores", {}).get(k, 3)))) for k in SCORE_DIMENSIONS}
    scores = {k: max(1, min(5, v)) for k, v in scores.items()}
    return {
        "scores": scores,
        "overall": round(sum(scores.values()) / len(scores), 1),
        "feedback": str(data.get("feedback", "")).strip(),
        "tips": [str(t) for t in (data.get("tips") or [])][:3],
    }


def _band(n: int, thresholds=(1, 2, 3, 4)) -> int:
    """Map a count to a 1-5 score by thresholds."""
    score = 1
    for t in thresholds:
        if n >= t:
            score += 1
    return min(5, score)


_PRODUCT_TIP = {
    co.ARTHREX: "Name specific products and specs (e.g. 2.6 mm FiberTak, 4.75 mm SwiveLock, 1.7 mm FiberTape).",
    co.STRYKER: "Name specific products and specs (e.g. ReelX STT, 2.3 mm Iconix all-suture, AlphaVent anchors).",
    co.SMITH_NEPHEW: "Name specific products and specs (e.g. HEALICOIL PRO, Q-FIX all-suture, FOOTPRINT Ultra).",
}


def _heuristic_score(message: str, company: str = co.DEFAULT_COMPANY) -> dict:
    text = _normalize_phrases(message or "")
    toks = set(re.findall(r"[a-z0-9-]+", text))

    n_prod = len(co.product_terms(company) & toks)
    n_evid = len(_EVIDENCE_TERMS & toks)
    n_ack = sum(1 for p in _ACK_TERMS if p in text)
    n_hedge = sum(1 for h in _HEDGES if h in text)
    n_q = message.count("?")
    n_disc = sum(1 for p in _DISCOVERY_PHRASES if p in text)
    n_close = sum(1 for p in _CLOSE_PHRASES if p in text)
    words = len(message.split())
    has_spec = bool(re.search(r"\b\d+(?:\.\d+)?\s*mm\b", message, re.I))

    # rapport: acknowledge + composure. Reward empathy phrases and a steady tone;
    # penalize a thin reply or one drowning in hedges.
    rapport = max(1, min(5, 2 + n_ack + (1 if (words >= 12 and n_hedge == 0) else 0)
                         - (1 if n_hedge >= 2 else 0) - (1 if words < 6 else 0)))
    # discovery: questions + phrases that probe the surgeon's world.
    discovery = _band(n_disc + n_q)
    # objection_handling: acknowledge the concern, then bring evidence.
    objection_handling = _band(n_ack + n_evid)
    # product_knowledge: correct products + concrete specs.
    product_knowledge = _band(n_prod + (1 if has_spec else 0))
    # closing: did the rep ask for a concrete next step? (sparse, so weight it heavily)
    closing = 1 if n_close == 0 else min(5, 2 + 2 * n_close)

    scores = {
        "rapport": rapport,
        "discovery": discovery,
        "objection_handling": objection_handling,
        "product_knowledge": product_knowledge,
        "closing": closing,
    }
    tips = []
    if rapport < 3:
        tips.append("Open by acknowledging the surgeon and their experience before you pitch.")
    if discovery < 3:
        tips.append("Ask discovery questions — current technique, case volume, what they'd change — before presenting.")
    if objection_handling < 3:
        tips.append("Acknowledge the surgeon's concern, then counter with concrete evidence or data.")
    if product_knowledge < 3:
        tips.append(_PRODUCT_TIP.get(company, _PRODUCT_TIP[co.DEFAULT_COMPANY]))
    if closing < 3:
        tips.append("Always close — ask for a single trial case, an in-service, or a side-by-side evaluation.")
    if not tips:
        tips.append("Strong, complete rep turn — keep that acknowledge → evidence → next-step structure.")

    overall = round(sum(scores.values()) / len(scores), 1)
    quality = "strong" if overall >= 4 else "solid" if overall >= 3 else "needs work"
    feedback = (
        f"{quality.capitalize()} response ({overall}/5). "
        f"Discovery {discovery}/5, objection handling {objection_handling}/5, closing {closing}/5."
    )
    return {"scores": scores, "overall": overall, "feedback": feedback, "tips": tips[:3]}
