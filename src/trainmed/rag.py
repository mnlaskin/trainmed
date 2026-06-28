"""TrainMed RAG engine — the single source of truth for retrieval + generation.

Both the CLI (`scripts/test_rag.py`) and the web app (`app/server.py`) import from
here, so there is exactly one copy of the chunk loading, tokenization, retrievers,
backend selection, prompt, and answer generation.

Backend-flexible and auto-selecting (see resolve_embed_backend / resolve_gen_backend):
  embeddings: openai -> voyage -> sentence-transformers -> tfidf
  generation: anthropic -> openai -> extractive
Neural embeddings are cached to data/kb/embeddings_<backend>_<model>.npz.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

# rag.py lives at src/trainmed/rag.py -> repo root is parents[2].
ROOT = Path(__file__).resolve().parents[2]
CHUNKS_DIR = ROOT / "data" / "kb" / "chunks"
KB_DIR = ROOT / "data" / "kb"

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
]


# ── chunk loading + tokenization ──────────────────────────────────────────────


def load_chunks() -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(CHUNKS_DIR.glob("*.json"))]


def _normalize_phrases(text: str) -> str:
    text = text.lower()
    for pat, repl in _PHRASE_NORMALIZE:
        text = pat.sub(repl, text)
    return text


def _tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", _normalize_phrases(text))
    return [t for t in toks if len(t) > 1 and t not in _STOPWORDS]


def _fingerprint(chunks: list[dict]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["chunk_id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(c["text"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ── retrievers ────────────────────────────────────────────────────────────────


class TfidfRetriever:
    name = "tfidf (pure-python)"
    cached = False

    def __init__(self, chunks: list[dict]):
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


class EmbeddingRetriever:
    """Neural embeddings (OpenAI / Voyage / sentence-transformers) + cosine, with a
    persistent on-disk cache keyed by a fingerprint of the chunk texts."""

    def __init__(self, chunks: list[dict], backend: str, use_cache: bool = True, model: str | None = None):
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

        fp = _fingerprint(chunks)
        slug = re.sub(r"[^a-z0-9]+", "-", f"{backend}_{self.model}".lower()).strip("-")
        cache_path = KB_DIR / f"embeddings_{slug}.npz"
        self.cached = False
        if use_cache and cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            if str(data["fingerprint"]) == fp:
                self.matrix = data["matrix"]
                self.cached = True
        if not self.cached:
            mat = self._embed([c["text"] for c in chunks], is_query=False)
            self.matrix = self._normalize(np.asarray(mat, dtype=float))
            KB_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, matrix=self.matrix, fingerprint=fp)

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


def build_retriever(chunks: list[dict], backend: str, use_cache: bool = True, model: str | None = None):
    if backend == "tfidf":
        return TfidfRetriever(chunks)
    return EmbeddingRetriever(chunks, backend, use_cache=use_cache, model=model)


# ── generation ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior Arthrex clinical sales specialist talking a surgeon through ROTATOR CUFF "
    "and SHOULDER REPAIR at a case review. Confident, concise, practical — a rep who has been in "
    "the OR a thousand times. Answer only from the numbered source excerpts; cite every claim "
    "inline with [n].\n\n"
    "SCOPE: Stay strictly on rotator cuff / shoulder repair. Do NOT bring up unrelated procedures "
    "(ankle, knee, biceps tenodesis, labral/instability, arthroplasty, etc.) unless the question "
    "explicitly asks about them. If the excerpts don't cover what's asked, say so plainly.\n\n"
    "SPECS — zero hallucination: give exact figures (drill/punch/reamer diameters, anchor sizes, "
    "FiberTape/suture specs) ONLY when they appear verbatim in the sources, each cited [n]. If a "
    "spec isn't in the excerpts, tell the surgeon to confirm it in the technique guide — never "
    "guess a number.\n\n"
    "FORMAT — keep it short and complete:\n"
    "- Bottom line first, in one sentence.\n"
    "- Then the essentials only: numbered steps for a procedure, or 2-4 tight bullets otherwise.\n"
    "- Add one practical 'OR pearl' only when a source supports it.\n"
    "- **Bold** key implants, instruments, and specs; use exact Arthrex product names.\n"
    "- No preamble, no restating the question, no marketing adjectives."
)


def _loc(c: dict) -> str:
    if c.get("timestamp_range"):
        return c["timestamp_range"]["label"]
    if c.get("source_type") == "pdf":
        return f"PDF, {c.get('page_count') or '?'} pp"
    return "n/a"


def _kind(c: dict) -> str:
    return "video" if c.get("source_type") == "youtube" else "surgical guide"


def _context_block(retrieved: list[dict]) -> str:
    parts = []
    for n, c in enumerate(retrieved, 1):
        parts.append(f"[{n}] {c['source_title']} ({_kind(c)}, {_loc(c)})\n{c['text']}")
    return "\n\n".join(parts)


def _user_prompt(question: str, retrieved: list[dict]) -> str:
    return f"Source excerpts:\n{_context_block(retrieved)}\n\nQuestion: {question}"


def generate_answer(question: str, retrieved: list[dict], backend: str, model: str | None = None) -> str:
    user = _user_prompt(question, retrieved)
    if backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model or GEN_MODEL_ANTHROPIC,
            max_tokens=700,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()
    if backend == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model or GEN_MODEL_OPENAI,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=700,
        )
        return resp.choices[0].message.content.strip()
    return _extractive_answer(retrieved)


def stream_answer(question: str, retrieved: list[dict], backend: str, model: str | None = None):
    """Yield answer text deltas. Mirrors generate_answer but streams (for the web UI)."""
    user = _user_prompt(question, retrieved)
    if backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        with client.messages.stream(
            model=model or GEN_MODEL_ANTHROPIC,
            max_tokens=700,
            system=SYSTEM_PROMPT,
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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=700,
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


def deep_link(c: dict) -> str:
    """YouTube → exact-timestamp link; PDF → its source URL."""
    url = c.get("source_url") or ""
    if c.get("source_type") == "youtube" and c.get("timestamp_range"):
        start = int(c["timestamp_range"]["start_seconds"])
        vid = c.get("source_id")
        if vid:
            return f"https://www.youtube.com/watch?v={vid}&t={start}s"
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}t={start}s"
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
    },
]

SCENARIO_BY_ID = {s["id"]: s for s in SCENARIOS}

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


def roleplay_system(scenario: dict, role: str, difficulty: str, context: str) -> str:
    persona = scenario.get("persona", "a skeptical orthopedic surgeon")
    goal = scenario.get("goal", "")
    r = ROLES.get(role, ROLES["sales_rep"])
    d = DIFFICULTY.get(difficulty, DIFFICULTY["medium"])
    return (
        f"You are role-playing as {persona}, speaking out loud to a TrainMed user who is a {r} "
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
        f"make or challenge a technical claim, rely on the Arthrex reference excerpts below so your "
        f"facts are correct; do not invent specifications. Do NOT break character, do NOT coach the "
        f"rep, do NOT hand them the answer, and never use [n] citation markers — you are the customer, "
        f"not a document.\n\n"
        f"(The rep's objective, for your awareness: {goal})\n\n"
        f"Arthrex reference excerpts (for your technical accuracy only):\n{context}"
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


def stream_roleplay(scenario, role, difficulty, history, message, retrieved, backend, model=None):
    """Stream the AI surgeon's reply to the rep's latest message."""
    system = roleplay_system(scenario, role, difficulty, _context_block(retrieved))
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

SCORE_DIMENSIONS = ["technique", "product_knowledge", "objection_handling", "confidence", "completeness"]

COACH_SYSTEM = (
    "You are a tough but fair Arthrex sales-training coach. Score the REP's latest message in a "
    "roleplay with a surgeon, using the Arthrex source excerpts as the ground truth of what is "
    "accurate. Reward specific, correct, well-structured, confident answers; penalize vagueness, "
    "wrong/unsupported specs, hedging, and dodging the objection. Respond with ONLY a JSON object, "
    "no prose, of exactly this shape:\n"
    '{"scores":{"technique":1-5,"product_knowledge":1-5,"objection_handling":1-5,'
    '"confidence":1-5,"completeness":1-5},"feedback":"1-2 sentence critique","tips":["tip","tip"]}'
)


def score_exchange(scenario, message, retrieved, backend, model=None) -> dict:
    """Score a rep message. LLM coach when a key is set; deterministic heuristic otherwise."""
    if backend in ("anthropic", "openai"):
        card = _llm_score(scenario, message, retrieved, backend, model)
        if card:
            card["method"] = backend
            return card
    card = _heuristic_score(message)
    card["method"] = "heuristic"
    return card


def _llm_score(scenario, message, retrieved, backend, model):
    user = (
        f"Scenario: {scenario.get('label')} — surgeon persona: {scenario.get('persona')}\n\n"
        f"Arthrex source excerpts (ground truth):\n{_context_block(retrieved)}\n\n"
        f"REP's latest message to score:\n\"\"\"{message}\"\"\""
    )
    try:
        if backend == "anthropic":
            import anthropic

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=model or GEN_MODEL_ANTHROPIC, max_tokens=400,
                system=COACH_SYSTEM, messages=[{"role": "user", "content": user}],
            )
            raw = "".join(b.text for b in msg.content if b.type == "text")
        else:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model=model or GEN_MODEL_OPENAI,
                messages=[{"role": "system", "content": COACH_SYSTEM}, {"role": "user", "content": user}],
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


def _heuristic_score(message: str) -> dict:
    text = _normalize_phrases(message or "")
    toks = set(re.findall(r"[a-z0-9-]+", text))

    n_prod = len(_PRODUCT_TERMS & toks)
    n_tech = len(_TECH_TERMS & toks)
    n_evid = len(_EVIDENCE_TERMS & toks)
    n_ack = sum(1 for p in _ACK_TERMS if p in text)
    n_hedge = sum(1 for h in _HEDGES if h in text)
    words = len(message.split())
    has_spec = bool(re.search(r"\b\d+(?:\.\d+)?\s*mm\b", message, re.I))

    product_knowledge = _band(n_prod + (1 if has_spec else 0))
    technique = _band(n_tech)
    objection_handling = _band(n_ack + n_evid)
    confidence = max(1, min(5, 4 + (1 if (n_evid or has_spec) and n_hedge == 0 else 0) - n_hedge))
    if words < 8:
        completeness = 1
    elif words < 18:
        completeness = 2
    elif words < 45:
        completeness = 3
    else:
        completeness = 4
    if product_knowledge >= 4 and technique >= 3 and (n_evid or has_spec):
        completeness = min(5, completeness + 1)

    scores = {
        "technique": technique,
        "product_knowledge": product_knowledge,
        "objection_handling": objection_handling,
        "confidence": confidence,
        "completeness": completeness,
    }
    tips = []
    if product_knowledge < 3:
        tips.append("Name specific products and specs (e.g. 2.6 mm FiberTak, 4.75 mm SwiveLock, 1.7 mm FiberTape).")
    if objection_handling < 3:
        tips.append("Acknowledge the surgeon's concern, then counter with concrete evidence or data.")
    if confidence < 4:
        tips.append("Cut the hedging ('I think', 'maybe') — state your points directly.")
    if completeness < 3:
        tips.append("Go deeper: address the actual question and add a supporting detail or step.")
    if not tips:
        tips.append("Strong answer — close with a clear next step (a trial case or a follow-up).")

    overall = round(sum(scores.values()) / len(scores), 1)
    quality = "strong" if overall >= 4 else "solid" if overall >= 3 else "needs work"
    feedback = (
        f"{quality.capitalize()} response ({overall}/5). "
        f"Product specificity {product_knowledge}/5, objection handling {objection_handling}/5, "
        f"confidence {confidence}/5."
    )
    return {"scores": scores, "overall": overall, "feedback": feedback, "tips": tips[:3]}
