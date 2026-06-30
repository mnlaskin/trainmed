# TrainMed — Multi-Company Knowledge Base

TrainMed now serves **multiple medical-device companies** (Arthrex, Stryker, Smith &
Nephew) from one app, with **strict separation** so a product answer or sales roleplay
for one company can never be contaminated by another's content.

Two product surfaces, both company-scoped:

1. **Knowledge chatbot** — accurate product knowledge, answered only from the selected
   company's sources.
2. **Competitive sales roleplay** — practice against an AI surgeon grounded in the
   selected company's KB, plus a curated cross-company `competitive_insights` layer for
   head-to-head scenarios.

---

## Current coverage (as of 2026-06-29, post spec gap-fill)

| Company | Chunks | Source PDFs | Product families covered |
| --- | --- | --- | --- |
| **Arthrex** | **242** | 63 PDF + 15 video | SpeedBridge, Instability (Bankart/SLAP/remplissage/Latarjet/glenoid bone loss), FiberTak, ArthroFLEX, FiberTape/FiberWire/SutureTape, SCR, SutureBridge, Biceps Tenodesis, PushLock, Corkscrew, CuffMend, SutureTak, SpeedFix, SwiveLock |
| **Stryker** | **80** | 43 PDF | InSpace, AlphaVent (hard-body + knotless), Cobra, Instability (Iconix/NanoTack 1.4 mm platform), Omega Knotless, XBraidTT/Force Fiber, Iconix, Cinchlock, Champion Slingshot, Knotilus+, PEEK IntraLine, PEEK Zip, ReelX STT |
| **Smith & Nephew** | **103** | 37 PDF | REGENETEN, Q-FIX, Instability (SUTUREFIX/Latarjet/Bankart), HEALICOIL, FOOTPRINT Ultra, MULTIFIX, MICRORAPTOR, REGENESORB, TWINFIX Ultra |
| Cross-company | 8 | — | `competitive_insights` (sentinel `_competitive`) |
| **Total** | **425 + 8** | | |

**Spec coverage filled:** anchor diameters/lengths/drill sizes & materials (PEEK / all-suture /
BioComposite / HA), suture & tape widths and types (FiberTape 1.7 & 2 mm, SutureTape 1.3 mm,
XBraid TT 1.4/1.8/2.2 mm, Force Fiber #2), fixation methods (knotless, self-punching,
tensionable, double-row/TOE), design/biomechanical features, and clinical/evidence summaries.

**Gap-fill method (reproducible, conservative):** official manufacturer PDFs only (no video
streams). Stryker via deterministic `pdf_ingest.scrape_pdf_links` over stryker.com product
pages; Arthrex via WebFetch of arthrex.com product pages → `/resources/LT1-*` & `/DOC1-*`
technique-guide/evidence PDFs; Smith & Nephew via search of smith-nephew.com +
`smith-nephew.stylelabs.cloud` CDN. Every candidate was `%PDF`-gated, deduped by URL and by
extracted-text SHA, and scope-checked (knee/ACL/hip/foot/arthroplasty excluded). Seed lists:
`data/urls/{arthrex,stryker,smithnephew}_shoulder_gap*.txt`.

---

## 1. The chunk schema

Every chunk is one JSON file carrying the original fields **plus** the multi-company
metadata. The new fields are in **bold**:

| Field | Type | Example |
| --- | --- | --- |
| `chunk_id` | str | `"omega-20product-20brochure-updated_000"` |
| **`company`** | `"Arthrex" \| "Stryker" \| "SmithNephew"` | `"Stryker"` |
| `source_id` / `source_type` / `source_title` / `source_url` | str | … |
| `channel` / `upload_date` / `page_count` / `language` | str / null | … |
| `procedure_family` | str | `"rotator_cuff"` |
| **`product_line`** | str | `"Rotator Cuff"`, `"Shoulder"` |
| **`product_family`** | str | `"SpeedBridge"`, `"Omega Knotless"`, `"HEALICOIL"` |
| **`product_name`** | str | `"Omega Knotless"`, `"2.6 FiberTak RC"` |
| **`category`** | `anchor \| suture_tape \| drill \| implant \| technique_guide \| clinical_study` | `"anchor"` |
| **`technique`** | list | `["knotless", "double_row"]` |
| **`advantages`** | list | `["all-PEEK knotless", "decoupled eyelet"]` |
| **`disadvantages`** | list | `["non-resorbable PEEK"]` |
| **`clinical_references`** | list | `["Omega Evidence Matters (LBF)"]` |
| `chunk_index` / `chunk_count` / `word_count` | int | … |
| `timestamp_range` / `topics` | obj / list | … |
| `text` | str | the chunk body |

A real migrated/ingested chunk:

```json
{
  "chunk_id": "omega-20product-20brochure-updated_000",
  "company": "Stryker",
  "source_type": "pdf",
  "source_title": "Omega Knotless Anchor System brochure / sell sheet",
  "source_url": "https://www.stryker.com/.../Omega product brochure-updated.pdf",
  "procedure_family": "rotator_cuff",
  "product_line": "Rotator Cuff",
  "product_family": "Omega Knotless",
  "product_name": "Omega Knotless",
  "category": "anchor",
  "technique": ["knotless", "double_row"],
  "advantages": [],
  "disadvantages": [],
  "clinical_references": [],
  "topics": ["Knotless fixation", "Suture anchors"],
  "text": "..."
}
```

`product_*`, `category`, and `technique` are **inferred deterministically** by
`trainmed.companies.infer_product_fields` (the same code for live ingestion and the
backfill). `advantages` / `disadvantages` / `clinical_references` default to `[]` and are
filled from structured PDF front-matter or future structured ingestion — they are never
guessed.

### Storage layout

```
data/
  transcripts/                 # legacy Arthrex YouTube transcripts (flat = Arthrex)
  transcripts/<Company>/       # other companies' transcripts
  pdfs/                        # legacy Arthrex extracted PDFs (flat = Arthrex)
  pdfs/<Company>/              # other companies' extracted PDFs
  kb/
    chunks/                    # legacy Arthrex chunks (flat = Arthrex)
    <Company>/chunks/          # e.g. data/kb/Stryker/chunks/*.json
    competitive_insights/      # cross-company collection (SEPARATE)
    <company>_<family>_index.md
  competitive/insights.json    # source for the competitive_insights collection
  urls/                        # per-company PDF seed lists
```

Arthrex stays in the legacy flat dirs (the committed KB is undisturbed); every other
company gets its own `<Company>/` subtree.

---

## 2. How company filtering / the firewall works

The firewall is enforced in **`src/trainmed/rag.py`**:

- **`load_chunks(company)`** returns *only* that company's chunks. The directory is
  authoritative for `<Company>/chunks/`; a chunk whose `company` field disagrees with its
  directory is **dropped and warned** (fail-closed) rather than misrouted. A non-Arthrex
  chunk found in the legacy flat dir is likewise dropped.
- **One retriever per company**, built once at startup over only that company's chunks.
  A request resolves its company from an **allowlist** and reads `RETRIEVERS[company]` —
  competitor chunks are never in the candidate set.
- **Retrievers own their chunks.** `retriever.retrieve(q, k)` returns chunk *dicts*
  directly, so no caller indexes a shared global list (the old positional-index coupling
  was the biggest silent-leak risk).
- **`enforce_company(chunks, company)`** is a belt-and-suspenders guard: before any chunk
  reaches a prompt or citation, a chunk whose `company` ≠ the request's is dropped + warned.
- **Embeddings cache is namespaced** by company in *both* the filename and the fingerprint
  (`embeddings_<company>_<backend>_<model>.npz`) and the company is stored in the `.npz`
  and re-checked on load — a cache built for one company can never be reused for another.
- **Prompts, roleplay personas, the coach, and the heuristic scorer vocab are
  parameterized by company** (`system_prompt(company)`, `scenarios_for(company)`,
  `coach_system(company)`, `companies.product_terms(company)`).

### Server endpoints (all company-aware)

| Endpoint | Company source |
| --- | --- |
| `GET /api/companies` | lists every known company + whether it has an ingested KB |
| `GET /api/meta?company=Stryker` | per-company KB stats |
| `GET /api/ask?q=…&company=Stryker` | answers from Stryker only (or a friendly "ingest first" message) |
| `GET /api/roleplay/scenarios?company=Stryker` | company-specific scenario set |
| `POST /api/roleplay/turn` `{ "company": "Stryker", … }` | grounds the surgeon + coach in Stryker only |

The header **company switcher** in the web UI drives all of the above.

### Isolation is tested

`scripts/test_isolation.py` asserts the firewall and exits non-zero on any leak (wire it
into CI). It currently passes for Arthrex + Stryker, including cross-company probes — e.g.
a Stryker query for "SpeedBridge FiberTape" (an Arthrex construct) returns **only** Stryker
chunks, and an Arthrex query for "Omega knotless" returns **only** Arthrex chunks.

```bash
PYTHONPATH=src python scripts/test_isolation.py
```

---

## 3. The `competitive_insights` collection

Cross-company comparisons live **only** in `data/kb/competitive_insights/` and are loaded
by `load_competitive_insights()` — never by `load_chunks(company)`. Each insight chunk is
tagged `company: "_competitive"` (a sentinel that can never match a real company filter)
and `collection: "competitive"`.

They are used **only** for explicit comparison and **competitive roleplay** scenarios
(the surgeon who currently uses a rival system). They are never blended into a
single-company product answer.

Each insight carries: `topic`, `companies`, `summary`, `arthrex_position`,
`stryker_position`, `smithnephew_position`, `rep_talk_track`, `evidence`, `sources`,
`confidence`, `notes`.

Rebuild from source:

```bash
python scripts/ingest_competitive_insights.py --clean   # reads data/competitive/insights.json
```

---

## 4. CLI / scripts

```bash
# 1. Download + extract a company's PDFs  (writes data/pdfs/<Company>/*.md)
python -m trainmed.pdf_ingest --company Stryker --from-file data/urls/stryker_shoulder_pdfs.txt

# 2. Chunk that company into the KB        (writes data/kb/<Company>/chunks/*.json + index)
python scripts/ingest_to_kb.py --company Stryker

# 3. Backfill the schema onto pre-existing chunks (idempotent; Arthrex already done)
python scripts/migrate_add_company.py --dry-run
python scripts/migrate_add_company.py

# 4. Build the cross-company competitive_insights collection
python scripts/ingest_competitive_insights.py --clean

# 5. Test one company's RAG end to end
PYTHONPATH=src python scripts/test_rag.py --company Stryker --backend tfidf --gen extractive

# 6. Prove isolation (CI gate)
PYTHONPATH=src python scripts/test_isolation.py
```

`pdf_ingest` and `ingest_to_kb` default to `--company Arthrex` (backward-compatible).
`--clean` on `ingest_to_kb` is company-scoped (only that company's chunks).

---

## 5. Targeted ingestion plan

### Stryker (priority — shoulder / rotator cuff) — STARTED ✅

Verified, reachable seed list in [`data/urls/stryker_shoulder_pdfs.txt`](../data/urls/stryker_shoulder_pdfs.txt)
(direct PDFs) and the full reference + product taxonomy in
[`data/urls/stryker_shoulder_sources.md`](../data/urls/stryker_shoulder_sources.md).

Ingested: **21 official Stryker.com PDFs → 36 chunks** in `data/kb/Stryker/chunks/`, across
7 product families: **Omega Knotless** (incl. Double-Double double-row guide + "Evidence
Matters" data), **AlphaVent** (knotless + hard-body + AlphaVent-to-Omega RCR technique
guide), **Iconix** (all-suture + Knotless + HA+ + SPEED-C RCR), **ReelX STT**, **Knotilus+**
(PEEK knotless), **Instability** (Iconix/NanoTack shoulder-instability guides), and **InSpace**
(subacromial balloon for irreparable cuff tears + 2 clinical literature summaries). Category
mix: 19 technique_guide / 10 anchor / 7 clinical_study. All video assets (VuMedi/YouTube) are
metadata-only — never stream-ingested; near-duplicate and foot/ankle/arthroplasty PDFs were
excluded. Curated seed: [`data/urls/stryker_shoulder_pdfs.txt`](../data/urls/stryker_shoulder_pdfs.txt).
Next candidates: verified open-access PubMed studies (listed in the sources file). No official
Stryker shoulder **biceps-tenodesis / SCR** PDFs exist (Citrelock is foot/ankle only).

### Smith & Nephew (later) — SEEDED

Seed list in [`data/urls/smithnephew_shoulder_pdfs.txt`](../data/urls/smithnephew_shoulder_pdfs.txt)
and [`data/urls/smithnephew_shoulder_sources.md`](../data/urls/smithnephew_shoulder_sources.md).
Priorities: **HEALICOIL** (PRO / REGENESORB / KNOTLESS), **Q-FIX** all-suture, FOOTPRINT
Ultra, REGENESORB. Run the same two-step ingest with `--company SmithNephew`.

> Verifier flags (e.g. a misattributed HEALICOIL citation, BIORAPTOR knotless naming) are
> recorded in each `*_sources.md` — review before sales-training use.

---

## 6. Adding a new company

1. Add an entry to `COMPANIES` (display name, portfolio, scorer vocab) and, if you want
   product inference, a `TAXONOMY` block in `src/trainmed/companies.py`.
2. Seed `data/urls/<company>_shoulder_pdfs.txt`.
3. `pdf_ingest --company <Company> --from-file …` then `ingest_to_kb.py --company <Company>`.
4. Restart the server — it auto-discovers any company that has chunks.

No other code changes; nothing else hard-codes a brand.

---

## 7. Backward compatibility

- The 109 original Arthrex chunks were backfilled in place (now 101 after removing 8
  stray macOS duplicate files); the Arthrex flat dirs and committed KB are untouched in
  layout.
- `pdf_ingest` / `ingest_to_kb` / `test_rag` default to Arthrex.
- `SYSTEM_PROMPT`, `COACH_SYSTEM`, `SCENARIOS`, `SCENARIO_BY_ID` still exist as Arthrex
  constants for any external importer.
- The web UI defaults to Arthrex; with no company param, every endpoint behaves as before.
