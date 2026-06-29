# Smith & Nephew — Shoulder / Rotator Cuff source list

_Updated 2026-06-29. Expansion to match the Arthrex / Stryker rotator-cuff focus._
_Method: discovery (per-product web research + the repo's `scrape_pdf_links`) → **every URL
re-verified by the real ingest fetch (urllib + %PDF magic-byte gate)** → ingest. All ingestible
items below returned `application/pdf` with a genuine `%PDF` header. **No video streams / .m3u8 /
.ts / transcripts were ingested** — video pages are recorded as metadata only._

- **Ingested PDFs:** 27 (26 official S&N CDN + 1 authorized distributor) → **72 KB chunks, 48,618 words**
- **Provenance:** `smith-nephew.stylelabs.cloud` (official CDN) for all but the FOOTPRINT PK
  technique guide, which is on `ortovit.eu` (authorized S&N distributor) — same provenance
  pattern already used for the BIORAPTOR guide.

## Direct PDFs — ingested (priority targets)

### HEALICOIL REGENESORB family + HEALICOIL Knotless
| Title | Kind |
| --- | --- |
| HEALICOIL Suture Anchor Family brochure (REGENESORB; Ultratape/Minitape/Ultrabraid) | brochure |
| HEALICOIL Knotless Suture Anchor reference/technique guide — self-tapping | technique_guide |
| HEALICOIL Knotless Suture Anchor reference/technique guide — non-self-tapping | technique_guide |
| HEALICOIL Knotless Suture Anchor reference guide — self-tapping (suture-threader workflow) | technique_guide |
| HEALICOIL REGENESORB Evidence in Focus — fixation properties & stress distribution | evidence |
| HEALICOIL REGENESORB biocomposite suture anchor — Evidence in Focus | evidence |

### Q-FIX All-Suture Anchor family (shoulder)
| Title | Kind |
| --- | --- |
| Q-FIX All-Suture Anchor Family brochure (shoulder repair portfolio) | brochure |
| Q-FIX KNOTLESS All-Suture Anchor — shoulder surgical technique guide | technique_guide |
| Q-FIX KNOTLESS All-Suture Anchor — suture-passing / drill step-by-step guide | technique_guide |
| Q-FIX — arthroscopic cinching-loop technique for biceps tenodesis | technique_guide |
| Q-FIX — arthroscopic technique for biceps tenodesis | technique_guide |
| Q-FIX 1.8mm All-Suture Anchor — Evidence in Focus (low incidence of pullout) | evidence |

### MULTIFIX S ULTRA Knotless
| Title | Kind |
| --- | --- |
| MULTIFIX S ULTRA Knotless Suture Anchor — product brochure | brochure |
| MULTIFIX S ULTRA Knotless — arthroscopic rotator cuff repair shoulder technique guide | technique_guide |

### FOOTPRINT Ultra / FOOTPRINT PK
| Title | Kind |
| --- | --- |
| TWINFIX Ultra & FOOTPRINT Ultra Suture Anchor family sales sheet (double-row) | brochure |
| FOOTPRINT PK Suture Anchor — arthroscopic shoulder repair surgical technique guide (via ortovit.eu) | technique_guide |

### REGENESORB material + Advanced Healing Solutions portfolio
| Title | Kind |
| --- | --- |
| REGENESORB Absorbable Biocomposite Material brochure | brochure |
| Advanced Healing Solutions — Rotator Cuff Repair portfolio brochure (HEALICOIL + REGENETEN) | brochure |

### REGENETEN Bioinductive Implant + rotator-cuff clinical evidence
| Title | Kind |
| --- | --- |
| REGENETEN Advanced Healing Solutions — clinical evidence collection (35 pp) | clinical_evidence |
| REGENETEN clinical summary — a decade of clinical evidence | clinical_evidence |
| REGENETEN 2-year RCT — re-tear reduction in full-thickness rotator cuff tears | clinical_evidence |
| REGENETEN landmark RCT data — partial-thickness rotator cuff tears | clinical_evidence |
| REGENETEN and the AAOS clinical practice guidelines summary | clinical_evidence |
| REGENETEN data on pain, function and early clinical outcomes | clinical_evidence |
| REGENETEN — partial-thickness tear size reduction & progression | clinical_evidence |
| REGENETEN 10-year cost analysis summary | clinical_evidence |
| Evidence in Focus — REGENETEN Bioinductive Implant | clinical_evidence |

_Exact URLs: see `data/urls/smithnephew_shoulder_pdfs.txt`._

## Video / animation pages — METADATA ONLY (never stream-ingested, per compliance rules)

Stored as title + official page URL only. Notably these cover the technique gaps that have
**no** ingestible PDF (UltraCONTACT / Ultra Contact double-row knotless suture-bridge; REGENETEN
double-row augmentation; triple-row HEALICOIL Knotless):

- UltraCONTACT / Ultra Contact — Double-Row Knotless Rotator Cuff Repair (HEALICOIL Knotless + FOOTPRINT/MULTIFIX) — smith-nephew.com & VuMedi S&N channel
- Triple-Row Rotator Cuff Repair using HEALICOIL Knotless — smith-nephew.com education
- ULTRALOCK Arthroscopic Rotator Cuff Repair using HEALICOIL Knotless REGENESORB — VuMedi S&N channel
- Double-Row Rotator Cuff Repair with REGENETEN Bioinductive Augmentation — VuMedi S&N channel
- Double-Row Rotator Cuff Repair featuring HEALICOIL, MULTIFIX S ULTRA — VuMedi S&N channel
- Q-FIX CURVED / KNOTLESS technique animations (shoulder; Bankart + Remplissage) — VuMedi S&N channel
- Smith+Nephew Shoulder Repair video collection + INSPIRE Live Surgery education — VuMedi / educationunlimited.smith-nephew.com

## Reference product / procedure / press pages (HTML — context, not ingested)

HEALICOIL REGENESORB, HEALICOIL Knotless, Q-FIX (shoulder), MULTIFIX S/P, FOOTPRINT Ultra,
REGENETEN (product + partial-/full-thickness procedure + campaign hub), MICRORAPTOR REGENESORB,
and the HEALICOIL Knotless + Q-FIX expansion press releases — all on smith-nephew.com.

## Verifier flags / decisions (read before sales use)

- **Competitive comparison content is present (by design, and compliant).** Several S&N
  brochures/evidence PDFs (HEALICOIL family, Advanced Healing Solutions, TWINFIX/FOOTPRINT sales
  sheet, REGENETEN evidence) contain **S&N's own head-to-head claims and citations naming Arthrex
  anchors** (SwiveLock, PushLock, SutureTak, Corkscrew). These chunks are correctly tagged
  `company=SmithNephew` and the retrieval firewall isolation test passes — they are S&N-authored
  content, not cross-contamination. Side effect: the generic, Arthrex-named `topics` index tags
  fire on those competitor mentions (cosmetic only; `product_family`/`company` are authoritative).
- **EXCLUDED — not a PDF:** the “MICRORAPTOR Reference Guide” CDN URL
  (`…/1183446afe48…`) actually returns a **JPEG**, not a PDF (a discovery agent mis-claimed `%PDF`;
  caught by both the critic and the real ingest gate). Excluded.
- **EXCLUDED — off rotator-cuff focus:** ULTRABRIDGE Achilles (foot/ankle) brochure that surfaces
  in FOOTPRINT Ultra searches; hip-indication Q-FIX docs (hip family brochure, CURVED XL hip ref
  guide, hip labral STG, ISHA white paper); MICRORAPTOR/BIORAPTOR (shoulder **instability**, not
  rotator cuff). Available if instability scope is later added.
- **EXCLUDED — unfetchable:** Bushnell et al. REGENETEN registry PDF on PMC sits behind an
  anti-bot “Preparing to download…” interstitial — not curl/urllib-ingestible. Use the official
  S&N REGENETEN evidence summaries (ingested above) instead.
- **De-duplicated:** the “Advanced Healing Solutions” brochure appeared 3× (HEALICOIL/REGENETEN/
  delivery-host) and the HEALICOIL family brochure 2× — same `content_id`, byte-identical by
  SHA-256; ingested once each.
- **Region note:** a few REGENETEN materials carry ANZ labelling; the ANZ patient brochure and
  rehab protocol were left out (consumer-facing / regional). Indications can differ by region.
