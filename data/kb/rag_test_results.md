# TrainMed RAG test results

_RAG test — 105 chunks (18 video, 87 pdf) | embeddings: tfidf (pure-python) (no embedding API key — set OPENAI_API_KEY (or VOYAGE_API_KEY) for neural) | generation: anthropic (ANTHROPIC_API_KEY set) | top-k=5_

## Q1. What is the Arthrex SpeedBridge and which anchors and sutures does it use to repair a rotator cuff tear?

**Answer:**

## Arthrex SpeedBridge™ — Bottom Line

SpeedBridge is a knotless double-row rotator cuff repair construct that pairs **fully threaded SwiveLock** anchors with **FiberTape** to compress the rotator cuff footprint. Here's exactly how it works and what it uses.

---

## Construct Components

| Role | Implant |
|---|---|
| Medial row anchors | **SwiveLock** anchor loaded with a ready-to-use **FiberTape** Loop [3] |
| Lateral row anchors | **BioComposite SwiveLock** anchor, preloaded with **FiberTape** tails from medial anchors [3] |
| Supplemental soft tissue fixation | **FiberLink** + fourth **SwiveLock** anchor for any loose cuff edge [3] |

> **Note:** Exact anchor diameters and FiberTape dimensions are not stated in these source excerpts. Confirm those specs in the technique guide before the case.

---

## Procedure — Ordered Steps

1. Prepare the rotator cuff and humerus for anchor insertion [3].
2. Punch the first medial bone socket; insert the **SwiveLock** anchor pre-loaded with a **FiberTape** Loop [3].
3. Release the **FiberTape** Loop; use the **FastPass Scorpion** suture passer to pass the **FiberTape** through the rotator cuff and retrieve it [3].
4. Repeat steps 2–3 for the second medial **SwiveLock** anchor [3].
5. Trim the **FiberTape** Loop at the splice to separate the tails [3].
6. Retrieve **one FiberTape tail from each medial anchor** and preload both into a third **SwiveLock** anchor eyelet [3][5].
7. Prepare the lateral bone socket and insert that **BioComposite SwiveLock** anchor, compressing the footprint [3].
8. If a loose cuff edge remains, secure it with a **FiberLink** and the two remaining **FiberTape** tails using a **fourth SwiveLock** anchor [3].
9. Repair is complete — four anchors total for a standard construct [3].

---

## Key Technical Notes from the Technique Guide

- Tension each **FiberTape** suture limb **individually** before fully seating the lateral anchors to remove slack beneath the tendon [5].
- Advance the driver **beyond the first laser line** until the anchor body contacts bone before rotating to final seating [5].
- Do **not** apply tension to sutures with the eyelet already in the bone socket [5].
- On hard bone, use a punch to break the cortex before inserting the self-punching **SwiveLock** [5].
- Cut **FiberTape** tails with a dedicated **FiberTape cutter** [5].

**Sources:**
```
  [1] (YOUTUBE) Rotator Cuff Repair with Arthrex® SutureBridge™  ·  00:00:00-00:02:00  ·  score 0.200
      https://www.youtube.com/watch?v=LNSfL8g2PtM&t=0s
  [2] (YOUTUBE) Rotator Cuff Repair with SutureBridge™ Technique  ·  00:00:00-00:02:06  ·  score 0.200
      https://www.youtube.com/watch?v=IOxUH3PVNmw&t=0s
  [3] (YOUTUBE) Rotator Cuff Repair with Arthrex® SpeedBridge™  ·  00:00:00-00:02:31  ·  score 0.182
      https://www.youtube.com/watch?v=3OaWXXEdS7g&t=0s
  [4] (YOUTUBE) Knotless Rotator Cuff Repair with Arthrex® SpeedFix™  ·  00:00:00-00:01:26  ·  score 0.176
      https://www.youtube.com/watch?v=ztk8JlLtX70&t=0s
  [5] (PDF) FiberTak SpeedBridge Rotator Cuff Repair (technique guide)  ·  PDF, 20 pp  ·  score 0.159
      https://www.arthrex.com/resources/LT1-000204-en-US/fibertak-speedbridge-rotator-cuff-repair
```

## Q2. How does the SutureBridge double-row technique differ from the SpeedBridge for rotator cuff repair?

**Answer:**

## SutureBridge vs. SpeedBridge: The Core Difference Is Knots

**SutureBridge** ties medial knots; **SpeedBridge** is fully knotless — and that single distinction drives measurable differences in biomechanics, retear rates, and OR time.

---

### SutureBridge — How It Works
1. Place two medial **Corkscrew FT** anchors at the articular margin; pass sutures through the cuff bottom to top [2][3]
2. **Tie medial knots** to secure the tendon down — this is the defining feature [2][3]
3. Cross sutures laterally and secure to bone with knotless **PushLock** anchors [2][3]

---

### SpeedBridge — How It Differs
1. Same medial anchor placement and suture passage [2][3]
2. **No medial knots tied** — completely knotless construct throughout [1][4]
3. Lateral fixation uses **SwiveLock** anchors with crossed **FiberTape** [4][5]

---

### Why It Matters Clinically

**Biomechanics — self-reinforcement:**
- In a knotless construct, footprint contact pressure *increases* under load (self-reinforcement effect). Medial knots actively inhibit this — they create an adverse biomechanical effect by blocking the self-reinforcing mechanism [1]
- This was confirmed in two independent biomechanical studies [1]

**Retear rates:**
- In a 155-case retrospective review, retear rate was **33% (4/12)** in the knotted **SutureBridge** group vs. **7.5% (3/40)** in the knotless **SpeedBridge** group at mean 4.4 years on MRI [4]
- A prospective 110-patient study showed **94% healing** with knotless vs. **81%** with medial knots (P < .001), with 8 of 11 failures in the knotted group being medial (Type II) failures [1]

**Medial failures specifically:**
- Zero medial failures in the knotless group vs. 8 of 11 failures occurring at the medial row in the knotted group — the knot itself appears to be the failure point [1]

**OR efficiency:**
- Knotless **SpeedBridge** reduced procedure time to **43.5 minutes** vs. **80 minutes** for knotted technique, and OR costs were significantly lower (**$3,788.40 vs. $4,262.90**, P < .01) — even with more anchors used [4][5]

**Long-term outcomes (10-year data):**
- Both techniques show durable results: only **5.5% revision rate** at minimum 10 years, with mean ASES of 93 and SANE of 88 — so the functional floor

**Sources:**
```
  [1] (PDF) Knotless Double-Row RCR — Scientific Update (biomechanics)  ·  PDF, 6 pp  ·  score 0.191
      https://www.arthrex.com/resources/DOC1-000217-en-US/knotless-double-row-rotator-cuff-repair-scientific-update
  [2] (YOUTUBE) Rotator Cuff Repair with Arthrex® SutureBridge™  ·  00:00:00-00:02:00  ·  score 0.185
      https://www.youtube.com/watch?v=LNSfL8g2PtM&t=0s
  [3] (YOUTUBE) Rotator Cuff Repair with SutureBridge™ Technique  ·  00:00:00-00:02:06  ·  score 0.185
      https://www.youtube.com/watch?v=IOxUH3PVNmw&t=0s
  [4] (PDF) Knotless Double-Row RCR — Scientific Update (biomechanics)  ·  PDF, 6 pp  ·  score 0.183
      https://www.arthrex.com/resources/DOC1-000217-en-US/knotless-double-row-rotator-cuff-repair-scientific-update
  [5] (PDF) Knotless Double-Row RCR — Scientific Update (biomechanics)  ·  PDF, 6 pp  ·  score 0.169
      https://www.arthrex.com/resources/DOC1-000217-en-US/knotless-double-row-rotator-cuff-repair-scientific-update
```

## Q3. What graft and fixation are used in superior capsular reconstruction, and when is it indicated?

**Answer:**

## Superior Capsular Reconstruction: Graft, Fixation, and Indications

**Bottom line:** SCR is indicated for chronic, massive, irreparable posterosuperior rotator cuff tears with superior capsule defects in a joint without significant glenohumeral arthritis. The graft of choice is a **dermal allograft**, fixed medially to the superior glenoid and laterally to the greater tuberosity with knotless double-row fixation.

---

### Indications

SCR is appropriate when [5]:
- Massive posterosuperior rotator cuff tear that is **irreparable** or very poor quality/unlikely to heal
- Glenohumeral joint with **minimal degenerative changes**
- Failed previous cuff repair
- Patient profile where joint preservation is preferred over reverse shoulder arthroplasty (age, activity requirements)

Avoid this technique in patients with severe glenohumeral osteoarthritis or significant bone defects in the glenoid or humeral head [2].

---

### Graft Selection

- **Acellular dermal allograft** is preferred over tensor fascia lata autograft — eliminates donor-site morbidity and reduces operative time [5]
- Originally described with fascia lata autograft by Mihata et al.; multiple groups have since adopted allograft [3][5]

---

### Fixation Construct — Step by Step

1. **Patient positioning:** Beach chair, arm in 20°–30° abduction and 20°–30° forward flexion [5]
2. **Diagnostic arthroscopy:** Assess subscapularis; repair if torn. Perform biceps tenodesis in most cases [5]
3. **Subacromial access:** Place a **10-mm flexible cannula (PassPort; Arthrex)** in the lateral portal for suture management and graft passage [5]
4. **Limited subacromial decompression:** Preserve the coracoacromial arch [5]
5. **Posterior interval slide** if needed to improve exposure of the superior glenoid or assess repairability [5]
6. **Medial (glenoid) fixation:** Place anchors at the superior glenoid just medial to the superior labrum — **2 anchors at the 10 o'clock and 2 o'clock positions** via the Neviaser portal [2]; suture limbs passed **2 mm anterior and 2 mm posterior** to their respective marks on the graft using an antegrade suture passer [4]
7. **Graft delivery:** Use the medial anchor eyelets as pulleys — tie one limb from each glenoid anchor over a switching stick, pull the two remaining limbs to advance the graft to the cannula orifice, then fold and introduce with a tissue grasper [4]
8. **Medial graft seating:** Once positioned on the superior glenoid, tie the two remaining suture limbs as a static knot in the subacromial space [

**Sources:**
```
  [1] (PDF) SCR With Superimposition of Acellular Dermal Allograft  ·  PDF, 5 pp  ·  score 0.138
      https://www.jorgechahlamd.com/wp-content/uploads/2019/12/SCR.pdf
  [2] (PDF) SCR With Superimposition of Acellular Dermal Allograft  ·  PDF, 5 pp  ·  score 0.119
      https://www.jorgechahlamd.com/wp-content/uploads/2019/12/SCR.pdf
  [3] (PDF) SCR With Superimposition of Acellular Dermal Allograft  ·  PDF, 5 pp  ·  score 0.109
      https://www.jorgechahlamd.com/wp-content/uploads/2019/12/SCR.pdf
  [4] (PDF) The Arthroscopic Superior Capsular Reconstruction (AJO)  ·  PDF, 5 pp  ·  score 0.107
      https://cdn.mdedge.com/files/s3fs-public/issues/articles/content_29e3e01c19e206a3bf_ajo045070320.PDF
  [5] (PDF) The Arthroscopic Superior Capsular Reconstruction (AJO)  ·  PDF, 5 pp  ·  score 0.104
      https://cdn.mdedge.com/files/s3fs-public/issues/articles/content_29e3e01c19e206a3bf_ajo045070320.PDF
```

## Q4. How is a subscapularis tendon tear repaired with a knotless technique?

**Answer:**

## Knotless Arthroscopic Subscapularis Repair — Bottom Line

This technique uses a looped **FiberSnare** (Arthrex) as a traction stitch to facilitate suture passage, then achieves knotless fixation at the lesser tuberosity. It works well for partial to full-thickness tears using standard portals and a **30° arthroscope** in most cases [4].

---

## Step-by-Step Technique

**1. Patient Positioning & Diagnostic Arthroscopy**
Beach-chair position. Standard posterior viewing portal with a **30° arthroscope**. Use the "shuck maneuver" (internal rotation + posterior humeral displacement) to evaluate the subscapularis and lesser tuberosity. In cases with significant retraction, a **70° arthroscope** may be needed [4][5].

**2. Address Biceps Pathology First**
Tenotomy or tenodesis per your preference — do this before starting subscapularis repair [5].

**3. Portal & Cannula Placement**
- **Anterior portal**: approach medial to lateral, just above the subscapularis, aimed at the lesser tuberosity. Minimum **7-mm Twist-In cannula** (Arthrex); the **Gemini cannula (8.25 mm)** (Arthrex) works well for retracting anterior capsule soft tissue [5].
- **Anterolateral portal**: placed in the rotator interval just anterior to the leading edge of the supraspinatus (or through a full-thickness supraspinatus tear if present). A **5- to 6-mm Twist-In cannula** is typically sufficient; a **7-mm cannula** provides least resistance if space allows [5].

**4. Tissue Preparation**
Debride subscapularis with an arthroscopic shaver. Use arthroscopic scissors from the anterolateral portal to release adhesions on anterior, superior, and posterior tendon surfaces if needed. Expose and prepare the lesser tuberosity footprint with the shaver [5].

**5. First Suture Passage — Traction Stitch**
Introduce a **90° SutureLasso** (Arthrex) through the anterior portal; puncture the proximal subscapularis tissue anterior to posterior. Retrieve the lasso wire through the anterolateral portal. Shuttle the non-looped end of the **FiberSnare** (Arthrex) through the proximal tendon. Retrieve and cinch through the anterolateral portal — this becomes your **traction stitch**. Alternatively, a **Scorpion** suture-passing device (Arthrex) from the anterolateral portal can be used [5].

**6. Second Suture Passage — Repair Stitch**
Apply traction on the first **FiberSnare** through the anterolateral portal to improve visualization, aid countertraction, and reduce the tendon. Introduce the **SutureLasso** again through the anterior portal; penet

**Sources:**
```
  [1] (YOUTUBE) Subscapularis SpeedFix™ Technique  ·  00:00:00-00:01:08  ·  score 0.323
      https://www.youtube.com/watch?v=24BqkL1orIk&t=0s
  [2] (PDF) Knotless Arthroscopic Repair of Subscapularis Tendon Tears  ·  PDF, 5 pp  ·  score 0.193
      https://www.mammothortho.com/pdf/subscap%20technique.pdf
  [3] (PDF) Knotless Arthroscopic Repair of Subscapularis Tendon Tears  ·  PDF, 5 pp  ·  score 0.161
      https://www.mammothortho.com/pdf/subscap%20technique.pdf
  [4] (PDF) Knotless Arthroscopic Repair of Subscapularis Tendon Tears  ·  PDF, 5 pp  ·  score 0.153
      https://www.mammothortho.com/pdf/subscap%20technique.pdf
  [5] (PDF) Knotless Arthroscopic Repair of Subscapularis Tendon Tears  ·  PDF, 5 pp  ·  score 0.130
      https://www.mammothortho.com/pdf/subscap%20technique.pdf
```

## Q5. What biological or augmentation options can improve rotator cuff healing?

**Answer:**

## Biological & Augmentation Options for Rotator Cuff Healing

**Bottom line:** Repairs fail because you're recreating a scar tissue interface, not the native fibrocartilaginous tendon-to-bone zone. Biological augmentation targets that gap — but product selection and technique execution matter more than the concept itself.

---

### Why Augmentation Is Needed

Retear rates run **20–80% in the literature**, now more realistically **20–40%**, and intact repairs yield superior long-term outcomes — particularly for strength. The fundamental problem is that tendon heals to bone via scar tissue with diminished biological and biomechanical properties, not through the normal tendon → unmineralized fibrocartilage → mineralized fibrocartilage → bone interface [2].

---

### Option 1: PRP (Platelet-Rich Plasma)

Use it cautiously. Multiple meta-analyses and systematic reviews show **no improvement in clinical outcome or retear rates using PRP alone** [2]. The concept isn't wrong — the preparation variability is. Different manufacturers, inconsistent concentrations, unknown release kinetics — you don't actually know what you're injecting [2]. PRP can still serve as a carrier or adjunct, but don't build your augmentation strategy around it.

---

### Option 2: Demineralized Bone Matrix (DBM)

More defensible biology. The demineralization process releases osteoinductive growth factors and cytokines from the bone collagen matrix in **physiological amounts and concentrations, released over time** [2]. Key distinction: the DBM product **must be osteoinductive, not merely osteoconductive** — and preparation methodology can reduce or eliminate that activity [2].

The product currently in use is **Flex Draft** (distributed by Arthrex; provided by Palatinate Health), which offers:
- Multiple sizes suited to rotator cuff repair
- Naturally absorbs PRP or BMC
- 100% demineralized, sterile, room-temperature storage
- **Every lot tested for osteoinductivity** via intramuscular assay [2]

**Technique with DBM:**
1. Standard footprint preparation and tendon mobilization
2. Place medial row anchors, complete medial row repair
3. Perform **footprint drilling / nanofracture / multiple channeling** — go **deeper than standard microfracture** to access stem cells and cytokines below the subchondral bone level [1][2]
4. Place **Flex Draft** underneath the repaired tendon
5. Close in **suture bridge fashion**, locking the graft at the footprint [2]

**Clinical data:** Two Level III studies show **significant reduction in retear rate** with footprint drilling/multiple channeling. One Level I study showed no difference; one study (Milano) showed **improved healing in large tears** [1]. The variability likely relates to chronicity — stem cell populations at the repair site diminish the longer the defect has been present [1].

---

### Option 3: Acellular Dermal

**Sources:**
```
  [1] (YOUTUBE) Biological Treatment Options in Rotator Cuff Repair  ·  00:08:37-00:10:53  ·  score 0.146
      https://www.youtube.com/watch?v=l0A5F53MMa4&t=517s
  [2] (YOUTUBE) Biological Treatment Options in Rotator Cuff Repair  ·  00:00:05-00:05:32  ·  score 0.115
      https://www.youtube.com/watch?v=l0A5F53MMa4&t=5s
  [3] (PDF) Rotator Cuff Repair with ArthroFLEX Augmentation  ·  PDF, 3 pp  ·  score 0.105
      https://lifenethealth.org/sites/default/files/files/68-20-248%20ArthroFlex%20RCR.pdf
  [4] (PDF) SCR With Superimposition of Acellular Dermal Allograft  ·  PDF, 5 pp  ·  score 0.088
      https://www.jorgechahlamd.com/wp-content/uploads/2019/12/SCR.pdf
  [5] (PDF) SCR With Superimposition of Acellular Dermal Allograft  ·  PDF, 5 pp  ·  score 0.088
      https://www.jorgechahlamd.com/wp-content/uploads/2019/12/SCR.pdf
```
