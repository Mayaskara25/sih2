# SatQuery AI — Build Plan (v2)

**SIH 2026 · Problem Statement 26167 (ISRO) · Single source of truth for every agent and session working in this repo.**

> **If you are an agent picking up work:** read §0–§6 in full, then read only your own work order in §7. Do not read `old_files/readme.md` — see §2.4. Do not start until you have confirmed your work order's *Depends on* row is satisfied.

Revision: v2 (2026-08-29). v1 archived at `old_files/PLAN.v1.md`.

---

## 0. Mission

Take a natural-language query plus one or two remote-sensing images (optical/multispectral or SAR, single-date or bi-temporal) and return an evidence-grounded answer: classify what the query asks, run the correct specialist model(s), return the result plus an auditable record of what ran and why.

Final grading uses **real ISRO Cartosat-2S optical + RISAT SAR pairs the system has never seen**. Everything here has to survive that domain shift, not just look good on our own demo images.

**Locked project parameters** (answered 2026-08-29): deadline 2–6 weeks → plan is laid out on a **4-week schedule with a 2-week fallback line**; execution is **one human + agents**, so work orders are written to be run one-at-a-time or fanned out, with strict file ownership either way; **free-tier GPU only** (Colab/Kaggle T4/P100), which is what forces the adaptation strategy in §3.

---

## 1. Hard Requirements (violating any of these risks disqualification)

| # | Requirement | Satisfied by |
|---|---|---|
| R1 | At least one visual/VL component shows **measurable fine-tuning or adaptation** on BigEarthNet.txt or other open RS data. A stock pretrained VLM called zero-shot does **not** count — the PS says so explicitly. | W2 (Track A mandatory, Track B target) |
| R2 | System ingests **two genuinely distinct modalities** (optical + SAR) for the cross-modal task — not one image processed twice and relabeled. | W0 IO layer + W5 |
| R3 | System ingests **two dates of the same modality** for bi-temporal, on a code path distinct from cross-modal. | W4 |
| R4 | Task routing is driven by **query text**, not just by how many files were uploaded. | W6 |
| R5 | Every run writes an **auditable execution trace to disk**: task selected, model/tool names, key parameters, outputs. | W6 |
| R6 | Delivery is an **interactive GUI/web app**, not a terminal script. | W7 |
| R7 | Grounding (bounding boxes) is **reachable at runtime** from a grounding-style query — not dead code. | W3 + W6 |
| R8 | Single-image VQA is mandatory **plus** one more single-image task (we do both captioning and grounding). | W3 |

Every work order's acceptance test in §7 maps back to one or more of these. §9 is the pre-submission audit that checks all eight.

---

## 2. Verified Ground Truth (checked 2026-08-29 — do not re-derive, do not contradict)

### 2.1 Hardware
- **Local:** GTX 1650, **4 GB VRAM** (verified via `nvidia-smi`). Note: `old_files/readme.md` claims 6 GB — it is wrong. Plan for 4 GB.
- 4 GB is enough for 4-bit inference of **one** small model at a time. It is not enough for two resident models, and not enough to train anything.
- **Cloud:** Colab / Kaggle free tier (T4 16 GB / P100 16 GB), session time limits. All training happens here. Every training run must fit one session and checkpoint to Drive / a Kaggle Dataset so a disconnect costs minutes, not hours.
- System Python is **3.14.7** — too new for the torch/transformers stack. W0 pins **Python 3.11** in a `uv` venv. No agent picks its own interpreter.

### 2.2 BigEarthNet — the acquisition problem (**SOLVED — see §2.2b**)

`BigEarthNet.txt` is **text only**. Confirmed layout:

| Artifact | Where | Size |
|---|---|---|
| `BigEarthNet.txt` annotations | `huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` (parquet, 9,553,962 rows) | 467 MB |
| BigEarthNet v2.0 **metadata.parquet** (patch IDs, s1↔s2 name mapping, labels, country/season) | `zenodo.org/records/10891137` | 3.6 MB |
| BigEarthNet v2.0 **S1 imagery** `BigEarthNet-S1.tar.zst` | same Zenodo record | **54.4 GB** |
| BigEarthNet v2.0 **S2 imagery** `BigEarthNet-S2.tar.zst` | same Zenodo record | **63.3 GB** |
| `Reference_Maps.tar.zst` | same Zenodo record | 282 MB |

The image archives are **monolithic `.tar.zst`, not sharded**. There is no per-patch HTTP access and no HuggingFace image mirror. v1 of this plan collapsed all of this into one checkbox ("BigEarthNet.txt slice downloaded + parsed"); that checkbox is a trap and W1 exists to defuse it.

**Consequence:** you cannot "download a slice." You can only *stream* the archive and keep what you want.

### 2.2b SOLVED — measured 2026-08-29, supersedes the three-tier guesswork

The archives were probed directly rather than reasoned about. Findings, all empirical:

**Internal layout** (confirmed by streaming the first 8 MB and listing tar members):
```
BigEarthNet-S2/<acquisition_folder>/<patch_name>/<patch_name>_B01.tif …_B12.tif   (12 bands)
BigEarthNet-S1/<acquisition_folder>/<patch_name>/<patch_name>_VV.tif, _VH.tif     (2 bands)
```
One S2 patch ≈ **165 KB** on disk (4×29 KB @10 m, 6×7.5 KB @20 m, 2×1.2 KB @60 m); one S1 patch ≈ **116 KB**.

**`metadata.parquet` (3.5 MB) is the join key.** Columns: `patch_id` (= the S2 patch name), `s1_name`, `labels`, **`split`** (official train/validation/test — 237,871 / 122,342 / 119,825 over 480,038 rows), `country`, snow/cloud flags. Use the official split; do not invent one.

**Range requests are NOT supported** — Zenodo returns `200` with full content-length for a `Range` header, so **there is no resume**. A stream that dies restarts from zero. This is why the bounded-prefix strategy below matters: it needs minutes, not hours, so a failed stream costs little.

**The pairing trap is real and confirmed.** S1 and S2 order their archives by *different* acquisition folders — S2 starts at `S2A_…20170613T101031_…T33UUP`, S1 at `S1A_…20170613T165043`, and the S1 partner of the first S2 patch lives in a *different* S1 folder (`S1B_…20170612T165809`). **Naively early-stopping both archives yields unpaired patches.** Both archives are ordered alphabetically by acquisition folder, so paired yield must be computed, not assumed:

| Read first *k* folders of **each** archive | Paired patches | Stream cost | Countries | Land-cover classes |
|---|---|---|---|---|
| 1 | 1,564 | ~0.6 GB | 1 | — |
| 5 | 7,010 | ~5.1 GB | 2 (IE, AT) | 17 / 19 |
| **10** | **13,630** | **~8.6 GB** | **4 (IE, PT, FI, AT)** | **19 / 19** ✅ |
| 20 | 19,646 | ~15.9 GB | — | 19 / 19 |

**DECISION: stream the first 10 acquisition folders of each archive.** 13,630 paired S1+S2 patches for ~8.6 GB of streaming and **~3.8 GB stored**. k=10 is the smallest prefix containing **all 19 land-cover classes** — k=5 misses two, which would leave a contrastive encoder blind to them. Subsample to ~5,000 pairs after extraction if disk gets tight; extract once, keep forever.

**Known limitation, must be disclosed in the writeup (§5.9):** the slice is 4 European countries, not a global sample. It is a prefix of the archive, not a stratified draw, because the archive format makes a stratified draw cost 118 GB. This is a defensible engineering trade-off — say so plainly rather than implying the sample is representative.

**Net effect: BigEarthNet is no longer the project's #1 schedule risk.** It is a ~10-minute stream, and it can run locally; Colab is not required for acquisition (still required for training).

**Note the PS wording:** "fine-tuned or otherwise adapted using BigEarthNet.txt **or any open source training data**," so substituting an open corpus is permitted if ever needed. Given §2.2b it should not be needed for BigEarthNet — but it is the sanctioned move for RSVQA-HR, whose test images cannot be separated from its 13.5 GB train archive.

### 2.2a Disk budget — **33 GB free on `/home` (re-measured 2026-08-29 after the user freed space).**

The 118 GB of BigEarthNet archives **never touch the local disk**, and neither does the extracted slice. Split every dataset into one of three tiers and respect the tier:

| Tier | What | Where it lives | Local cost |
|---|---|---|---|
| **Never local** | BEN `S1.tar.zst` (54.4 GB), `S2.tar.zst` (63.3 GB) | streamed on Colab, discarded | **0** |
| **Never local** | Extracted BEN 5k-pair slice (~2–3 GB), VRSBench **train** split, RSVQA **train** split, all training checkpoints | Google Drive / Kaggle Dataset | **0** |
| **Local** | `.venv` (torch+cu130 dominates) — **measured 5.6 GB after `uv sync`** | `.venv/` | 5.6 GB |
| **Local** | HF model cache (Qwen2-VL-2B ~4.4 GB fp16, Grounding DINO tiny ~0.7 GB, CLIP base ~0.6 GB, `benclip` adapter ~50 MB) | `~/.cache/huggingface` | ~6 GB |
| **Local** | VRSBench **test** split only, RSVQA LR (~0.2 GB), SECOND/CDVQA (~1–2 GB) | `data/` | ~3–4 GB |
| **Local** | Bhoonidhi Cartosat + RISAT samples (5–10 scenes) | `data/bhoonidhi/` | ~3 GB |
| **Local** | `runs/` traces, masks, overlays | `runs/` | ~1 GB |

**Measured 2026-08-29 after the user freed space: 33 GB free, `.venv` = 5.6 GB.** Remaining local demand is ~13–14 GB (models + eval data + Bhoonidhi + runs), leaving roughly **19 GB of slack**. Comfortable — but the tiering rules below still bind, because the thing they prevent (pulling a train split or a BEN archive locally) is measured in tens of GB and would eat that slack in one command. The `verify.py` disk floor stays at 5 GB.

Also verified on this environment (W0): Python 3.11.15, torch 2.13.0+cu130, CUDA available, GTX 1650 reporting **3.9 GB usable VRAM**. Note `transformers` resolved to **5.16.x** — a major version, so the Grounding DINO `box_threshold` → `threshold` rename and the Qwen2-VL loading API are the 5.x forms. W3 must not copy the 4.x-era calls from `old_files/models_registry.py` verbatim.

Rules this implies, binding on W1 and W2:
- **Download eval/test splits locally; never download a train split locally.** Track A and Track B both train in the cloud, so training data has no reason to be on this machine.
- The BEN slice stays in Drive/Kaggle permanently. If W2 needs to sanity-check `benclip` locally, pull **a handful of patches**, not the slice.
- Set `HF_HOME` explicitly and check its size before each new model download. The HF cache silently keeps old revisions; `huggingface-cli delete-cache` when it grows.
- If free space drops below **5 GB**, stop and escalate rather than deleting things ad hoc.
- Nothing in `data/` is ever committed (§5.7) — the manifests in `data/manifests/` are what make the slice reproducible.

### 2.3 The hidden evaluation set
- **Cartosat-2S**: panchromatic ~0.65 m (**1 band**) and multispectral ~2 m (**4 bands**). Not 3-band RGB.
- **RISAT SAR**: single-band intensity, possibly dual-pol (**1–2 bands**), high dynamic range, speckled.
- Therefore `load_image`'s hardcoded `src.read([1,2,3])` in both old files **hard-crashes or silently corrupts on the actual grading data**. Handling 1-band, 2-band, 4-band and 12-band rasters correctly is the cheapest available point on the cross-modal rubric row, and it is a W0 deliverable, not an afterthought.

### 2.4 `old_files/` is reference, not spec — read this before you look at it
`old_files/*.py` may be read for reference. **`old_files/readme.md` must not be treated as a requirements document.** It describes features that do not exist in the code:

| readme claim | reality in code |
|---|---|
| "Grad-CAM style saliency / neural attention maps" | `cv2.GaussianBlur` over the change mask + a colormap. No model gradients involved. |
| "Adaptive Lee Filtering for SAR speckle" | `scipy.ndimage.uniform_filter` is imported and **never called**. |
| "Simulates RISAT-1A backscatter" | `cv2.threshold` on the **optical** grayscale. No radar data anywhere. |
| "SAR-guided cloud-free inpainting" | Does not exist. |
| "100% top-level alignment" | Unsupported; the registration code has the bugs in §2.5. |
| "6GB VRAM" | Machine has 4 GB. |

**Rule: source requirements only from this PLAN.md.** If an agent implements something because the old readme claimed it, that is a plan violation.

### 2.5 Confirmed bugs in `old_files/` — fix or discard, never carry forward
If any work order reuses this code rather than rewriting:

- `BT_CM.register_images`: the affine transform is fit on the **unfiltered** `matches` list even though a filtered `good` list is computed for the confidence score. Fit on `good`.
- No null-check after `cv2.estimateAffinePartial2D` — returns `None` on low-feature scenes (water, farmland, uniform terrain) and the next line crashes.
- No guard for `orb.detectAndCompute` returning empty descriptors on a near-blank scene → `bf.match` crashes.
- `load_image` duplicated in **both** `BT_CM.py` and `models_registry.py`, already diverging. §5 forbids this.
- `models_registry` caches Qwen2-VL **and** Grounding DINO simultaneously in module globals → guaranteed OOM at 4 GB. §4.3 forbids this.
- `run_grounding` has a clean contract and is **never imported or called** by `final_test.py`. `get_execution_metadata` is imported and **never called** — so the audit JSON contains no task/model/parameter record at all. This is exactly the drift §5.4 exists to prevent.
- Confidence scores are uncalibrated guesses (`run_vqa`'s word-count heuristic; `BT_CM`'s 0.4/0.6-weighted "trust"). Acceptable as v0 placeholders **only if labelled as heuristic in the trace**; W8 validates or replaces them.
- CLI + matplotlib popup does not satisfy R6.
- Duplicated line in `run_grounding`: `score = ...` appears twice.

---

## 3. Locked Technical Decisions

These are decided. Do not re-open them inside a work order; if you believe one is wrong, stop and raise it (§5.5).

### 3.1 Adaptation strategy — two tracks, in priority order

**Track A (MANDATORY — satisfies R1 on its own): `benclip`, an RS-adapted multi-sensor image–text encoder.**

- Base: `openai/clip-vit-base-patch32` (or `google/siglip-base-patch16-224` if the agent finds it trains more stably — either is acceptable, record which).
- Modify the patch-embedding stem to accept **12 Sentinel-2 bands + 2 Sentinel-1 bands**, initialising new input channels from the mean of the pretrained RGB weights. Train the new stem fully; LoRA the rest of the vision tower; freeze the text tower.
- Contrastive training against `BigEarthNet.txt` captions on the W1 data slice.
- **Before/after metric:** zero-shot image→text retrieval R@1/R@5 on a held-out BEN split, plus linear-probe mAP on BEN 19-class multi-label. This maps directly onto the rubric's *"Feature Representation Loss / Domain Fit"* row.

Why this is primary and not VLM-QLoRA: it fits a free T4 comfortably, gives a clean defensible before/after number, and — because it ingests S1 — produces an encoder that **genuinely understands SAR**, which is then reused by W4 (change) and W5 (fusion). One training run pays for three rubric rows.

**Track B (TARGET, not mandatory): QLoRA on `Qwen/Qwen2-VL-2B-Instruct`** over VRSBench + RSVQA instruction data, to lift the captioning/VQA rows. 2B in 4-bit with small-rank LoRA is feasible in one T4 session on a few thousand samples. **Start only after Track A has a saved checkpoint.** If the calendar slips, Track B is the thing that gets cut.

**BIFOLD ships pretrained BEN v2.0 encoders on HuggingFace** (`BIFOLD-BigEarthNetv2-0/resnet50-s1-*`, `-s2-*`, `-all-*`). Use these as (a) a sanity baseline for Track A, and (b) an emergency substitute for the S1 tower if W1's data acquisition degrades. They are RS-pretrained, so leaning on them is defensible — but they are **not by themselves sufficient for R1**, because R1 requires *we* adapted something. Track A must still run.

### 3.2 Model registry (locked; env-var overridable, defaults are these)

| Role | Model | Precision | Resident where |
|---|---|---|---|
| VQA / captioning | `Qwen/Qwen2-VL-2B-Instruct` | 4-bit NF4 | local, one at a time |
| Grounding | `IDEA-Research/grounding-dino-tiny` | fp16 | local, one at a time |
| RS encoder | `benclip` (our Track A checkpoint) | fp16 | local, small enough to stay resident |
| Change backbone | classical CV + `benclip` labels | — | local |

**GeoChat is dropped as a local baseline.** It is LLaVA-1.5-7B class; 4-bit weights alone will not fit 4 GB with a display attached. It may be referenced in the writeup as related work; it is not in the runtime path.

### 3.3 How each rubric row gets earned

| Rubric row | Our path |
|---|---|
| Single-image captioning & grounding (VRSBench, BLEU/ROUGE/CIDEr/IoU) | W3: Qwen2-VL-2B (+ Track B LoRA if it lands) for caption; Grounding DINO for boxes; `benclip` land-cover labels injected into the caption prompt as grounded evidence |
| VQA (RSVQA, accuracy) | W3, same VLM, with `benclip` label evidence in the prompt |
| Multi-image change (CDVQA) | W4: registration → change mask → `benclip` land-cover labels **on both dates** → structured change evidence → VLM verbalises. The evidence is model-derived, not prompt-stuffed statistics. |
| Domain adaptation (BigEarthNet.txt, feature representation / domain fit) | W2 Track A before/after retrieval + linear-probe numbers |
| Joint cross-modal (Cartosat + RISAT) | W5: optical → `benclip` S2 tower, SAR → `benclip` S1 tower, per-class agreement/disagreement map + SAR-physics water & built-up detection |
| Agentic orchestration (routing accuracy, auditable summary) | W6: deterministic rules + exemplar-embedding tiebreak, scored against a hand-labelled 100-query routing set; full trace JSON per run |

### 3.4 Routing approach (locked)
Deterministic keyword/pattern rules first; if no rule fires with confidence, fall back to nearest-neighbour over embedded task exemplars. **The trace records which mechanism fired.** No LLM-in-the-loop routing as the primary path — it adds latency and nondeterminism to the one rubric row that rewards being auditable. An LLM tiebreak may be added later as a third tier if the routing test set shows the first two are insufficient.

---

## 4. Frozen Contracts

**These are frozen.** Changing any signature or key requires the §5.5 protocol. `satquery/contracts.py` is the machine-readable copy and must stay identical to this section.

### 4.1 Specialist functions

```python
run_vqa(image_path: str, query: str, *, evidence: dict | None = None) -> {
    "text_response": str,
    "confidence": float,          # [0,1]
    "confidence_basis": str,      # "stub" | "heuristic" | "calibrated" | "model_logprob"
    "evidence": dict,             # what was fed in / derived; {} if none
}

run_caption(image_path: str, *, evidence: dict | None = None) -> {
    "text_response": str,
    "confidence": float,
    "confidence_basis": str,
    "evidence": dict,
}

run_grounding(image_path: str, target_text: str) -> {
    "text_response": str,
    "bounding_boxes": [
        {"label": str, "box_2d": [ymin, xmin, ymax, xmax], "confidence": float}
    ],                            # pixel coords, ymin/xmin/ymax/xmax order — do not change
    "overlay_path": str | None,   # rendered image with boxes drawn
    "confidence": float,
    "confidence_basis": str,
}

run_change(image_path_t0: str, image_path_t1: str, query: str) -> {
    "text_response": str,
    "change_mask_path": str | None,
    "overlay_path": str | None,
    "metrics": dict,              # registration_confidence, changed_area_fraction, per-class deltas
    "confidence": float,
    "confidence_basis": str,
    "evidence": dict,
}

run_fusion(optical_path: str, sar_path: str, query: str) -> {
    "text_response": str,
    "agreement_map_path": str | None,
    "overlay_path": str | None,
    "evidence": dict,             # per-modality labels, agreement/disagreement summary
    "confidence": float,
    "confidence_basis": str,
}
```

`confidence_basis` is new in v2 and mandatory. It exists so an uncalibrated heuristic is *disclosed in the trace* rather than presented as a real probability. Judges probing confidence numbers is a predictable question; this is the honest answer.

### 4.2 Execution trace (written to `runs/<run_id>/trace.json` every run, no exceptions)

```json
{
  "run_id": "str",
  "timestamp": "ISO-8601",
  "query": "str",
  "task_selected": "vqa | caption | grounding | change | fusion",
  "routing": {"mechanism": "rule | exemplar_nn", "matched": "str", "score": 0.0,
              "alternatives_considered": ["str"]},
  "inputs": [{"path": "str", "modality": "optical|msi|sar|unknown", "bands": 0,
              "shape": [0,0], "format": "str", "crs": "str|null", "checks_passed": true}],
  "validation": {"passed": true, "warnings": ["str"], "errors": ["str"]},
  "models_used": [{"role": "str", "name": "str", "revision": "str", "precision": "str",
                   "adapter": "str|null", "device": "cuda|cpu"}],
  "parameters": {},
  "result": {},
  "artifacts": {"mask": "path|null", "overlay": "path|null", "report": "path|null"},
  "timings_ms": {"routing": 0, "validation": 0, "inference": 0, "total": 0}
}
```

### 4.3 Runtime rule — one heavy model resident at a time
`satquery/runtime/modelpool.py` (W0) owns all model loading. Contract: `acquire(role)` loads, `release()` frees and empties the CUDA cache, and acquiring a second heavy role auto-releases the first. **No specialist may cache a model in its own module globals** — that is precisely what OOMs the 4 GB card in the old code. `benclip` is small enough to be exempt and stays resident.

### 4.4 Image IO contract
`satquery/io/raster.py` (W0) is the **only** place a raster is opened. It must handle:
- 1 band (Cartosat panchromatic, RISAT single-pol) — replicate to 3 for RGB-expecting models
- 2 bands (RISAT dual-pol) — plus a synthesised third channel for display
- 3 bands (PNG/JPEG benchmark inputs)
- 4 bands (Cartosat-2S MSI)
- 12–13 bands (Sentinel-2)
- SAR: **dB/log scaling** before any percentile stretch — a linear stretch of a SAR intensity image is unreadable

Returns a `RasterInput` dataclass: `array`, `modality`, `band_count`, `band_names`, `crs`, `transform`, `is_georeferenced`, `source_path`, `display_rgb`, plus `modality_decision` (the `ModalityDecision` from `io/modality.py`, carrying which precedence tier decided and why, so W6 can drop it into the trace without re-deriving it). *(`modality_decision` added 2026-08-29 via §5.5 — additive, no existing field changed.)*

**Modality is explicit, never guessed silently.** Order of precedence: (1) user selection in the UI, (2) filename/metadata heuristic, (3) band-count heuristic — and whichever was used is recorded in the trace. A wrong silent guess on the hidden eval set costs the whole cross-modal row.

### 4.5 `benclip` band-mapping contract — **the W2↔W3↔W5 integration seam**

`benclip` trains on a 14-channel stem (12×S2 + 2×S1), but its callers hand it whatever the user uploaded: W3 passes **3-band VRSBench PNGs**, W5 passes **1-band Cartosat PAN, 4-band Cartosat MSI, and 1–2-band RISAT**. If each caller invents its own padding, W3's VRSBench numbers stop being comparable to W5's Bhoonidhi behaviour and the mismatch surfaces in week 3.

**W2 owns the band-mapping policy, and it lives entirely inside `benclip.py`. Callers pass a `RasterInput` and nothing else.**

```python
predict_labels(raster: RasterInput) -> {
    "labels": [{"label": str, "score": float}],
    "band_mapping": {                # what W2 actually did with the input
        "slots_filled": ["B04", "B03", "B02"],
        "slots_absent": ["B01", "B05", ...],
        "fill_strategy": "per_channel_training_mean",
        "source_modality": "optical|msi|sar|unknown",
    },
}
embed_optical(raster: RasterInput) -> np.ndarray   # same policy applies
embed_sar(raster: RasterInput)     -> np.ndarray
```

Required of W2's policy:
- **RGB (3-band)** → S2 slots B04/B03/B02. **PAN (1-band)** → replicated across B04/B03/B02. **Cartosat MSI (4-band)** → B02/B03/B04/B08 (blue/green/red/NIR). **SAR (1-band)** → VV slot; **SAR (2-band)** → VV/VH.

> ⚠️ **The Cartosat-2S 4-band order above is an UNVERIFIED ASSUMPTION written into this plan, not a sourced fact.** `io/raster.py` deliberately mirrors it so the two modules never disagree about what channel 0..3 means — but that consistency is not verification, and citing §4.5 as the source is circular. It must be checked against a real Bhoonidhi product's metadata XML (`docs/bhoonidhi_registration.md` step 5). If it's wrong, `raster.py`'s display selection and `benclip`'s slot mapping are wrong **together**, silently, on the graded sensor. W1 confirms or corrects it; W2 and W5 must not treat it as settled until then.
>
> Related: a **4-band quad-pol RISAT** scene (HH/HV/VH/VV) has the same band count as Cartosat MSI. Band count alone cannot separate them — only the resolved modality can, which is why the SAR/optical branch is driven by modality and never by band count.
- **Absent slots are filled with the per-channel training mean, not zeros** — zeros are far outside the training distribution and will silently degrade the embedding. Pick this and state it; do not leave it to the caller.
- `band_mapping` is returned on every call and propagates into the trace's `evidence`, so a weak out-of-domain result is explainable rather than mysterious.

Callers (W3, W4, W5) **must not** pad, replicate, or reorder bands themselves. If you find yourself writing band logic outside `benclip.py`, that's a §5.5 contract issue — stop and report it.

**`source_modality` accepts `"unknown"`, and `benclip` must fail soft when it sees it.** §4.4's precedence chain can legitimately fail to decide a modality, and an unrecognised TIFF on the hidden eval set is exactly when that happens. `predict_labels` must not refuse in that case: fall back to mapping by **band count alone** (the §4.5 table above), return the labels, and report `source_modality: "unknown"` so the trace discloses the uncertainty. A degraded, labelled-uncertain answer scores; a hard failure on the graded set scores zero. *(Resolved 2026-08-29 via §5.5 — W0 flagged that §4.4 permits `unknown` while §4.5 originally did not.)*

**All payload numbers must be Python scalars, not numpy types.** `validate_*` rejects `np.float32`/`np.int64`, deliberately — the trace is written as JSON and `json.dump` rejects them too. Every module returning a contract payload (W2–W6) must coerce with `float(...)` / `int(...)` at the boundary. This is the single most likely cause of a late integration failure; do it at the point of return, not at the point of writing.

---

## 5. Rules of Engagement for Agents

### 5.1 File ownership — one owner per path, no exceptions

| Path | Owner | Others |
|---|---|---|
| `PLAN.md` | human | read-only |
| `pyproject.toml`, `.gitignore`, `satquery/contracts.py`, `satquery/io/**`, `satquery/runtime/**` | **W0** | read-only |
| **every `__init__.py`** — W0 pre-declares all module exports pointing at stubs; frozen thereafter | **W0** | read-only |
| `requirements/extra-<ws>.txt` | that workstream | — |
| `scripts/data/**`, `data/manifests/**` | **W1** | read-only |
| `train/**`, `satquery/adapters/**` | **W2** | read-only |
| `satquery/specialists/vqa.py`, `grounding.py` | **W3** | read-only |
| `satquery/specialists/change.py` | **W4** | read-only |
| `satquery/specialists/fusion.py` | **W5** | read-only |
| `satquery/controller/**` | **W6** | read-only |
| `app/**`, `satquery/report.py` | **W7** | read-only |
| `eval/**`, `docs/RESULTS.md` | **W8** | read-only |
| `tests/test_w<N>_*.py` | that workstream | read-only |
| `docs/status/W<N>.md` | that workstream | read-only |
| `old_files/**` | nobody | **read `.py` only; never edit, never import, never trust the readme** |

If your work order requires a change to a file you do not own, **stop and report it** — do not edit it.

### 5.2 Stub-first
W0 ships all five specialist functions as **contract-honouring stubs** that return valid dummy payloads on day one. This is what actually unlocks parallel work: W6 and W7 build against stubs instead of waiting on W2–W5. When a real implementation lands, it replaces the stub body and nothing downstream changes.

### 5.3 No specialist imports another specialist
Shared code goes in `satquery/io/`, `satquery/runtime/`, or `satquery/adapters/` — all W0/W2-owned. The duplicated-and-diverging `load_image` in the old code is the exact failure mode this prevents. Band handling in particular belongs to `benclip.py` alone (§4.5).

### 5.4 Dead code is a defect
Every function you add must have a live caller or a test that calls it, in the same change. `run_grounding` and `get_execution_metadata` sitting unreferenced in the old draft is how R5 and R7 quietly went unmet. §9's audit pass greps for this.

### 5.5 Contract-change protocol
To change anything in §4: edit `satquery/contracts.py` **and** §4 of this file in the same commit, state the change in your status file, and flag it to the human. Never change one without the other, and never change a contract to make your own module easier — that breaks someone else's tests silently.

### 5.6 Dependencies
Do not edit `pyproject.toml`. Declare your extra deps in `requirements/extra-w<N>.txt`. W0's lockfile pulls them all in. Zero merge conflicts by construction.

### 5.7 Git
- The repo currently has **zero commits**. W0's first act is `.gitignore` + initial commit.
- Commit only paths you own.
- Branch per workstream: `w<N>-<short-name>`. Merge to `main` only after your acceptance test passes.
- **Merge order: W0 merges to `main` first.** *(Done — W0 landed directly on `main` as the initial commit, since the repo had no commits to branch from. Its branch pointer was deleted; `main` is unambiguously the trunk. W1+ branch normally.)* Every other branch rebases on `main` before merging. With one owner per path this should be conflict-free by construction — if you hit a conflict, someone edited a file they don't own, so stop and report rather than resolving it.
- `.gitignore` must cover: `data/` (except `data/manifests/`), `runs/`, `checkpoints/`, `*.safetensors`, `*.pt`, `__pycache__/`, `.venv/`, `old_files/__pycache__/`.

### 5.8 Status reporting
Write `docs/status/W<N>.md`: what's done, what's stubbed, what's blocked, any contract pressure, actual measured numbers. **Do not append to PLAN.md's progress log** — that's human-owned and multi-writer appends conflict.

### 5.9 Honesty rule
Do not write a number you did not measure, and do not name a technique you did not implement. If a metric is a placeholder, label it `PLACEHOLDER`. §2.4 exists because the previous iteration's documentation drifted from its code; the grading here is against real held-out data and the drift will be visible.

---

## 6. Repository Layout (W0 creates this exactly)

```
sih2/
├── PLAN.md                     # this file — source of truth
├── pyproject.toml              # uv, python 3.11 pinned          [W0]
├── requirements/extra-w*.txt   # per-workstream deps
├── .gitignore                                                    [W0]
├── satquery/
│   ├── contracts.py            # §4, machine-readable. FROZEN    [W0]
│   ├── io/
│   │   ├── raster.py           # the ONLY raster reader          [W0]
│   │   └── modality.py         # explicit modality tagging       [W0]
│   ├── runtime/
│   │   └── modelpool.py        # single-resident model manager   [W0]
│   ├── adapters/
│   │   └── benclip.py          # loads the Track A checkpoint    [W2]
│   ├── specialists/
│   │   ├── vqa.py              # run_vqa, run_caption            [W3]
│   │   ├── grounding.py        # run_grounding                   [W3]
│   │   ├── change.py           # run_change                      [W4]
│   │   └── fusion.py           # run_fusion                      [W5]
│   ├── controller/
│   │   ├── router.py           # query text -> task              [W6]
│   │   ├── validate.py         # input compatibility checks      [W6]
│   │   └── trace.py            # §4.2 trace writer               [W6]
│   └── report.py               # downloadable PDF/HTML report    [W7]
├── train/                      # Colab/Kaggle notebooks+scripts  [W2]
├── scripts/data/               # dataset fetch/prep              [W1]
├── eval/                       # benchmark harnesses             [W8]
├── app/                        # Streamlit front end             [W7]
├── data/                       # gitignored except manifests/
├── runs/                       # gitignored — execution traces
├── docs/status/W*.md           # per-workstream status
├── tests/test_w*_*.py
└── old_files/                  # READ-ONLY REFERENCE. See §2.4.
```

Every `__init__.py` (omitted above for brevity) is created by **W0** with all module exports pre-declared, and is frozen thereafter (§5.1). No other workstream edits one.

---

## 7. Work Orders

Each is written to be handed to an agent verbatim. **Acceptance test is the definition of done** — not "code written."

---

### W0 — Foundations, Contracts, IO, Stubs · **BLOCKS EVERYTHING** · ~1 day

**Depends on:** nothing. Run this first, alone.
**Owns:** `pyproject.toml`, `.gitignore`, `satquery/contracts.py`, `satquery/io/**`, `satquery/runtime/**`, every `__init__.py`, `tests/test_w0_*.py`. **Also creates the initial specialist stubs (`vqa.py`, `grounding.py`, `change.py`, `fusion.py`) — ownership of those four files transfers to W3/W4/W5 on merge.**
**Read-only:** `old_files/*.py`

**Goal:** every other work order has a stable, correct foundation to build against, and nobody has to guess an interface.

Tasks:
1. `git init` state is clean: write `.gitignore` per §5.7, make the initial commit.
2. `uv` project pinned to **Python 3.11**, with torch/transformers/rasterio/opencv/streamlit. Verify `import torch; torch.cuda.is_available()` is `True` on the 4 GB card.
3. `satquery/contracts.py` — §4.1 and §4.2 as dataclasses or TypedDicts plus a `validate_*` function per contract, importable by everyone.
4. `satquery/io/raster.py` — the §4.4 contract. This is the highest-value file in the repo; take the time. **Explicitly test 1-, 2-, 4-, and 12-band paths and SAR dB scaling**, because the hidden eval set is 1-band Cartosat PAN and 1–2-band RISAT (§2.3).
5. `satquery/io/modality.py` — explicit tagging with the §4.4 precedence order; always reports which mechanism decided.
6. `satquery/runtime/modelpool.py` — §4.3 single-resident manager with `acquire`/`release` and CUDA cache emptying.
7. Stub `vqa.py`, `grounding.py`, `change.py`, `fusion.py` — real signatures, valid dummy payloads, `confidence_basis: "stub"`. Each raises nothing and passes the contract validators.
8. **Every `__init__.py`, with all module exports pre-declared** (pointing at the stubs). These are frozen after W0. This is deliberate: it means W3/W4/W5 replacing a stub body changes nothing at package level, so no two agents ever edit the same `__init__.py`.
9. `docs/status/W0.md`.

**Acceptance test:** `pytest tests/test_w0_*` passes, covering — contract validators accept the stubs' output and reject a malformed payload; `raster.py` correctly opens a synthetic 1-band, 2-band, 3-band, 4-band and 12-band GeoTIFF plus a PNG; SAR dB scaling produces a visually reasonable dynamic range on a synthetic speckled image; `modelpool` acquiring role B after role A leaves exactly one model resident (assert on `torch.cuda.memory_allocated`).

---

### W1 — Data Acquisition · **parallel with everything, start immediately** · ~2–4 days wall-clock

**Depends on:** nothing (no code dependency on W0).
**Owns:** `scripts/data/**`, `data/manifests/**`
**Read-only:** everything else

**Goal:** a durable, re-downloadable-once local/Drive corpus. Every dataset gets a manifest (file list + checksums + row counts) committed to git so the slice is reproducible even though the data isn't in the repo.

**Measured facts (2026-08-29). Use these; do not re-derive, and do not trust any conflicting number in a recon doc.**

| Item | Exact size | Direct URL | Local? |
|---|---|---|---|
| BEN `metadata.parquet` (480,038 rows; join + official split) | 3.5 MB | `zenodo.org/records/10891137/files/metadata.parquet?download=1` | yes |
| `BigEarthNet.txt.parquet` (annotations; types: binary/mcq/captioning/bounding_box) | **466,819,745 B** | `huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/resolve/main/BigEarthNet.txt.parquet` — plain HTTP 200, **no `git clone` needed** | yes |
| BEN S2 archive | 63,251,710,377 B | `zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1` | **stream only** |
| BEN S1 archive | 54,439,153,171 B | same pattern, `BigEarthNet-S1.tar.zst` | **stream only** |
| RSVQA-LR (images + QA JSONs) | ~150 MB | Zenodo record 6344334 | yes — test split cleanly separable |
| RSVQA-HR | 14.4 GB (13.5 GB monolithic image tar) | Zenodo record 6344366 | **NO** — test images inseparable from train; violates §2.2a. Use LR only. |
| VRSBench | ~12.5 GB total; eval needs the ~4 GB validation image set | HuggingFace | yes, but it is the single largest local item — fetch nothing beyond the eval images |

**The S1↔S2 join column is `s1_name`, and `patch_id` is the S2 patch name.** (`docs/status/W1-recon-cdvqa-ben.md` says `patch_id_s1` — that is wrong; the column does not exist. Verified against the real parquet.)

**Still NOT FOUND — resolve before writing those fetchers:** SECOND dataset total size; a direct HTTP link for CDVQA annotations and for SECOND (it is distributed via Google Drive / Baidu, so a plain `curl` will not work and the fetcher needs `gdown` or a manual step).

**BigEarthNet acquisition — see §2.2b. The three-tier guesswork is superseded:** stream the first 10 acquisition folders of each archive for 13,630 paired patches (~8.6 GB streamed, ~3.8 GB stored, all 19 classes). No abort trigger needed; this is a ~10-minute job that runs locally.

<details><summary>Superseded: the original three-tier strategy and its abort trigger</summary>

- **Tier 1 (attempt first, timebox 1 Colab session per modality):** stream-filter. Pull `metadata.parquet` (3.6 MB), choose a **stratified target set of ~5,000 patch IDs** (spread across countries/seasons/climate zones so the slice isn't geographically degenerate), then stream each `.tar.zst` from Zenodo through `zstd -dc | tar -x` extracting only members in the target set, writing to Drive/Kaggle. Disk cost is a few GB; the cost is bandwidth and wall-clock. **Do not naively early-stop after N members** — tar order is alphabetical and S1 and S2 patch names differ, so early-stopping the two archives independently gives you *unpaired* patches, which destroys the whole point.
- **Tier 2 (if Tier 1 exceeds ~2 sessions per archive):** S2-only via Tier 1 on the smaller archive; take the S1/SAR side from a substitute pairing corpus (SEN12MS or SEN1-2, both far smaller and both co-registered S1/S2).
- **Tier 3 (abort trigger — if by end of week 1 no S1+S2 pairs are on disk):** switch to RSVQAxBEN (BigEarthNet-derived, much smaller) + a substitute SAR corpus, and record the substitution in `docs/status/W1.md` and in the final writeup. **The PS explicitly permits "any open source training data" — this is a legal move, not a failure.** Escalate to the human when the trigger fires; do not silently downgrade.

</details>

Other datasets (all straightforward, do these first since they're quick wins):
- **`BigEarthNet.txt` parquet** — `huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`. **Not `txt.bigearth.net`** — v1 of this plan had the wrong host. 467 MB, download in full, filter locally to the W1 patch-ID slice.
- **VRSBench** — `github.com/lx709/VRSBench` / `vrsbench.github.io`. 29,614 images total, so **download the prescribed test split locally and nothing else** (§2.2a); if the release is a single archive, extract only the test images and delete the rest before it lands on disk. Use their split verbatim — it's graded. The train split, if Track B needs it, goes straight to Drive.
- **RSVQA LR** (start with LR, it's smaller) — `rsvqa.sylvainlobry.com`, `github.com/syvlo/RSVQA`. Test split local; train split Drive-only.
- **CDVQA + SECOND** — `github.com/YZHJessica/CDVQA`. SECOND is ~4.6k bi-temporal 512×512 pairs; small and fully downloadable.
- **Bhoonidhi** — `bhoonidhi.nrsc.gov.in`, free registration. Pull **5–10 real Cartosat + RISAT scenes**. This is the single most valuable item in W1: it is the only thing that tells you before judging day whether the Sentinel→Cartosat/RISAT domain gap breaks you. Registration can take a day, **so start it on day 1** even though the data is needed in week 3.

**Acceptance test:** for each dataset, `data/manifests/<name>.json` exists in git with file counts, total bytes, checksums and split sizes; a `scripts/data/verify.py` re-validates a local copy against every manifest and exits 0; at least one matched S1+S2 BEN pair loads through W0's `raster.py` and displays; at least one real Bhoonidhi Cartosat scene and one RISAT scene load through `raster.py` **with correct band counts** (this is the §2.3 check that matters most); and `df -h /home` shows **≥5 GB still free** with every local-tier dataset in place — if not, you downloaded something that belongs in the Drive tier (§2.2a).

---

### W2 — Domain Adaptation (`benclip` + optional VLM LoRA) · **highest priority after W0/W1** · ~1.5 weeks

**Depends on:** W0 contracts, W1 BEN slice (Track A) / VRSBench+RSVQA (Track B).
**Owns:** `train/**`, `satquery/adapters/benclip.py`, `requirements/extra-w2.txt`
**Read-only:** everything else

**Goal:** satisfy R1 with a **measured** before/after number, and produce the multi-sensor encoder that W3/W4/W5 all consume.

**Track A — `benclip` (mandatory):** as specified in §3.1. Deliver:
1. A **baseline** measurement first: stock CLIP/SigLIP zero-shot retrieval R@1/R@5 and linear-probe mAP on the held-out BEN split. Record this before training anything. *This "before" number is the deliverable, not a formality* — without it R1 has no evidence.
2. Modified 14-channel stem (12×S2 + 2×S1), new channels initialised from the mean of pretrained RGB weights.
3. LoRA on the vision tower, text tower frozen, contrastive loss against BigEarthNet.txt captions.
4. Checkpoint to Drive/Kaggle Dataset **every N steps** — free-tier sessions die without warning.
5. `satquery/adapters/benclip.py`: `load_benclip()`, `embed_optical(raster)`, `embed_sar(raster)`, `predict_labels(raster)` — **per the §4.5 contract, which you own**. These are what W3/W4/W5 actually call; design them for those callers, not for the training loop. In particular §4.5's band-mapping policy is your responsibility: callers hand you 1-, 2-, 3-, 4- and 12-band rasters and must never do band logic themselves.
6. After/before table in `docs/status/W2.md`.

**Track B — Qwen2-VL-2B QLoRA (target, start only once Track A has a checkpoint):** 4-bit base, small rank, VRSBench + RSVQA instruction-formatted subsample, one T4 session. Adapter must be loadable by W3's `vqa.py` behind a flag so the untuned path still works if the adapter is bad.

**Acceptance test:** `docs/status/W2.md` contains a before/after table with **real measured numbers** for zero-shot retrieval R@1/R@5 and linear-probe mAP, with after > before; the checkpoint loads on the local 4 GB card via `load_benclip()` in under 30 s; `predict_labels()` returns sane land-cover labels for a known BEN patch; **`predict_labels()` returns a valid §4.5 payload (labels + populated `band_mapping`) for 1-band, 2-band, 3-band, 4-band and 14-band inputs** — this is the integration seam, test it explicitly; and it runs without crashing on a real Bhoonidhi Cartosat scene (it may be *wrong* there — that's the domain gap, and knowing its size is the point).

---

### W3 — Single-Image Specialists: VQA, Captioning, Grounding · ~4 days

**Depends on:** W0. Uses W2's `benclip` if available, must degrade gracefully if not.
**Owns:** `satquery/specialists/vqa.py`, `satquery/specialists/grounding.py`, `requirements/extra-w3.txt`
**Read-only:** everything else

**Goal:** R7 and R8. VQA, captioning and grounding all genuinely callable and returning §4.1-shaped payloads.

Tasks:
1. `run_vqa` / `run_caption` on Qwen2-VL-2B via `modelpool` — **never a module-global cache** (§4.3).
2. Inject `benclip.predict_labels()` output into the prompt as *evidence*, and put the same labels in the returned `evidence` dict. This is what makes the answer evidence-grounded rather than a VLM guess, and it's what carries the adaptation into the VQA rows.
3. Load W2's Track B LoRA adapter if present; fall back to base weights with a logged warning if not.
4. `run_grounding` on Grounding DINO. Fix the duplicated `score` line and the label-coercion mess from the old code. **Render an overlay image and return `overlay_path`** — R7 requires grounding be visibly reachable, and the app needs a picture.
5. Replace the word-count confidence heuristic, or keep it and set `confidence_basis: "heuristic"` honestly. Prefer mean token log-prob where the VLM exposes it — that's `"model_logprob"` and it's defensible.

**Acceptance test:** all three functions run end-to-end on a real VRSBench sample and pass the W0 contract validators; grounding on a VRSBench image with a known referring expression produces ≥1 box with IoU > 0.3 against the reference; peak VRAM stays under 3.5 GB throughout (assert it); the `evidence` dict is non-empty when `benclip` is available.

---

### W4 — Bi-Temporal Change Analysis · ~4 days

**Depends on:** W0, W1 (CDVQA/SECOND). Uses W2's `benclip`.
**Owns:** `satquery/specialists/change.py`, `requirements/extra-w4.txt`
**Read-only:** everything else

**Goal:** R3. A real change path validated against CDVQA — not VQA with statistics stuffed into a prompt.

Tasks:
1. Registration. You may adapt `old_files/BT_CM.py` **only after fixing every §2.5 bug**: fit the affine on the *filtered* matches; null-check `estimateAffinePartial2D` and fall back to no-warp with a low-confidence flag rather than crashing; guard empty ORB descriptors. Prefer skipping registration entirely when both inputs are georeferenced and already co-registered — check the CRS/transform from `raster.py` first.
2. Radiometric normalisation (histogram matching) before differencing.
3. Change mask: differencing + morphological cleanup, saved to disk.
4. **The part that makes this real:** run `benclip.predict_labels()` on **both dates**, produce a structured per-class delta (e.g. built-up +8%, water −3%), and pass *that* as evidence to the VLM. The change answer is then grounded in an RS-adapted model's read of both scenes, not in pixel arithmetic alone.
5. Overlay render for the app.
6. Featureless-scene handling: uniform water/farmland is exactly where ORB dies. Test it deliberately.

**Acceptance test:** `run_change` scores on ≥100 CDVQA test questions with accuracy recorded in `docs/status/W4.md`; runs without crashing on a deliberately featureless synthetic pair (uniform + noise) and reports low registration confidence rather than throwing; change mask precision reported against ≥20 SECOND reference masks.

---

### W5 — Cross-Modal Optical–SAR Analysis · ~4 days · **highest-risk rubric row**

**Depends on:** W0, W1 (Bhoonidhi samples), W2 (`benclip` S1 tower).
**Owns:** `satquery/specialists/fusion.py`, `requirements/extra-w5.txt`
**Read-only:** everything else

**Goal:** R2. Genuine two-modality ingestion. **The fabricated "SAR proxy" from the old code is deleted, not ported** — it thresholds the optical image and will fail instantly against real RISAT input, which is exactly what the hidden eval set contains.

Tasks:
1. Require two inputs with **distinct, explicitly tagged** modalities. Refuse (with a clear message in `validation.errors`) if both are tagged optical — do not silently proceed.
2. Real SAR preprocessing: dB/log scaling (mandatory), speckle filtering (Lee or Refined Lee) if time allows. **If you don't implement Lee, don't mention Lee anywhere** (§2.4, §5.9).
3. Embed optical through `benclip`'s S2 path and SAR through its S1 path. Produce per-class predictions from each and a structured **agreement / disagreement** summary — that's the "joint alignment & feature fusion" the rubric names.
4. SAR-physics evidence that stands alone if `benclip` underperforms out of domain: low backscatter → smooth water; high backscatter / double-bounce → built-up. Simple, physically defensible, and it degrades gracefully on Cartosat/RISAT.
5. Render an agreement map for the app.
6. **Sanity-check against the real Bhoonidhi Cartosat + RISAT pair from W1.** This is the whole point of pulling that data. If it breaks there, it breaks in judging.

**Acceptance test:** `run_fusion` completes on a real Bhoonidhi Cartosat(1-band PAN or 4-band MSI) + RISAT(1–2-band) pair and returns a non-empty `evidence` dict naming both modalities; the SAR path is verifiably reading the SAR file (assert the result changes when the SAR input is swapped for a different scene — this is the direct test that we're not faking the second modality); rejects a two-optical-input call with a clear error.

---

### W6 — Agentic Controller & Execution Trace · ~4 days, can start early against stubs

**Depends on:** W0 stubs only. Does **not** need W2–W5 finished.
**Owns:** `satquery/controller/**`, `requirements/extra-w6.txt`, `eval/routing_testset.json` (hand-written, W6 authors it; W8 scores it)
**Read-only:** everything else

**Goal:** R4 and R5.

Tasks:
1. `router.py` — §3.4. Query text → `{vqa, caption, grounding, change, fusion}`. Deterministic rules first, exemplar-NN tiebreak, record which fired and what the alternatives were.
2. **Query text must be able to override input count.** Two images + "describe the first image" routes to `caption` on image 1, not to `change`. This is exactly the R4 trap and the rubric's "task routing accuracy" row will probe it.
3. `validate.py` — before dispatch: file exists, format supported, band count sane for the claimed modality, image count matches the task, both images same size/CRS for change, distinct modalities for fusion. Warnings vs. hard errors, both into the trace.
4. `trace.py` — write §4.2 JSON to `runs/<run_id>/trace.json` **every run, including failed ones**. A crashed run with a trace explaining why is worth more than a silent crash.
5. Wire **all five** specialists — including `run_grounding`, which was dead code in the draft (§2.5). Grep your own output to confirm every specialist has a live call path.
6. Author `eval/routing_testset.json`: ~100 hand-written queries with gold task labels, spanning all five tasks and including the adversarial cases (query contradicts input count; ambiguous phrasing; the PS's five representative queries verbatim).

**Acceptance test:** routing accuracy ≥90% on the 100-query set, printed with a per-task confusion matrix; a trace JSON is written for a successful run, a validation-failure run **and** a mid-inference crash; every one of the five specialists is reachable via at least one test query (assert programmatically, don't eyeball); a change run and a fusion run each produce **≥2 entries in `models_used`** — the PS requires selecting "one or more models or tools" and sequencing them, so this must be *demonstrable from the trace*, not merely true internally; the trace validates against `contracts.py`.

---

### W7 — Web App & Report Generation · ~4 days, can start early against stubs

**Depends on:** W0 stubs + W6 controller (can develop against the stub controller).
**Owns:** `app/**`, `satquery/report.py`, `requirements/extra-w7.txt`
**Read-only:** everything else

**Goal:** R6 and the "visual evidence, confidence, execution summary, downloadable report" deliverable.

Tasks:
1. Streamlit app: upload 1–2 images, **explicit modality selector per image** (§4.4 precedence tier 1) with the auto-detected value pre-filled, query box, run button.
2. Result view: text answer, confidence **with its `confidence_basis` shown next to it** (do not display a heuristic as if it were a probability), and the visual evidence — boxes / change mask / agreement map overlays.
3. **Execution summary panel** rendering the trace: task selected, why it routed that way, models used, parameters, timings. This is a graded rubric row; make it prominent and readable, not a collapsed JSON blob.
4. `report.py` — downloadable PDF or HTML per query, containing query, inputs, answer, visuals, and the full execution trace.
5. Handle the slow path honestly: model load is tens of seconds on this hardware. Progress indicators, not a frozen page.

**Acceptance test:** the app runs `streamlit run app/main.py`, completes a full single-image VQA round trip and a full bi-temporal round trip through the real controller, and produces a downloadable report file containing the trace; the modality selector demonstrably changes routing/validation behaviour; the CLI/matplotlib flow from `old_files/final_test.py` is not used anywhere.

---

### W8 — Evaluation Harness · runs continuously from the moment W3 lands

**Depends on:** W1 data, plus whichever specialists exist. Design it to run against stubs and report `PLACEHOLDER`.
**Owns:** `eval/**` (except `routing_testset.json`, authored by W6), `requirements/extra-w8.txt`
**Read-only:** everything else

**Goal:** real numbers for every rubric row, re-runnable with one command.

Tasks:
1. `eval/vrsbench.py` — BLEU, ROUGE, CIDEr on captioning; IoU on grounding. Use the prescribed test split.
2. `eval/rsvqa.py` — accuracy, answer match rate, BLEU.
3. `eval/cdvqa.py` — change VQA accuracy; change mask precision against SECOND references.
4. `eval/routing.py` — accuracy + confusion matrix over W6's test set.
5. `eval/adaptation.py` — W2's before/after retrieval and linear-probe numbers, re-runnable rather than copy-pasted from a notebook.
6. `eval/run_all.py` → a single `docs/RESULTS.md` table mapped **row-for-row onto the rubric in §3.3**, with sample counts and dates. Anything not yet measured says `PLACEHOLDER`, never a guess (§5.9).

**Acceptance test:** `python eval/run_all.py` produces `docs/RESULTS.md` with a row per rubric line, each carrying either a real number with its sample count or an explicit `PLACEHOLDER`; every harness runs on a ≥100-sample subset without manual intervention; results are reproducible across two runs (fixed seeds, greedy decoding).

---

## 8. Schedule

**Four-week line (target):**

| Week | Primary | Parallel |
|---|---|---|
| 1 | **W0** (day 1–2, blocking), then **W1** tiers 1–2. Bhoonidhi registration **day 1**. | W6 + W7 start against stubs |
| 2 | **W2 Track A** — baseline measured, then trained, checkpoint saved | W3 lands; W8 harnesses come online |
| 3 | **W4** and **W5** against the real `benclip`. W2 Track B if Track A is clean. | W8 fills in real numbers; W7 wires the real controller |
| 4 | Integration, §9 audit pass, edge cases, demo rehearsal on Bhoonidhi data | `docs/RESULTS.md` finalised |

**Two-week fallback line** (if the calendar compresses): W0 → W1 tier 3 → W2 Track A only → W3 → W6 → W7, with W4 and W5 built on classical CV plus `benclip` labels and **no** trained change/fusion head. Every hard requirement R1–R8 is still met on this line; the benchmark numbers are just weaker. **Cut Track B first, then W4/W5 sophistication. Never cut W6's trace (R5) or W7's app (R6) — those are pass/fail requirements, not scored ones.**

**Ordering constraints that actually matter:**
- W0 blocks everything. Do not fan out before it merges.
- W1's Bhoonidhi registration has a human-latency tail — start it on day 1, use the data in week 3.
- W2's baseline measurement must happen **before** training, or R1 has no "before."
- W6 and W7 do **not** need real specialists. Starting them in week 1 against stubs is the main source of parallelism available to a solo operator.

---

## 9. Pre-Submission Audit (run this before submitting; it is not optional)

- [ ] **R1** — `docs/status/W2.md` has real before/after numbers; adapter checkpoint exists and loads
- [ ] **R2** — swap the SAR input on a fusion call; the output changes. (Proves the second modality is genuinely read.)
- [ ] **R3** — change path is distinct code from fusion path; both reachable
- [ ] **R4** — two images + "describe the first image" routes to `caption`, not `change`
- [ ] **R5** — `runs/` contains a trace for every run made during rehearsal, each with task, models, parameters
- [ ] **R6** — app runs; `old_files/final_test.py`'s CLI flow is referenced nowhere
- [ ] **R7** — a grounding query through the app returns drawn boxes
- [ ] **R8** — VQA + captioning + grounding all reachable
- [ ] **Dead code sweep** — every function defined has a live caller or a test that calls it (the check that would have caught `run_grounding` and `get_execution_metadata`)
- [ ] **Claim sweep** — every claim in every README/report has a matching code path. Re-read §2.4 and confirm we didn't recreate it.
- [ ] **Edge cases** — featureless image, missing file, wrong format, 1-band SAR TIFF, mismatched image sizes, corrupt upload: all handled, all traced
- [ ] **Domain-shift rehearsal** — full demo run end-to-end on the real Bhoonidhi Cartosat + RISAT pair, not on Sentinel data
- [ ] `docs/RESULTS.md` has a real number or an honest `PLACEHOLDER` for every rubric row

---

## 10. Agent Brief Template

Copy this when dispatching. Fill the brackets, paste verbatim.

```
You are executing work order [W_] from PLAN.md in /home/mayaskara/projects/sih2.

BEFORE YOU START: read PLAN.md sections 0-6 in full, then section 7's [W_] entry.
Do not read old_files/readme.md — PLAN.md §2.4 explains why.

GOAL: [paste the work order's Goal line]

YOU OWN (edit freely):    [paste Owns row]
READ-ONLY (never edit):   everything else in the repo, especially satquery/contracts.py,
                          PLAN.md, and any other workstream's files.
BRANCH:                   w[N]-[short-name]

CONTRACTS: your outputs must satisfy PLAN.md §4 exactly. If you believe a contract
needs to change, STOP and report it — follow §5.5, do not edit contracts.py unilaterally.

DEPENDENCIES: declare new packages in requirements/extra-w[N].txt only.
Do not touch pyproject.toml.

RULES: PLAN.md §5 applies in full. In particular:
 - No dead code: every function you add gets a live caller or a test in the same change (§5.4)
 - No module-global model caching — use satquery/runtime/modelpool.py (§4.3)
 - Do not import another specialist (§5.3)
 - Do not write a number you did not measure or name a technique you did not
   implement; label placeholders PLACEHOLDER (§5.9)

DONE MEANS: [paste the Acceptance test verbatim]. Not "code written" — the acceptance
test passing.

WHEN FINISHED: write docs/status/W[N].md with what's done, what's stubbed, what's
blocked, any contract pressure, and real measured numbers. Do not edit PLAN.md.
```

---

## 11. Progress Log

*Human-owned. Agents write `docs/status/W<N>.md` instead (§5.8).*

- 2026-08-29 — PLAN v2 written. Verified: 4 GB VRAM (not 6); BigEarthNet v2.0 imagery is 54.4 GB + 63.3 GB monolithic `.tar.zst` on Zenodo with no sharded or HF-mirrored access; BigEarthNet.txt lives on HuggingFace, not `txt.bigearth.net`; Cartosat-2S is 1-band PAN / 4-band MSI and RISAT is 1–2 band, so the old `read([1,2,3])` breaks on the actual grading data. Locked: `benclip` multi-sensor encoder as primary adaptation, Qwen2-VL-2B QLoRA as target; GeoChat dropped as a local baseline. Repo still has zero commits — W0 makes the first.
