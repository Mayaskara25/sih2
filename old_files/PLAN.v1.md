# SatQuery AI — Build Plan
**SIH 2026 · Problem Statement 26167 (ISRO) · Source of truth for all agents/sessions working on this repo.**

Read this whole file before touching code. If something you're about to build isn't described here, stop and add it here first — don't let scope drift silently between sessions.

---

## 0. Mission Statement

Build an agentic vision-language assistant that takes a natural-language query plus one or two remote-sensing images (optical/multispectral or SAR, single date or bi-temporal) and returns an evidence-grounded answer: it must classify what the query is asking, run the correct specialist model(s), and return a result plus an auditable record of what it did and why. Final grading uses **real ISRO Cartosat-2S optical + RISAT SAR pairs** that the system has never seen during training — everything we build has to survive that domain shift, not just look good on our own test images.

## 1. Non-Negotiable Requirements

These are hard constraints, not nice-to-haves. Violating any of these risks disqualification regardless of how polished the rest looks:

- [ ] At least one visual/vision-language component must show **measurable fine-tuning or adaptation** on BigEarthNet.txt or equivalent open RS data. A stock pretrained VLM called zero-shot does **not** satisfy this — the problem statement says so explicitly.
- [ ] The system must genuinely ingest **two distinct modalities** (optical + SAR) for the cross-modal task — not a single-modality image processed twice and relabeled.
- [ ] The system must genuinely ingest **two dates of the same modality** for the bi-temporal task, distinct from the cross-modal path.
- [ ] Task routing must be driven by the **query text**, not just by how many files were uploaded.
- [ ] Every run must produce an **auditable execution trace**: selected task, model/tool names used, key parameters — written to disk, not just printed to console.
- [ ] Delivery format must be an **interactive GUI or web app**, not a terminal script.
- [ ] Grounding (bounding-box output) must actually be reachable at runtime for grounding-style queries, not just exist as unused code.

## 2. Environment & Known Constraints

- Local: GTX 1650, 4GB VRAM. Enough for 4-bit inference of small VLMs (Qwen2-VL-2B class) and classical CV, **not** enough for training/fine-tuning any VLM.
- Cloud: Colab / Kaggle free-tier (T4/P100, ~15GB VRAM, session time limits). This is where all fine-tuning happens. Plan training runs to fit inside a single session (a few hours), checkpointing adapters to Drive/Kaggle datasets so a disconnect doesn't lose progress.
- Given the VRAM ceiling, fine-tuning means **LoRA/QLoRA only** (4-bit base, small rank, few epochs, subsampled data) — not full fine-tuning of any base model.

### Known issues in the existing draft code (`models_registry.py`, `BT_CM.py`, `final_test.py`)
If any phase below builds on top of these files rather than rewriting from scratch, these must be fixed, not carried forward:

- [ ] `register_images`: affine transform is fit on the *unfiltered* match list even though a filtered `good` list is computed for the confidence score — fit the transform on the filtered matches.
- [ ] No null-check after `cv2.estimateAffinePartial2D` — it returns `None` on low-feature images (water, farmland, uniform terrain) and the next line crashes. Add a fallback (skip warp, flag low registration confidence) instead of crashing.
- [ ] No guard against `orb.detectAndCompute` returning empty descriptors on a near-blank scene — crashes `bf.match`.
- [ ] `generate_visual_modalities`'s "SAR" panel is a plain grayscale threshold of the *optical* image, not real radar data — replace with a genuine second-modality ingestion path (see Phase 6). Don't ship this as-is; it will fail immediately against real SAR inputs.
- [ ] Readme claims a Lee filter for SAR speckle; `uniform_filter` is imported but never called anywhere. Either implement it for real or remove the claim.
- [ ] `run_grounding` exists with a clean contract but is never imported or called in `final_test.py` — wire it into the controller (Phase 7) so grounding-style queries actually return bounding boxes.
- [ ] `get_execution_metadata` is imported but never called — the audit JSON currently has no task/model/parameter record. Wire it in (Phase 7).
- [ ] `load_image` always reads bands `[1,2,3]` as RGB — discards NIR/SWIR bands that matter for LULC tasks. Make band selection configurable per sensor.
- [ ] Confidence scores (`run_vqa`'s word-count heuristic, `BT_CM`'s 0.4/0.6-weighted "trust") are uncalibrated guesses. Fine for a v0 placeholder, but Phase 7/8 should replace or validate these against actual correctness on a held-out set before final submission.
- [ ] CLI (`input()` + matplotlib popup) does not satisfy the GUI/web-app requirement — Phase 9 wraps this in Streamlit or Gradio.

## 3. Shared Contracts (freeze before building modules)

Every specialist function returns JSON with these keys, no exceptions — agents building different modules must not invent their own shapes:

```
run_vqa(image_path, prompt) -> {"text_response": str, "confidence": float}
run_grounding(image_path, target_text) -> {"text_response": str, "bounding_boxes": [{"label","box_2d":[ymin,xmin,ymax,xmax],"confidence"}], "confidence": float}
run_change(image_path_t0, image_path_t1, query) -> {"text_response": str, "change_mask_path": str, "metrics": {...}, "confidence": float}
run_fusion(optical_path, sar_path, query) -> {"text_response": str, "evidence": {...}, "confidence": float}
```

Controller output (the audit trace, written to disk every run):
```
{"task_selected": str, "inputs": {...}, "models_used": {...}, "parameters": {...}, "result": {...}, "timestamp": str}
```

If a phase needs to add a field, add it here first, then propagate — don't let modules silently drift out of sync (this is exactly what happened with `run_grounding` and `get_execution_metadata` in the current draft).

## 4. Data Sources

| Dataset | Purpose | Source | Used in Phase |
|---|---|---|---|
| BigEarthNet.txt | Mandatory domain adaptation (Sentinel-1 SAR + Sentinel-2 MSI, captions/VQA/referring) | txt.bigearth.net (paper: arxiv.org/abs/2603.29630) | 2, 4 |
| VRSBench | Single-image captioning, grounding, VQA benchmark | vrsbench.github.io · github.com/lx709/VRSBench | 2, 3, 4, 8 |
| RSVQA (LR/HR) + RSVQAxBEN | VQA benchmark / extra fine-tuning pairs | rsvqa.sylvainlobry.com · github.com/syvlo/RSVQA · github.com/syvlo/RSVQAxBEN | 2, 8 |
| CDVQA (built on SECOND) | Bi-temporal change VQA benchmark | github.com/YZHJessica/CDVQA | 2, 5, 8 |
| Bhoonidhi (ISRO) | Real Cartosat-2S / RISAT samples for domain-shift sanity checks — not for training, for calibration | bhoonidhi.nrsc.gov.in (free registration) | 2, 6, 9 |
| GeoChat (reference baseline) | Pretrained RS-VLM to start adaptation from instead of training from scratch | github.com/mbzuai-oryx/GeoChat | 3, 4 |

**Subsampling note:** none of these need to be downloaded/trained on in full — BigEarthNet.txt alone is ~464K image pairs / 9.6M annotations, far beyond what a T4/P100 session can process. Pull a stratified few-thousand-sample slice per dataset; the goal is demonstrable adaptation and a believable benchmark number, not SOTA.

---

## 5. Phases

### Phase 1 — Foundations & Contracts
**Goal:** Repo skeleton, environment set up (local + Colab/Kaggle), Section 3's contracts written into a shared `contracts.py` or `.md`, so every later phase/agent builds against the same interface instead of discovering mismatches at integration time.
**Status:** Not started

- [ ] Repo structure: `/specialists`, `/controller`, `/adapters`, `/eval`, `/app`, `/data`
- [ ] `contracts.py` (or equivalent) encoding Section 3's schemas, importable by every module
- [ ] Local + Colab/Kaggle environments both installing from the same `requirements.txt`

### Phase 2 — Data Acquisition
**Goal:** Working local/Drive copies of subsampled slices of all four datasets, plus a handful of real Bhoonidhi Cartosat/RISAT samples. Can run in parallel with Phase 1.
**Status:** Not started

- [ ] BigEarthNet.txt slice downloaded + parsed
- [ ] VRSBench train/eval splits downloaded
- [ ] RSVQA (LR or HR) downloaded
- [ ] CDVQA/SECOND downloaded
- [ ] 5–10 real Cartosat/RISAT pairs pulled from Bhoonidhi for later sanity checks

### Phase 3 — Baseline End-to-End Skeleton (zero-shot, no fine-tuning yet)
**Goal:** A single query → single specialist call → single answer path works end to end, using an off-the-shelf pretrained VLM (GeoChat or Qwen2-VL), before any adaptation work starts. This proves the wiring works and gives a baseline number to improve on. Do not skip this — it's what catches integration gaps early instead of at the end.
**Status:** Not started

- [ ] `run_vqa` callable end to end on a VRSBench sample
- [ ] Baseline (untuned) score recorded — this is your "before" number for Phase 4

### Phase 4 — Domain Adaptation / Fine-Tuning
**Goal:** Satisfy the non-negotiable adaptation requirement with a measurable before/after improvement. This is the single highest-priority phase — start it as soon as Phase 2/3 data and baseline exist, since cloud training sessions are the biggest time sink.
**Status:** Not started

- [ ] QLoRA config chosen (small rank, 4-bit base)
- [ ] Training run on BigEarthNet.txt + VRSBench subsample, on Colab/Kaggle
- [ ] Adapter checkpoint saved and reloadable from the local pipeline
- [ ] Before/after comparison recorded against Phase 3's baseline

### Phase 5 — Change Detection Module (bi-temporal, same modality)
**Goal:** A real Change-VQA path (not just VQA-with-stats-stuffed-into-the-prompt), validated against CDVQA. Can reuse the existing classical-CV registration pipeline as a first pass, but the known bugs (Section 2) must be fixed first.
**Status:** Not started

- [ ] Registration bugs fixed (filtered matches used for the actual fit, `None`-matrix guarded)
- [ ] `run_change` implemented per the Section 3 contract
- [ ] Validated against a CDVQA test sample, not just eyeballed

### Phase 6 — Cross-Modal Optical–SAR Module
**Goal:** Genuine two-modality ingestion — replace the fabricated "SAR proxy" entirely. This is the least-served-by-existing-checkpoints task and the one judges will probe hardest, given the hidden eval set is real Cartosat+RISAT.
**Status:** Not started

- [ ] Explicit modality tagging at upload (band count / metadata / user-specified, not inferred)
- [ ] Real SAR preprocessing (at minimum dB/log scaling; speckle filtering if time allows)
- [ ] `run_fusion` implemented per the Section 3 contract
- [ ] Sanity-checked against the real Bhoonidhi Cartosat/RISAT samples from Phase 2

### Phase 7 — Agentic Controller & Auditable Execution Trace
**Goal:** Query text actually decides which specialist runs — this can be stubbed early (Phase 1) and filled in as Phases 4–6 produce real specialists to dispatch to.
**Status:** Not started

- [ ] Task classifier: query text → {vqa, grounding, change, fusion}
- [ ] Input compatibility checks (right number of images, right modality, right format) before dispatch
- [ ] `run_grounding` actually wired into the live path (currently dead code in the draft)
- [ ] Execution trace written to disk every run per Section 3's schema — task, models, parameters, not just the final answer
- [ ] Confidence scores reviewed — replace pure heuristics with something checked against actual correctness on a held-out sample, or at minimum document the limitation

### Phase 8 — Evaluation Harness & Benchmarking
**Goal:** Real numbers against the named benchmarks, not vibes. Can run continuously once each specialist from Phases 4–6 exists.
**Status:** Not started

- [ ] Batch eval script: BLEU/ROUGE/CIDEr/IoU on VRSBench
- [ ] Batch eval script: Accuracy on RSVQA
- [ ] Batch eval script: Change VQA Accuracy / Change Mask Precision on CDVQA
- [ ] Results table compiled — this is what you can defend if judges ask "what's your actual accuracy"

### Phase 9 — GUI, Report Generation & Integration Hardening
**Goal:** Everything from Phases 1–8 wrapped in an actual web app, plus a final pass that verifies every claimed feature has a real code path behind it — the same check that would have caught `run_grounding` and `get_execution_metadata` being dead code in the draft.
**Status:** Not started

- [ ] Streamlit or Gradio front end replacing the CLI/matplotlib flow
- [ ] Downloadable report (PDF or similar) generated per query
- [ ] Full audit pass: grep for every function defined-but-never-called, every readme claim without a matching code path
- [ ] Edge-case test pass: featureless images, missing files, wrong format, single-band SAR TIFFs
- [ ] Dry-run demo rehearsal against the real Bhoonidhi samples

## 6. Suggested Parallelization

Phase 1 and 2 run together first. Phase 3 (baseline skeleton) comes before Phase 4 so there's a "before" number, and before Phases 5/6 so there's proven wiring to slot them into. Once Phase 3 is done, Phases 4, 5, and 6 can run as three parallel tracks (different agents/sessions) since they're independent specialists sharing only the Section 3 contracts. Phase 7 needs stub versions of 4–6 to develop against but doesn't need them finished. Phase 8 runs continuously once any specialist is ready. Phase 9 is last and touches everything, so it should not start in earnest until 4–7 are at least functionally complete.

## 7. Final Submission Checklist (map back to the rubric before submitting)

- [ ] Single-Image Captioning & Grounding — VRSBench numbers in hand
- [ ] VQA — RSVQA numbers in hand
- [ ] Multi-Image Change Analysis — CDVQA numbers in hand
- [ ] Domain Adaptation & Fine-Tuning — before/after comparison in hand, adapter checkpoint exists
- [ ] Joint Cross-Modal Analysis — tested against real Cartosat/RISAT samples, not just simulated data
- [ ] Agentic Workflow Orchestration — execution trace JSON actually contains task/model/parameters for every run

## 8. Progress Log
*(append dated entries here as phases complete or plans change — keep this the single source of truth rather than status updates scattered across chats)*

- YYYY-MM-DD — 
