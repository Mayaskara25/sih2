# Demo pack — everything needed to exercise all five specialists

Nine images, 2.0 MB, committed to the repo so they arrive with the clone. Nothing here needs
the 17 GB of datasets. Pair this with `benclip_state.zip` (sent separately) and you can run
the whole system.

Start the app, then work down this page:

```bash
.venv/bin/python -m streamlit run app/main.py
```

Reference answers for every image are in [`ground_truth.json`](ground_truth.json).

---

## Before you start — two things that will otherwise look like bugs

**1. The first query of each type is slow.** Model weights download from HuggingFace on first
use (~9.6 GB total across all three models) and only one heavy model stays resident at a time
(PLAN.md §4.3), so switching between query types reloads. A ~20 s stall on a switch is the
design working, not hanging. `SATQUERY_CO_RESIDENT_MODELS=1` trades that for VRAM — see the
main README.

**2. Counting questions are a trap, not a target.** RSVQA ground truth includes answers like
"403 roads" and "1067 buildings" — exhaustive annotation counts. A zero-shot VLM will not
reproduce them and is not expected to. The checkable questions are `presence` (yes/no) and
`rural_urban`. Measured normalised accuracy is **0.4323 on n=155** (`docs/RESULTS.md` row 2);
judge against that, not against a perfect score.

---

## 1 · Single image — VQA, captioning, grounding

Upload **one** image from `01_single_image/`.

| Query | Routes to | Try it on | Expect |
|---|---|---|---|
| `Is it a rural or an urban area?` | `vqa` | `rsvqa_232_urban.tif` | **urban** |
| `Is it a rural or an urban area?` | `vqa` | `rsvqa_235_rural.tif` | **rural** |
| `Is there a road?` | `vqa` | `rsvqa_232_urban.tif` | **yes** |
| `Describe this satellite image in detail.` | `caption` | `vrsbench_P0003_0002.png` | a caption mentioning yellow buses / a parking lot; compare to the reference caption in `ground_truth.json` |
| `Find all the buildings in this image.` | `grounding` | `rsvqa_232_urban.tif` | boxes drawn on the image |
| `Locate every road in the image.` | `grounding` | `rsvqa_232_urban.tif` | boxes |

The `rural_urban` pair is worth running first: that whole question category was **100/100
misrouted** to grounding until W9 added a Tier-1 rule, and is now 0/100. If those two queries
land on `vqa` in the execution summary, the fix is live in your checkout.

Routing verified for all six queries against `satquery.controller.router.route` — task and
selected image index as listed.

---

## 2 · Two dates, same modality — bi-temporal change (R3)

Upload **both** `second_00014_t0.png` and `second_00014_t1.png` from `02_change_pair/`.

| Query | Routes to |
|---|---|
| `What changed between these two dates?` | `change` |
| `Compare the two dates and describe the land cover change.` | `change` |

**Ground truth: 47.1% of pixels changed** (fraction where the per-date semantic label differs).
The two `*_GTlabel.png` files are the reference masks — for your eyes only, the system never
sees them. Expect over-segmentation: this is classical differencing with no trained change
head, measured precision **0.2492** against a **0.1832** trivial floor.

---

## 3 · Optical + SAR — cross-modal fusion (R2)

Upload **both** files from `03_cross_modal/`. They are the same 120×120 ground footprint on the
same date — Sentinel-2 optical and Sentinel-1 radar, co-registered.

| Query | Routes to |
|---|---|
| `Combine the optical and radar images to analyse this area.` | `fusion` |
| `Use the SAR and optical data together to identify water bodies.` | `fusion` |

Modality resolves automatically here — **verified: the 3-band stack → `optical`, the 2-band
stack → `sar`, and fusion validation passes with zero warnings.** BigEarthNet ground-truth
labels for this patch: *Arable land, Broad-leaved forest, Inland waters*.

---

## 4 · The two things that are graded but easy to miss

### R2 — refusing a fake cross-modal run

Upload **two optical images** (`rsvqa_232_urban.tif` + `rsvqa_235_rural.tif`) and ask
`Combine the optical and radar images to analyse this area.`

**It must refuse.** Verified error text:

```
fusion requires two genuinely distinct modalities with a SAR input (R2);
both inputs resolved as optical-family modalities ['optical', 'optical']
— refusing the cross-modal run
```

R2 says the system must ingest two genuinely distinct modalities, "not one image processed
twice and relabeled". A system that happily *pretends* to fuse two optical images fails that
requirement. The refusal is the feature.

### R4 — query text overrides file count

Upload the **two** change-pair images, then ask `Describe the second image.`

Two files uploaded, but it routes to **`caption`** on **image index 1** — not to `change`.
Verified. R4 requires routing to be driven by query text, not by how many files arrived.

---

## 5 · Every run writes a trace (R5)

After any query, check `runs/<run_id>/trace.json`. It records the task selected, why the router
picked it, the model names, key parameters, the validation result, and the outputs — on success,
on refusal, and on crash, no exceptions.

Two things to look for in the evidence:

- **`confidence_basis`** is always shown next to a confidence number. `"heuristic"` means the
  number came from classical image processing, not a calibrated model. It is never dressed up
  as a probability.
- **`benclip_skipped`** or **`benclip_t0_error`** means the checkpoint is missing or was skipped
  for VRAM. The evidence layer is *disclosed as absent* rather than silently omitted — if you
  see this and did install the checkpoint, check `SATQUERY_BENCLIP_PATH`.

---

## What's in the pack

```
01_single_image/
  rsvqa_232_urban.tif        256×256 RGB · RSVQA-LR test img 232 · GT: urban
  rsvqa_235_rural.tif        256×256 RGB · RSVQA-LR test img 235 · GT: rural
  vrsbench_P0003_0002.png    512×512 RGB · VRSBench val · caption + VQA + referring boxes
02_change_pair/
  second_00014_t0.png        512×512 · SECOND test pair 00014, date 1
  second_00014_t1.png        512×512 · SECOND test pair 00014, date 2
  second_00014_t0_GTlabel.png   reference semantic labels, date 1
  second_00014_t1_GTlabel.png   reference semantic labels, date 2
03_cross_modal/
  S2_optical_RGB_61_39.tif   120×120×3 · Sentinel-2 B04/B03/B02 · → optical
  S1_SAR_VV_VH_61_39.tif     120×120×2 · Sentinel-1 VV/VH · → sar
ground_truth.json            reference answers for all of the above
```

Sources: RSVQA (Lobry et al.), VRSBench (`xiang709/VRSBench`), SECOND
(`captain-whu.github.io/SCD`), BigEarthNet v2.0 (`BIFOLD-BigEarthNetv2-0`). All open academic
releases; these are single-sample excerpts for testing. The two `03_cross_modal/` files are
band stacks built from the original per-band BigEarthNet tifs — band order matters (PLAN.md
§4.5) and these are in the order `benclip` expects.
