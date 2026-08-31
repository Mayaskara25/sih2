# SatQuery AI

Agentic vision–language assistant for remote-sensing imagery, driven by natural-language
queries. Smart India Hackathon 2026, ISRO problem statement **26167**.

You give it one or two satellite images and a question in plain English. A router picks the
right specialist (VQA, captioning, grounding, bi-temporal change, or optical–SAR fusion),
runs it, and writes an auditable execution trace to `runs/<run_id>/trace.json` — every run,
success or refusal or crash, no exceptions.

---

## Status (2026-08-31)

| | |
|---|---|
| Hard requirements (PLAN.md §1) | **8 / 8** |
| Test suite | **378 passed, 0 failed** on the dev machine (full data + GPU) |
| Stubs remaining | none — every specialist is real |
| Measured results | [`docs/RESULTS.md`](docs/RESULTS.md) |

Numbers in `docs/RESULTS.md` are measured or say `PLACEHOLDER` and name their blocker.
That rule (PLAN.md §5.9) applies to this README too: every figure below was measured on the
machine named, and is a *observation*, not a *requirement*, unless it says otherwise.

---

## ⚠️ Read this before you run the test suite

The full suite **peaked hard enough to hang a 13 GB-RAM desktop for ~15 minutes** (system-wide
OOM, killed processes with exit 137). It was not GPU memory — it was host RAM, because loading
4-bit Qwen2-VL streams weights through system memory. Firefox was open at the time.

**Close your browser and other heavy apps before `pytest`.** On a laptop with 8 GB RAM,
expect the model-loading tests to fail or hang; use the no-GPU quick check below instead.

---

## Dev machine (what the numbers above were measured on)

| | |
|---|---|
| GPU | NVIDIA GTX 1650, 4096 MiB total, **3.64 GiB usable** |
| RAM | 13 GB — **this, not the GPU, is the binding constraint** |
| Python | 3.11.15 (`requires-python = "==3.11.*"`) |
| uv | 0.12.5 |
| `.venv` | 5.7 GB after `uv sync` |
| HF model cache | 9.6 GB (Qwen2-VL-2B 7.8 G + CLIP 1.2 G + Grounding-DINO-tiny 659 M) |
| Datasets on disk | 17 GB (all six; not needed for most work — see tiers) |

The whole system was designed around 4 GB of VRAM: one heavy model resident at a time
(PLAN.md §4.3), 4-bit NF4 quantisation for the VLM. **If you have more VRAM than this, it
still works** — nothing assumes a small card. If you have less, it won't.

---

## Setup

```bash
git clone <this repo> && cd sih2
uv sync                 # creates .venv from uv.lock, Python 3.11
```

No `pip install -r requirements.txt` — `requirements/extra-w*.txt` are per-work-order
dependency *declarations* (PLAN.md §5.6), not install lists. `pyproject.toml` + `uv.lock` are
the real dependency set.

### Tier 0 — verify your install (no data, no GPU, no weights, ~25 s)

```bash
CUDA_VISIBLE_DEVICES="" uv run pytest -q -rs
```

**Measured on a fresh clone with no `data/` and no checkpoint: 326 passed, 52 skipped,
0 failed, 23.8 s.** Everything that needs a GPU, a dataset, or the benclip checkpoint skips
with a reason instead of failing. If you see failures here, your environment is wrong — not
the code.

### Tier 1 — run the app

```bash
uv run streamlit run app/main.py
```

Upload one image (VQA / captioning / grounding) or two (change / fusion), pick the modality
for each, ask a question. Model weights download from HuggingFace on first use (~9.6 GB total,
cached in `~/.cache/huggingface`), so the first query of each type is slow.

**Without the benclip checkpoint** (see below) everything still runs — you lose the
land-cover evidence layer: VQA and captioning report `benclip_skipped`, change reports
`benclip_t0_error` with empty per-class deltas, fusion reports no cross-modal agreement.
The app does not crash, and the trace says so explicitly rather than silently omitting it.

### Tier 2 — run the full test suite (needs GPU + datasets)

```bash
uv run pytest -q          # ~8 min (486 s measured), 378 passed on the dev machine
```

### Tier 3 — reproduce `docs/RESULTS.md` (needs all six datasets)

```bash
uv run python eval/run_all.py
uv run python eval/adaptation.py --full   # the authoritative R1 numbers, full split
```

---

## The benclip checkpoint — ⚠️ ACTION NEEDED BEFORE HANDOVER

`checkpoints/benclip/benclip_state.pt` is **384 MB and gitignored** — it is not in this repo
and you cannot get it by cloning. It is the domain-adapted CLIP vision tower (LoRA, trained on
BigEarthNet S1+S2 pairs on a Colab T4) that earns rubric row 4.

<!-- MAYA: host benclip_state.pt (Drive / HF / release asset) and paste the link here.
     Until this line is filled in, the handover is incomplete for anyone but you. -->

**Download link: _____________________ (not yet published)**

Place it at `checkpoints/benclip/benclip_state.pt`, or point elsewhere:

```bash
export SATQUERY_BENCLIP_PATH=/path/to/dir-containing-benclip_state.pt
```

Fallback if you can't get the file: retrain it. `train/` holds the training code; it ran on a
free Colab T4 and the measured result is in `docs/status/W2.md` (mAP 0.42095 → 0.43134,
macro-F1 0.26999 → 0.30542, `n_train` 6180 / `n_test` 3394). Retraining needs the BigEarthNet
slice (7.4 GB) on the training machine.

---

## Datasets

All gitignored (17 GB). Only what a given task needs — Tier 0 and Tier 1 need none.

| Dataset | Size | How to get it |
|---|---|---|
| BigEarthNet S1+S2 slice | 7.4 GB | `uv run python scripts/data/fetch_bigearthnet.py` — reproduces the identical 13,630 patches |
| RSVQA-LR test | 243 MB | `uv run python scripts/data/fetch_rsvqa.py` |
| VRSBench val | 4.0 GB | HF `xiang709/VRSBench` — `Images_val` + `Annotations_val`. Use `hf_hub_download` (chunked), **not** plain `curl`: single-stream was throttled to 5.3 Mbps vs 120 Mbps measured on the same line |
| SECOND | 4.9 GB | `captain-whu.github.io/SCD` (Google Drive). Keep the **train** split — CDVQA is built on it |
| CDVQA | 53 MB | `git clone https://github.com/YZHJessica/CDVQA` — annotations only; images come from SECOND train |
| Bhoonidhi (Cartosat/RISAT) | — | **Not acquired.** See below |

There is no fetch script for VRSBench, SECOND or CDVQA — those three were pulled by hand.
Provenance and schema notes are in `docs/status/W1-recon-*.md`.

After fetching: `uv run python scripts/data/verify.py` checks sha256 against
`data/manifests/`.

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `SATQUERY_BENCLIP_PATH` | `checkpoints/benclip` | Directory holding `benclip_state.pt` |
| `SATQUERY_VQA_MODEL` | `Qwen/Qwen2-VL-2B-Instruct` | VQA / captioning backbone |
| `SATQUERY_GROUNDING_MODEL` | `IDEA-Research/grounding-dino-tiny` | Detector |
| `SATQUERY_GROUND_THRESHOLD` | see `grounding.py` | Box confidence threshold |
| `SATQUERY_GROUND_TEXT_THRESHOLD` | see `grounding.py` | Text-match threshold |
| `SATQUERY_CO_RESIDENT_MODELS` | unset (off) | `=1` keeps VQA + grounding both resident. Removes a ~20 s reload stall on every caption↔grounding switch (measured 20 s → 0.75 s warm). **Cost:** peak 3089 MB on a 3.64 GiB card, so benclip is skipped in captions — disclosed in the trace as `benclip_skipped`, never silently. Worth considering for a live demo; off by default |

---

## Repository map

```
PLAN.md            source of truth — contracts, requirements, rules. READ §4 BEFORE EDITING CODE
app/main.py        Streamlit UI (thin; logic lives in satquery/)
satquery/
  controller/      router + validation + trace writing — the single entry point run_query()
  specialists/     vqa, caption, grounding, change, fusion — one file each, never import each other (§5.3)
  adapters/        benclip.py — the ONLY place band-mapping logic lives (§4.5)
  runtime/         modelpool.py — one heavy model resident at a time (§4.3)
  io/              raster.py — the ONLY place rasters are opened (§4.4)
eval/              measurement harness; run_all.py writes docs/RESULTS.md
train/             benclip training (ran on Colab T4)
scripts/data/      dataset fetchers + sha256 verifier
tests/             378 tests
docs/status/W*.md  per-work-order evidence logs — what was measured and how
docs/briefs/W*.md  the agent work orders those logs answer
old_files/         PRIOR-ART REFERENCE ONLY, NOT SPEC — PLAN.md §2.4/§2.5 list confirmed bugs in it. Do not copy from it
runs/              execution traces (gitignored)
```

**Doc precedence:** `PLAN.md` (contracts) → `docs/RESULTS.md` (measured numbers) →
`docs/status/W*.md` (how each number was obtained). If they disagree, `docs/RESULTS.md` is the
newest for numbers; PLAN.md §11's progress log is human-maintained and lags.

---

## Known limitations — say these before a judge finds them

1. **Cartosat/RISAT band order is assumed, not verified.** PLAN.md §4.5 assumes Sentinel-2
   band semantics. Bhoonidhi turned out to gate the real data: Cartosat-2S is **priced**, not
   open, and RISAT-1 under Open Data offers only low-resolution microwave with no selectable
   products. See `docs/bhoonidhi_registration.md` (correction banner at the top). The nearest
   open substitutes are Resourcesat LISS-III/AWiFS and Cartosat-1 ortho — and **LISS-III has no
   blue band**, which §4.5's B02/B03/B04/B08 mapping assumes.
2. **Change-mask precision is 0.2492 against a 0.1832 trivial floor.** Real signal at 1.36×
   the "everything changed" baseline, but weak — expected of classical differencing with no
   trained change head. The detector over-flags (0.259 predicted vs 0.183 actual).
3. **CDVQA change-VQA accuracy is unpublished on purpose.** 1,230 questions were run; the score
   is invalid, not missing. `run_change` speaks BigEarthNet's 19-class vocabulary and CDVQA
   expects yes/no or one of SECOND's six tokens — containment 0.0512 vs a 0.3163 majority floor,
   i.e. *below* the trivial baseline, the signature of a format mismatch. Fixing it needs a VQA
   head over CDVQA's closed vocabulary (post-Round-1).
4. **Captioning and grounding are zero-shot.** Stock Qwen2-VL-2B and stock Grounding DINO,
   not fine-tuned on VRSBench — not comparable to VRSBench leaderboard numbers. CIDEr is a
   stdlib proxy implementation, labelled as such, not the reference `pycocoevalcap` scorer.
5. **Routing accuracy 1.000 is self-graded** — `eval/routing_testset.json` was written by the
   same work order as the router. Independent checks give 92% (hand-written probe) and 98.78%
   (all 10,004 RSVQA test questions pre-routed, after W9 fixed the `rural_urban`
   category that was 100/100 misrouted). Treat 100% as "no known regression".

---

## If you are adding code

Non-negotiables from PLAN.md §5, all of which cost us real damage when broken:

- **Never `git add -A` or `git add .`** (§5.7). Stage owned paths by name. A broad add once
  swept another agent's in-progress files into an unrelated commit.
- **`PLAN.md` is read-only** except to Maya (§5.1). Agents write `docs/status/W<N>.md`.
- **Never write a number you did not measure** (§5.9). Label it `PLACEHOLDER` and name the
  blocker instead. Every rubric row in `docs/RESULTS.md` follows this.
- **Specialists never import each other** (§5.3). Shared logic goes in `satquery/adapters/`
  or `satquery/io/`.
- **Dead code is a defect** (§5.4).
- **Re-run before you believe a report.** Three separate agent status docs overstated results
  on this project — a "requirement satisfied" where the metric had regressed, a self-graded
  100%, and a "378 passing" that was 375/3. Each was caught by re-running, not by reading.
