You are executing work order **W9 — Router fidelity fix + model-switch latency** in /home/mayaskara/projects/sih2.

BEFORE YOU START: read PLAN.md §0–§6, then §7's W6 entry, then `docs/status/W6.md` and `docs/status/W8.md` (W8's "Routing-fidelity finding" section is the measurement that motivates task 1). Do NOT read `old_files/readme.md` — PLAN.md §2.4 explains why it is fiction.

Project state: **8/8 hard requirements met, 367 tests passing, no stubs.** Nothing here is blocking a requirement — both tasks are quality fixes ahead of a demo. **Do not regress the 367.**

## YOU OWN — create/edit ONLY these files
```
satquery/controller/router.py          (task 1)
satquery/runtime/modelpool.py          (task 2 — see the FROZEN warning below)
tests/test_w9_*.py
docs/status/W9.md
```

## DO NOT TOUCH
`satquery/contracts.py`, `satquery/io/**` (W0, frozen) · `satquery/specialists/**` (W3/W4/W5) · `satquery/adapters/**`, `train/**`, `checkpoints/**` (W2) · `app/**`, `satquery/report.py` (W7) · `eval/**` and `docs/RESULTS.md` (W8) · `scripts/data/**`, `data/**` (W1) · `PLAN.md` (read-only for every subagent, §5.1) · `pyproject.toml` (§5.6).

**`eval/routing_testset.json` is W6-authored and MUST NOT be edited.** It is the scoring set. Editing it to make your change pass would be marking your own homework — the exact circularity that made its 100/100 score untrustworthy in the first place (see task 1). Add new cases to `tests/test_w9_*.py` instead.

---

# TASK 1 — `rural_urban` queries are 100% misrouted to grounding

## The measurement (already done, do not re-derive)
W8 pre-routed **all 10,004 active RSVQA test questions**:

| type | misrouted | note |
|---|---|---|
| `rural_urban` | **100 / 100** | every one → `grounding`. Entire category unscored for VQA |
| `presence` | 115 / 2955 | object word beats the yes/no verb |
| `comp` | 7 / 4002 | |
| `count` | 0 / 2947 | |
| **total** | **222 / 10,004 (2.2%)** | real-query routing fidelity **97.8%** |

`eval/routing_testset.json` reports **100/100**, but W6 wrote both that set and `router.py`, so it is self-graded and an upper bound. The 97.8% figure is the trustworthy one.

## Root cause (verified by reading the code — confirm before changing)
`satquery/controller/router.py` routes in two tiers: Tier-1 regex rules, then Tier-2 idf-weighted cosine nearest-neighbour over `_EXEMPLARS`.

The RSVQA question is **"Is it a rural or an urban area?"**. No Tier-1 rule matches it, so it falls to Tier-2, where:

- `_EXEMPLARS["grounding"]` contains **"find the urban area"** (~line 280)
- `_EXEMPLARS["vqa"]` contains **"what fraction of the image is urban area"** (~line 271)

The query shares the high-idf tokens `urban` + `area` with the *short* grounding exemplar, and short documents score higher under cosine similarity. The longer vqa exemplar dilutes. Grounding wins.

## The fix
Add a **narrow Tier-1 rule** routing rural-vs-urban classification to `vqa`. Requiring **both** tokens is the safe form:

```python
Tier1Rule("vqa", "vqa_rural_urban", r"\brural\b.*\burban\b|\burban\b.*\brural\b")
```

**Why both tokens, and why this cannot be widened casually:** a rule on `\burban\b` alone would capture legitimate non-VQA queries that already route correctly —
- `"find the urban area"` → **grounding** (correct)
- `"use optical together with sar to map the urban extent"` → **fusion** (correct, and it is in the scoring set at `routing_testset.json:393`)
- `"what fraction of the image is urban area"` → **vqa**

Neither of the first two contains `rural`, so the paired form cannot touch them. **Verify this claim yourself by running those three queries before and after your change.**

Check the Tier-1 priority ordering (`router.py` ~line 90: *"grounding beats caption, caption beats vqa"*) and confirm no grounding Tier-1 rule fires on the rural/urban phrasing. It should not — the question contains none of `highlight` / `locate` / `localise` / `where is`.

## Optional second fix — `presence` (115/2955)
Only attempt after task 1 is landed, tested, and committed. Queries like *"Is a forest present in the image?"* route to grounding because the object word outweighs the yes/no verb. A Tier-1 rule anchored on the interrogative yes/no form (`^is|are\b .* \bpresent\b`) is the likely shape. **This one is higher-risk** — it sits close to genuine grounding phrasing. If your change moves any currently-correct query, stop and revert; 115/2955 (3.9%) is not worth breaking grounding for.

## Acceptance for task 1
1. `"Is it a rural or an urban area?"` routes to **vqa**, asserted in `tests/test_w9_*.py`.
2. All three queries above still route to grounding / fusion / vqa respectively — asserted, not eyeballed.
3. `eval/routing_testset.json` is **unmodified** (`git diff --stat` must not list it).
4. Re-run W8's routing harness and record the new number: `uv run --no-sync python eval/routing.py`.
5. **Re-measure real-query fidelity** over the full RSVQA split and report the new total (it should improve from 222/10,004). W8's method: pre-route every active question in `data/rsvqa_lr/*test*.json` and count those not landing on `vqa`. Do not edit `eval/**` to do this — write a throwaway script under `/tmp`.
6. Existing **367 tests still pass**.

---

# TASK 2 — model-switch latency (~20 s stall when the demo changes task type)

## The measurement (already done)
PLAN.md §4.3 keeps **one heavy model resident**: *"acquiring a different heavy role auto-releases the currently resident one"* (`satquery/runtime/modelpool.py` line ~17).

W8's VRSBench harness originally interleaved captioning (Qwen2-VL) and grounding (Grounding DINO) per item. Measured consequence: **GPU utilisation 0–5%, VRAM sawtoothing 761 ↔ 2100 MiB**, ~2N model loads for N items. Restructuring into two sequential passes (all captions, then all groundings) took it to **2 loads and ~100% GPU**. That fix is already committed in `eval/vrsbench.py` — read it as the reference case.

**The same behaviour hits the live app.** A judge who asks a captioning question and then a grounding question pays a **~20 s reload each way**. This is not a bug — the pool is doing exactly what §4.3 says on a small card — but it is a bad demo moment.

## Hard constraint: measure before you design
The local card is **3.64 GiB usable** (the old readme's "6 GB" is wrong — do not trust it). Measured residency:

| role | precision | measured VRAM | exempt |
|---|---|---|---|
| `benclip` | fp16 | **0.60 GiB** | **yes** (§4.3) |
| `vqa` (Qwen2-VL-2B) | 4bit-nf4 | ~2.1–2.4 GiB | no |
| `grounding` (grounding-dino-tiny) | **fp32 at runtime** | measure it | no |

⚠️ **`grounding` is registered fp16 in `DEFAULT_REGISTRY` but `satquery/specialists/grounding.py` re-registers it as fp32 at runtime**, because fp16 grounding-dino-tiny crashes on this stack (`grid_sample` mixes a float32 sampling grid with half feature maps). **Do not "fix" this by forcing fp16 — it is a deliberate, documented workaround.** Read the comment at the top of `grounding.py` first.

Measure actual peak residency before choosing an approach:
```python
import torch; torch.cuda.reset_peak_memory_stats()
# ... acquire role, run one query ...
print(torch.cuda.max_memory_allocated()/1024**3)
```

## Approaches, best first
1. **Free `benclip` under pressure.** It is `exempt=True` and stays resident at 0.60 GiB — 16% of the card. If releasing it lets Qwen and DINO coexist, that alone removes the stall. This is the cheapest change and the first thing to test.
2. **Make `grounding` co-resident** if the arithmetic works (Qwen ~2.4 + DINO-tiny + benclip 0.60 against 3.64 total — it is marginal; measure, do not assume).
3. **Warm-start the likely-next model** in the background while the user reads the current answer. Hides latency without changing residency rules.
4. **If none fit: change nothing in the pool and document it.** An honest "model switching costs ~20 s on a 3.6 GB card; the app shows a spinner" is a perfectly good answer. §4.3 exists because the card is small.

## ⚠️ `satquery/runtime/modelpool.py` is W0-FROZEN (§5.1)
You are granted a **narrow exception** to change residency policy there, and nothing else. Specifically:
- **Do not change `RoleSpec`'s field names or the `acquire`/`release` signatures** — `app/**`, `eval/**` and every specialist call them.
- `release()` deliberately raises if a caller still holds a live reference. **Keep that.** It is what stops a freed model being used.
- If your change needs a contract or signature change, **STOP and report it** (§5.5). Do not redesign the pool.

## Acceptance for task 2
1. A measured before/after table in `docs/status/W9.md`: seconds for `caption → grounding → caption` through `satquery.controller.run_query`, plus peak VRAM. **Numbers you measured, not estimates** (§5.9).
2. No OOM on the 3.64 GiB card across a 10-query alternating sequence — run it.
3. If you conclude the stall cannot be removed within the VRAM budget, **say so with the measurements that show it**. That is a successful outcome, not a failure — do not force a change that risks OOM mid-demo.
4. Existing **367 tests still pass**.

---

## RULES (PLAN.md §5)
- **No dead code** (§5.4): every function gets a live caller or a test.
- **Do not write a number you did not measure**; label placeholders `PLACEHOLDER` (§5.9).
- Tests must not download weights unconditionally or dispatch a real model ungated — gate on `torch.cuda.is_available()` plus data presence so a GPU-less clone stays green.
- ⚠️ **VRAM in tests**: `tests/test_w0_stubs.py` and `tests/test_w6_trace.py` carry autouse fixtures that release benclip between tests, because the card is too small to hold benclip plus a heavy model across serial test cases. If you add GPU tests, do the same or you will get spurious `torch.OutOfMemoryError`.

## GIT
- Work on the **current branch** (`w1-data`). Do not create or switch branches.
- **NEVER `git add -A` or `git add .`** (§5.7). Stage only paths you own, by name. A broad add previously swept another agent's in-progress files into an unrelated commit and had to be undone with a soft reset.
- Prefer not to commit; leave it to the orchestrator.

## ENVIRONMENT
```
cd /home/mayaskara/projects/sih2
uv run --no-sync pytest tests/ -q                    # ~10-14 min, expect 367 passed
uv run --no-sync python eval/routing.py
```
Python 3.11, torch 2.13.0+cu130, transformers **5.16.x** (major version — use 5.x APIs, not 4.x-era calls). GPU: GTX 1650, **3.64 GiB usable**. benclip loads in ~25 s; Qwen2-VL in ~20 s.

## REPORT BACK
Write `docs/status/W9.md` with: the routing change and exactly which queries you verified unchanged; the new real-query fidelity number over all 10,004 RSVQA questions; the measured latency before/after table with peak VRAM; and — if you left the stall in place — the measurements proving it does not fit. State plainly whether `eval/routing_testset.json` was touched (it must not be). Do not edit PLAN.md.
