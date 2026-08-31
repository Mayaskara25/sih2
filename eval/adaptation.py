"""R1 adaptation (benclip) before/after harness — re-runnble, not copy-pasted.

PLAN.md §3.1 / W2's acceptance require a REAL before/after retrieval + linear-
probe mAP on the held-out BEN split. The numbers belong to ``train/``:
``train/measure_baseline.py`` produces the *before* (stock CLIP on RGB) and
``train/measure_benclip.py`` the *after* (trained 14-channel benclip), both via
the identical ``train/benclip_eval.evaluate_retrieval_and_probe`` protocol.

W8's job is to RE-RUN that protocol so ``docs/RESULTS.md`` regenerates real
numbers every time — never to copy the committed JSON values into the table
(PLAN.md §5.9: a number is only real if this run measured it). This harness
imports the two W2 scripts' ``main()`` functions (the W8 brief: "shell out to /
import train.measure_benclip"), feeds them a temp out-file, and reads back their
measured payload.

Caps: the full protocol is 6,180 train / 3,394 test patches (~8 min). Every W8
harness must run on a >=100-sample subset without manual intervention, so the
default here is a deterministic prefix subsample (train cap + test cap), with
``--full`` available to rerun the whole scale. All rows carry their n.

``PLACEHOLDER`` when the benclip checkpoint is absent from disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as _c  # noqa: E402

DEFAULT_CHECKPOINT = os.path.join(_c.REPO_ROOT, "checkpoints", "benclip")


def checkpoint_present(checkpoint: str = DEFAULT_CHECKPOINT) -> bool:
    marker = os.path.join(checkpoint, "benclip_state.pt")
    if os.path.isfile(marker):
        return True
    # Some layouts keep the state file directly under the checkpoint dir name.
    return os.path.isfile(checkpoint) or os.path.isfile(os.path.join(checkpoint, "..", "benclip_state.pt"))


def _measure(which: str, out_path: str, checkpoint: str, train_cap: Optional[int],
             test_cap: Optional[int], device: Optional[str]) -> dict:
    """Run the W2 measurement script's main() via sys.argv substitution so
    the script's own argparse parses exactly the same flags a shell call would
    (the W8 brief: "shell out to / import train.measure_benclip")."""
    argv = ["train.measure_baseline" if which == "before" else "train.measure_benclip"]
    if which == "before":
        argv += ["--out", out_path]
    else:
        argv += ["--checkpoint", checkpoint, "--out", out_path]
    if train_cap is not None:
        argv += ["--train-cap", str(train_cap)]
    if test_cap is not None:
        argv += ["--test-cap", str(test_cap)]
    if device is not None:
        argv += ["--device", device]

    saved = sys.argv
    sys.argv = argv
    try:
        if which == "before":
            from train.measure_baseline import main as baseline_main
            baseline_main()
        else:
            from train.measure_benclip import main as benclip_main
            benclip_main()
    finally:
        sys.argv = saved

    with open(out_path) as fh:
        return json.load(fh)


def _to_metric_row(payload: dict, encoder_label: str) -> dict:
    return {
        "encoder": payload.get("encoder", encoder_label),
        "retrieval_r1": payload["retrieval_r1"],
        "retrieval_r5": payload["retrieval_r5"],
        "map": payload["map"],
        "macro_f1": payload["macro_f1"],
        "n_train": payload["n_train"],
        "n_test": payload["n_test"],
    }


def run(train_cap: int = 600, test_cap: int = 300, checkpoint: str = DEFAULT_CHECKPOINT,
        device: Optional[str] = None) -> dict:
    """Measure before+after and return the paired table (or PLACEHOLDER)."""
    if not checkpoint_present(checkpoint):
        return {
            "status": _c.PLACEHOLDER,
            "blocker": (
                f"benclip checkpoint absent at {checkpoint}/benclip_state.pt — W2's "
                f"after-model has not landed; cannot measure the R1 'after' row."
            ),
            "date": _c.today(),
        }
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Device gate: the protocol needs a CUDA card for embeddings on the local
    # GTX 1650; CPU is supported by the W2 scripts and allowed here.
    print(f"[adaptation] measuring before (stock CLIP, device={dev}) ...")
    with tempfile.TemporaryDirectory() as tmp:
        before_path = os.path.join(tmp, "benclip_before_run.json")
        before = _measure("before", before_path, checkpoint, train_cap, test_cap, dev)

        print(f"[adaptation] measuring after (benclip {checkpoint}, device={dev}) ...")
        after_path = os.path.join(tmp, "benclip_after_run.json")
        after = _measure("after", after_path, checkpoint, train_cap, test_cap, dev)

    row_before = _to_metric_row(before, "openai/clip-vit-base-patch32 (stock)")
    row_after = _to_metric_row(after, "benclip (14-channel adapted)")

    return {
        "status": "measured",
        "date": _c.today(),
        "n_train": row_after["n_train"],
        "n_test": row_after["n_test"],
        "checkpoint": checkpoint,
        "before": row_before,
        "after": row_after,
        "delta": {
            "retrieval_r1": row_after["retrieval_r1"] - row_before["retrieval_r1"],
            "retrieval_r5": row_after["retrieval_r5"] - row_before["retrieval_r5"],
            "map": row_after["map"] - row_before["map"],
            "macro_f1": row_after["macro_f1"] - row_before["macro_f1"],
        },
    }


def format_report(res: dict) -> str:
    if res.get("status") == _c.PLACEHOLDER:
        return f"adaptation: {_c.PLACEHOLDER} — {res['blocker']}"
    lines = [
        f"R1 adaptation before/after (n_train={res['n_train']}, n_test={res['n_test']}, "
        f"date {res['date']})",
        "| encoder | R@1 | R@5 | mAP | macro-F1 |",
        "|---|---|---|---|---|",
        f"| before ({res['before']['encoder']}) | {res['before']['retrieval_r1']:.5f} | "
        f"{res['before']['retrieval_r5']:.5f} | {res['before']['map']:.4f} | "
        f"{res['before']['macro_f1']:.4f} |",
        f"| after ({res['after']['encoder']}) | {res['after']['retrieval_r1']:.5f} | "
        f"{res['after']['retrieval_r5']:.5f} | {res['after']['map']:.4f} | "
        f"{res['after']['macro_f1']:.4f} |",
        f"| delta | {res['delta']['retrieval_r1']:+.5f} | {res['delta']['retrieval_r5']:+.5f} | "
        f"{res['delta']['map']:+.4f} | {res['delta']['macro_f1']:+.4f} |",
    ]
    return "\n".join(lines)


def main(out_json: str | None = None, train_cap: int = 600, test_cap: int = 300) -> dict:
    res = run(train_cap=train_cap, test_cap=test_cap)
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(res, fh, indent=2)
    print(format_report(res))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="R1 adaptation before/after harness")
    ap.add_argument("--train-cap", type=int, default=600)
    ap.add_argument("--test-cap", type=int, default=300)
    ap.add_argument("--full", action="store_true", help="full-scale run (no caps)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tc = None if args.full else args.train_cap
    tt = None if args.full else args.test_cap
    main(out_json=args.out, train_cap=tc, test_cap=tt)