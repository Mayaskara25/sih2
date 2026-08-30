#!/usr/bin/env python3
"""Measure the STOCK CLIP *before* baseline on the held-out BEN split.

This is R1's "before": zero-shot image->text retrieval R@1/R@5 and linear-probe
mAP for the unadapted openai/clip-vit-base-patch32 on the same held-out split
and identical protocol that the trained benclip ("after") will use. Run this
FIRST, before any training, and record the numbers in docs/status/W2.md.

Local, small-batch, no-grad — safe on the 3.9 GB GTX 1650.

Usage:
  python -m train.measure_baseline [--train-cap N] [--test-cap N]
                                   [--out docs/status/benclip_before.json]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from train.benclip_data import (default_ben_paths, load_caption_map, load_targets,
                                split_patches)
from train.benclip_eval import evaluate_retrieval_and_probe, stock_clip_embedder


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the stock CLIP 'before' baseline.")
    ap.add_argument("--train-cap", type=int, default=None)
    ap.add_argument("--test-cap", type=int, default=None)
    ap.add_argument("--out", default="docs/status/benclip_before.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # Steering clear of CUDA if it isn't usable.
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = default_ben_paths()
    patches = load_targets(paths["targets"])
    all_ids = {p["patch_id"] for p in patches}
    caption_map = load_caption_map(paths["parquet"], all_ids)

    print(f"[benclip-baseline] device={args.device}")
    print(f"[benclip-baseline] train split: "
          f"{len(split_patches(patches,'train'))}, test: {len(split_patches(patches,'test'))}")

    embedder = stock_clip_embedder(device=args.device)
    t0 = time.time()
    results = evaluate_retrieval_and_probe(
        embedder, patches, caption_map,
        batch_size=8, train_cap=args.train_cap, test_cap=args.test_cap,
    )
    results["encoder"] = "openai/clip-vit-base-patch32"
    results["input_bands"] = "B04/B03/B02 (RGB composite, 3 channels)"
    results["seconds"] = round(time.time() - t0, 1)

    print(json.dumps(results, indent=2))
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[benclip-baseline] wrote {args.out}")


if __name__ == "__main__":
    main()
