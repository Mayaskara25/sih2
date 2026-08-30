#!/usr/bin/env python3
"""Measure the TRAINED benclip *after* numbers on the held-out BEN split.

This is R1's "after": retrieval R@1/R@5 + linear-probe mAP for the adapted
14-channel benclip, computed with the SAME protocol and held-out test split as
``measure_baseline.py`` (the "before"), so the two are directly comparable.

Usage:
  python -m train.measure_benclip --checkpoint checkpoints/benclip \
      [--train-cap N] [--test-cap N] \
      --out docs/status/benclip_after.json
"""

from __future__ import annotations

import argparse
import json
import time

from train.benclip_data import (default_ben_paths, load_caption_map, load_targets,
                                split_patches)
from train.benclip_eval import benclip_embedder, evaluate_retrieval_and_probe


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the trained benclip 'after'.")
    ap.add_argument("--checkpoint", default="checkpoints/benclip")
    ap.add_argument("--train-cap", type=int, default=None)
    ap.add_argument("--test-cap", type=int, default=None)
    ap.add_argument("--out", default="docs/status/benclip_after.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from satquery.adapters.benclip import load_benclip

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    paths = default_ben_paths()
    patches = load_targets(paths["targets"])
    all_ids = {p["patch_id"] for p in patches}
    caption_map = load_caption_map(paths["parquet"], all_ids)

    print(f"[benclip-after] loading checkpoint {args.checkpoint} (device={device})")
    bc = load_benclip(args.checkpoint, device=device)
    print(f"[benclip-after] base={bc.meta.get('base_model_id')} "
          f"n_channels={bc.meta.get('num_channels')}")

    embedder = benclip_embedder(bc)
    print(f"[benclip-after] train split: {len(split_patches(patches,'train'))}, "
          f"test: {len(split_patches(patches,'test'))}")

    t0 = time.time()
    results = evaluate_retrieval_and_probe(
        embedder, patches, caption_map,
        batch_size=8, train_cap=args.train_cap, test_cap=args.test_cap,
    )
    results["encoder"] = "benclip (14-channel adapted)"
    results["input_bands"] = "12 S2 + 2 S1 (14 channels)"
    results["seconds"] = round(time.time() - t0, 1)

    print(json.dumps(results, indent=2))
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[benclip-after] wrote {args.out}")


if __name__ == "__main__":
    main()
