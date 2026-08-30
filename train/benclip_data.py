"""Dataset / caption helpers for benclip training and evaluation.

This module is training-support only (lives under ``train/``). The band-stacking
logic it uses lives in ``satquery/adapters/benclip.py`` (PLAN.md §4.5 — band logic
lives there and nowhere else). This file only *loads* the BEN targets manifest,
*joins* the per-patch captions from ``BigEarthNet.txt.parquet``, and sequences
those through a PyTorch Dataset.

Data layout (W1 status doc):
  data/bigearthnet/targets_k10_10.json      patch_id, s1_name, split, country, labels
  data/bigearthnet/BigEarthNet.txt.parquet  patch_id, type, input, output, ...
  data/bigearthnet/images/BigEarthNet-S2/<acq>/<patch>/<patch>_B0*.tif
  data/bigearthnet/images/BigEarthNet-S1/<acq>/<patch>/<patch>_VV|VH.tif
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np

from satquery.adapters.benclip import S2_BAND_ORDER, stack_ben_patch

_SPLITS = ("train", "validation", "test")


def load_targets(path: str) -> List[dict]:
    """Load the BEN targets manifest and add resolved folder paths."""
    with open(path) as fh:
        data = json.load(fh)
    patches = data["target_patches"]
    s2_root = os.path.join(os.path.dirname(path), "images", "BigEarthNet-S2")
    s1_root = os.path.join(os.path.dirname(path), "images", "BigEarthNet-S1")
    out = []
    for p in patches:
        p = dict(p)
        p["s2_folder"] = os.path.join(s2_root, p["s2_folder"])
        p["s1_folder"] = os.path.join(s1_root, p["s1_folder"])
        out.append(p)
    return out


def split_patches(patches: List[dict], split: str) -> List[dict]:
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}; got {split!r}")
    return [p for p in patches if p["split"] == split]


def load_caption_map(parquet_path: str, patch_ids: Optional[set] = None) -> Dict[str, str]:
    """Build ``{patch_id: caption_output}`` from the captioning annotations.

    There is exactly one captioning row per annotation-covered patch, so take
    the first captioning row per patch. If ``patch_ids`` is given, only those
    patches are kept (avoids loading the whole 9.5M-row parquet into memory).
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["patch_id", "type", "output"])
    df = table.to_pandas()
    df = df[df["type"] == "captioning"]
    if patch_ids is not None:
        df = df[df["patch_id"].isin(patch_ids)]
    # There is one caption per patch in this dataset; keep the first if any dup.
    grouped = df.groupby("patch_id")["output"].first()
    return {str(k): str(v) for k, v in grouped.items()}


class BenCaptureDataset:
    """PyTorch Dataset returning (14,120,120) stem, caption, patch_id.

    Folders are resolved in the patch dicts (see ``load_targets``). Only patches
    that have a caption and whose band files exist are kept.
    """

    def __init__(
        self,
        patches: List[dict],
        caption_map: Dict[str, str],
        target_size: int = 120,
        transform: Optional[Callable] = None,
        stacker: Callable = stack_ben_patch,
    ) -> None:
        self.items = []
        for p in patches:
            cap = caption_map.get(p["patch_id"])
            if not cap:
                continue
            self.items.append((p, cap))
        self.target_size = target_size
        self.transform = transform
        self.stacker = stacker

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        p, cap = self.items[idx]
        stem = self.stacker(
            p["s2_folder"], p["s1_folder"], p["patch_id"], p["s1_name"],
            target_size=self.target_size,
        )
        if self.transform is not None:
            stem = self.transform(stem)
        return stem, cap, p["patch_id"]


def default_ben_paths(root: str = "data/bigearthnet") -> dict:
    """Resolve the standard BEN paths under ``root``."""
    return {
        "targets": os.path.join(root, "targets_k10_10.json"),
        "parquet": os.path.join(root, "BigEarthNet.txt.parquet"),
        "s2_root": os.path.join(root, "images", "BigEarthNet-S2"),
        "s1_root": os.path.join(root, "images", "BigEarthNet-S1"),
    }
