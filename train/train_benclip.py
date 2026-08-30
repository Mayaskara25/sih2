#!/usr/bin/env python3
"""Train benclip: a 14-channel CLIP ViT-B/32 adapted to multi-sensor RS data.

PLAN.md §3.1 Track A:
  - Replace the 3-channel CLIP stem with a 14-channel stem (12 S2 + 2 S1),
    new channels initialised from the mean of the pretrained RGB weights.
  - Train the new stem fully; LoRA the rest of the vision tower; freeze text.
  - Contrastive (InfoNCE) loss against BigEarthNet.txt captions.

Design notes:
  - The text tower is FROZEN, so the caption text embeddings for the whole
    training subset are precomputed once and cached to disk; each training step
    therefore only runs the (adapted) vision tower + the contrastive sum. This
    is what makes a full 6k-patch T4 run finish in one session, and it keeps the
    local 3.9 GB card viable for a small smoke run.
  - 14-channel stems are stacked once and cached as .npy so epochs reuse them.
  - Checkpoints are the fully-MERGED adapted vision tower + new stem + channel
    stats + label text embeddings, loadable by
    satquery.adapters.benclip.load_benclip with no peft at inference time.
  - Every ``--ckpt-every`` steps writes a checkpoint so a dropped Colab/Kaggle
    session costs minutes, not hours.

Usage (local smoke, small):
  python -m train.train_benclip --n-train 300 --epochs 2 --batch-size 8 \
      --checkpoint-dir checkpoints/benclip --device cuda

Usage (Colab/T4, full):
  python -m train.train_benclip --n-train 6000 --epochs 10 --batch-size 64 \
      --checkpoint-dir /content/drive/MyDrive/benclip --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from satquery.adapters.benclip import (
    BEN_14_SLOTS,
    S1_BAND_ORDER,
    BenClipConfig,
    _default_mean,
    _default_std,
    _get_pooled,
    build_benclip_model,
    compute_channel_stats,
    stack_ben_patch,
)
from train.benclip_data import (default_ben_paths, load_caption_map,
                                load_targets, split_patches)

BEN_CLASS_NAMES: List[str] = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]


# --------------------------------------------------------------------------- #
# Data: cache 14-channel stems to disk (stacked once), precompute text embeddings
# --------------------------------------------------------------------------- #


def _cache_subset(patches: List[dict], caption_map: Dict[str, str],
                  cache_dir: str) -> List[int]:
    """Stack the 14-channel stems to disk as .npy and return the list of item
    indices (those with a caption). Returns indices into ``patches`` that are
    usable for training, in a stable order, with the caption alongside."""
    os.makedirs(cache_dir, exist_ok=True)
    usable = [i for i, p in enumerate(patches) if p["patch_id"] in caption_map]
    meta_path = os.path.join(cache_dir, "order.json")
    done = set()
    if os.path.exists(meta_path):
        try:
            done = set(json.load(open(meta_path))["cached"])
        except Exception:
            done = set()
    for rank, i in enumerate(usable):
        name = f"stem_{rank}.npy"
        if name not in done:
            p = patches[i]
            stem = stack_ben_patch(p["s2_folder"], p["s1_folder"],
                                   p["patch_id"], p["s1_name"])
            np.save(os.path.join(cache_dir, name), stem)
            done.add(name)
    with open(meta_path, "w") as fh:
        json.dump({"cached": sorted(done), "n": len(usable),
                   "patch_ids": [patches[i]["patch_id"] for i in usable]}, fh)
    return usable


def _load_stem(cache_dir: str, rank: int) -> np.ndarray:
    return np.load(os.path.join(cache_dir, f"stem_{rank}.npy"))


def precompute_text_embeddings(processor, model, texts: List[str],
                               device: str, scale: float) -> np.ndarray:
    """Encode caption strings once through the frozen text tower and scale them
    by ``scale`` (so their norm equals the CLIP temperature-scaled logits), then
    L2-normalise. Returns (N, D)."""
    import torch

    embs = []
    B = 64
    for start in range(0, len(texts), B):
        chunk = texts[start:start + B]
        with torch.no_grad():
            tokens = processor(text=chunk, padding=True, truncation=True,
                               return_tensors="pt").to(device)
            feats = _get_pooled(model.get_text_features(**tokens)).float()
        embs.append(feats)
    text = torch.cat(embs, dim=0) * scale
    text = text / text.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return text


def encode_class_texts(model, processor, class_names: List[str],
                       device: str) -> np.ndarray:
    """Encode the 19 class names through the frozen text tower; unit-length."""
    import torch

    prompts = [f"a satellite image of {n.lower()}" for n in class_names]
    with torch.no_grad():
        tokens = processor(text=prompts, padding=True, truncation=True,
                           return_tensors="pt").to(device)
        feats = _get_pooled(model.get_text_features(**tokens)).float().cpu().numpy()
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return feats / norms


# --------------------------------------------------------------------------- #
# Model + LoRA
# --------------------------------------------------------------------------- #


def apply_lora(model, r: int, alpha: int) -> None:
    """Freeze everything except the new 14-channel stem and LoRA on the VISION
    tower's attention projections; the text tower stays fully frozen."""
    import torch
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha, target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
        lora_dropout=0.05, bias="none",
    ))
    # Everything frozen by default from peft; explicitly unfreeze the stem only.
    for name, param in model.named_parameters():
        if "text_model" in name:
            param.requires_grad = False
        if "patch_embedding" in name:
            param.requires_grad = True
    return model


def stem_normalize(stem: np.ndarray, means: Dict[str, float], stds: Dict[str, float],
                   input_size: int, device: str) -> torch.Tensor:
    """Resize + per-channel standardise a (14, H, W) stem to a model-ready
    (1, 14, input_size, input_size) tensor — IDENTICAL to the inference-time
    transform in satquery.adapters.benclip, so training sees the same
    distribution inference will."""
    import torch
    t = torch.from_numpy(stem).float().unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(input_size, input_size),
                                        mode="bilinear", align_corners=False)
    for i, band in enumerate(BEN_14_SLOTS):
        m = float(means.get(band, _default_mean(band)))
        s = float(stds.get(band, _default_std(band)))
        if s == 0:
            s = 1.0
        t[:, i] = (t[:, i] - m) / s
    t = t / 3.0
    return t.clamp(-1.0, 1.0).to(device)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


def save_merged_checkpoint(peft_model, config: BenClipConfig,
                           means: Dict[str, float], stds: Dict[str, float],
                           class_names: List[str], class_text_emb: np.ndarray,
                           out_dir: str) -> str:
    """Merge LoRA into the vision tower and write a loadable benclip_state.pt."""
    import torch

    merged = peft_model.merge_and_unload()
    merged = merged.base_model if hasattr(merged, "base_model") else merged
    vision_state = {
        k: v.float().cpu().clone() for k, v in merged.vision_model.state_dict().items()
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "benclip_state.pt")
    torch.save({
        "base_model_id": config.base_model_id,
        "base_path": config.base_path,
        "num_channels": config.num_channels,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "freeze_text": config.freeze_text,
        "vision_state_dict": vision_state,
        "channel_stats": {b: {"mean": float(means.get(b, 0.0)),
                              "std": float(stds.get(b, 1.0))} for b in BEN_14_SLOTS},
        "class_names": list(class_names),
        "class_text_embeddings": class_text_emb.astype(np.float32),
        "model_metadata": {"base_model_id": config.base_model_id,
                           "num_channels": config.num_channels,
                           "stem_initialization":
                               "rgb_exact_plus_mean_of_rgb_for_extra_channels",
                           "input_size": config.input_size},
    }, path)
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the benclip 14-channel CLIP encoder.")
    ap.add_argument("--data-root", default="data/bigearthnet")
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--checkpoint-dir", default="checkpoints/benclip")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--stats-sample", type=int, default=200)
    ap.add_argument("--base-path", default=None)
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(args.seed)

    paths = default_ben_paths(args.data_root)
    pre_base = os.path.join(args.data_root, "images")
    cache_dir = os.path.join(args.data_root, "train_cache")
    os.makedirs(cache_dir, exist_ok=True)

    patches = load_targets(paths["targets"])
    train = split_patches(patches, "train")
    rng.shuffle(train)
    train = train[:args.n_train]
    train_ids = {p["patch_id"] for p in train}
    caption_map = load_caption_map(paths["parquet"], train_ids)

    # Channel stats from a sample of the (real) train patches. The patch dicts
    # from load_targets already carry absolute s2_folder/s1_folder paths.
    stats_sample = [p for p in train if p["patch_id"] in caption_map][:args.stats_sample]
    stats = compute_channel_stats(stats_sample)
    means = {b: stats[b]["mean"] for b in BEN_14_SLOTS}
    stds = {b: stats[b]["std"] for b in BEN_14_SLOTS}

    # Stack stems once (cached), keep captions in the same order.
    usable = _cache_subset(train, caption_map, cache_dir)
    caption_order = [caption_map[train[i]["patch_id"]] for i in usable]

    config = BenClipConfig(base_path=args.base_path, lora_r=args.lora_r,
                           lora_alpha=args.lora_alpha)
    print(f"[benclip-train] base={config.base_model_id} n_train={len(usable)} "
          f"device={device}")

    model, _ = build_benclip_model(config, device=device)
    processor = model.processor if hasattr(model, "processor") else None
    if processor is None:
        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained(config.base_path or config.base_model_id)
    model.processor = processor  # type: ignore[attr-defined]
    model.config = model.config  # type: ignore[attr-defined]

    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    # Precompute caption text embeddings ONCE (text tower frozen).
    print("[benclip-train] precomputing caption text embeddings...")
    text_emb = precompute_text_embeddings(
        processor, model, caption_order, device, args.scale
    )  # (N, D) tensor on device

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"[benclip-train] trainable params: {len(trainable)} tensors, "
          f"{sum(p.numel() for p in trainable)} params")

    n = len(usable)
    per_epoch = max(n // args.batch_size, 1)
    total_steps = per_epoch * args.epochs
    global_step = 0
    t0 = time.time()

    def run_step(batch_idx: List[int]) -> float:
        nonlocal global_step
        opt.zero_grad()
        img_feats = []
        for rank in batch_idx:
            stem = _load_stem(cache_dir, rank)
            inp = stem_normalize(stem, means, stds, config.input_size, device)
            out = model.get_image_features(pixel_values=inp)
            img_feats.append(_get_pooled(out))
        img = torch.cat(img_feats, dim=0)
        img = img / img.norm(dim=1, keepdim=True).clamp_min(1e-6)
        txt = text_emb[batch_idx]
        logits = img @ txt.T  # already temperature-scaled into text_emb
        labels = torch.arange(len(batch_idx), device=device)
        loss = (torch.nn.functional.cross_entropy(logits, labels)
                + torch.nn.functional.cross_entropy(logits.T, labels)) / 2
        loss.backward()
        opt.step()
        global_step += 1
        return float(loss.item())

    for epoch in range(args.epochs):
        order = np.random.RandomState(args.seed + epoch).permutation(n)
        for start in range(0, n, args.batch_size):
            batch = order[start:start + args.batch_size].tolist()
            loss = run_step(batch)
            if global_step % 10 == 0 or global_step == total_steps:
                avg = time.time() - t0
                print(f"[benclip-train] epoch {epoch} step {global_step}/{total_steps} "
                      f"loss={loss:.4f} elapsed={avg:.0f}s")
            if global_step % args.ckpt_every == 0:
                _ = checkpoint(model, config, means, stds, processor, args, tag=f"step{global_step}")

    final = checkpoint(model, config, means, stds, processor, args, tag="final", final=True)
    print(f"[benclip-train] done in {time.time()-t0:.0f}s; saved {final}")


def checkpoint(model, config, means, stds, processor, args, tag="", final=False) -> str:
    """Save a merged checkpoint; the final one goes at checkpoint-dir root."""
    class_text_emb = encode_class_texts(model, processor, BEN_CLASS_NAMES, args.device)
    out_dir = args.checkpoint_dir if final else os.path.join(args.checkpoint_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    path = save_merged_checkpoint(model, config, means, stds, BEN_CLASS_NAMES,
                                  class_text_emb, out_dir)
    print(f"[benclip-train] saved checkpoint {path}")
    return path


if __name__ == "__main__":
    main()
