"""benclip — a BigEarthNet-adapted multi-sensor image–text encoder (PLAN.md §3.1 Track A).

W2 owns this module and the §4.5 band-mapping policy that lives entirely here.
W3, W4 and W5 never touch band logic; they hand this module a `RasterInput` and
get back an embedding or a labelled ``predict_labels`` payload.

Band-mapping policy (PLAN.md §4.5, all of it lives here):
  - RGB (3-band)          -> S2 slots B04/B03/B02
  - PAN (1-band)          -> replicated across B04/B03/B02
  - Cartosat MSI (4-band) -> B02/B03/B04/B08 (blue/green/red/NIR)
  - SAR (1-band)          -> VV slot
  - SAR (2-band)          -> VV/VH slots
  - all other slots, plus any the input cannot supply, are filled with the
    PER-CHANNEL TRAINING MEAN (never zeros) — see ``training_means``.
  - a 14-channel input (an already-stacked BEN patch, see ``stack_ben_patch``)
    passes straight through channel-for-channel.
  - ``source_modality`` may be "unknown": then we map by band count alone and
    fail soft, returning the labels with the uncertainty disclosed.

The model is a CLIP ViT-B/32 with its 3-channel patch-embedding stem replaced by
a 14-channel stem; new channels are initialised from the mean of the pretrained
RGB weights (``build_benclip_model``). Training applies LoRA to the vision tower
and freezes the text tower (``train/train_benclip.py``), and the resulting
checkpoint stores the fully-merged adapted vision tower plus the per-channel
statistics and the text embeddings used for label prediction — so
``load_benclip`` needs nothing beyond torch/transformers at inference time.

NOTE on the Cartosat-2S 4-band order (B02/B03/B04/B08): this is the UNVERIFIED
assumption written into PLAN.md §4.5 and mirrored by satquery/io/raster.py. It
has NOT been independently verified against a real Bhoonidhi product (W1's
registration is still in progress). Do not present it as established fact.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # pragma: no cover - doc-only import guard
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    CLIPModel = None  # type: ignore[assignment,misc]
    CLIPProcessor = None  # type: ignore[assignment,misc]

from satquery.contracts import validate_band_mapping

# --------------------------------------------------------------------------- #
# Band slot ordering — the single source of truth for the 14-channel stem.
# --------------------------------------------------------------------------- #

# Sentinel-2 L2A bands as shipped by BigEarthNet v2.0. B10 (cirrus) is absent
# from L2A, so the 12 S2 bands are B01..B09, B11, B12, B8A.
S2_BAND_ORDER: List[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A",
]
S1_BAND_ORDER: List[str] = ["VV", "VH"]
BEN_14_SLOTS: List[str] = S2_BAND_ORDER + S1_BAND_ORDER  # 14 slots, stem input order

# Sentinel-1 band files as named on disk.
_S1_DISK_NAMES: List[str] = ["VV", "VH"]

# The three S2 spatial resolutions in the slice (W1 status doc): 10 m = 120,
# 20 m = 60, 60 m = 20. All stacks are resampled up to the 10 m grid.
_S2_RES_10M = {"B02", "B03", "B04", "B08"}
_S2_RES_20M = {"B05", "B06", "B07", "B11", "B12", "B8A"}
_S2_RES_60M = {"B01", "B09"}


def _default_mean(band: str) -> float:
    """Fallback per-channel mean (only used before training has computed one)."""
    return 900.0 if band not in S1_BAND_ORDER else 0.08


def _default_std(band: str) -> float:
    """Fallback per-channel std (only used before training has computed one)."""
    return 900.0 if band not in S1_BAND_ORDER else 0.04


@dataclass
class BenClipModel:
    """Loaded benclip: a CLIPModel with a 14-channel stem plus the runtime
    policy state (per-channel stats, label text embeddings, meta).

    This is the object ``load_benclip`` returns and the one W3/W4/W5 feed a
    ``RasterInput`` to. Band logic lives here and only here.
    """

    model: Any
    processor: Any
    training_means: Dict[str, float]
    training_stds: Dict[str, float]
    class_text_embeddings: Optional[np.ndarray] = None  # (n_classes, proj_dim)
    class_names: Optional[List[str]] = None
    device: str = "cpu"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenClipConfig:
    """Configuration for building training/loading a benclip model."""

    base_model_id: str = "openai/clip-vit-base-patch32"
    num_channels: int = 14
    input_size: int = 224          # CLIP ViT-B/32 native input
    lora_r: int = 8
    lora_alpha: int = 16
    freeze_text: bool = True
    base_path: Optional[str] = None  # locally-cached base CLIP checkpoint


# --------------------------------------------------------------------------- #
# BEN patch stacking (band logic lives here, nowhere else — PLAN.md §4.5)
# --------------------------------------------------------------------------- #


def stack_ben_patch(s2_folder: str, s1_folder: str, patch_id: str, s1_name: str,
                    target_size: int = 120) -> np.ndarray:
    """Stack one labelled BEN S1+S2 patch into a (14, target_size, target_size) array.

    BigEarthNet v2.0 stores one band per GeoTIFF at three resolutions
    (PLAN.md §2.2b / W1 status doc). This reads the 12 S2 GeoTIFFs and the
    VV/VH S1 GeoTIFFs and resamples the 20 m (60) and 60 m (20) S2 bands up to
    the 10 m (120) grid with bilinear interpolation, returning the 14 channels
    in ``BEN_14_SLOTS`` order. S1 float32 amplitude passes through unchanged
    (no dB scaling — that is a display concern owned by satquery.io.raster, and
    the encoder is trained on raw amplitude).
    """
    import rasterio
    from rasterio.enums import Resampling as RstResampling

    channels = np.zeros(
        (len(BEN_14_SLOTS), target_size, target_size), dtype=np.float32
    )
    patch_dir = os.path.join(s2_folder, patch_id)
    for i, band in enumerate(S2_BAND_ORDER):
        path = os.path.join(patch_dir, f"{patch_id}_{band}.tif")
        if not os.path.exists(path):
            raise FileNotFoundError(f"BEN S2 band file not found: {path}")
        with rasterio.open(path) as src:
            arr = src.read(1)
            if arr.shape == (target_size, target_size):
                channels[i] = arr.astype(np.float32)
            else:
                channels[i] = src.read(
                    1, out_shape=(1, target_size, target_size),
                    resampling=RstResampling.bilinear,
                )[0].astype(np.float32)

    s1_patch_dir = os.path.join(s1_folder, s1_name)
    for j, disk in enumerate(_S1_DISK_NAMES):
        band = S1_BAND_ORDER[j]
        path = os.path.join(s1_patch_dir, f"{s1_name}_{disk}.tif")
        if not os.path.exists(path):
            raise FileNotFoundError(f"BEN S1 band file not found: {path}")
        with rasterio.open(path) as src:
            arr = src.read(1)
            if arr.shape == (target_size, target_size):
                channels[len(S2_BAND_ORDER) + j] = arr.astype(np.float32)
            else:
                channels[len(S2_BAND_ORDER) + j] = src.read(
                    1, out_shape=(1, target_size, target_size),
                    resampling=RstResampling.bilinear,
                )[0].astype(np.float32)
    return channels


def compute_channel_stats(patches: List[dict],
                          sample: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """Compute per-channel mean/std (14 slots) over a list of BEN patches.

    ``patches`` entries need ``patch_id``/``s1_name``/``s2_folder``/``s1_folder``
    (the shape produced by ``train/benclip_data.load_targets`` — the folders are
    resolved inside each patch dict). This is the fill strategy for absent slots
    (mean) and the per-channel standardisation used by
    ``_normalize_stem_to_model``. Pass the official TRAIN split; ``sample``
    limits how many patches are read.
    """
    if sample is not None and sample < len(patches):
        patches = patches[:sample]
    sums = np.zeros(len(BEN_14_SLOTS), dtype=np.float64)
    sumsq = np.zeros(len(BEN_14_SLOTS), dtype=np.float64)
    valid = np.zeros(len(BEN_14_SLOTS), dtype=np.int64)
    for p in patches:
        stacked = stack_ben_patch(
            p["s2_folder"], p["s1_folder"], p["patch_id"], p["s1_name"]
        )
        flat = stacked.reshape(len(BEN_14_SLOTS), -1).astype(np.float64)
        finite = np.isfinite(flat)
        for c in range(len(BEN_14_SLOTS)):
            vals = flat[c][finite[c]]
            if vals.size:
                sums[c] += float(vals.sum())
                sumsq[c] += float((vals * vals).sum())
                valid[c] += int(vals.size)
    out: Dict[str, Dict[str, float]] = {}
    for i, band in enumerate(BEN_14_SLOTS):
        n = valid[i]
        if n == 0:
            out[band] = {"mean": _default_mean(band), "std": _default_std(band)}
            continue
        mean = sums[i] / n
        var = max(sumsq[i] / n - mean * mean, 0.0)
        out[band] = {"mean": float(mean), "std": float(np.sqrt(var))}
    return out


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #


def _build_stem(reference_weight: torch.Tensor, num_channels: int) -> nn.Conv2d:
    """14-channel patch-embedding stem; new channels init from mean of RGB.

    ``reference_weight`` is the pretrained CLIP Conv2d.weight of shape
    (embed_dim, 3, patch, patch). Stem channels 0/1/2 map to B04/B03/B02 ->
    R/G/B and are copied exactly; channels >= 3 are initialised from the mean
    of the three RGB channels so no input pathway is zero-initialised.
    """
    embed_dim, _, ps, _ = reference_weight.shape
    stem = nn.Conv2d(num_channels, embed_dim, kernel_size=ps, stride=ps, bias=False)
    with torch.no_grad():
        new_weight = stem.weight.clone()  # (embed_dim, num_channels, ps, ps)
        for c in range(embed_dim):
            rgb = reference_weight[c]  # (3, ps, ps)
            new_weight[c, 0:3] = rgb
            new_weight[c, 3:] = rgb.mean(dim=0, keepdim=True).expand(
                num_channels - 3, ps, ps
            )
        stem.weight.copy_(new_weight)
    return stem


def build_benclip_model(config: BenClipConfig = BenClipConfig(),
                        device: Optional[Any] = None) -> Tuple[Any, Dict[str, Any]]:
    """Construct the benclip model from a base CLIP checkpoint.

    Replaces the 3-channel CLIP vision stem with a 14-channel stem and returns
    ``(model, metadata)``. No LoRA is applied here — that happens during
    training and the result is saved merged, so the load-and-infer path never
    needs peft.
    """
    if torch is None:
        raise ImportError(
            "torch/transformers are not importable; benclip cannot build a model"
        )
    source = config.base_path or config.base_model_id
    load_kwargs: Dict[str, Any] = {}
    if config.base_path:
        load_kwargs["local_files_only"] = True
    model = CLIPModel.from_pretrained(source, **load_kwargs)
    model.eval()
    stem = _build_stem(
        model.vision_model.embeddings.patch_embedding.weight, config.num_channels
    )
    model.vision_model.embeddings.patch_embedding = stem
    if device is not None:
        model = model.to(device)
    metadata = {
        "base_model_id": config.base_model_id,
        "base_path": config.base_path,
        "num_channels": config.num_channels,
        "stem_initialization": "rgb_exact_plus_mean_of_rgb_for_extra_channels",
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "freeze_text": config.freeze_text,
    }
    return model, metadata


# --------------------------------------------------------------------------- #
# Checkpoint load
# --------------------------------------------------------------------------- #


def _checkpoint_state_path(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "benclip_state.pt")
    return path


def load_benclip(path: str = "checkpoints/benclip",
                 device: Optional[str] = None) -> BenClipModel:
    """Load a trained benclip checkpoint for embeddings/labels.

    The checkpoint is a ``benclip_state.pt`` whose ``vision_state_dict`` holds
    the fully-merged adapted vision tower (new stem + LoRA baked in), plus the
    per-channel stats and label text embeddings. Loading builds the base CLIP,
    replaces the stem, loads the merged vision weights, and leaves the (frozen)
    base text tower in place — no peft needed at inference time.

    Raises FileNotFoundError if the checkpoint is absent, so a misconfigured
    runtime fails with an actionable message rather than silently degrading
    (modelpool's ``_load_benclip`` depends on this).
    """
    state_path = _checkpoint_state_path(path)
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"benclip checkpoint not found at '{state_path}'. Training has not "
            "produced a checkpoint here yet (see train/train_benclip.py), or "
            "the SATQUERY_BENCLIP_PATH / path argument is wrong."
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)

    config = BenClipConfig(
        base_model_id=state.get("base_model_id", "openai/clip-vit-base-patch32"),
        num_channels=state.get("num_channels", 14),
        lora_r=state.get("lora_r", 8),
        lora_alpha=state.get("lora_alpha", 16),
        freeze_text=state.get("freeze_text", True),
        base_path=state.get("base_path"),
    )
    if device is None:
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    model, _ = build_benclip_model(config, device=device)
    model.vision_model.load_state_dict(state["vision_state_dict"], strict=True)
    model.eval()

    processor = CLIPProcessor.from_pretrained(config.base_path or config.base_model_id)

    stats = state.get("channel_stats", {})
    training_means = {
        b: float(stats.get(b, {}).get("mean", _default_mean(b))) for b in BEN_14_SLOTS
    }
    training_stds = {
        b: float(stats.get(b, {}).get("std", _default_std(b))) for b in BEN_14_SLOTS
    }

    return BenClipModel(
        model=model,
        processor=processor,
        training_means=training_means,
        training_stds=training_stds,
        class_text_embeddings=(
            np.asarray(state["class_text_embeddings"], dtype=np.float32)
            if "class_text_embeddings" in state else None
        ),
        class_names=list(state["class_names"]) if "class_names" in state else None,
        device=device,
        meta=state.get("model_metadata", {}),
    )


# --------------------------------------------------------------------------- #
# Runtime band mapping — the §4.5 policy (band logic lives here, nowhere else)
# --------------------------------------------------------------------------- #


def _make_mapping(filled: List[str], absent: List[str],
                  source_modality: str) -> Dict[str, Any]:
    mapping = {
        "slots_filled": sorted(filled),
        "slots_absent": sorted(absent),
        "fill_strategy": "per_channel_training_mean",
        "source_modality": source_modality,
    }
    return validate_band_mapping(mapping)


def _map_source_modality_by_band(band_count: int) -> str:
    """Band-count-only source_modality for the fail-soft ("unknown") path.
    Mirrors satquery.io.modality's band-count tier."""
    if band_count == 3:
        return "optical"
    if band_count in (4, 12, 13, 14):
        return "msi"
    return "unknown"


def _raster_to_stem(bc: BenClipModel, raster) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """Map a RasterInput to a (14, H, W) stem using the §4.5 policy.

    Returns ``(stem_array, band_mapping, source_modality)``. This is the single
    routine implementing the band-mapping contract; all public entry points
    (``predict_labels``, ``embed_optical``, ``embed_sar``) route through it.
    """
    arr = np.asarray(raster.array, dtype=np.float32)  # (H, W, C)
    if arr.ndim != 3:
        raise ValueError(f"benclip expects (H, W, C); got {arr.shape}")
    band_count = arr.shape[2]
    H, W = arr.shape[0], arr.shape[1]
    resolved_modality = str(getattr(raster, "modality", "unknown") or "unknown")

    # A 14-channel input already in BEN slot order passes straight through.
    if band_count == 14:
        stem = np.transpose(arr, (2, 0, 1))
        src = resolved_modality if resolved_modality != "unknown" else "msi"
        mapping = _make_mapping(list(BEN_14_SLOTS), [], src)
        return stem, mapping, src

    stem = np.zeros((14, H, W), dtype=np.float32)
    filled: List[str] = []

    def fill(slot: str, ch: np.ndarray) -> None:
        stem[BEN_14_SLOTS.index(slot)] = ch.astype(np.float32)
        filled.append(slot)

    if resolved_modality == "sar":
        # Give BV of a SAR 1-band -> VV; SAR 2-band -> VV/VH.
        if band_count >= 1:
            fill("VV", arr[..., 0])
        if band_count >= 2:
            fill("VH", arr[..., 1])
        src = "sar"
    elif band_count == 3:
        # RGB -> B04/B03/B02 (R/G/B).
        fill("B04", arr[..., 0])
        fill("B03", arr[..., 1])
        fill("B02", arr[..., 2])
        src = _map_source_modality_by_band(band_count)
    elif band_count == 4:
        # Cartosat MSI order assumption: B02/B03/B04/B08 (unverified, §4.5).
        fill("B02", arr[..., 0])
        fill("B03", arr[..., 1])
        fill("B04", arr[..., 2])
        fill("B08", arr[..., 3])
        src = _map_source_modality_by_band(band_count)
    elif band_count == 1:
        # PAN (or unknown 1-band): replicate across B04/B03/B02.
        ch = arr[..., 0]
        fill("B04", ch)
        fill("B03", ch)
        fill("B02", ch)
        src = _map_source_modality_by_band(band_count)
    elif band_count == 2:
        # Unknown 2-band: ambiguous SAR/PAN; fill VV/VH (SAR-shaped guess).
        fill("VV", arr[..., 0])
        fill("VH", arr[..., 1])
        src = _map_source_modality_by_band(band_count)
    else:
        # 12/13-band Sentinel-like stack: map B02/B03/B04/B08 if present.
        if band_count >= 4:
            fill("B02", arr[..., 0])
            fill("B03", arr[..., 1])
            fill("B04", arr[..., 2])
            fill("B08", arr[..., 3])
        src = _map_source_modality_by_band(band_count)

    # Fill absent slots with per-channel training mean (never zeros).
    absent = [b for b in BEN_14_SLOTS if b not in filled]
    for slot in absent:
        stem[BEN_14_SLOTS.index(slot)] = float(bc.training_means.get(slot, 0.0))

    mapping = _make_mapping(filled, absent, src)
    return stem, mapping, src


def _normalize_stem_to_model(bc: BenClipModel, stem: np.ndarray) -> torch.Tensor:
    """Resize + per-channel standardise a (14, H, W) stem to a
    (1, 14, input_size, input_size) torch tensor.

    The 14 bands span wildly different scales (S2 uint16 reflectances vs S1
    float32 amplitude). Each channel is standardised to zero mean / unit
    variance with the checkpoint's per-channel stats, then scaled to a fixed
    amplitude compatible with the pretrained CLIP vision distribution and
    clamped to [-1, 1]. Training uses the identical transform, so inference
    sees the same distribution the model was adapted to.
    """
    t = torch.from_numpy(stem).float().unsqueeze(0)
    size = bc.meta.get("input_size", 224)
    t = torch.nn.functional.interpolate(
        t, size=(size, size), mode="bilinear", align_corners=False
    )
    for i, band in enumerate(BEN_14_SLOTS):
        mu = float(bc.training_means.get(band, 0.0))
        sd = float(bc.training_stds.get(band, 1.0))
        if sd == 0:
            sd = 1.0
        t[:, i] = (t[:, i] - mu) / sd
    t = t / 3.0
    return t.clamp(-1.0, 1.0)


def _get_pooled(output: Any) -> torch.Tensor:
    """Extract the pooled (CLS-style) embedding from a transformers 5.x CLIP
    forward output. In transformers 5.x ``get_image_features`` returns a
    ``BaseModelOutputWithPooling``; in older revisions it returned a raw
    tensor. Accept both so the adapter works across the pinned version."""
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    return output


def _image_embedding(bc: BenClipModel, stem: np.ndarray) -> np.ndarray:
    """Vision-encode a (14, H, W) stem -> unit-length 1-D numpy embedding."""
    t = _normalize_stem_to_model(bc, stem).to(bc.device)
    with torch.no_grad():
        feats = _get_pooled(bc.model.get_image_features(pixel_values=t))
    feats = feats.float().cpu().numpy()[0]  # drop the batch dim -> (D,)
    norm = float(np.linalg.norm(feats))
    if norm == 0:
        norm = 1.0
    return feats / norm


# --------------------------------------------------------------------------- #
# Public runtime API (what W3/W4/W5 call)
# --------------------------------------------------------------------------- #


def _default() -> BenClipModel:
    global _default_model
    if _default_model is None:
        _default_model = load_benclip()
    return _default_model


_default_model: Optional[BenClipModel] = None


def reset_default() -> None:
    """Clear the module singleton (used by tests)."""
    global _default_model
    _default_model = None


def embed_optical(raster, bc: Optional[BenClipModel] = None) -> np.ndarray:
    """Embed an optical/MSI raster into a unit-length image embedding.

    The input is mapped to S2 slots per §4.5. ``bc`` defaults to the module
    singleton. For a non-optical raster the optical slots degrade to per-channel
    means but never fail.
    """
    if bc is None:
        bc = _default()
    stem, _mapping, _src = _raster_to_stem(bc, raster)
    return _image_embedding(bc, stem)


def embed_sar(raster, bc: Optional[BenClipModel] = None) -> np.ndarray:
    """Embed a SAR raster into a unit-length image embedding.

    Maps SAR bands to VV/VH per §4.5. ``bc`` defaults to the module singleton.
    For a non-SAR raster the SAR slots degrade to per-channel means but never
    fail.
    """
    if bc is None:
        bc = _default()
    stem, _mapping, _src = _raster_to_stem(bc, raster)
    return _image_embedding(bc, stem)


def predict_labels(raster, bc: Optional[BenClipModel] = None,
                   top_k: int = 5) -> Dict[str, Any]:
    """Return per-class land-cover scores plus the §4.5 band_mapping payload.

    Args:
        raster: any ``RasterInput`` (1/2/3/4/12/13/14-band).
        bc: an already-loaded BenClipModel, or None for the default singleton.
        top_k: how many labels to return in ``labels``.

    Returns a dict conforming to PLAN.md §4.5 / satquery.contracts:
    ``{"labels": [{"label": str, "score": float}], "band_mapping": {...}}``.
    Every number is coerced with ``float()`` and ``band_mapping`` passes
    ``validate_band_mapping``. ``source_modality`` may be "unknown" — then we
    map by band count, return the labels, and disclose the uncertainty.
    """
    if bc is None:
        bc = _default()
    stem, mapping, _src = _raster_to_stem(bc, raster)
    feat = _image_embedding(bc, stem)

    if bc.class_text_embeddings is None:
        text_emb = _zero_shot_class_embeddings(bc)
    else:
        text_emb = bc.class_text_embeddings

    text_emb = text_emb.astype(np.float32)
    norms = np.linalg.norm(text_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    text_emb = text_emb / norms

    logits = (text_emb @ feat).ravel()  # (n_classes,), cosine sim
    order = np.argsort(-logits)[:top_k]
    class_names = bc.class_names or _BEN_CLASS_NAMES

    labels = []
    for j in order:
        name = class_names[j] if j < len(class_names) else f"class_{j}"
        labels.append({"label": str(name), "score": float(logits[j])})

    return {"labels": labels, "band_mapping": mapping}


def _zero_shot_class_embeddings(bc: BenClipModel) -> np.ndarray:
    """19-class CLIP text embeddings from BEN class names (fallback when the
    checkpoint has no trained text embeddings)."""
    names = bc.class_names or _BEN_CLASS_NAMES
    prompts = [f"a satellite image of {n.lower()}" for n in names]
    with torch.no_grad():
        tokens = bc.processor(
            text=prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(bc.device)
        text_feats = _get_pooled(bc.model.get_text_features(**tokens))
    text_feats = text_feats.float().cpu().numpy()
    norms = np.linalg.norm(text_feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return text_feats / norms


_BEN_CLASS_NAMES: List[str] = [
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
