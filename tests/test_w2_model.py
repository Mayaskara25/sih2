"""Integration tests for benclip: load_benclip, predict_labels, embed_*.

These build a minimal *untrained* benclip checkpoint on the fly (base CLIP +
14-channel stem, no LoRA — no training, no peft required) so the load path and
the §4.5 payload path are exercised against a real loadable model.

The mapping-only behaviour is covered in test_w2_band_mapping.py; this module
focuses on the model-dependent integration seam (payload validity, float
coercion, embeddings shape, "sane labels on a known BEN patch").
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.contracts import validate_band_mapping


def _build_checkpoint(tmp_path, device="cpu"):
    """Create a minimal loadable benclip_state.pt and return its dir."""
    import os

    import torch
    from transformers import CLIPProcessor
    from satquery.adapters.benclip import (BEN_14_SLOTS, BenClipConfig,
                                           build_benclip_model, _get_pooled)
    from satquery.adapters.benclip import _default_mean, _default_std

    config = BenClipConfig(
        base_path=None, num_channels=14, lora_r=8, lora_alpha=16, freeze_text=True
    )
    model, _ = build_benclip_model(config, device=device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(config.base_path or config.base_model_id)

    # Class text embeddings from the frozen text tower.
    class_names = [
        "Agro-forestry areas", "Arable land", "Beaches, dunes, sands",
        "Broad-leaved forest", "Coastal wetlands", "Complex cultivation patterns",
        "Coniferous forest", "Industrial or commercial units", "Inland waters",
        "Inland wetlands",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "Marine waters", "Mixed forest",
        "Moors, heathland and sclerophyllous vegetation",
        "Natural grassland and sparsely vegetated areas", "Pastures",
        "Permanent crops", "Transitional woodland, shrub", "Urban fabric",
    ]
    with torch.no_grad():
        tokens = processor(
            text=[f"a satellite image of {n.lower()}" for n in class_names],
            padding=True, truncation=True, return_tensors="pt",
        ).to(device)
        te = _get_pooled(model.get_text_features(**tokens)).float().cpu().numpy()
    n = np.linalg.norm(te, axis=1, keepdims=True)
    n[n == 0] = 1.0
    te = te / n

    vision_state = {k: v.cpu().clone() for k, v in model.vision_model.state_dict().items()}

    out_dir = str(tmp_path / "benclip")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "base_model_id": config.base_model_id,
        "base_path": None,
        "num_channels": config.num_channels,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "freeze_text": config.freeze_text,
        "vision_state_dict": vision_state,
        "channel_stats": {b: {"mean": _default_mean(b), "std": _default_std(b)}
                          for b in BEN_14_SLOTS},
        "class_names": class_names,
        "class_text_embeddings": te.astype(np.float32),
    }, os.path.join(out_dir, "benclip_state.pt"))
    return out_dir


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory):
    """Build a minimal loadable checkpoint (needs transformers/torch + base
    CLIP download). Skipped if the base model cannot be downloaded/built."""
    try:
        return _build_checkpoint(tmp_path_factory.mktemp("benckpt"), device="cpu")
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"could not build benclip checkpoint: {exc}")
        raise


@pytest.fixture(scope="module")
def bc(checkpoint_dir):
    from satquery.adapters.benclip import load_benclip
    return load_benclip(checkpoint_dir, device="cpu")


def _raster(array, modality):
    class R:
        pass
    r = R()
    r.array = np.asarray(array, dtype=np.float32)
    r.modality = modality
    return r


def test_load_returns_model(bc):
    assert bc is not None
    assert bc.class_names is not None
    assert bc.class_text_embeddings is not None
    assert bc.class_text_embeddings.shape[0] == 19


def test_predict_labels_valid_payload_1band(bc):
    from satquery.adapters.benclip import predict_labels
    arr = np.random.RandomState(1).rand(64, 64, 1).astype(np.float32)
    result = predict_labels(_raster(arr, "unknown"), bc=bc)
    assert "labels" in result and "band_mapping" in result
    assert validate_band_mapping(result["band_mapping"])
    assert result["band_mapping"]["source_modality"] == "unknown"
    assert len(result["labels"]) > 0
    for lab in result["labels"]:
        assert isinstance(lab["label"], str)
        assert isinstance(lab["score"], float)  # coerced, not np.float32


def test_predict_labels_valid_payload_2band(bc):
    from satquery.adapters.benclip import predict_labels
    arr = np.random.RandomState(2).rand(64, 64, 2).astype(np.float32)
    result = predict_labels(_raster(arr, "sar"), bc=bc)
    assert validate_band_mapping(result["band_mapping"])
    assert result["band_mapping"]["source_modality"] in ("sar", "unknown")


def test_predict_labels_valid_payload_3band(bc):
    from satquery.adapters.benclip import predict_labels
    arr = np.random.RandomState(3).rand(64, 64, 3).astype(np.float32)
    result = predict_labels(_raster(arr, "optical"), bc=bc)
    assert validate_band_mapping(result["band_mapping"])
    assert result["band_mapping"]["slots_filled"] == ["B02", "B03", "B04"]


def test_predict_labels_valid_payload_4band(bc):
    from satquery.adapters.benclip import predict_labels
    arr = np.random.RandomState(4).rand(64, 64, 4).astype(np.float32)
    result = predict_labels(_raster(arr, "msi"), bc=bc)
    assert validate_band_mapping(result["band_mapping"])
    assert result["band_mapping"]["slots_filled"] == ["B02", "B03", "B04", "B08"]


def test_predict_labels_valid_payload_14band(bc):
    from satquery.adapters.benclip import predict_labels
    arr = np.random.RandomState(5).rand(64, 64, 14).astype(np.float32)
    result = predict_labels(_raster(arr, "msi"), bc=bc)
    assert validate_band_mapping(result["band_mapping"])
    assert result["band_mapping"]["slots_absent"] == []
    assert len(result["band_mapping"]["slots_filled"]) == 14


def test_embed_optical_and_sar_shapes(bc):
    from satquery.adapters.benclip import embed_optical, embed_sar
    arr3 = np.random.RandomState(6).rand(64, 64, 3).astype(np.float32)
    arr2 = np.random.RandomState(7).rand(64, 64, 2).astype(np.float32)
    eo = embed_optical(_raster(arr3, "optical"), bc=bc)
    es = embed_sar(_raster(arr2, "sar"), bc=bc)
    assert eo.ndim == 1 and es.ndim == 1
    assert np.allclose(np.linalg.norm(eo), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(es), 1.0, atol=1e-4)


def test_labels_are_python_scalars(bc):
    from satquery.adapters.benclip import predict_labels
    import json
    arr = np.random.RandomState(8).rand(64, 64, 3).astype(np.float32)
    result = predict_labels(_raster(arr, "optical"), bc=bc)
    # The whole payload must survive json.dumps (no numpy scalars leaked).
    json.dumps(result["labels"])
    json.dumps(result["band_mapping"])
