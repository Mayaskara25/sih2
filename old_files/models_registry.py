"""
SatQuery AI -- Model Registry (Person 5: Single-Image VQA & Text-Guided Grounding)

Unified inference engine exposing the exact JSON/function contract expected by the
Agentic Controller (Pair 2). This module is intentionally provider-agnostic:
it lazily loads models on first use and caches them module-wide so that Person 6's
bi-temporal / cross-modal functions can reuse the same Qwen2-VL model instance
(shared _load_vlm() -- load once, never twice, to avoid CUDA OOM on 6GB VRAM).

Exported functions (contract -- DO NOT change keys):
    run_vqa(image_path, prompt)     -> {"text_response": str, "confidence": float}
    run_grounding(image_path, target)
        -> {"text_response": str,
            "bounding_boxes": [{"label", "box_2d": [ymin, xmin, ymax, xmax], "confidence"}],
            "confidence": float}
"""

from __future__ import annotations

import os
import time
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
# VLM used for VQA / scene captioning (and reused by Person 6 for multi-image
# change & cross-modal text summaries).
VLM_REPO: str = os.environ.get("SATQUERY_VLM", "Qwen/Qwen2-VL-2B-Instruct")

# Prefer the compact -tiny variant for Grounding DINO. On a 6GB laptop GPU this
# is the safe choice; it still handles single-object grounding reliably.
GROUNDING_REPO: str = os.environ.get(
    "SATQUERY_GROUNDING", "IDEA-Research/grounding-dino-tiny"
)

GROUNDING_BOX_THRESHOLD: float = float(os.environ.get("SATQUERY_GROUND_THRESHOLD", "0.18"))
GROUNDING_TEXT_THRESHOLD: float = float(os.environ.get("SATQUERY_GROUND_TEXT_THRESHOLD", "0.18"))

# Sane defaults for short satellite-domain answers.
VLM_MAX_NEW_TOKENS: int = int(os.environ.get("SATQUERY_VLM_MAX_TOKENS", "256"))

# --------------------------------------------------------------------------- #
# GLOBAL LAZY CACHES
# --------------------------------------------------------------------------- #
_VLM: Optional[Any] = None
_VLM_PROCESSOR: Optional[Any] = None
_GROUNDING: Optional[Any] = None
_GROUNDING_PROCESSOR: Optional[Any] = None
_IS_CUDA: Optional[bool] = None


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def _cuda_available() -> bool:
    """Cache CUDA availability so we only probe torch once."""
    global _IS_CUDA
    if _IS_CUDA is None:
        try:
            import torch

            _IS_CUDA = bool(torch.cuda.is_available())
        except Exception:  # torch not importable -> definitely CPU
            _IS_CUDA = False
    return _IS_CUDA


def load_image(image_path: str) -> "Image.Image":
    """
    Load an image into an RGB PIL image.

    Supports common rasters (.tif/.tiff handled by taking the first 3 bands and
    normalizing to 0-255) and any PIL-readable format (.png/.jpg/...).
    Raises FileNotFoundError on a missing path.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()

    if ext in (".tif", ".tiff"):
        try:
            import rasterio

            with rasterio.open(image_path) as src:
                band_count = src.count
                idx = [1, 2, 3] if band_count >= 3 else [1] * min(band_count, 1)
                arr = src.read(idx)  # shape (bands, H, W)
                arr = np.transpose(arr, (1, 2, 0))  # (H, W, bands)
                arr = arr.astype(np.float32)
                lo, hi = np.percentile(arr, (2, 98))
                if hi > lo:
                    arr = (arr - lo) / (hi - lo)
                arr = np.clip(arr, 0, 1)
                arr = (arr * 255).astype(np.uint8)
                if arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                return Image.fromarray(arr).convert("RGB")
        except ImportError:
            warnings.warn("rasterio not installed; falling back to PIL read of TIF.")

    return Image.open(image_path).convert("RGB")


def _confidence_to_float(value: Any) -> float:
    """Safely coerce a model score into a bounded float in [0, 1]."""
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# SHARED VLM LOADER (used by Person 5 AND Person 6)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# SAFE VLM LOADER (Fix for Meta Tensor Error)
# --------------------------------------------------------------------------- #
def _load_vlm() -> Tuple[Any, Any]:
    global _VLM, _VLM_PROCESSOR

    if _VLM is not None and _VLM_PROCESSOR is not None:
        return _VLM, _VLM_PROCESSOR

    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2VLForConditionalGeneration,
    )

    print(f"[SatQuery] Initializing Safe-Load for Qwen2-VL...")

    if _cuda_available():
        # Using 4-bit for 6GB VRAM, but with more stable settings
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        load_kwargs = {
            # "sequential" often avoids the 'meta tensor' bug better than "auto"
            "device_map": "sequential", 
            "quantization_config": quant_config,
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }
    else:
        load_kwargs = {
            "device_map": "cpu",
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
        }

    # Load Model
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(VLM_REPO, **load_kwargs)
        processor = AutoProcessor.from_pretrained(VLM_REPO, trust_remote_code=True)
    except RuntimeError as e:
        if "meta tensors" in str(e):
            print("[SatQuery] Auto-device-map failed. Retrying with explicit CUDA mapping...")
            # FORCE ALL to CUDA (Will only work if you have ~5-6GB free)
            load_kwargs["device_map"] = {"": "cuda:0"} 
            model = Qwen2VLForConditionalGeneration.from_pretrained(VLM_REPO, **load_kwargs)
            processor = AutoProcessor.from_pretrained(VLM_REPO, trust_remote_code=True)
        else:
            raise e

    model.eval()
    _VLM, _VLM_PROCESSOR = model, processor
    return _VLM, _VLM_PROCESSOR


def _run_vlm_generation(image: "Image.Image", prompt: str) -> str:
    """
    Shared low-level Qwen2-VL generation. Returns the raw model text.
    Person 6 reuses this for bi-temporal / cross-modal textual summaries.
    """
    import torch

    model, processor = _load_vlm()

    chat = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    if _cuda_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=VLM_MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    output = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output.strip()


# --------------------------------------------------------------------------- #
# TASK 1 -- VQA & SCENE CAPTIONING
# --------------------------------------------------------------------------- #
def run_vqa(image_path: str, prompt: str) -> Dict[str, Any]:
    """
    Answer a natural-language question (or caption a scene) for a single image.

    Contract:
        {"text_response": str, "confidence": float}
    """
    t0 = time.time()
    image = load_image(image_path)

    # Satellite-domain system-style instruction folded into the user turn so it
    # works with plain masked LM / instruct chat templates.
    system = (
        "You are SatQuery, a remote-sensing vision assistant. You analyse aerial "
        "and satellite imagery (optical and SAR). Answer the user's question about "
        "this satellite image. Be specific about objects, their count, locations, "
        "and land cover. If you are unsure of an exact measurement, say so. "
        "Keep the answer concise (1-4 sentences)."
    )
    full_prompt = f"{system}\n\nUser query: {prompt}"

    # If the prompt does not ask for an image description, and it is not a
    # close-ended object question, fall back to a descriptive caption request.
    if "describe" in prompt.lower() or "caption" in prompt.lower():
        task_hint = "Describe the scene and the dominant land cover / features."
        full_prompt = f"{system}\n\n{task_hint}\nUser query: {prompt}"

    text_response = _run_vlm_generation(image, full_prompt)

    # Heuristic confidence: non-empty, reasonably long, and not an explicit
    # "I cannot see" refusal. Higher length -> more substantive answer.
    conf = 0.85
    if not text_response:
        text_response = "The model produced no answer for this image."
        conf = 0.1
    else:
        words = len(text_response.split())
        if words < 2:
            conf = 0.4
        elif words < 6:
            conf = 0.65
        lowered = text_response.lower()
        if any(w in lowered for w in ("cannot see", "can't see", "unable", "i don't know")):
            conf = 0.3

    return {
        "text_response": text_response,
        "confidence": _confidence_to_float(conf),
    }


# --------------------------------------------------------------------------- #
# GROUNDING DINO LOADER
# --------------------------------------------------------------------------- #
def _load_grounding() -> Tuple[Any, Any]:
    """Lazy, cached loader for Grounding DINO."""
    global _GROUNDING, _GROUNDING_PROCESSOR

    if _GROUNDING is not None and _GROUNDING_PROCESSOR is not None:
        return _GROUNDING, _GROUNDING_PROCESSOR

    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    print(f"[SatQuery] Loading Grounding DINO: {GROUNDING_REPO} ...")
    t0 = time.time()

    if _cuda_available():
        _GROUNDING = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_REPO, trust_remote_code=True
        ).to("cuda")
    else:
        _GROUNDING = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_REPO, trust_remote_code=True
        ).to("cpu")

    _GROUNDING.eval()
    _GROUNDING_PROCESSOR = AutoProcessor.from_pretrained(
        GROUNDING_REPO, trust_remote_code=True
    )
    print(f"[SatQuery] Grounding DINO loaded in {time.time() - t0:.1f}s")

    return _GROUNDING, _GROUNDING_PROCESSOR


# --------------------------------------------------------------------------- #
# TASK 2 -- TEXT-GUIDED VISUAL GROUNDING
# --------------------------------------------------------------------------- #
def run_grounding(image_path: str, target_object: str) -> Dict[str, Any]:
    """
    Localize instances of a text-described object in a single image.

    Contract:
        {"text_response": str,
         "bounding_boxes": [{"label": str,
                             "box_2d": [ymin, xmin, ymax, xmax],
                             "confidence": float}],
         "confidence": float}
    """
    import torch

    t0 = time.time()
    image = load_image(image_path)
    width, height = image.size

    model, processor = _load_grounding()
    query = target_object.strip() or "object"

    inputs = processor(images=image, text=query, return_tensors="pt")
    if _cuda_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # transformers >= 5.x renamed `box_threshold` -> `threshold`; try both.
    kwargs = dict(
        input_ids=inputs["input_ids"],
        target_sizes=[(height, width)],
        text_threshold=GROUNDING_TEXT_THRESHOLD,
    )
    try:
        results = processor.post_process_grounded_object_detection(
            outputs, threshold=GROUNDING_BOX_THRESHOLD, **kwargs
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs, box_threshold=GROUNDING_BOX_THRESHOLD, **kwargs
        )[0]

    bounding_boxes: List[Dict[str, Any]] = []
    scores = results.get("scores", [])
    boxes = results.get("boxes", [])
    # transformers >= v4.51 exposes string names under `text_labels`; older
    # versions returned them under `labels`. Both may occasionally be int ids
    # or empty strings, so coerce defensively and fall back to the query text.
    labels = results.get("text_labels") or results.get("labels", [])

    for i, box in enumerate(boxes):
        # boxes arrive as [xmin, ymin, xmax, ymax] in pixel coordinates.
        xmin, ymin, xmax, ymax = [float(v) for v in box]
        raw_label = labels[i] if i < len(labels) else None
        label = (
            str(raw_label)
            if raw_label is not None and str(raw_label) != "" and not str(raw_label).isdigit()
            else query
        )
        score = _confidence_to_float(scores[i]) if i < len(scores) else 0.0
        score = _confidence_to_float(scores[i]) if i < len(scores) else 0.0
        bounding_boxes.append(
            {
                "label": label,
                "box_2d": [ymin, xmin, ymax, xmax],  # contract order!
                "confidence": score,
            }
        )

    if bounding_boxes:
        n = len(bounding_boxes)
        text_response = (
            f"Detected {n} instance(s) of '{target_object}' in the image. "
            f"Highest-confidence detection scores {max(b['confidence'] for b in bounding_boxes):.2f}."
        )
        confidence = max(b["confidence"] for b in bounding_boxes)
    else:
        text_response = (
            f"No '{target_object}' detected above the confidence threshold "
            f"({GROUNDING_BOX_THRESHOLD}). Consider uploading higher-resolution imagery."
        )
        confidence = 0.0

    return {
        "text_response": text_response,
        "bounding_boxes": bounding_boxes,
        "confidence": _confidence_to_float(confidence),
    }


# --------------------------------------------------------------------------- #
# METADATA EXPORTS (for Pair 2's execution trace and Pair 3 coordination)
# --------------------------------------------------------------------------- #
def get_execution_metadata() -> Dict[str, Any]:
    """Model identifiers / version strings for the agentic execution trace."""
    return {
        "vqa_model": VLM_REPO,
        "grounding_model": GROUNDING_REPO,
        "quantization": "4bit-nf4" if _cuda_available() else "none (cpu fp32)",
        "device": "cuda" if _cuda_available() else "cpu",
    }


def flush_models() -> None:
    """Release all cached models (frees VRAM). Useful between heavy jobs."""
    global _VLM, _VLM_PROCESSOR, _GROUNDING, _GROUNDING_PROCESSOR
    import gc

    try:
        import torch

        del _VLM, _VLM_PROCESSOR, _GROUNDING, _GROUNDING_PROCESSOR
        gc.collect()
        if _cuda_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _VLM = _VLM_PROCESSOR = _GROUNDING = _GROUNDING_PROCESSOR = None


# Allow running as a script: `python models_registry.py image.png "describe"`
if __name__ == "__main__":
    import sys

    _img = sys.argv[1] if len(sys.argv) > 1 else "RS/before.png"
    try:
        _res = run_vqa(_img, "Describe this satellite image")
    except Exception as exc:  # model may not be downloaded / no network
        _res = {"text_response": f"VQA unavailable here: {exc}", "confidence": 0.0}
    print(json.dumps(_res, indent=2, default=str))
