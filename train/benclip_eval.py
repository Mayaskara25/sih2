"""Shared benclip evaluation: zero-shot image->text retrieval and linear-probe mAP.

These are the before/after numbers that satisfy R1 (PLAN.md §3.1, W2 task 1&6).
The SAME functions compute the *before* (stock CLIP on the RGB composite) and
the *after* (trained benclip on all 14 bands), so the two numbers are directly
comparable — identical protocol, identical held-out split, only the encoder and
the input bands differ.

Hosted under ``train/`` (not ``eval/``, which is W8-owned). W8 is expected to
re-run this from ``eval/adaptation.py``; this module is the single implementation
of the protocol so both W2's status doc and W8's re-run agree.

Protocol:
  Retrieval R@1/R@5 — for every test-split patch, rank ALL test captions by
  cosine similarity in the shared text space; R@k = fraction where the paired
  caption is in the top-k.
  Linear-probe mAP — fit a multi-label linear head on TRAIN-split embeddings to
  the 19-class ground truth, then mean average-precision over classes on the
  test split.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import numpy as np

from train.benclip_data import load_caption_map, load_targets, split_patches

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


class Embedder:
    """Encapsulates image+text encoding so before/after share one protocol.

    ``image_factory(arrays)`` maps a list of (C, H, W) arrays to a batch of
    normalized image embeddings; ``text_factory(texts)`` maps a list of strings
    to normalized text embeddings. Both return (N, D) float arrays.
    """

    def __init__(self, image_factory: Callable[[List[np.ndarray]], np.ndarray],
                 text_factory: Callable[[List[str]], np.ndarray]) -> None:
        self.image_factory = image_factory
        self.text_factory = text_factory

    def embed_images(self, arrays: List[np.ndarray]) -> np.ndarray:
        return _ensure_ndim(self.image_factory(arrays))

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return _ensure_ndim(self.text_factory(texts))


def _ensure_ndim(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return x


def _label_matrix(patches: List[dict]) -> np.ndarray:
    """(N, 19) binary label matrix in BEN_CLASS_NAMES order."""
    mat = np.zeros((len(patches), len(BEN_CLASS_NAMES)), dtype=np.float32)
    index = {name: i for i, name in enumerate(BEN_CLASS_NAMES)}
    for r, p in enumerate(patches):
        for lab in p["labels"]:
            if lab in index:
                mat[r, index[lab]] = 1.0
    return mat


def retrieval_recall(image_feats: np.ndarray, text_feats: np.ndarray) -> Tuple[float, float]:
    """R@1 and R@5 for image->text retrieval.

    ``image_feats`` and ``text_feats`` are (N, D) with paired rows. Each image
    ranks all texts by cosine similarity (both inputs already unit-normalised);
    returns (R@1, R@5) as fractions in [0,1].
    """
    sims = image_feats @ text_feats.T  # (N, N)
    n = sims.shape[0]
    r1 = 0.0
    r5 = 0.0
    for i in range(n):
        order = np.argsort(-sims[i])
        pos = int(np.where(order == i)[0][0])
        if pos == 0:
            r1 += 1.0
        if pos < 5:
            r5 += 1.0
    return r1 / n, r5 / n


def linear_probe(train_feats: np.ndarray, train_labels: np.ndarray,
                 test_feats: np.ndarray, test_labels: np.ndarray,
                 C: float = 1.0) -> Tuple[float, float]:
    """Fit a multi-label linear head and report test mAP and macro-F1.

    Uses a per-class independent L2-regularised logistic model (sklearn) — the
    standard, reproducible choice for BEN multi-label linear probing. Returns
    ``(mAP, macro_f1)``. mAP is the mean over classes of average precision
    (the rubric's "feature representation" number); macro-F1 is reported for
    the status doc as a secondary measure on the same head.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score

    n_classes = train_labels.shape[1]
    scores = np.zeros((test_feats.shape[0], n_classes), dtype=np.float64)
    for c in range(n_classes):
        y = train_labels[:, c]
        # A class with all-same labels (e.g. no positive samples in the train
        # subset, common in tiny smoke runs) cannot be fit by a binary
        # classifier. Score 0 for that class (no evidence) and move on; the
        # AP for it is treated as 0 by nanmean.
        if len(np.unique(y)) < 2:
            continue
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(train_feats, y)
        scores[:, c] = clf.decision_function(test_feats)

    ap = average_precision_score(test_labels, scores, average=None)
    map_score = float(np.nanmean(ap))
    macro_f1 = float(f1_score(test_labels, (scores > 0).astype(int), average="macro"))
    return map_score, macro_f1


def evaluate_retrieval_and_probe(embedder: Embedder, patches: List[dict],
                                 caption_map: dict, batch_size: int = 32,
                                 train_cap: int = None, test_cap: int = None
                                 ) -> dict:
    """Run the full before/after protocol and return a dict of metrics.

    Args:
        embedder: ``Embedder`` with image/task factories for this encoder.
        patches: ALL target patches (already has split). The train/validation/test
            sub-lists are derived via the official ``split`` column.
        caption_map: ``{patch_id: caption}`` for the involved patches.
        batch_size: embedding batch size (keeps VRAM bounded on the 3.9 GB card).
        train_cap / test_cap: optional caps on the number of train/test patches
            used (useful for a quick smoke run; the full run leaves them None).

    Returns a dict with ``retrieval_r1``/``retrieval_r5``/``map``/``macro_f1``
    plus ``n_train``/``n_test``.
    """
    train = split_patches(patches, "train")
    test = split_patches(patches, "test")
    if train_cap:
        train = train[:train_cap]
    if test_cap:
        test = test[:test_cap]

    train = [p for p in train if p["patch_id"] in caption_map]
    test = [p for p in test if p["patch_id"] in caption_map]

    # Image embeddings for retrieval (only test needed for retrieval) and for
    # both training and testing the linear probe.
    test_images = _embed_from_stack(embedder, test, "image")
    test_captions = embedder.embed_texts([caption_map[p["patch_id"]] for p in test])
    train_images = _embed_from_stack(embedder, train, "image")

    r1, r5 = retrieval_recall(test_images, test_captions)

    train_labels = _label_matrix(train)
    test_labels = _label_matrix(test)
    map_score, macro_f1 = linear_probe(train_images, train_labels, test_images, test_labels)

    return {
        "retrieval_r1": float(r1),
        "retrieval_r5": float(r5),
        "map": float(map_score),
        "macro_f1": float(macro_f1),
        "n_train": len(train),
        "n_test": len(test),
    }


def _embed_from_stack(embedder: Embedder, patches: List[dict],
                      which: str) -> np.ndarray:
    """Embed a list of patch dicts (re-stacking 14-band stems) in batches."""
    from satquery.adapters.benclip import stack_ben_patch

    feats = []
    batch_arrays: List[np.ndarray] = []
    batch_meta: List[dict] = []
    for p in patches:
        stem = stack_ben_patch(p["s2_folder"], p["s1_folder"], p["patch_id"], p["s1_name"])
        batch_arrays.append(stem)
        batch_meta.append(p)
        if len(batch_arrays) >= 32:
            feats.append(embedder.embed_images(batch_arrays))
            batch_arrays = []
    if batch_arrays:
        feats.append(embedder.embed_images(batch_arrays))
    if not feats:
        return np.zeros((0, 512), dtype=np.float32)
    return np.concatenate(feats, axis=0)


def embed_ben_14_from_stem(bc, stems: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
    """Embed a list of (14,120,120) stems through a loaded BenClipModel in batches."""
    import torch
    from satquery.adapters.benclip import _image_embedding

    feats = []
    for i in range(0, len(stems), batch_size):
        chunk = stems[i:i + batch_size]
        for stem in chunk:
            feats.append(_image_embedding(bc, stem))
    if not feats:
        return np.zeros((0, 512), dtype=np.float32)
    return np.stack(feats, axis=0)


def benclip_embedder(bc) -> Embedder:
    """Build the ``Embedder`` for a trained benclip model.

    Images are encoded from their full 14-band stems; texts through the
    (frozen, base) CLIP text tower — matching the training-time text encoder so
    the retrieval space is the same one the contrastive loss aligned into.
    """
    from satquery.adapters.benclip import _get_pooled, _image_embedding
    import torch

    def img_factory(arrays):
        feats = []
        for a in arrays:
            feats.append(_image_embedding(bc, a))
        return np.stack(feats, axis=0)

    def txt_factory(texts):
        # Chunk the text batch so encoding thousands of captions never spikes the
        # 3.9 GB card (a single 3.5k-text forward OOMs; 256 at a time is safe).
        import torch
        outs = []
        for i in range(0, len(texts), 256):
            chunk = list(texts[i:i + 256])
            with torch.no_grad():
                tokens = bc.processor(text=chunk, padding=True, truncation=True,
                                      return_tensors="pt").to(bc.device)
                tf = _get_pooled(bc.model.get_text_features(**tokens)).float().cpu().numpy()
            n = np.linalg.norm(tf, axis=1, keepdims=True)
            n[n == 0] = 1.0
            outs.append(tf / n)
        return np.concatenate(outs, axis=0)

    return Embedder(img_factory, txt_factory)


def _to_uint8(struct: np.ndarray) -> np.ndarray:
    """Percentile-stretch a multi-channel float array to uint8 (2-98 per band)."""
    h, w = struct.shape[0], struct.shape[1]
    out = np.zeros((h, w, struct.shape[2]), dtype=np.uint8)
    for c in range(struct.shape[2]):
        band = struct[..., c].astype(np.float64)
        lo, hi = np.percentile(band, (2, 98))
        if hi <= lo:
            lo, hi = float(band.min()), float(band.max())
        if hi <= lo:
            out[..., c] = 127
            continue
        out[..., c] = ((band - lo) / (hi - lo) * 255).round().astype(np.uint8)
    return out


def stock_clip_embedder(device: str = "cpu") -> Embedder:
    """Build the ``Embedder`` for STOCK CLIP on the RGB (B04/B03/B02) composite.

    This is the *before* baseline. Stock CLIP can only ingest 3 bands, so we
    feed it the same visual content benclip gets but as B04/B03/B02 = R/G/B.
    """
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor

    from satquery.adapters.benclip import _get_pooled, BEN_14_SLOTS, S2_BAND_ORDER

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_id)

    def _rgb_from_14(stem: np.ndarray) -> np.ndarray:
        # stem is (14, H, W) in BEN_14_SLOTS order; B04/B03/B02 indices.
        idx = {b: BEN_14_SLOTS.index(b) for b in S2_BAND_ORDER}
        r = stem[idx["B04"]]
        g = stem[idx["B03"]]
        b = stem[idx["B02"]]
        return np.stack([r, g, b], axis=-1)  # (H, W, 3)

    def img_factory(arrays):
        import torch.nn.functional as F

        feats = []
        for a in arrays:
            rgb = _rgb_from_14(a)
            # Normalise each band's dynamic range to uint8 for display/processing.
            rgb = _to_uint8(rgb)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)
            t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            rgb224 = t[0].clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
            inputs = proc(images=rgb224, return_tensors="pt").to(device)
            with torch.no_grad():
                f = _get_pooled(model.get_image_features(**inputs)).float().cpu().numpy()
            n = np.linalg.norm(f, axis=1, keepdims=True)
            n[n == 0] = 1.0
            feats.append(f / n)
        return np.concatenate(feats, axis=0)

    def txt_factory(texts):
        # Chunk the text batch (see benclip_embedder.txt_factory) to keep the
        # peak on the 3.9 GB card bounded when encoding the full 3.5k captions.
        import torch
        outs = []
        for i in range(0, len(texts), 256):
            chunk = list(texts[i:i + 256])
            with torch.no_grad():
                tokens = proc(text=chunk, padding=True, truncation=True,
                              return_tensors="pt").to(device)
                tf = _get_pooled(model.get_text_features(**tokens)).float().cpu().numpy()
            n = np.linalg.norm(tf, axis=1, keepdims=True)
            n[n == 0] = 1.0
            outs.append(tf / n)
        return np.concatenate(outs, axis=0)

    return Embedder(img_factory, txt_factory)
