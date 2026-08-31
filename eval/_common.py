"""Shared helpers for the W8 evaluation harnesses (eval/*).

W8 owns everything under ``eval/`` except ``routing_testset.json`` (W6).
This module is pure stdlib + numpy, imports nothing from the frozen
``satquery`` code, and is used by every harness so they all carry
sample counts, dates and normalisation recipes consistently.

No ``eval/__init__.py`` is created (PLAN.md §5.1: every __init__.py is
W0-owned and frozen), so harnesses add this directory to ``sys.path`` and
``import _common`` as a top-level module — matching how ``python eval/*.py``
runs them as scripts.
"""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SEED = 20260831

# Marker for rows/grid cells that have not been measured yet (§5.9).
PLACEHOLDER = "PLACEHOLDER"


def today() -> str:
    """ISO date of the run, for the RESULTS.md rows."""
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# Text normalisation + output metric helpers (RSVQA / caption matching).
# ---------------------------------------------------------------------------


def strip_norm(text: str) -> str:
    """Lowercase-normalise and strip both sides.

    This is the normalisation the RSVQA harness applies before comparing a
    predicted answer to the gold reference. `run_vqa` returns answers with
    inconsistent casing (`Yes` vs `yes`); exact string match on raw output
    would therefore report spuriously low accuracy. We lower-case, strip
    surrounding whitespace, and collapse inner whitespace so equivalent
    tokens compare equal.
    """
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokenise(text: str) -> List[str]:
    """Split on non-alphanumeric boundaries for n-gram metrics."""
    return [t for t in re.split(r"[^0-9a-z]+", strip_norm(text)) if t]


# ---------------------------------------------------------------------------
# Lightweight BLEU (smoothing + brevity penalty), pure stdlib.
# ---------------------------------------------------------------------------


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    """BLEU with method-1 (Lin & Och) additive smoothing and a brevity penalty.

    Self-contained because the venv does not carry sacrebleu/nltk and W8's
    harnesses must run under ``uv run --no-sync``. Implements the standard
    modified n-gram precision: each level's count is clipped to the reference
    n-gram counts; levels where either side has no n-grams get (+1) additive
    smoothing so a 0/0 never crops up. The brevity penalty applies when the
    candidate is shorter than the reference, per Papineni et al.

    Note on short answers: for a single-token candidate (the RSVQA case),
    levels 2..N receive full smoothing credit, so BLEU-4 is ill-posed and hugs
    ~1 regardless of correctness. Prefer ``max_n=1`` (BLEU-1) there — see
    ``bleu1``.
    """
    ref = tokenise(reference)
    cand = tokenise(candidate)
    if not ref or not cand:
        return 0.0
    ref_counts: Dict[tuple, int] = {}
    for k in range(1, max_n + 1):
        for g in _ngrams(ref, k):
            ref_counts[g] = ref_counts.get(g, 0) + 1

    precisions: List[float] = []
    for k in range(1, max_n + 1):
        cand_ngrams = _ngrams(cand, k)
        clipped = sum(1 for g in cand_ngrams if ref_counts.get(g, 0) > 0)
        if cand_ngrams:
            precisions.append(clipped / len(cand_ngrams))
        elif ref_counts:  # no candidate n-grams at this level -> not scored
            precisions.append(1.0)

    if not precisions or any(p == 0.0 for p in precisions):
        return 0.0
    log_prec = sum(p ** (1.0 / len(precisions)) for p in precisions) / len(precisions)
    bp = 1.0 if len(cand) > len(ref) else 2.718281828 ** (1.0 - len(ref) / max(len(cand), 1))
    return min(1.0, bp * log_prec)


def bleu1(reference: str, candidate: str) -> float:
    """BLEU-1 (unigram precision + brevity penalty) — the meaningful BLEU on
    RSVQA's single-token answer vocabulary."""
    return bleu(reference, candidate, max_n=1)


def bleu_corpus(references: List[str], candidates: List[str], max_n: int = 4) -> float:
    if not references:
        return 0.0
    return sum(bleu(r, c, max_n=max_n) for r, c in zip(references, candidates)) / len(references)


def bleu1_corpus(references: List[str], candidates: List[str]) -> float:
    return bleu_corpus(references, candidates, max_n=1)


# ---------------------------------------------------------------------------
# ROUGE-L (F1 over the longest common subsequence), pure stdlib.
# ---------------------------------------------------------------------------


def _lcs(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l(reference: str, candidate: str) -> float:
    ref = tokenise(reference)
    cand = tokenise(candidate)
    if not ref or not cand:
        return 0.0
    lcs = _lcs(ref, cand)
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def rouge_l_corpus(references: List[str], candidates: List[str]) -> float:
    if not references:
        return 0.0
    return sum(rouge_l(r, c) for r, c in zip(references, candidates)) / len(references)


def cidre(references: List[str], candidates: List[str]) -> float:
    """A minimal stand-in for CIDEr: mean cosine of TF-IDF n-gram vectors.

    Declared explicitly as a *proxy*, not the official MS-COCO CIDEr, because
    the official CIDEr requires a reference TF-IDF corpus trained on a document
    collection that VRSBench does not define verbatim. The value is only ever
    used when VRSBench data lands; it reports 0.0 here (placeholder path).
    """
    def vec(tokens: List[str]) -> Dict[str, float]:
        counts: Dict[str, float] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0.0) + 1.0
        norm = sum(counts.values()) or 1.0
        return {k: v / norm for k, v in counts.items()}

    total = 0.0
    for r, c in zip(references, candidates):
        vr, vc = vec(tokenise(r)), vec(tokenise(c))
        keys = set(vr) | set(vc)
        dot = sum(vr.get(k, 0.0) * vc.get(k, 0.0) for k in keys)
        nr = (sum(v ** 2 for v in vr.values()) ** 0.5) or 1.0
        nc = (sum(v ** 2 for v in vc.values()) ** 0.5) or 1.0
        total += dot / (nr * nc)
    return total / len(references) if references else 0.0


# ---------------------------------------------------------------------------
# Deterministic subsampling (fixed seed => reproducible across runs).
# ---------------------------------------------------------------------------


def seeded_subset(items: List[dict], sample_size: int, seed: int = DEFAULT_SEED) -> List[dict]:
    """Deterministically select ``sample_size`` items (or all if fewer)."""
    if sample_size <= 0 or sample_size >= len(items):
        return list(items)
    import random

    rng = random.Random(seed)
    return rng.sample(items, sample_size)
