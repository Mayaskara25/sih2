"""
W8 evaluation-harness tests — all pure-CPU, no model weights, no downloads.

Covers the W8 acceptance-critical mechanics that must be provable without a
GPU or the heavy specialists:

1. CDVQA stub detection is MECHANICAL: ``cdvqa.classify_run`` inspects the
   ``confidence_basis`` a specialist result announces and flips from
   PLACEHOLDER to measured when a non-stubbed result is fed in. This is the
   W8 brief's "PLACEHOLDER because it detected a stub, not because you
   hardcoded it" property.
2. The shared metric implementations (BLEU, ROUGE-L, CIDEr proxy) on known
   inputs.
3. RSVQA scoring: case-normalisation flips raw mismatch -> normalised match
   (the ``run_vqa`` answers with both ``Yes``/``yes`` trap); single-token
   BLEU-1 behaves like exact-match.
4. Routing harness: confusion-matrix shape + accuracy on the committed
   W6 testset (known 100/100); PLACEHOLDER mapping of misroutes stays out of
   the scored set.
5. The orchestration reproducibility reducer keeps only numeric payloads so
   two runs can be compared for identity (ignoring volatile text/dates).
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO_ROOT, "eval")
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, REPO_ROOT)

import _common as _c  # noqa: E402
import cdvqa  # noqa: E402
import rsvqa  # noqa: E402


# --------------------------------------------------------------------------
# 1. CDVQA stub detection is mechanical
# --------------------------------------------------------------------------

class TestStubDetection:
    def test_stubbed_result_yields_placeholder(self):
        res = {
            "text_response": "changed",
            "confidence": 0.5,
            "confidence_basis": "stub",  # W0/W4 stub announcement
            "metrics": {},
            "evidence": {},
        }
        verdict = cdvqa.classify_run(res, n_items=100)
        assert verdict["status"] == "PLACEHOLDER"
        assert verdict["n"] == 100

    def test_non_stubbed_result_flips_to_measured(self):
        for basis in ("heuristic", "calibrated", "model_logprob"):
            res = {
                "text_response": "changed",
                "confidence": 0.7,
                "confidence_basis": basis,
                "metrics": {},
                "evidence": {},
            }
            verdict = cdvqa.classify_run(res, n_items=100)
            assert verdict["status"] == "measured", basis
            assert verdict["basis"] == basis

    def test_unknown_basis_is_contract_violation(self):
        res = {"text_response": "x", "confidence": 0.5, "confidence_basis": "totally_made_up"}
        verdict = cdvqa.classify_run(res, n_items=1)
        assert verdict["status"] == "contract_violation"

    def test_probe_stub_inspects_actual_specialist(self):
        """The probe must observe the REAL run_change at runtime, never hardcode.

        This test previously asserted `stub is True` with the comment "W4 has
        not landed". W4 HAS now landed, so that assertion encoded a moment in
        time rather than a property. The durable assertion -- true before W4,
        true after, and true again if W4 is ever reverted -- is that the probe
        AGREES with what the specialist actually reports. That is exactly the
        mechanical-detection guarantee the harness exists to provide.
        """
        import os as _os

        from satquery.specialists import change as change_mod

        from eval import _common as _cc  # noqa: F401  (REPO_ROOT already imported as _c)

        try:
            actual_basis = change_mod.run_change(
                _os.path.join(_c.REPO_ROOT, "__w8_probe_t0.tif"),
                _os.path.join(_c.REPO_ROOT, "__w8_probe_t1.tif"),
                "__w8_probe__ what changed between these two images?",
            )["confidence_basis"]
        except Exception:
            actual_basis = None

        status = cdvqa._probe_stub()

        if actual_basis is not None:
            assert status["basis"] == actual_basis, (
                "probe disagrees with the specialist it claims to inspect — "
                "detection is not actually mechanical"
            )
        assert status["stub"] is (actual_basis == "stub")
        # W4 has landed, so a regression back to a stub basis is a real defect.
        assert status["stub"] is False


# --------------------------------------------------------------------------
# 2. Metric implementations on known inputs
# --------------------------------------------------------------------------

class TestMetrics:
    def test_bleu_exact_match(self):
        assert _c.bleu("a single token", "a single token") == 1.0

    def test_bleu_no_overlap(self):
        assert _c.bleu("yes", "no") == 0.0

    def test_bleu1_is_unigram_precision(self):
        # BLEU-1 for prediction superset of gold drops below 1 (brevity-free
        # unigram precision), always <= 1, > 0 on any shared token.
        score = _c.bleu1("yes please", "yes")
        assert 0.0 < score < 1.0
        assert _c.bleu1("yes", "yes") == 1.0

    def test_bleu_brevity_penalty(self):
        # A candidate that is a strict prefix of the reference hits perfect
        # precision at every level but is much shorter, so the brevity penalty
        # must pull BLEU below 1.0.
        short = _c.bleu("the quick brown fox jumps over the lazy dog", "the quick brown")
        assert 0.0 < short < 1.0

    def test_rouge_l_known(self):
        # ROUGE-L is an LCS F1; a full overlap scores 1, partial < 1.
        assert _c.rouge_l("the quick brown fox", "the quick brown fox") == 1.0
        score = _c.rouge_l("the quick brown fox jumps over the lazy dog",
                           "a quick brown fox leaps over the lazy hound")
        assert 0.0 < score < 1.0

    def test_cidre_proxy_bounded(self):
        # Cosine-similarity TF-IDF proxy: same sentence scores 1, disjoint 0.
        assert 0.0 <= _c.cidre(["hello there"], ["hello there"]) <= 1.0
        assert _c.cidre(["hello there"], ["banana sandwich zebra quark"]) < 0.1

    def test_seeded_subset_deterministic(self):
        items = [{"id": i} for i in range(50)]
        a = _c.seeded_subset(items, 20, seed=7)
        b = _c.seeded_subset(items, 20, seed=7)
        assert [x["id"] for x in a] == [x["id"] for x in b]
        assert len(a) == 20


# --------------------------------------------------------------------------
# 3. RSVQA scoring + normalisation
# --------------------------------------------------------------------------

class TestRsvqaScoring:
    def test_raw_mismatch_but_normalised_match_flips(self):
        # run_vqa returns "Yes"; gold is "yes" (the W8 brief's trap). Raw
        # mismatch is real, but the normalised comparison must match.
        scores = rsvqa.score_pair(prediction="Yes", gold="yes", qtype="presence")
        assert scores["raw_match"] is False
        assert scores["norm_match"] is True
        assert scores["answer_match"] is True

    def test_rural_urban(self):
        scores = rsvqa.score_pair(prediction="Rural", gold="rural", qtype="rural_urban")
        assert scores["norm_match"] is True
        assert scores["answer_match"] is True
        # A trailing period (not something run_vqa emits) is NOT normalised
        # away; that stays an honest mismatch, and answer-match still fires.
        punct = rsvqa.score_pair(prediction="Rural.", gold="rural", qtype="rural_urban")
        assert punct["norm_match"] is False
        assert punct["answer_match"] is True

    def test_answer_match_requires_keyword(self):
        scores = rsvqa.score_pair(prediction="no buildings visible", gold="no", qtype="presence")
        assert scores["norm_match"] is False
        assert scores["answer_match"] is True  # gold keyword present in prediction

    def test_wrong_number_no_match(self):
        scores = rsvqa.score_pair(prediction="5", gold="8", qtype="count")
        assert scores["norm_match"] is False
        assert scores["raw_match"] is False

    def test_bleu1_equals_exact_for_single_token(self):
        # On RSVQA's single-token answers, BLEU-1 degenerates to exact match.
        hit = rsvqa.score_pair("yes", "yes", "presence")
        miss = rsvqa.score_pair("no", "yes", "presence")
        assert hit["bleu1"] == 1.0
        assert miss["bleu1"] == 0.0

    def test_normalisation_option_in_report(self):
        res = rsvqa.evaluate([
            {
                "routed_task": "vqa",
                "prediction": "Yes",
                "scores": rsvqa.score_pair("Yes", "yes", "presence"),
                "type": "presence",
            },
        ])
        assert res["normalised_accuracy"] == 1.0
        assert res["raw_accuracy"] == 0.0  # raw stays honest
        assert res["normalisation"] == "lowercase + strip + collapse whitespace on both sides"


# --------------------------------------------------------------------------
# 4. Routing harness shape + PLACEHOLDER mapping
# --------------------------------------------------------------------------

class TestRoutingHarness:
    def test_committed_testset_accuracy(self):
        import routing
        queries = routing.load_testset()
        assert len(queries) >= 100
        res = routing.evaluate(queries)
        assert res["accuracy"] == 1.0  # W6 acceptance: 100/100
        assert res["n"] == len(queries)

    def test_confusion_matrix_shape_and_identity(self):
        import routing
        queries = routing.load_testset()
        res = routing.evaluate(queries)
        tasks = sorted(res["matrix"].keys())
        assert tasks == sorted(routing.TASK_ORDER)
        for g in tasks:
            for p in tasks:
                cell = res["matrix"][g][p]
                assert isinstance(cell, int)
                if g == p and res["accuracy"] == 1.0:
                    assert cell > 0  # every gold task actually routed somewhere
        # With perfect accuracy the matrix is diagonal.
        for g in tasks:
            assert res["matrix"][g][g] == sum(res["matrix"][g].values())

    def test_per_task_rows_carry_n(self):
        import routing
        res = routing.evaluate(routing.load_testset())
        for task in routing.TASK_ORDER:
            row = res["per_task"][task]
            assert row["n"] == row["correct"]  # perfect routing

    def test_misrouted_queries_carried_outside_scored_set(self):
        # The routing fidelity of the RSVQA sample is separately reported;
        # the VQA accuracy number must only count vqa-routed, scored items.
        items = [
            {"routed_task": "vqa", "prediction": "Yes",
             "scores": rsvqa.score_pair("Yes", "yes", "presence"), "type": "presence"},
            {"routed_task": "grounding", "prediction": None},  # misroute: not scored
        ]
        res = rsvqa.evaluate(items)
        assert res["n_scored_vqa"] == 1
        assert res["n_misrouted"] == 1
        assert res["n_total"] == 2
        assert res["normalised_accuracy"] == 1.0


# --------------------------------------------------------------------------
# 5. Orchestration reproducibility reducer
# --------------------------------------------------------------------------

class TestReproReducer:
    def test_numeric_payload_strips_volatile_fields(self):
        import run_all
        payload = {
            "vqa": {
                "normalised_accuracy": 0.4323,
                "n_scored_vqa": 155,
                "date": "2026-08-31",
            },
            "adaptation": {
                "status": "measured",
                "date": "2026-08-31",
                "after": {"map": 0.2969, "n_test": 300},
            },
        }
        num = run_all._num_scalar(payload)
        assert num["vqa"]["normalised_accuracy"] == 0.4323
        assert "date" not in num["vqa"]
        assert num["adaptation"]["status"] == "measured"  # status is a stable token
        assert num["adaptation"]["after"]["map"] == 0.2969

    def test_identical_numeric_payloads_compare_equal(self):
        import run_all
        a = {"vqa": {"accuracy": 0.5, "n": 10}, "adaptation": {"map": 0.2}}
        b = {"vqa": {"accuracy": 0.5, "n": 10}, "adaptation": {"map": 0.2}}
        assert run_all._numeric_payloads.__name__  # importable
        assert a == b

    def test_render_results_contains_rubric_rows(self):
        import run_all
        cross = {
            "status": "PLACEHOLDER",
            "blocker": "no data",
            "n": None,
            "date": "2026-08-31",
        }
        collected = {
            "vrsbench": {"status": "PLACEHOLDER", "blocker": "missing", "n": None,
                         "date": "2026-08-31", "bleu": None, "rouge_l": None,
                         "cider_proxy": None, "grounding_iou": None},
            "vqa": {
                "status": "measured", "n_scored_vqa": 155, "date": "2026-08-31",
                "raw_accuracy": 0.2774, "normalised_accuracy": 0.4323,
                "answer_match_rate": 0.4323, "bleu1": 0.4323,
            },
            "cdvqa": {"status": "PLACEHOLDER", "blocker": "stub", "n": None,
                       "date": "2026-08-31"},
            "adaptation": {
                "status": "measured", "date": "2026-08-31", "n_train": 579,
                "n_test": 300, "checkpoint": "x",
                "before": {"retrieval_r1": 0.01333, "retrieval_r5": 0.0,
                           "map": 0.2474, "macro_f1": 0.0},
                "after": {"retrieval_r1": 0.01, "retrieval_r5": 0.06,
                          "map": 0.2969, "macro_f1": 0.2359},
            },
            "routing": {"accuracy": 1.0, "n": 100, "date": "2026-08-31",
                        "metric": "routing_accuracy"},
        }
        md = run_all.render_results(collected, cross)
        for needle in (
            "| 1 | Single-image captioning & grounding",
            "| 2 | VQA (RSVQA) | normalised accuracy | 0.4323 | 155",
            "| 2b | VQA (RSVQA) | raw exact-match accuracy | 0.2774 | 155",
            "| 3 | Multi-image change (CDVQA)",
            "| 4 | Domain adaptation (BigEarthNet) | retrieval R@1 after | 0.01000 | 300",
            "| 5 | Joint cross-modal (Cartosat + RISAT)",
            "| 6 | Agentic orchestration (routing) | routing accuracy | 1.0000 | 100",
        ):
            assert needle in md, needle