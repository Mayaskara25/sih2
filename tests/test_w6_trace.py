"""
W6 execution-trace + orchestrator tests (PLAN.md §4.2, R5: every run writes an
auditable trace; §4.3 models_used).

Traces are written under tmp_path and validated against the contract.

NOTE (updated after W3): this file used to say "nothing downloads". That stopped
being true the moment W3 replaced the vqa/caption/grounding stubs with real
models -- dispatching them pulls ~5 GB of weights. Tests that actually dispatch
are now gated behind CUDA; routing itself stays model-free in test_w6_router.py.
"""

import json

import numpy as np
import pytest
import rasterio
from affine import Affine

from satquery.contracts import validate_execution_trace
from satquery.controller.trace import TASK_ROLES, _merge_models, run_query, write_trace
from satquery.runtime.modelpool import ModelPool, RoleSpec
import torch  # noqa: E402
from satquery.contracts import CONFIDENCE_BASES  # noqa: E402

# Gate anything that dispatches a real specialist. Before this, `pytest tests/`
# downloaded ~5 GB unconditionally -- on a fresh clone, on a CPU-only box, in CI.
_CUDA = torch.cuda.is_available()
requires_real_specialists = pytest.mark.skipif(
    not _CUDA, reason="dispatching real specialists needs CUDA + downloaded weights"
)

_T = Affine.translation(10.0, 20.0) @ Affine.scale(0.5, -0.5)


def _write_geotiff(path, bands=3, crs="EPSG:4326", height=16, width=16):
    array = np.zeros((height, width, bands), np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="uint8",
        crs=crs,
        transform=_T,
    ) as dst:
        for i in range(bands):
            dst.write(array[:, :, i], i + 1)


@pytest.fixture
def two_imgs(tmp_path):
    p1 = tmp_path / "t0.tif"
    p2 = tmp_path / "t1.tif"
    _write_geotiff(str(p1))
    _write_geotiff(str(p2))
    return str(p1), str(p2)


def _assert_trace_valid(trace):
    validate_execution_trace(trace)
    json.dumps(trace)  # must survive json (no numpy), ensuring json.dump of file
    for key in ("routing", "validation", "inference", "total"):
        assert isinstance(trace["timings_ms"][key], int)
    assert isinstance(trace["routing"]["score"], float)


# --------------------------------------------------------------------------- #
# The orchestrator reaches every specialist and writes an auditable trace.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query,task,path_count,forced,need",
    [
        ("how many buildings are there in this image", "vqa", 1, None, 1),
        ("Describe the land-cover and major objects visible in this image.", "caption", 1, None, 1),
        ("Highlight the water body referred to in the query.", "grounding", 1, None, 1),
        ("What changed between these two dates, and where did the change occur?", "change", 2, None, 2),
        ("Use the optical and SAR images together.", "fusion", 2, ["optical", "sar"], 2),
    ],
)
@requires_real_specialists
def test_all_five_specialists_reachable_via_run_query(
    query, task, path_count, forced, need, two_imgs, tmp_path
):
    paths = [two_imgs[0]] if path_count == 1 else list(two_imgs)
    trace = run_query(
        query,
        paths,
        forced_modalities=forced,
        runs_root=str(tmp_path / "runs"),
    )
    assert trace["task_selected"] == task
    # R7: assert a specialist ACTUALLY RAN, not that it is still a stub. The
    # original `== "stub"` inverted the meaning of success the moment W3 made
    # three of these real -- see the §5.2 corollary in PLAN.md.
    assert trace["result"]["confidence_basis"] in CONFIDENCE_BASES, (
        f"{task}: no specialist produced a valid confidence_basis; "
        f"got {trace['result'].get('confidence_basis')!r}"
    )
    assert len(trace["models_used"]) >= need, (
        f"{task} must show >= {need} models_used entries, got {trace['models_used']}"
    )
    _assert_trace_valid(trace)
    # the file on disk equals what was returned
    on_disk = json.load(open(str(tmp_path / "runs" / trace["run_id"] / "trace.json")))
    assert on_disk == trace


# --------------------------------------------------------------------------- #
# R5: traces written on EVERY run, including failures and crashes.
# --------------------------------------------------------------------------- #


def test_validation_failure_still_writes_trace(two_imgs, tmp_path):
    # change routed from text but only one image supplied -> refused, trace kept
    trace = run_query("What changed between these two dates?", [two_imgs[0]],
                      runs_root=str(tmp_path / "runs"))
    assert trace["task_selected"] == "change"
    assert trace["validation"]["passed"] is False
    assert trace["result"]["status"] == "validation_failed"
    _assert_trace_valid(trace)


def test_mid_inference_crash_writes_trace_then_reraises(two_imgs, tmp_path, monkeypatch):
    import satquery.specialists.vqa as vqa_mod

    def boom(*a, **k):
        raise RuntimeError("simulated specialist crash")

    monkeypatch.setattr(vqa_mod, "run_vqa", boom)

    with pytest.raises(RuntimeError, match="simulated specialist crash"):
        run_query(
            "how many buildings are there in this image",
            [two_imgs[0]],
            runs_root=str(tmp_path / "runs"),
        )

    # exactly one trace must exist, and it must record the error + attempt.
    runs_dir = tmp_path / "runs"
    entry = list(runs_dir.iterdir())[0]
    trace = json.loads((entry / "trace.json").read_text())
    validate_execution_trace(trace)
    assert trace["result"]["status"] == "error"
    assert "simulated specialist crash" in trace["result"]["error"]
    assert trace["task_selected"] == "vqa"
    assert trace["timings_ms"]["inference"] >= 0


# --------------------------------------------------------------------------- #
# models_used: task-planned roles, merged with real pool metadata
# --------------------------------------------------------------------------- #


def test_models_used_for_change_and_fusion_are_two_entries():
    assert TASK_ROLES["change"] == ("benclip", "vqa")
    assert TASK_ROLES["fusion"] == ("benclip", "vqa")


def test_merge_models_planned_entries_when_pool_empty():
    merged = _merge_models("change", ModelPool())
    roles = [m["role"] for m in merged]
    assert roles == ["benclip", "vqa"]
    for m in merged:
        for key in ("role", "name", "revision", "precision", "adapter", "device"):
            assert key in m


def test_merge_models_pool_actual_overrides_plan():
    pool = ModelPool()

    def loader(spec):
        return object(), object()

    pool.register(
        RoleSpec(role="vqa", model_id="fake/vqa-real", precision="fp32", loader=loader)
    )
    pool.acquire("vqa")
    merged = _merge_models("change", pool)
    roles = [m["role"] for m in merged]
    assert roles == ["benclip", "vqa"]
    vqa_entry = next(m for m in merged if m["role"] == "vqa")
    assert vqa_entry["name"] == "fake/vqa-real"
    # role only reported for tasks in the plan: other tasks stay lean
    merged_vqa_only = _merge_models("vqa", pool)
    assert [m["role"] for m in merged_vqa_only] == ["vqa"]


# --------------------------------------------------------------------------- #
# artefact mapping & timings sanity
# --------------------------------------------------------------------------- #


def test_change_result_artifacts_mapped(two_imgs, tmp_path):
    trace = run_query("what changed?", list(two_imgs), runs_root=str(tmp_path / "runs"))
    assert trace["artifacts"]["mask"] is None
    assert set(trace["artifacts"].keys()) >= {"mask", "overlay", "report"}
    assert trace["timings_ms"]["total"] >= 0


def test_write_trace_writes_to_runs_root(tmp_path, two_imgs):
    # Uses a CHANGE query deliberately: this test is about where write_trace puts
    # the file, not about inference, and run_change is still a stub -- so it stays
    # model-free. The original "describe the scene" routed to caption, which after
    # W3 loads Qwen2-VL and hung the CPU-only suite (PLAN.md §5.2 corollary 2).
    trace = run_query(
        "what changed between these two dates?", list(two_imgs), runs_root=str(tmp_path / "runs")
    )
    path = write_trace(trace, runs_root=str(tmp_path / "replayed"))
    assert path == str(tmp_path / "replayed" / trace["run_id"] / "trace.json")
    assert json.load(open(path)) == trace