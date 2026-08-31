"""
W9 model-switch latency: co-resident vqa + grounding.

These tests are CPU-only when CUDA is absent (gated). When CUDA is present
they verify the W9 residency policy without downloading real weights unless
the checkpoint is on disk. The heavy/GPU tests gate on both CUDA and data
presence so a GPU-less clone stays green (PLAN.md §5.2 corollary 2).

VRAM in tests: the autouse fixture releases benclip between tests because
the card is 3.64 GiB usable (see test_w0_stubs.py).
"""

import gc
import os

import pytest
import torch

from satquery.runtime.modelpool import ModelPool, RoleSpec

CUDA = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(not CUDA, reason="requires CUDA for VRAM assertions")

_FIXTURE_MB = 50


class _Fake(torch.nn.Module):
    def __init__(self, mb=_FIXTURE_MB):
        super().__init__()
        n = (mb * 1024 * 1024) // 4
        self.p = torch.nn.Parameter(torch.zeros(n, device="cuda"))


def _loader(mb=_FIXTURE_MB):
    def fn(spec):
        return _Fake(mb), object()
    return fn


@pytest.fixture(autouse=True)
def _free_vram():
    yield
    try:
        from satquery.adapters import benclip as _bc
        _bc.reset_default()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass



@pytest.fixture(autouse=True)
def _opt_in_to_co_residency(monkeypatch):
    """Co-residency is OFF by default (it exceeds W3's 2.5 GiB grounding budget
    and costs benclip evidence in the trace). These tests exercise the opt-in
    path, so they enable it explicitly rather than assuming the default."""
    monkeypatch.setenv("SATQUERY_CO_RESIDENT_MODELS", "1")


def test_vqa_and_grounding_co_reside():
    """The W9 whitelist: vqa + grounding keep each other resident."""
    pool = ModelPool()
    pool.register(RoleSpec(role="vqa", model_id="fake/vqa", precision="fp32", loader=_loader()))
    pool.register(RoleSpec(role="grounding", model_id="fake/gnd", precision="fp32", loader=_loader()))
    pool.acquire("vqa")
    assert pool.resident_heavy_role == "vqa"
    pool.acquire("grounding")
    # Both should be resident after the whitelist fix
    assert set(pool.resident_roles) == {"vqa", "grounding"}
    assert pool.resident_heavy_role == "grounding"  # most recent
    # Re-acquiring vqa must not reload
    calls = {}
    # Use call counting loader to verify no reload
    pool2 = ModelPool()
    cnt = {"vqa": 0, "grounding": 0}

    def counting_loader(key):
        def fn(spec):
            cnt[key] += 1
            return _Fake(), object()
        return fn

    pool2.register(RoleSpec(role="vqa", model_id="fake/vqa", precision="fp32", loader=counting_loader("vqa")))
    pool2.register(RoleSpec(role="grounding", model_id="fake/gnd", precision="fp32", loader=counting_loader("grounding")))
    pool2.acquire("vqa")
    pool2.acquire("grounding")
    pool2.acquire("vqa")
    assert cnt == {"vqa": 1, "grounding": 1}


def test_fake_heavy_roles_remain_single_resident():
    """Non-whitelisted heavies still evict each other (preserves W0 contract)."""
    pool = ModelPool()
    pool.register(RoleSpec(role="fake_a", model_id="fake/a", precision="fp32", loader=_loader()))
    pool.register(RoleSpec(role="fake_b", model_id="fake/b", precision="fp32", loader=_loader()))
    pool.acquire("fake_a")
    pool.acquire("fake_b")
    assert pool.resident_roles == ["fake_b"]
    assert pool.resident_heavy_role == "fake_b"


def test_co_resident_evicts_on_third_non_whitelisted_heavy():
    pool = ModelPool()
    pool.register(RoleSpec(role="vqa", model_id="fake/vqa", precision="fp32", loader=_loader()))
    pool.register(RoleSpec(role="grounding", model_id="fake/gnd", precision="fp32", loader=_loader()))
    pool.register(RoleSpec(role="fake_x", model_id="fake/x", precision="fp32", loader=_loader()))
    pool.acquire("vqa")
    pool.acquire("grounding")
    assert set(pool.resident_roles) == {"vqa", "grounding"}
    pool.acquire("fake_x")
    # Acquiring a non-whitelisted heavy must evict the co-resident pair
    assert pool.resident_roles == ["fake_x"]


def test_benclip_evicts_when_second_heavy_co_resides():
    """Acquiring the second heavy of the pair evicts pool-exempt benclip."""
    pool = ModelPool()
    pool.register(RoleSpec(role="vqa", model_id="fake/vqa", precision="fp32", loader=_loader()))
    pool.register(RoleSpec(role="grounding", model_id="fake/gnd", precision="fp32", loader=_loader()))
    pool.register(RoleSpec(role="benclip", model_id="fake/bc", precision="fp16", loader=_loader(), exempt=True))
    pool.acquire("benclip")
    pool.acquire("vqa")
    assert "benclip" in pool.resident_roles
    pool.acquire("grounding")
    # benclip should have been evicted to free VRAM
    assert "benclip" not in pool.resident_roles
    assert set(pool.resident_roles) == {"vqa", "grounding"}
