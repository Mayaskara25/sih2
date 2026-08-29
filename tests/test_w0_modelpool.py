"""
Tests for satquery/runtime/modelpool.py (PLAN.md §4.3, W0 acceptance test).

Key acceptance criterion: acquiring role B after role A leaves exactly ONE
model resident on the GPU -- asserted on torch.cuda.memory_allocated, not
eyeballed.

These tests never download a real model. Fake heavy roles are registered
with loaders that allocate a deliberately large-ish CUDA tensor (~200 MB)
wrapped in a torch.nn.Module (so device-detection via `.parameters()` works
the same way it would for a real model).
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from satquery.runtime.modelpool import (
    DEFAULT_REGISTRY,
    ModelPool,
    RoleSpec,
    _load_benclip,
    _load_grounding,
    _load_vqa,
)

CUDA_AVAILABLE = torch.cuda.is_available()

_FAKE_MB = 200
_FAKE_BYTES = _FAKE_MB * 1024 * 1024


class _FakeHeavyModel(torch.nn.Module):
    """Stand-in for a real heavy model: holds one big CUDA parameter."""

    def __init__(self, size_mb: int = _FAKE_MB):
        super().__init__()
        n_floats = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
        self.big = torch.nn.Parameter(
            torch.zeros(n_floats, device="cuda"), requires_grad=False
        )


def _make_cuda_loader(call_counter: dict, key: str, size_mb: int = _FAKE_MB):
    def loader(spec: RoleSpec):
        call_counter[key] = call_counter.get(key, 0) + 1
        return _FakeHeavyModel(size_mb), object()

    return loader


def _make_cpu_loader(call_counter: dict, key: str):
    class _FakeCPUModel:
        pass

    def loader(spec: RoleSpec):
        call_counter[key] = call_counter.get(key, 0) + 1
        return _FakeCPUModel(), object()

    return loader


# --------------------------------------------------------------------------- #
# The key acceptance criterion, GPU version
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a CUDA device")
def test_single_heavy_resident_on_cuda_memory():
    pool = ModelPool()
    calls: dict = {}
    pool.register(
        RoleSpec(
            role="fake_a",
            model_id="fake/a",
            precision="fp32",
            loader=_make_cuda_loader(calls, "a"),
        )
    )
    pool.register(
        RoleSpec(
            role="fake_b",
            model_id="fake/b",
            precision="fp32",
            loader=_make_cuda_loader(calls, "b"),
        )
    )

    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()

    pool.acquire("fake_a")
    after_a = torch.cuda.memory_allocated()
    delta_a = after_a - baseline
    # allocator overhead means this won't be exact; require it's in the
    # right ballpark for one ~200MB tensor.
    assert 0.7 * _FAKE_BYTES < delta_a < 1.5 * _FAKE_BYTES, (
        f"expected ~{_FAKE_BYTES} bytes resident after acquiring fake_a, got {delta_a}"
    )
    assert pool.resident_heavy_role == "fake_a"

    pool.acquire("fake_b")
    after_b = torch.cuda.memory_allocated()
    delta_b = after_b - baseline
    # If both fake_a and fake_b were resident simultaneously this would be
    # ~2x delta_a (~400MB). Assert it stayed in one-model territory instead.
    assert delta_b < 1.5 * _FAKE_BYTES, (
        f"memory after acquiring fake_b ({delta_b} bytes) looks like TWO models "
        f"are resident, not one (one-model ballpark is ~{_FAKE_BYTES} bytes)"
    )
    assert 0.7 * _FAKE_BYTES < delta_b, "fake_b does not appear to be resident at all"

    assert pool.resident_heavy_role == "fake_b"
    assert pool.resident_roles == ["fake_b"]
    assert calls == {"a": 1, "b": 1}

    pool.release()
    gc.collect()
    torch.cuda.empty_cache()
    after_release = torch.cuda.memory_allocated()
    assert after_release - baseline < 0.3 * _FAKE_BYTES
    assert pool.resident_heavy_role is None


# --------------------------------------------------------------------------- #
# CPU-side equivalent: meaningful even with no GPU present.
# --------------------------------------------------------------------------- #


def test_single_heavy_resident_cpu_weakref():
    pool = ModelPool()
    calls: dict = {}
    refs: dict = {}

    def make_tracked_loader(key: str):
        class _FakeCPUModel:
            pass

        def loader(spec: RoleSpec):
            calls[key] = calls.get(key, 0) + 1
            obj = _FakeCPUModel()
            refs[key] = weakref.ref(obj)
            return obj, object()

        return loader

    pool.register(
        RoleSpec(
            role="fake_a", model_id="fake/a", precision="fp32", loader=make_tracked_loader("a")
        )
    )
    pool.register(
        RoleSpec(
            role="fake_b", model_id="fake/b", precision="fp32", loader=make_tracked_loader("b")
        )
    )

    pool.acquire("fake_a")
    assert refs["a"]() is not None
    assert pool.resident_heavy_role == "fake_a"

    pool.acquire("fake_b")
    gc.collect()
    # fake_a's model must have been dropped -- no lingering reference anywhere
    # in the pool -- once fake_b (a different heavy role) was acquired.
    assert refs["a"]() is None, "pool kept a reference to the previous heavy model"
    assert refs["b"]() is not None
    assert pool.resident_heavy_role == "fake_b"
    assert pool.resident_roles == ["fake_b"]
    assert calls == {"a": 1, "b": 1}


# --------------------------------------------------------------------------- #
# Leak detection: a caller holding a reference past its acquire()/using()
# call must not let a second heavy model load silently on top of it.
# --------------------------------------------------------------------------- #


def test_acquire_raises_if_previous_model_still_referenced_cpu():
    pool = ModelPool()
    calls: dict = {}

    class _FakeModel:
        pass

    def make_loader(key: str):
        def loader(spec: RoleSpec):
            calls[key] = calls.get(key, 0) + 1
            return _FakeModel(), object()

        return loader

    pool.register(
        RoleSpec(role="fake_a", model_id="fake/a", precision="fp32", loader=make_loader("a"))
    )
    pool.register(
        RoleSpec(role="fake_b", model_id="fake/b", precision="fp32", loader=make_loader("b"))
    )

    # Caller keeps a live reference to fake_a's model -- exactly the pattern
    # (e.g. two `with model_pool.using(...)` blocks in one function) that
    # would otherwise leave a stale model resident when swapping roles.
    model_a, _proc_a = pool.acquire("fake_a")

    with pytest.raises(RuntimeError, match="still holds a reference"):
        pool.acquire("fake_b")

    # fake_b must never have been loaded -- we must fail BEFORE double-loading.
    assert calls == {"a": 1}
    assert pool.resident_heavy_role == "fake_a"

    # Dropping the caller's reference lets the swap succeed afterward.
    del model_a
    gc.collect()
    pool.acquire("fake_b")
    assert calls == {"a": 1, "b": 1}
    assert pool.resident_heavy_role == "fake_b"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a CUDA device")
def test_acquire_raises_if_previous_heavy_model_still_referenced_cuda():
    pool = ModelPool()
    calls: dict = {}
    pool.register(
        RoleSpec(
            role="fake_a", model_id="fake/a", precision="fp32", loader=_make_cuda_loader(calls, "a")
        )
    )
    pool.register(
        RoleSpec(
            role="fake_b", model_id="fake/b", precision="fp32", loader=_make_cuda_loader(calls, "b")
        )
    )

    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()

    model_a, _proc_a = pool.acquire("fake_a")  # caller keeps a live reference

    with pytest.raises(RuntimeError):
        pool.acquire("fake_b")

    after_attempt = torch.cuda.memory_allocated()
    # fake_b must never have actually been loaded onto the card.
    assert after_attempt - baseline < 1.5 * _FAKE_BYTES
    assert calls == {"a": 1}
    assert pool.resident_heavy_role == "fake_a"

    del model_a
    gc.collect()
    pool.acquire("fake_b")
    assert calls == {"a": 1, "b": 1}
    assert pool.resident_heavy_role == "fake_b"


# --------------------------------------------------------------------------- #
# release() safety and no-reload-on-reacquire
# --------------------------------------------------------------------------- #


def test_release_with_nothing_resident_is_safe():
    pool = ModelPool()
    pool.release()  # must not raise
    assert pool.resident_heavy_role is None
    assert pool.resident_roles == []


def test_reacquire_same_role_does_not_reload():
    pool = ModelPool()
    calls: dict = {}
    pool.register(
        RoleSpec(
            role="fake", model_id="fake/x", precision="fp32", loader=_make_cpu_loader(calls, "n")
        )
    )
    m1, p1 = pool.acquire("fake")
    m2, p2 = pool.acquire("fake")
    m3, p3 = pool.acquire("fake")
    assert calls == {"n": 1}
    assert m1 is m2 is m3
    assert p1 is p2 is p3


def test_unknown_role_raises_keyerror():
    pool = ModelPool()
    with pytest.raises(KeyError):
        pool.acquire("does_not_exist")


# --------------------------------------------------------------------------- #
# benclip exemption
# --------------------------------------------------------------------------- #


def test_benclip_exempt_coexists_with_heavy_role():
    pool = ModelPool()
    calls: dict = {}
    pool.register(
        RoleSpec(
            role="fake_heavy",
            model_id="fake/heavy",
            precision="fp32",
            loader=_make_cpu_loader(calls, "heavy"),
        )
    )
    pool.register(
        RoleSpec(
            role="benclip",
            model_id="fake/benclip-checkpoint",
            precision="fp16",
            loader=_make_cpu_loader(calls, "benclip"),
            exempt=True,
        )
    )

    pool.acquire("benclip")
    pool.acquire("fake_heavy")

    assert pool.resident_heavy_role == "fake_heavy"
    assert set(pool.resident_roles) == {"benclip", "fake_heavy"}
    assert calls == {"benclip": 1, "heavy": 1}

    # Releasing the heavy role must not evict the exempt benclip model.
    pool.release()
    assert pool.resident_heavy_role is None
    assert pool.resident_roles == ["benclip"]


def test_benclip_default_loader_fails_clearly_without_checkpoint():
    pool = ModelPool()  # uses the real DEFAULT_REGISTRY benclip spec
    with pytest.raises((FileNotFoundError, NotImplementedError)) as excinfo:
        pool.acquire("benclip")
    message = str(excinfo.value).lower()
    assert "benclip" in message
    assert "checkpoint" in message


# --------------------------------------------------------------------------- #
# using() context manager
# --------------------------------------------------------------------------- #


def test_using_context_manager_keeps_model_resident_after_exit():
    pool = ModelPool()
    calls: dict = {}

    def loader(spec: RoleSpec):
        calls["n"] = calls.get("n", 0) + 1
        return "MODEL", "PROC"

    pool.register(RoleSpec(role="fake", model_id="fake/x", precision="fp32", loader=loader))
    with pool.using("fake") as (model, processor):
        assert model == "MODEL"
        assert processor == "PROC"
    assert pool.resident_heavy_role == "fake"
    assert calls == {"n": 1}


# --------------------------------------------------------------------------- #
# get_execution_metadata()
# --------------------------------------------------------------------------- #


def test_get_execution_metadata_shape_and_content():
    pool = ModelPool()

    def loader(spec: RoleSpec):
        return object(), object()

    pool.register(
        RoleSpec(
            role="fake",
            model_id="fake/meta-model",
            precision="fp32",
            loader=loader,
            revision="abc123",
        )
    )
    pool.acquire("fake")

    metadata = pool.get_execution_metadata()
    assert isinstance(metadata, list)
    assert len(metadata) == 1
    entry = metadata[0]
    for key in ("role", "name", "revision", "precision", "adapter", "device"):
        assert key in entry
    assert entry["role"] == "fake"
    assert entry["name"] == "fake/meta-model"
    assert entry["revision"] == "abc123"
    assert entry["precision"] == "fp32"
    assert entry["adapter"] is None


def test_get_execution_metadata_empty_pool():
    pool = ModelPool()
    assert pool.get_execution_metadata() == []


def test_get_execution_metadata_survives_release_as_most_recent():
    pool = ModelPool()

    def loader(spec: RoleSpec):
        return object(), object()

    pool.register(
        RoleSpec(role="fake", model_id="fake/meta", precision="fp32", loader=loader)
    )
    pool.acquire("fake")
    pool.release()
    assert pool.resident_heavy_role is None
    metadata = pool.get_execution_metadata()
    assert len(metadata) == 1
    assert metadata[0]["role"] == "fake"


def test_get_execution_metadata_includes_exempt_and_heavy():
    pool = ModelPool()

    def loader(spec: RoleSpec):
        return object(), object()

    pool.register(RoleSpec(role="fake_heavy", model_id="fake/h", precision="fp32", loader=loader))
    pool.register(
        RoleSpec(
            role="benclip", model_id="fake/bc", precision="fp16", loader=loader, exempt=True
        )
    )
    pool.acquire("benclip")
    pool.acquire("fake_heavy")

    roles = {e["role"] for e in pool.get_execution_metadata()}
    assert roles == {"benclip", "fake_heavy"}


# --------------------------------------------------------------------------- #
# Default registry shape (touches, but never invokes, the real loaders --
# invoking them would download multi-GB weights).
# --------------------------------------------------------------------------- #


def test_default_registry_shape():
    assert DEFAULT_REGISTRY["vqa"].precision == "4bit-nf4"
    assert DEFAULT_REGISTRY["vqa"].loader is _load_vqa
    assert DEFAULT_REGISTRY["vqa"].exempt is False
    assert "Qwen2-VL" in DEFAULT_REGISTRY["vqa"].model_id

    assert DEFAULT_REGISTRY["grounding"].precision == "fp16"
    assert DEFAULT_REGISTRY["grounding"].loader is _load_grounding
    assert DEFAULT_REGISTRY["grounding"].exempt is False
    assert "grounding-dino" in DEFAULT_REGISTRY["grounding"].model_id

    assert DEFAULT_REGISTRY["benclip"].loader is _load_benclip
    assert DEFAULT_REGISTRY["benclip"].exempt is True


def test_model_ids_overridable_by_env_var(monkeypatch):
    monkeypatch.setenv("SATQUERY_VQA_MODEL", "some/override-model")
    # Re-import-equivalent: exercise the same helper the module uses at
    # import time, since the module-level DEFAULT_REGISTRY was already built
    # with the environment as it was when modelpool.py was first imported.
    from satquery.runtime.modelpool import _env_model_id

    assert _env_model_id("SATQUERY_VQA_MODEL", "default") == "some/override-model"
    assert _env_model_id("SATQUERY_UNSET_VAR_XYZ", "default") == "default"


# --------------------------------------------------------------------------- #
# Module-level singleton exists and is safe to construct/import.
# --------------------------------------------------------------------------- #


def test_module_singleton_importable_and_lazy():
    from satquery.runtime.modelpool import model_pool

    assert isinstance(model_pool, ModelPool)
    # Constructing/importing the pool must not have loaded anything.
    assert model_pool.resident_roles == []
