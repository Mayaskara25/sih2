"""
satquery/runtime/modelpool.py

Single-resident heavy-model manager (PLAN.md §4.3).

The local GPU is a GTX 1650 with 4 GB VRAM. That is enough for ONE small model
in 4-bit at a time; two resident heavy models OOM. This module is the sole
enforcement point for that rule: every specialist must load its model through
`acquire()` / `using()` here, and none may cache a model in its own module
globals (that is exactly the bug in `old_files/models_registry.py`, which
caches Qwen2-VL and Grounding DINO simultaneously).

Design:
- Construction is lazy. Building a ModelPool touches no GPU and downloads
  nothing; only calling `acquire()`/`using()` triggers a loader.
- Exactly one "heavy" role (vqa, grounding, ...) may be resident at a time.
  Acquiring a different heavy role auto-releases the currently resident one.
- `benclip` is registered as "exempt": small enough to stay resident alongside
  a heavy model, per PLAN.md §3.2/§4.3.
- Loaders are stored on the RoleSpec and are injectable/patchable, so tests
  can register fake roles with cheap loaders instead of downloading real
  multi-GB checkpoints.
"""

from __future__ import annotations

import gc
import os
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - torch is a pinned project dependency;
    # this guard only protects module import if it is ever missing (e.g. a
    # docs-only environment) so importing modelpool never itself requires CUDA.
    torch = None  # type: ignore[assignment]


def _cuda_available() -> bool:
    """CPU-safe CUDA probe. Never raises even if torch failed to import."""
    try:
        return torch is not None and bool(torch.cuda.is_available())
    except Exception:
        return False


def _env_model_id(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


@dataclass
class RoleSpec:
    """Static description of one loadable model role."""

    role: str
    model_id: str
    precision: str
    loader: Callable[["RoleSpec"], Tuple[Any, Any]]
    exempt: bool = False  # True only for benclip (PLAN.md §4.3)
    # The requested revision string (e.g. a HF `revision=` argument), NOT a
    # resolved commit hash -- nothing here pins or measures the actual
    # revision a loader ends up fetching. Default "main" is a placeholder.
    revision: str = "main"


@dataclass
class _Resident:
    """A currently-loaded model + processor, plus the spec that produced it."""

    role: str
    model: Any
    processor: Any
    spec: RoleSpec
    device: str
    loaded_at: float


# --------------------------------------------------------------------------- #
# Default loaders — real network-downloading implementations. Never invoked
# directly by tests (that would download multi-GB weights); tests register
# fake RoleSpecs with cheap loaders instead. These are wired into
# DEFAULT_REGISTRY below, which is the "live caller" that keeps them non-dead.
# --------------------------------------------------------------------------- #


def _load_vqa(spec: RoleSpec) -> Tuple[Any, Any]:
    """Load the VQA/captioning VLM (Qwen2-VL) in 4-bit NF4 on CUDA, fp32 on CPU."""
    import torch as _torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2VLForConditionalGeneration,
    )

    if _cuda_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            spec.model_id,
            quantization_config=quant_config,
            device_map={"": "cuda:0"},
            torch_dtype=_torch.float16,
            trust_remote_code=True,
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            spec.model_id,
            device_map="cpu",
            torch_dtype=_torch.float32,
            trust_remote_code=True,
        )
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.model_id, trust_remote_code=True)
    return model, processor


def _load_grounding(spec: RoleSpec) -> Tuple[Any, Any]:
    """Load Grounding DINO in fp16 on CUDA, fp32 on CPU."""
    import torch as _torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if _cuda_available() else "cpu"
    dtype = _torch.float16 if _cuda_available() else _torch.float32
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        spec.model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(spec.model_id, trust_remote_code=True)
    return model, processor


def _load_benclip(spec: RoleSpec) -> Tuple[Any, Any]:
    """
    Load the benclip Track A checkpoint (PLAN.md §3.1, §4.5).

    W2 owns `satquery/adapters/benclip.py` and the actual checkpoint; at W0
    time neither exists yet. Per the work order this must fail loudly and
    actionably rather than fake a model.
    """
    checkpoint_path = spec.model_id
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"benclip checkpoint not found at '{checkpoint_path}'. PLAN.md "
            "§3.1/§4.5 assigns training this checkpoint to W2, which has not "
            "produced one yet. Set SATQUERY_BENCLIP_PATH to a real checkpoint "
            "directory once W2 lands one, or use ModelPool.register() to "
            "inject a test double."
        )
    raise NotImplementedError(
        f"A path exists at '{checkpoint_path}' but modelpool.py does not itself "
        "know how to load a benclip checkpoint -- that loader belongs to W2's "
        "satquery/adapters/benclip.py (load_benclip()). Wire that function in "
        "as this role's RoleSpec.loader once it exists."
    )


DEFAULT_REGISTRY: Dict[str, RoleSpec] = {
    "vqa": RoleSpec(
        role="vqa",
        model_id=_env_model_id("SATQUERY_VQA_MODEL", "Qwen/Qwen2-VL-2B-Instruct"),
        precision="4bit-nf4",
        loader=_load_vqa,
        exempt=False,
    ),
    "grounding": RoleSpec(
        role="grounding",
        model_id=_env_model_id(
            "SATQUERY_GROUNDING_MODEL", "IDEA-Research/grounding-dino-tiny"
        ),
        precision="fp16",
        loader=_load_grounding,
        exempt=False,
    ),
    "benclip": RoleSpec(
        role="benclip",
        model_id=_env_model_id("SATQUERY_BENCLIP_PATH", "checkpoints/benclip"),
        precision="fp16",
        loader=_load_benclip,
        exempt=True,
    ),
}


class ModelPool:
    """
    Enforces PLAN.md §4.3: at most one non-exempt ("heavy") model resident at
    a time, with `benclip` exempt and allowed to coexist with a heavy model.

    Construction is lazy: building a ModelPool loads nothing and touches no
    GPU. Use `acquire(role)` / `release()` directly, or the ergonomic
    `with pool.using(role) as (model, processor): ...` form.
    """

    def __init__(self, registry: Optional[Dict[str, RoleSpec]] = None) -> None:
        self._registry: Dict[str, RoleSpec] = dict(
            registry if registry is not None else DEFAULT_REGISTRY
        )
        self._heavy: Optional[_Resident] = None
        self._exempt: Dict[str, _Resident] = {}
        self._lock = threading.Lock()
        # Sticky across a release() so get_execution_metadata() can still
        # report what was "most recently loaded" for the trace (PLAN.md §4.2)
        # even after the model has been freed.
        self._last_heavy_metadata: Optional[Dict[str, Any]] = None

    def register(self, spec: RoleSpec) -> None:
        """Add or override a role spec. Used by tests to inject fake loaders."""
        self._registry[spec.role] = spec

    @property
    def resident_heavy_role(self) -> Optional[str]:
        """The single resident heavy role's name, or None if none is loaded."""
        return self._heavy.role if self._heavy is not None else None

    @property
    def resident_roles(self) -> List[str]:
        """All currently resident roles: every exempt role plus the heavy one."""
        roles = list(self._exempt.keys())
        if self._heavy is not None:
            roles.append(self._heavy.role)
        return roles

    def acquire(self, role: str) -> Tuple[Any, Any]:
        """
        Load (or return the already-resident) model+processor for `role`.

        Acquiring a heavy role different from the one currently resident
        auto-releases the current one first, so at most one heavy model is
        ever resident. Acquiring an exempt role (benclip) never evicts a
        heavy role and vice versa. Re-acquiring the same resident role is a
        no-op, not a reload.
        """
        if role not in self._registry:
            raise KeyError(
                f"Unknown model role '{role}'. Known roles: {sorted(self._registry)}"
            )
        spec = self._registry[role]

        with self._lock:
            if spec.exempt:
                resident = self._exempt.get(role)
                if resident is not None:
                    return resident.model, resident.processor
                model, processor = spec.loader(spec)
                resident = _Resident(
                    role=role,
                    model=model,
                    processor=processor,
                    spec=spec,
                    device=self._device_of(model),
                    loaded_at=time.time(),
                )
                self._exempt[role] = resident
                return model, processor

            if self._heavy is not None and self._heavy.role == role:
                return self._heavy.model, self._heavy.processor

            # A different heavy role (or none) is resident: release it first
            # so exactly one heavy model is ever resident (PLAN.md §4.3).
            self._release_heavy_locked()

            model, processor = spec.loader(spec)
            resident = _Resident(
                role=role,
                model=model,
                processor=processor,
                spec=spec,
                device=self._device_of(model),
                loaded_at=time.time(),
            )
            self._heavy = resident
            self._last_heavy_metadata = self._metadata_entry(resident)
            return model, processor

    def release(self, role: Optional[str] = None) -> None:
        """
        Free a resident model, drop references, gc.collect(), and empty the
        CUDA cache.

        With no argument, releases the resident heavy model (if any). Pass an
        exempt role's name to release that specific exempt model instead.
        Safe to call when nothing is resident.

        Raises RuntimeError, and leaves the role marked resident, if a caller
        elsewhere still holds a reference to the model/processor after the
        pool drops its own -- see `_release_heavy_locked` for why this can't
        be a silent no-op.
        """
        with self._lock:
            if role is None:
                self._release_heavy_locked()
                return
            if self._heavy is not None and self._heavy.role == role:
                self._release_heavy_locked()
                return
            if role in self._exempt:
                del self._exempt[role]
                gc.collect()
                if _cuda_available():
                    torch.cuda.empty_cache()

    def _release_heavy_locked(self) -> None:
        """
        Caller must hold self._lock. Safe no-op if nothing heavy is resident.

        gc.collect() only actually frees the model if the pool's `_Resident`
        was the last strong reference to it. If a specialist kept a local
        variable bound to the model/processor past its `using()`/`acquire()`
        call (e.g. two `with model_pool.using(...)` blocks in the same
        function, where the first target variable is still in scope when the
        second is acquired), the object survives the collect and the tensor
        stays on the GPU -- silently, with nothing in the API telling anyone.
        Loading a second heavy model on top of that is precisely the 4GB OOM
        this module exists to prevent (PLAN.md §4.3), so this checks for that
        with a weakref and fails loudly instead of proceeding.
        """
        if self._heavy is None:
            return
        # Extract only the small, non-GPU fields up front. Deliberately do
        # NOT keep a local variable bound to the `_Resident` wrapper (or its
        # .model/.processor) across the check below -- that reference would
        # itself keep the model alive and make every release() look "leaked".
        role = self._heavy.role
        spec = self._heavy.spec
        device = self._heavy.device
        loaded_at = self._heavy.loaded_at
        model_ref = self._safe_weakref(self._heavy.model)
        processor_ref = self._safe_weakref(self._heavy.processor)

        self._heavy = None  # the pool's own, intended-to-be-only, reference
        gc.collect()
        if _cuda_available():
            torch.cuda.empty_cache()

        model_survivor = model_ref() if model_ref is not None else None
        processor_survivor = processor_ref() if processor_ref is not None else None
        if model_survivor is not None or processor_survivor is not None:
            # Something outside the pool still holds a reference, so the
            # model is, in fact, still resident in memory. Rebuild the
            # bookkeeping to reflect that rather than silently losing track
            # of it, and fail loudly.
            self._heavy = _Resident(
                role=role,
                model=model_survivor,
                processor=processor_survivor,
                spec=spec,
                device=device,
                loaded_at=loaded_at,
            )
            raise RuntimeError(
                f"modelpool: could not release heavy role '{role}' "
                f"(model_id='{spec.model_id}') -- something still holds a "
                "reference to its model or processor after gc.collect(). "
                "This role remains marked resident rather than being silently "
                "dropped. PLAN.md §4.3 allows exactly one heavy model resident "
                "at a time; proceeding to load a different heavy role now would "
                "double up VRAM usage on a 4GB card. Fix the caller so the "
                "model/processor from acquire()/using() is not kept alive past "
                "the call that needed it -- use `with model_pool.using(role) as "
                "(model, processor): ...` and let that scope end before "
                "acquiring a different heavy role."
            )

    @staticmethod
    def _safe_weakref(obj: Any) -> Optional["weakref.ReferenceType[Any]"]:
        """weakref.ref(), or None for objects that don't support it (e.g. a
        bare `object()`, as used by some tests/stub loaders) -- best-effort
        leak detection must never itself crash release()."""
        if obj is None:
            return None
        try:
            return weakref.ref(obj)
        except TypeError:
            return None

    @staticmethod
    def _device_of(model: Any) -> str:
        try:
            return str(next(model.parameters()).device)
        except Exception:
            return "cuda" if _cuda_available() else "cpu"

    @staticmethod
    def _metadata_entry(resident: _Resident) -> Dict[str, Any]:
        return {
            "role": resident.role,
            "name": resident.spec.model_id,
            "revision": resident.spec.revision,
            "precision": resident.spec.precision,
            "adapter": None,
            "device": resident.device,
        }

    def get_execution_metadata(self) -> List[Dict[str, Any]]:
        """
        Model identifiers/revision/precision/device for every currently (or,
        for the heavy slot, most recently) loaded role.

        Shaped to be appended directly into trace.json's `models_used` list
        (PLAN.md §4.2: role/name/revision/precision/adapter), with `device`
        included as an extra, controller-useful field. Callable at any point
        in a run -- including after release() -- so a specialist that has
        already freed its model still leaves a model record in the trace,
        unlike the old draft where an equivalent function existed and was
        simply never called (PLAN.md §2.5).
        """
        entries: List[Dict[str, Any]] = [
            self._metadata_entry(resident) for resident in self._exempt.values()
        ]
        if self._heavy is not None:
            entries.append(self._metadata_entry(self._heavy))
        elif self._last_heavy_metadata is not None:
            entries.append(dict(self._last_heavy_metadata))
        return entries

    @contextmanager
    def using(self, role: str) -> Iterator[Tuple[Any, Any]]:
        """
        Ergonomic default for specialists:

            with model_pool.using("vqa") as (model, processor):
                ...

        The model remains resident after the `with` block exits -- release
        happens lazily, on the next acquire() of a different heavy role, or
        explicitly via release(). This is what makes re-use across calls to
        the same specialist free instead of a reload every time.
        """
        model, processor = self.acquire(role)
        yield model, processor


# Module-level singleton. Constructing it is free (no GPU touch, no download);
# every specialist should import and use this instance rather than building
# its own ModelPool, so the single-resident guarantee is repo-wide.
model_pool = ModelPool()
