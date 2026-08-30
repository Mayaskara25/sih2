"""Pre-dispatch input compatibility checks (PLAN.md §4.1/§4.2/§4.4, W6).

Builds a ``ValidationInfo`` (passed / warnings / errors) and one ``InputInfo``
per image. All raster opening goes through ``satquery.io.load_raster`` — this
module never calls rasterio or PIL itself (§5.3; ``raster.py`` is the only
reader in the repo).

Check inventory (matters for the §2.3 hidden set and the R2 cross-modal row):

  * file exists; format decodes; band count plausible for the claimed modality
  * image count matches the routed task (change/fusion need exactly 2;
    vqa/caption/grounding need >=1)
  * change: both inputs comparable (same size/wavelength; same CRS if both
    georeferenced) — WARN, don't hard-fail, since W4 can register
  * fusion: the two inputs must be GENUINELY DISTINCT modalities. Two optical /
    msi / optical+msi inputs is a HARD ERROR (R2). One SAR input is required.
  * modality "unknown" is a WARNING, never an error — W0's resolver returns
    unknown for bare 1- or 2-band files that genuinely look like real Cartosat
    PAN and RISAT SAR products. Surface it loudly so the app can prompt, but
    never refuse to run on it.

Warnings are recorded and proceed; errors refuse dispatch but the trace is
still written. ``validate_inputs`` raises ``ValueError`` only for a bogus
task name or a mismatched ``forced_modalities`` list — never for bad images.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from satquery.contracts import InputInfo, TASK_NAMES, ValidationInfo
from satquery.io.raster import RasterReadError, load_raster
from satquery.io.modality import VALID_MODALITIES

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "PLAUSIBLE_BANDS",
    "ValidationResult",
    "inspect_input",
    "check_compatibility",
    "validate_inputs",
]

SUPPORTED_EXTENSIONS = frozenset(
    {"tif", "tiff", "png", "jpg", "jpeg", "webp", "bmp", "jp2"}
)

# Band counts this system is willing to call "plausible" for each claimed
# modality. Anything outside the set is a warning (W2's benclip maps by band
# count as a soft fallback) rather than an error — the graded ISRO data is
# exactly where odd band shapes live.
PLAUSIBLE_BANDS = {
    "optical": frozenset({1, 3, 4, 12, 13}),  # PAN, RGB, Cartosat MSI, S2
    "msi": frozenset({3, 4, 12, 13}),
    "sar": frozenset({1, 2, 4}),  # single-pol, dual-pol, quad-pol
    "unknown": None,  # anything is plausible for an unclaimed modality
}


def _format_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    if ext in ("tif", "tiff"):
        return "GeoTIFF"
    return ext.upper() if ext else "unknown"


def _empty_input_info(path: str, reason: str) -> InputInfo:
    return {
        "path": path,
        "modality": "unknown",
        "bands": 0,
        "shape": [0, 0],
        "format": _format_for(path),
        "crs": None,
        "checks_passed": False,
    }


@dataclass
class ValidationResult:
    """Per-run validation outcome: the trace's validation dict plus the
    per-image InputInfos the controller needs for dispatch and the trace."""

    passed: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    input_infos: List[InputInfo] = field(default_factory=list)

    def to_validation_info(self) -> ValidationInfo:
        return {
            "passed": self.passed,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def inspect_input(
    path: str,
    *,
    forced_modality: Optional[str] = None,
) -> tuple[InputInfo, List[str], List[str]]:
    """Inspect one image through ``load_raster`` and return its ``InputInfo``
    plus any per-image warnings/errors.

    ``forced_modality`` is the app's explicit selector (PLAN.md §4.4 tier 1),
    passed straight through to ``load_raster``; it is never re-derived here.
    """
    warnings: List[str] = []
    errors: List[str] = []
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    if ext and ext not in SUPPORTED_EXTENSIONS:
        warnings.append(
            f"{path}: extension '.{ext}' is not in the supported set "
            f"({sorted(SUPPORTED_EXTENSIONS)}); attempting decode anyway"
        )

    try:
        raster = load_raster(path, modality=forced_modality)
    except FileNotFoundError as exc:
        errors.append(str(exc))
        return _empty_input_info(path, str(exc)), warnings, errors
    except (RasterReadError, ValueError, OSError) as exc:
        errors.append(f"{path}: could not be read: {exc}")
        return _empty_input_info(path, str(exc)), warnings, errors
    except Exception as exc:  # pragmatic catch-all: a decode must not take the app down
        errors.append(f"{path}: unexpected failure while reading: {type(exc).__name__}: {exc}")
        return _empty_input_info(path, str(exc)), warnings, errors

    info: InputInfo = {
        "path": path,
        "modality": raster.modality,
        "bands": int(raster.band_count),
        "shape": [int(raster.array.shape[0]), int(raster.array.shape[1])],
        "format": _format_for(path),
        "crs": raster.crs,
        "checks_passed": True,
    }

    plausible = PLAUSIBLE_BANDS.get(forced_modality or raster.modality)
    if plausible is not None and raster.band_count not in plausible:
        warnings.append(
            f"{path}: {raster.band_count}-band input is unusual for modality "
            f"'{forced_modality or raster.modality}' (expected one of "
            f"{sorted(plausible)}); benclip will fall back to band-count "
            "mapping for embedding"
        )

    if raster.modality == "unknown":
        # WORK ORDER: unknown is a WARNING, not an error. W0's resolver
        # deliberately returns unknown for a bare 1- or 2-band file with no
        # filename hint — which is exactly what real Cartosat PAN and RISAT
        # products look like. Surface it loudly so the app can prompt.
        warnings.append(
            f"{path}: resolved modality is 'unknown' — a bare "
            f"{raster.band_count}-band raster with no filename/metadata hint "
            "(this is exactly what real Cartosat PAN and RISAT products look "
            "like). Set an explicit modality in the UI if you know it; the "
            "run will proceed with the ambiguity recorded."
        )

    if raster.modality not in VALID_MODALITIES:  # pragmatic guard, unreachable today
        warnings.append(
            f"{path}: resolver reported unhandled modality {raster.modality!r}"
        )

    return info, warnings, errors


def _fusion_modality_check(mods: List[str], warnings: List[str], errors: List[str]) -> None:
    """R2 check: fusion needs two GENUINELY DISTINCT modalities, i.e. at least
    one SAR input. Two optical-family inputs (optical/msi/optical) is a hard
    error; 'unknown' anywhere is a warning, never a refusal."""
    has_sar = any(m == "sar" for m in mods)
    known = [m for m in mods if m != "unknown"]
    if has_sar:
        if all(m == "sar" for m in mods):
            warnings.append(
                f"both fusion inputs resolved as SAR ({mods}); fusion assumes an "
                "optical/non-SAR input as its optical side — proceeding with the "
                "first input as the optical channel"
            )
        elif any(m == "unknown" for m in mods):
            warnings.append(
                f"one fusion input resolved to 'unknown' ({mods}); proceeding, but "
                "verify the pair really is optical + SAR"
            )
        return
    # No SAR input at all.
    if len(known) >= 1 and len(known) == len(mods):
        errors.append(
            f"fusion requires two genuinely distinct modalities with a SAR input "
            f"(R2); both inputs resolved as optical-family modalities {mods} — "
            "refusing the cross-modal run"
        )
    else:
        warnings.append(
            f"fusion inputs are {mods} — no input was confidently tagged SAR; "
            "proceeding but the cross-modal claim cannot be verified"
        )


def check_compatibility(
    input_infos: Sequence[InputInfo],
    task: str,
    *,
    query: str = "",
    selected_image_index: int = 0,
) -> ValidationResult:
    """Task-level checks over the already-inspected inputs."""
    warnings: List[str] = []
    errors: List[str] = []
    infos = list(input_infos)
    n = len(infos)

    min_count, max_count = (2, 2) if task in ("change", "fusion") else (1, 2)
    if n < min_count:
        errors.append(f"task '{task}' requires at least {min_count} image(s); got {n}")
    if n > max_count:
        errors.append(f"task '{task}' accepts at most {max_count} image(s); got {n}")

    if task in ("vqa", "caption", "grounding") and n > 0:
        if not (0 <= selected_image_index < n):
            errors.append(
                f"query selects image index {selected_image_index} for a "
                f"single-image task, but only {n} image(s) were supplied"
            )

    if n >= 2:
        a, b = infos[0], infos[1]
        mods = [a["modality"], b["modality"]]

        if task == "change":
            if a["shape"] != b["shape"]:
                warnings.append(
                    f"change inputs differ in size ({a['shape']} vs {b['shape']}); "
                    "W4 registration will resample/crop — proceeding"
                )
            if a["crs"] and b["crs"]:
                if a["crs"] != b["crs"]:
                    warnings.append(
                        f"change inputs have different CRS ({a['crs']} vs "
                        f"{b['crs']}); W4 will register — proceeding"
                    )
            elif (a["crs"] is None) != (b["crs"] is None):
                warnings.append(
                    "only one change input is georeferenced, so co-registration "
                    "cannot be skipped; W4 registration will run"
                )
            known_mods = [m for m in mods if m != "unknown"]
            if len(set(known_mods)) > 1:
                warnings.append(
                    f"change normally compares two dates of the SAME modality, "
                    f"but inputs resolved to {mods}; proceeding via registration, "
                    "and consider fusion if the intent is cross-modal"
                )

        elif task == "fusion":
            _fusion_modality_check(mods, warnings, errors)

    # Error count drives `passed`; warnings never do.
    return ValidationResult(passed=len(errors) == 0, warnings=warnings, errors=errors)


def validate_inputs(
    paths: Sequence[str],
    task: str,
    *,
    forced_modalities: Optional[Sequence[Optional[str]]] = None,
    query: str = "",
    selected_image_index: int = 0,
) -> ValidationResult:
    """Full pre-dispatch validation: inspect every input, then run the
    task-level compatibility checks. Returns a ``ValidationResult``; raises
    ``ValueError`` only for a bogus task name or a mismatched modality list."""
    if task not in TASK_NAMES:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASK_NAMES)}")

    paths = list(paths)
    if forced_modalities is None:
        forced_modalities = [None] * len(paths)
    forced = list(forced_modalities)
    if len(forced) != len(paths):
        raise ValueError(
            "forced_modalities length must match the number of image paths "
            f"(got {len(forced)} for {len(paths)} paths)"
        )
    for fm in forced:
        if fm is not None and fm not in VALID_MODALITIES:
            raise ValueError(
                f"invalid forced modality {fm!r}; expected one of {VALID_MODALITIES}"
            )

    infos: List[InputInfo] = []
    warnings: List[str] = []
    errors: List[str] = []
    for path, fm in zip(paths, forced):
        info, wi, ei = inspect_input(path, forced_modality=fm)
        infos.append(info)
        warnings.extend(wi)
        errors.extend(ei)

    compat = check_compatibility(
        infos,
        task,
        query=query,
        selected_image_index=selected_image_index,
    )
    warnings.extend(compat.warnings)
    errors.extend(compat.errors)
    return ValidationResult(
        passed=len(errors) == 0,
        warnings=warnings,
        errors=errors,
        input_infos=infos,
    )