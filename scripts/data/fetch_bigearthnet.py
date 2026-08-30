"""Fetch a paired Sentinel-1 + Sentinel-2 slice of BigEarthNet v2.0.

PLAN.md S2.2b (measured 2026-08-29) is the spec this file implements. The
short version: BigEarthNet-S1.tar.zst (54.4 GB) and BigEarthNet-S2.tar.zst
(63.3 GB) are monolithic, Zenodo ignores Range requests (no resume), and
the two archives are ordered alphabetically by *different* acquisition
folders -- so naively early-stopping both after N tar members yields
UNPAIRED patches. The measured fix: read the first ``k`` acquisition
folders of *each* archive independently (k=10 by default -- the smallest
prefix containing all 19 BigEarthNet land-cover classes, 13,630 pairs,
~8.6 GB streamed / ~3.8 GB stored), then verify pairing on disk afterwards
and drop whatever didn't survive.

Streaming approach chosen (see W1 work order, "two viable approaches"):
a deliberate HYBRID of the two options offered, not a pure pick of either:

  - HTTP fetch + zstd decompression are done by shelling out to the
    system ``curl`` and ``zstd`` binaries (subprocess), exactly as option
    (a) suggests. This avoids adding the ``zstandard`` PyPI dependency --
    ``zstd`` is already on this machine, and PLAN.md S5.6 says "prefer no
    new dependency" when one isn't truly needed.
  - The *tar* layer is handled by Python's stdlib ``tarfile`` in
    streaming mode (``mode="r|"``, per option (b)), reading directly from
    the zstd subprocess's stdout pipe, rather than shelling out to a
    second ``tar`` process with ``--wildcards``. Doing member selection
    and early-exit in Python -- instead of via shell-quoted wildcard
    patterns -- makes "stop as soon as we see a folder we didn't ask
    for" an explicit, unit-testable boundary condition (see
    ``stream_extract_archive`` and tests/test_w1_bigearthnet.py), and
    avoids passing 10+ shell-escaped patterns per archive on a command
    line.

Early exit is the single most important behaviour here: both archives
list members in ascending-alphabetical acquisition-folder order (verified
directly against the real archives before writing this file), so the
first member whose folder is not in the selected set means every
following member is also outside it. At that point the code breaks out
of the tarfile iterator and forcibly terminates the curl/zstd
subprocesses -- it does not let curl keep pulling the remaining ~90% of a
54-60 GB archive into a closed pipe.

Directory / manifest layout produced under ``<data-root>/bigearthnet/``:
    metadata.parquet              -- BEN v2.0 join table (join key: s1_name)
    BigEarthNet.txt.parquet       -- annotations (fetched in full)
    targets_k<k>.json             -- computed target patch list (S5, item 2)
    images/BigEarthNet-S2/<acq_folder>/<patch>/<patch>_<band>.tif
    images/BigEarthNet-S1/<acq_folder>/<patch>/<patch>_<band>.tif

Two manifests land in <manifests-dir>/: "bigearthnet_s1s2_slice" (the
paired image slice -- files, checksums, k, paired count, per-split and
per-country counts, class coverage) and "bigearthnet_annotations" (the
BigEarthNet.txt.parquet fetch).

Known, disclosed limitation (PLAN.md S5.9): this slice is an archive
*prefix*, not a stratified sample. It happens to cover whichever
countries/seasons appear in the first k acquisition folders of each
archive (4 European countries at k=10) -- it is not representative of the
full BigEarthNet distribution. This is recorded in the manifest's
``notes``, not implied away.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from scripts.data.common import (
    DEFAULT_MANIFESTS_DIR,
    MIN_FREE_BYTES,
    DiskBudgetExceeded,
    DownloadError,
    FileEntry,
    Manifest,
    Tier,
    download,
    require_free_space,
    sha256_file,
    utcnow_iso,
    write_manifest,
)

# --------------------------------------------------------------------------
# Measured facts (PLAN.md S2.2b, S7 W1). Do not re-derive, do not adjust
# without re-measuring against the live Zenodo/HF endpoints.
# --------------------------------------------------------------------------

METADATA_URL = "https://zenodo.org/records/10891137/files/metadata.parquet?download=1"
S2_URL = "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"
S1_URL = "https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst?download=1"
ANNOTATIONS_URL = (
    "https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/"
    "resolve/main/BigEarthNet.txt.parquet"
)

# metadata.parquet's exact byte count is not locked in PLAN.md (only "~3.5 MB"
# there); this value was measured directly via `curl` against the live URL
# above while writing this file (2026-08-30).
METADATA_BYTES = 3_616_349
S2_ARCHIVE_BYTES = 63_251_710_377
S1_ARCHIVE_BYTES = 54_439_153_171
ANNOTATIONS_BYTES = 466_819_745

# Approximate per-patch sizes from PLAN.md S2.2b ("~165 KB" / "~116 KB"),
# used only for the projected-stored-bytes estimate -- not exact per-file
# sizes, which vary slightly and are measured for real from disk after
# extraction.
S2_PATCH_APPROX_BYTES = 165_000
S1_PATCH_APPROX_BYTES = 116_000

# Confirmed by listing the first ~5 MB of the real archives directly
# (2026-08-30): 12 Sentinel-2 L2A bands (no B10) and 2 Sentinel-1 bands.
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S1_BANDS = ["VV", "VH"]

DEFAULT_K = 10  # smallest prefix with all 19 classes present (PLAN.md S2.2b)


# --------------------------------------------------------------------------
# Acquisition-folder derivation (PLAN.md S7 W1, item 2)
# --------------------------------------------------------------------------


def s2_acquisition_folder(patch_id: str) -> str:
    """S2 acquisition folder = patch_id minus the trailing "_<row>_<col>".

    e.g. "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"
      -> "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP"
    (verified against the real metadata.parquet and the real archive's tar
    listing while writing this file.)
    """
    return "_".join(patch_id.split("_")[:-2])


def s1_acquisition_folder(s1_name: str) -> str:
    """S1 acquisition folder = s1_name minus the trailing "_<tile>_<row>_<col>".

    e.g. "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57"
      -> "S1B_IW_GRDH_1SDV_20170612T165809"
    Unlike S2, the S1 acquisition folder does not carry a tile code, so
    three trailing underscore-tokens (tile, row, col) are dropped, not two.
    """
    return "_".join(s1_name.split("_")[:-3])


# --------------------------------------------------------------------------
# Step 2: target patch set
# --------------------------------------------------------------------------


@dataclass
class TargetSet:
    k: int  # legacy/display value == k_s2; kept so manifests stay comparable
    k_s2: int
    k_s1: int
    all_s2_folders: list[str]
    all_s1_folders: list[str]
    selected_s2_folders: set[str]
    selected_s1_folders: set[str]
    target_rows: pd.DataFrame  # both ranks < k -- the predicted paired set
    candidate_count_s2: int  # rows with s2_rank < k (S2 side would be extracted)
    candidate_count_s1: int  # rows with s1_rank < k (S1 side would be extracted)

    @property
    def paired_count(self) -> int:
        return len(self.target_rows)


def compute_target_set(
    metadata: pd.DataFrame, k: int, k_s1: int | None = None
) -> TargetSet:
    """Rank acquisition folders alphabetically and select patches with BOTH
    ranks < k. This is the fix for the pairing trap in PLAN.md S2.2b: S1 and
    S2 order their archives by different folders, so folder rank must be
    computed per-archive and intersected, never assumed.

    ``k`` bounds the S2 side and ``k_s1`` the S1 side; ``k_s1`` defaults to
    ``k`` (the original symmetric behaviour). They should NOT be equal in
    practice: the S2 archive packs far more patches per acquisition folder,
    so reading as many S2 folders as S1 folders streams gigabytes that
    contain no additional pairs. Measured: (k=5, k_s1=10) yields the SAME
    13,630 pairs and the same 19/19 class coverage as (10, 10) for 5.89 GB
    instead of 8.65 GB -- a 32% saving. That matters because Zenodo does not
    support Range, so there is no resume and a shorter stream is a
    materially likelier one to finish.
    """
    k_s1 = k if k_s1 is None else k_s1
    df = metadata.copy()
    df["s2_folder"] = df["patch_id"].map(s2_acquisition_folder)
    df["s1_folder"] = df["s1_name"].map(s1_acquisition_folder)

    all_s2_folders = sorted(df["s2_folder"].unique())
    all_s1_folders = sorted(df["s1_folder"].unique())
    selected_s2_folders = set(all_s2_folders[:k])
    selected_s1_folders = set(all_s1_folders[:k_s1])

    s2_rank = {folder: i for i, folder in enumerate(all_s2_folders)}
    s1_rank = {folder: i for i, folder in enumerate(all_s1_folders)}
    df["s2_rank"] = df["s2_folder"].map(s2_rank)
    df["s1_rank"] = df["s1_folder"].map(s1_rank)

    both = (df["s2_rank"] < k) & (df["s1_rank"] < k_s1)
    target_rows = df.loc[both].reset_index(drop=True)

    return TargetSet(
        k=k,
        k_s2=k,
        k_s1=k_s1,
        all_s2_folders=all_s2_folders,
        all_s1_folders=all_s1_folders,
        selected_s2_folders=selected_s2_folders,
        selected_s1_folders=selected_s1_folders,
        target_rows=target_rows,
        candidate_count_s2=int((df["s2_rank"] < k).sum()),
        candidate_count_s1=int((df["s1_rank"] < k_s1).sum()),
    )


def _labels_to_list(labels: Any) -> list[str]:
    if hasattr(labels, "tolist"):
        return list(labels.tolist())
    return list(labels)


def write_target_list(target: TargetSet, path: Path) -> None:
    """Write the computed target patch list to disk so extraction and the
    final manifest can both be checked against the same list (PLAN.md S7
    W1, item 2: "so the extraction step and the manifest agree exactly")."""
    cols = target.target_rows[["patch_id", "s1_name", "s2_folder", "s1_folder", "split", "country", "labels"]]
    records = []
    for row in cols.itertuples(index=False):
        records.append(
            {
                "patch_id": row.patch_id,
                "s1_name": row.s1_name,
                "s2_folder": row.s2_folder,
                "s1_folder": row.s1_folder,
                "split": row.split,
                "country": row.country,
                "labels": _labels_to_list(row.labels),
            }
        )
    payload = {
        "k": target.k,
        "generated_utc": utcnow_iso(),
        "selected_s2_folders": sorted(target.selected_s2_folders),
        "selected_s1_folders": sorted(target.selected_s1_folders),
        "predicted_paired_count": target.paired_count,
        "candidate_count_s2": target.candidate_count_s2,
        "candidate_count_s1": target.candidate_count_s1,
        "target_patches": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _all_classes(metadata: pd.DataFrame) -> set[str]:
    classes: set[str] = set()
    for labels in metadata["labels"]:
        classes.update(_labels_to_list(labels))
    return classes


def summarize(rows: pd.DataFrame, total_classes: int) -> dict[str, Any]:
    """Per-split / per-country counts and land-cover class coverage for a
    set of (predicted or actually-kept) patch rows."""
    if len(rows) == 0:
        per_split: dict[str, int] = {}
        per_country: dict[str, int] = {}
        covered: set[str] = set()
    else:
        per_split = {str(k): int(v) for k, v in rows["split"].value_counts().items()}
        per_country = {str(k): int(v) for k, v in rows["country"].value_counts().items()}
        covered = set()
        for labels in rows["labels"]:
            covered.update(_labels_to_list(labels))
    return {
        "per_split": per_split,
        "per_country": per_country,
        "classes_covered": sorted(covered),
        "classes_covered_count": len(covered),
        "classes_total": total_classes,
    }


def project_bytes(target: TargetSet) -> tuple[int, int]:
    """Estimate (streamed bytes, stored bytes) for reading k folders of
    each archive. Streamed bytes scale with the folder-count fraction of
    each *whole* archive's measured size; stored bytes use the
    per-patch-type approximate sizes from PLAN.md S2.2b."""
    frac_s2 = min(target.k_s2, len(target.all_s2_folders)) / len(target.all_s2_folders)
    frac_s1 = min(target.k_s1, len(target.all_s1_folders)) / len(target.all_s1_folders)
    stream_bytes = int(frac_s2 * S2_ARCHIVE_BYTES + frac_s1 * S1_ARCHIVE_BYTES)
    stored_bytes = int(target.paired_count * (S2_PATCH_APPROX_BYTES + S1_PATCH_APPROX_BYTES))
    return stream_bytes, stored_bytes


# --------------------------------------------------------------------------
# Step 1: metadata + annotations fetch (kept behind thin, monkeypatchable
# wrappers so tests never need real network access)
# --------------------------------------------------------------------------


def ensure_downloaded(url: str, dest: Path, expected_bytes: int | None, tier: Tier) -> Path:
    """Skip the network round-trip if ``dest`` already exists with the
    right size; otherwise delegate to common.download (disk-guarded)."""
    if dest.exists() and (expected_bytes is None or dest.stat().st_size == expected_bytes):
        return dest
    return download(url, dest, tier=tier, expected_bytes=expected_bytes)


def _fetch_metadata_df(ben_dir: Path) -> pd.DataFrame:
    path = ensure_downloaded(METADATA_URL, ben_dir / "metadata.parquet", METADATA_BYTES, tier=Tier.LOCAL)
    return pd.read_parquet(path)


def _fetch_annotations_file(ben_dir: Path) -> Path:
    return ensure_downloaded(ANNOTATIONS_URL, ben_dir / "BigEarthNet.txt.parquet", ANNOTATIONS_BYTES, tier=Tier.LOCAL)


# --------------------------------------------------------------------------
# Step 3: stream-extract, with the early-exit that matters most
# --------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    archive_root: str
    files_written: int = 0
    bytes_written: int = 0
    folders_seen: list[str] = field(default_factory=list)
    stopped_early: bool = False


def _spawn_decompressed_tar_stream(source: str, is_url: bool) -> tuple[list[subprocess.Popen], Any]:
    """Return (subprocesses, readable fileobj yielding decompressed tar bytes).

    is_url=True:  curl -sL --fail <source> | zstd -dc   (production: streams the
                  archive over HTTP without ever writing it to disk; --fail turns
                  an HTTP error into a curl exit code instead of a zstd parse
                  failure on an HTML error page).
    is_url=False: zstd -dc <source>              (tests: decompress a small local
                  .tar.zst fixture; no curl, no network).
    """
    if is_url:
        curl = subprocess.Popen(["curl", "-sL", "--fail", source], stdout=subprocess.PIPE)
        zstd = subprocess.Popen(["zstd", "-dc", "-q"], stdin=curl.stdout, stdout=subprocess.PIPE)
        assert curl.stdout is not None
        curl.stdout.close()  # let curl see SIGPIPE if zstd exits/closes first
        return [curl, zstd], zstd.stdout
    zstd = subprocess.Popen(["zstd", "-dc", "-q", str(source)], stdout=subprocess.PIPE)
    return [zstd], zstd.stdout


def stream_extract_archive(
    archive_root: str,
    source: str,
    is_url: bool,
    selected_folders: set[str],
    images_dir: Path,
    *,
    selected_patches: set[str] | None = None,
    min_free_bytes: int = MIN_FREE_BYTES,
) -> ExtractionResult:
    """Stream ``source`` (a curl-able URL, or a local .tar.zst path when
    is_url=False) through zstd and extract members whose acquisition folder
    is in ``selected_folders`` AND (if given) whose patch directory name is
    in ``selected_patches``.

    ``selected_patches`` is what keeps stored bytes matching the measured
    ~3.8 GB for k=10 (PLAN.md S2.2b): a folder in ``selected_folders`` can
    still contain patches whose *other* archive's folder rank is >= k (the
    exact pairing trap this file exists to defuse), and those would
    otherwise sit on disk forever as orphans invisible to the manifest.
    Pass the precomputed target patch set (S2: ``patch_id``, S1:
    ``s1_name``) so only genuinely-target patches are ever written; the
    post-hoc ``verify_pairing`` pass remains the authoritative check for
    the rarer case of a target patch missing from the archive itself.

    Archive members arrive in ascending-alphabetical acquisition-folder
    order (measured fact, PLAN.md S2.2b). ``selected_folders`` is exactly
    the first k folders alphabetically, so once a folder sorts *after* the
    largest selected folder, every following member is also outside the
    selection -- at that point this function stops reading and kills the
    subprocess pipeline rather than draining the rest of a 50-60 GB
    archive. A folder that is merely absent from ``selected_folders`` but
    sorts *before* the largest one (e.g. an acquisition present in the
    real archive but missing from metadata.parquet) is skipped, not
    treated as the end of the selection -- an early `break` on any
    non-selected folder would silently truncate the read in that case.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    procs, stream = _spawn_decompressed_tar_stream(source, is_url)
    result = ExtractionResult(archive_root=archive_root)
    since_check = 0
    max_selected_folder = max(selected_folders) if selected_folders else None

    try:
        tar = tarfile.open(fileobj=stream, mode="r|")
        try:
            for member in tar:
                parts = member.name.split("/", 3)
                if len(parts) < 2:
                    continue
                folder = parts[1]
                if folder not in selected_folders:
                    if max_selected_folder is None or folder > max_selected_folder:
                        result.stopped_early = True
                        break
                    continue  # a folder before/among the selection that metadata doesn't list
                if folder not in result.folders_seen:
                    result.folders_seen.append(folder)
                if not member.isfile() or not member.name.endswith(".tif"):
                    continue
                if selected_patches is not None:
                    if len(parts) < 3 or parts[2] not in selected_patches:
                        continue

                fh = tar.extractfile(member)
                if fh is None:
                    continue
                out_path = images_dir / member.name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as out:
                    shutil.copyfileobj(fh, out)

                result.files_written += 1
                result.bytes_written += member.size
                since_check += member.size
                if since_check >= 64 * 1024 * 1024:
                    since_check = 0
                    free = shutil.disk_usage(images_dir).free
                    if free < min_free_bytes:
                        raise DiskBudgetExceeded(
                            f"free space dropped to {free} bytes (< floor {min_free_bytes}) "
                            f"while extracting {archive_root}"
                        )
        except tarfile.ReadError as exc:
            last_folder = result.folders_seen[-1] if result.folders_seen else "(none yet)"
            raise DownloadError(
                f"{archive_root} stream ended unexpectedly while in/after folder "
                f"{last_folder!r} ({exc}). Zenodo does not support resumable downloads for "
                "this archive (PLAN.md S2.2b), so a dropped connection has no partial-"
                "progress path -- it must restart from the beginning of the archive. This "
                "is a transient network failure, not a code defect: re-run the fetch; files "
                "already written for completed folders will simply be overwritten."
            ) from exc
    finally:
        try:
            stream.close()
        except Exception:
            pass
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

    return result


# --------------------------------------------------------------------------
# Step 4: verify pairing on disk
# --------------------------------------------------------------------------


def _s2_patch_files(images_dir: Path, s2_folder: str, patch_id: str) -> list[Path]:
    d = images_dir / "BigEarthNet-S2" / s2_folder / patch_id
    return [d / f"{patch_id}_{band}.tif" for band in S2_BANDS]


def _s1_patch_files(images_dir: Path, s1_folder: str, s1_name: str) -> list[Path]:
    d = images_dir / "BigEarthNet-S1" / s1_folder / s1_name
    return [d / f"{s1_name}_{band}.tif" for band in S1_BANDS]


def verify_pairing(target_rows: pd.DataFrame, images_dir: Path) -> tuple[list[tuple[Any, list[Path]]], list[dict]]:
    """For every predicted-paired row, require ALL S2 band files AND ALL S1
    band files to exist on disk. A patch missing any part of either half is
    dropped and reported -- it must never reach the manifest (PLAN.md S7
    W1, item 4: "A patch counted as paired but missing its S1 half would
    corrupt W2's training set in a way that is very hard to notice later")."""
    cols = target_rows[["patch_id", "s1_name", "s2_folder", "s1_folder", "split", "country", "labels"]]
    kept: list[tuple[Any, list[Path]]] = []
    dropped: list[dict] = []
    for row in cols.itertuples(index=False):
        s2_paths = _s2_patch_files(images_dir, row.s2_folder, row.patch_id)
        s1_paths = _s1_patch_files(images_dir, row.s1_folder, row.s1_name)
        all_paths = s2_paths + s1_paths
        missing = [p for p in all_paths if not p.is_file()]
        if missing:
            dropped.append(
                {
                    "patch_id": row.patch_id,
                    "s1_name": row.s1_name,
                    "missing_s2_bands": sum(1 for p in s2_paths if not p.is_file()),
                    "missing_s1_bands": sum(1 for p in s1_paths if not p.is_file()),
                }
            )
            continue
        kept.append((row, all_paths))
    return kept, dropped


# --------------------------------------------------------------------------
# Step 5: manifests
# --------------------------------------------------------------------------


def build_image_manifest(
    kept: list[tuple[Any, list[Path]]],
    dropped: list[dict],
    images_dir: Path,
    data_root: Path,
    k: int,
    total_classes_count: int,
    s2_result: ExtractionResult,
    s1_result: ExtractionResult,
) -> Manifest:
    files = [
        FileEntry(path=str(p.relative_to(images_dir)), bytes=p.stat().st_size, sha256=sha256_file(p))
        for _row, paths in kept
        for p in paths
    ]
    rows_df = pd.DataFrame([row._asdict() for row, _ in kept]) if kept else pd.DataFrame(columns=["split", "country", "labels"])
    summary = summarize(rows_df, total_classes_count)
    countries = sorted(summary["per_country"].keys())

    notes = (
        f"Archive PREFIX: first k={k} acquisition folder(s) (alphabetical order) of each of "
        "BigEarthNet-S1.tar.zst and BigEarthNet-S2.tar.zst, streamed via curl|zstd and stopped "
        "reading as soon as the stream passed the last targeted folder -- NOT a stratified "
        f"sample. Covers {len(countries)} countr{'y' if len(countries) == 1 else 'ies'} "
        f"({', '.join(countries) if countries else 'none'}); not representative of the full "
        "BigEarthNet geographic/seasonal distribution. See PLAN.md S2.2b and S5.9."
    )

    return Manifest(
        dataset="bigearthnet_s1s2_slice",
        split="mixed",  # rows span the official train/validation/test split; see counts.split_*
        tier=Tier.LOCAL,
        root=str(images_dir.relative_to(data_root)),
        source_urls=[S2_URL, S1_URL, METADATA_URL],
        retrieved_utc=utcnow_iso(),
        files=files,
        counts={
            "k": k,
            "paired_patches": len(kept),
            "dropped_unpaired": len(dropped),
            "s2_files_extracted": s2_result.files_written,
            "s1_files_extracted": s1_result.files_written,
            "s2_bytes_extracted": s2_result.bytes_written,
            "s1_bytes_extracted": s1_result.bytes_written,
            **{f"split_{name}": count for name, count in summary["per_split"].items()},
            **{f"country_{name}": count for name, count in summary["per_country"].items()},
            "classes_covered": summary["classes_covered_count"],
            "classes_total": summary["classes_total"],
        },
        notes=notes,
    )


def build_annotations_manifest(ann_path: Path, ben_dir: Path, num_rows: int) -> Manifest:
    return Manifest(
        dataset="bigearthnet_annotations",
        split="all",
        tier=Tier.LOCAL,
        root=ben_dir.name,
        source_urls=[ANNOTATIONS_URL],
        retrieved_utc=utcnow_iso(),
        files=[
            FileEntry(
                path=ann_path.name,
                bytes=ann_path.stat().st_size,
                sha256=sha256_file(ann_path),
            )
        ],
        counts={"rows": num_rows},
        notes=(
            "Full BigEarthNet.txt annotations parquet (binary/mcq/captioning/bounding_box QA "
            "types over BigEarthNet), fetched in full via plain HTTP -- not filtered down to "
            "the S1+S2 image slice's patch IDs. See PLAN.md S2.2/S7 W1."
        ),
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a paired Sentinel-1 + Sentinel-2 slice of BigEarthNet v2.0 by streaming a "
            "bounded prefix of the two archives (PLAN.md S2.2b)."
        )
    )
    parser.add_argument(
        "--folders",
        "-k",
        type=int,
        default=DEFAULT_K,
        help=f"acquisition-folder rank cutoff per archive (default: {DEFAULT_K}, the measured "
        "all-19-classes minimum).",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--manifests-dir", type=Path, default=DEFAULT_MANIFESTS_DIR)
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fetch metadata and report the target set without streaming any archive "
        "(default: on; pass --no-dry-run to actually stream and extract).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="proceed even if projected stored bytes exceed --stored-ceiling-gb.",
    )
    parser.add_argument(
        "--s1-folders",
        type=int,
        default=None,
        help=(
            "S1 acquisition folders to read; defaults to --folders. Because the "
            "S2 archive packs far more patches per folder, the efficient setting "
            "is asymmetric: --folders 5 --s1-folders 10 yields the same 13,630 "
            "pairs and 19/19 classes as 10/10 for 5.89 GB instead of 8.65 GB."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help=(
            "Stream attempts per archive. Zenodo does not support Range, so a "
            "dropped connection cannot resume and the archive must be re-read "
            "from byte 0; retrying is the only recovery. Observed: a 6.76 GB "
            "single-shot stream failed with 'unexpected end of data'."
        ),
    )
    parser.add_argument("--stored-ceiling-gb", type=float, default=10.0)
    return parser.parse_args(argv)


def _print_report(target: TargetSet, summary: dict[str, Any], stream_bytes: int, stored_bytes: int) -> None:
    print(
        f"[bigearthnet] k={target.k}: predicted paired patches = {target.paired_count} "
        f"(S2-side candidates={target.candidate_count_s2}, S1-side candidates={target.candidate_count_s1})"
    )
    print(f"[bigearthnet] per-split: {summary['per_split']}")
    print(f"[bigearthnet] per-country: {summary['per_country']}")
    print(f"[bigearthnet] class coverage: {summary['classes_covered_count']}/{summary['classes_total']}")
    print(
        f"[bigearthnet] projected stream bytes: {stream_bytes / 1024**3:.2f} GB; "
        f"projected stored bytes: {stored_bytes / 1024**3:.2f} GB"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    manifests_dir = Path(args.manifests_dir)
    ben_dir = data_root / "bigearthnet"
    images_dir = ben_dir / "images"

    metadata = _fetch_metadata_df(ben_dir)
    target = compute_target_set(metadata, k=args.folders, k_s1=args.s1_folders)
    write_target_list(target, ben_dir / f"targets_k{target.k_s2}_{target.k_s1}.json")

    total_classes = len(_all_classes(metadata))
    predicted_summary = summarize(target.target_rows, total_classes)
    stream_bytes, stored_bytes = project_bytes(target)
    _print_report(target, predicted_summary, stream_bytes, stored_bytes)

    if target.paired_count == 0:
        raise SystemExit(
            f"[bigearthnet] 0 paired patches predicted for k={args.folders} folders -- "
            "refusing to stream tens of GB for nothing. Increase --folders."
        )

    if args.dry_run:
        print("[bigearthnet] --dry-run: stopping before any archive streaming.")
        return 0

    if stored_bytes > args.stored_ceiling_gb * 1024**3 and not args.force:
        raise SystemExit(
            f"[bigearthnet] projected stored bytes ({stored_bytes / 1024**3:.2f} GB) exceeds "
            f"the {args.stored_ceiling_gb:.1f} GB ceiling -- stopping. Pass --force to override."
        )

    require_free_space(stored_bytes + MIN_FREE_BYTES, data_root)

    ann_path = _fetch_annotations_file(ben_dir)
    ann_rows = pq.ParquetFile(ann_path).metadata.num_rows
    write_manifest("bigearthnet_annotations", build_annotations_manifest(ann_path, ben_dir, ann_rows), manifests_dir)

    def _stream_with_retry(label, url, folders, patches, attempts):
        """Zenodo does not honour Range, so a dropped stream cannot resume --
        the archive must be re-read from byte 0. Retrying is the only
        recovery available, and it is needed in practice: a single-shot
        6.76 GB stream failed here with zstd 'premature end' followed by
        tarfile 'unexpected end of data'. Already-extracted patch dirs are
        left in place between attempts, so a retry re-reads the bytes but
        does not redo the disk writes.
        """
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return stream_extract_archive(
                    label, url, True, folders, images_dir, selected_patches=patches
                )
            except (tarfile.ReadError, OSError, RuntimeError) as exc:
                last = exc
                print(
                    f"[bigearthnet] {label} stream attempt {attempt}/{attempts} "
                    f"failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                if attempt < attempts:
                    print(f"[bigearthnet] retrying {label} from byte 0 ...", file=sys.stderr)
        raise RuntimeError(
            f"{label}: all {attempts} stream attempts failed; last error: {last}"
        )

    s2_result = _stream_with_retry(
        "BigEarthNet-S2", S2_URL, target.selected_s2_folders,
        set(target.target_rows["patch_id"]), args.retries,
    )
    s1_result = _stream_with_retry(
        "BigEarthNet-S1", S1_URL, target.selected_s1_folders,
        set(target.target_rows["s1_name"]), args.retries,
    )

    for result, label in ((s2_result, "S2"), (s1_result, "S1")):
        if not result.stopped_early:
            raise SystemExit(
                f"[bigearthnet] {label} stream reached end-of-archive without ever seeing a "
                f"folder past the selected {args.folders} -- the real archive has far more "
                "folders than that, so this means the connection was cut mid-stream (Zenodo "
                "has no resume) and the extraction is silently truncated, not complete. "
                "Refusing to build a manifest from it -- retry from scratch."
            )

    kept, dropped = verify_pairing(target.target_rows, images_dir)
    if not kept:
        raise SystemExit(
            "[bigearthnet] 0 patches survived pairing verification after extraction -- "
            "something is wrong upstream (archive layout changed?); refusing to write a manifest."
        )
    if len(kept) < target.paired_count:
        print(
            f"[bigearthnet] {len(dropped)}/{target.paired_count} predicted pairs were dropped "
            "as unpaired remainder (present in one archive's selected folders, but the "
            "matching half was missing on disk after extraction)."
        )

    image_manifest = build_image_manifest(
        kept, dropped, images_dir, data_root, args.folders, total_classes, s2_result, s1_result
    )
    write_manifest("bigearthnet_s1s2_slice", image_manifest, manifests_dir)

    print(f"[bigearthnet] done: {len(kept)} paired patches kept, {len(dropped)} dropped as unpaired remainder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
