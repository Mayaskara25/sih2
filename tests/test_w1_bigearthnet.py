"""Tests for scripts/data/fetch_bigearthnet.py.

No real network, and no real archive is ever streamed. Synthetic
`.tar.zst` fixtures are built with the real system `zstd` binary
(compressing a small in-memory tar), mimicking the real BigEarthNet tar
layout confirmed against the live archives (BigEarthNet-<S1|S2>/
<acquisition_folder>/<patch_name>/<patch_name>_<band>.tif). Anything that
would otherwise hit the network (metadata.parquet, BigEarthNet.txt.parquet,
the S1/S2 archives) is monkeypatched to a local fixture instead.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.data import fetch_bigearthnet as ben
from scripts.data.common import Tier, load_manifest, sha256_file

pytestmark = pytest.mark.skipif(
    subprocess.run(["which", "zstd"], capture_output=True).returncode != 0,
    reason="zstd binary not available",
)


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _make_tar_zst(tmp_path: Path, archive_root: str, members: dict[str, bytes]) -> Path:
    """Build a small real .tar.zst at tmp_path/<archive_root>.tar.zst.

    ``members`` maps a full tar member name (already including the
    "<archive_root>/<folder>/<patch>/<file>" prefix) to its content bytes.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    tar_path = tmp_path / f"{archive_root}.tar"
    with tarfile.open(tar_path, "w") as tar:
        for name in sorted(members):
            data = members[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    zst_path = tmp_path / f"{archive_root}.tar.zst"
    subprocess.run(["zstd", "-q", "-f", str(tar_path), "-o", str(zst_path)], check=True)
    return zst_path


def _s2_members(archive_root: str, folder: str, patch_id: str) -> dict[str, bytes]:
    return {
        f"{archive_root}/{folder}/{patch_id}/{patch_id}_{band}.tif": f"s2-{patch_id}-{band}".encode()
        for band in ben.S2_BANDS
    }


def _s1_members(archive_root: str, folder: str, s1_name: str) -> dict[str, bytes]:
    return {
        f"{archive_root}/{folder}/{s1_name}/{s1_name}_{band}.tif": f"s1-{s1_name}-{band}".encode()
        for band in ben.S1_BANDS
    }


def _row(patch_id, s1_name, split, country, labels):
    return {
        "patch_id": patch_id,
        "s1_name": s1_name,
        "labels": labels,
        "split": split,
        "country": country,
    }


# --------------------------------------------------------------------------
# Acquisition-folder derivation, checked against real measured strings
# --------------------------------------------------------------------------


def test_s2_acquisition_folder_strips_row_col():
    patch_id = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"
    assert ben.s2_acquisition_folder(patch_id) == "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP"


def test_s1_acquisition_folder_strips_tile_row_col():
    s1_name = "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57"
    assert ben.s1_acquisition_folder(s1_name) == "S1B_IW_GRDH_1SDV_20170612T165809"


# --------------------------------------------------------------------------
# The shared synthetic scenario used by several tests below:
#
#   S2 folders (alphabetical): F2_A, F2_B, F2_C, F2_D
#   S1 folders (alphabetical): F1_A, F1_B, F1_C, F1_D
#   k = 2  ->  selected_s2_folders = {F2_A, F2_B}, selected_s1_folders = {F1_A, F1_B}
#
#   patch1: s2_folder=F2_A, s1_folder=F1_A -> both ranks < 2 -> predicted AND
#           genuinely paired (both halves land on disk)
#   patch2: s2_folder=F2_A, s1_folder=F1_D -> s2_rank=0<2, s1_rank=3 !< 2 ->
#           NOT predicted (folder-rank orphan on the S2 side)
#   patch3: s2_folder=F2_D, s1_folder=F1_B -> s2_rank=3 !< 2, s1_rank=1<2 ->
#           NOT predicted (folder-rank orphan on the S1 side)
#   patch4: s2_folder=F2_B, s1_folder=F1_B -> both ranks < 2 -> PREDICTED
#           paired, but its S1 half is deliberately absent from the S1
#           archive fixture -- this is the pairing-guarantee test.
#   patch5: s2_folder=F2_C, s1_folder=F1_C -> both ranks !< 2 -> never
#           extracted at all; also used to prove early-exit (F2_C/F2_D and
#           F1_C/F1_D must never reach disk).
# --------------------------------------------------------------------------


def _scenario_metadata() -> pd.DataFrame:
    rows = [
        _row("F2_A_1_1", "F1_A_T1_1_1", "train", "Ireland", ["urban"]),
        _row("F2_A_2_2", "F1_D_T1_2_2", "train", "Ireland", ["water"]),
        _row("F2_D_3_3", "F1_B_T1_3_3", "test", "Finland", ["forest"]),
        _row("F2_B_4_4", "F1_B_T1_4_4", "validation", "Austria", ["forest", "urban"]),
        _row("F2_C_5_5", "F1_C_T1_5_5", "test", "Portugal", ["water"]),
    ]
    return pd.DataFrame(rows)


def _build_scenario_archives(tmp_path: Path, *, drop_patch4_s1: bool) -> tuple[Path, Path]:
    s2_members: dict[str, bytes] = {}
    s2_members.update(_s2_members("BigEarthNet-S2", "F2_A", "F2_A_1_1"))
    s2_members.update(_s2_members("BigEarthNet-S2", "F2_A", "F2_A_2_2"))
    s2_members.update(_s2_members("BigEarthNet-S2", "F2_B", "F2_B_4_4"))
    s2_members.update(_s2_members("BigEarthNet-S2", "F2_C", "F2_C_5_5"))
    s2_members.update(_s2_members("BigEarthNet-S2", "F2_D", "F2_D_3_3"))

    s1_members: dict[str, bytes] = {}
    s1_members.update(_s1_members("BigEarthNet-S1", "F1_A", "F1_A_T1_1_1"))
    if not drop_patch4_s1:
        s1_members.update(_s1_members("BigEarthNet-S1", "F1_B", "F1_B_T1_4_4"))
    s1_members.update(_s1_members("BigEarthNet-S1", "F1_B", "F1_B_T1_3_3"))
    s1_members.update(_s1_members("BigEarthNet-S1", "F1_C", "F1_C_T1_5_5"))
    s1_members.update(_s1_members("BigEarthNet-S1", "F1_D", "F1_D_T1_2_2"))

    s2_path = _make_tar_zst(tmp_path, "BigEarthNet-S2", s2_members)
    s1_path = _make_tar_zst(tmp_path, "BigEarthNet-S1", s1_members)
    return s2_path, s1_path


# --------------------------------------------------------------------------
# 1. Folder-rank selection picks exactly the patches whose both ranks < k.
# --------------------------------------------------------------------------


def test_compute_target_set_selects_only_both_ranks_below_k():
    target = ben.compute_target_set(_scenario_metadata(), k=2)

    assert target.selected_s2_folders == {"F2_A", "F2_B"}
    assert target.selected_s1_folders == {"F1_A", "F1_B"}
    assert set(target.target_rows["patch_id"]) == {"F2_A_1_1", "F2_B_4_4"}
    assert target.paired_count == 2
    # patch2 (s2 rank 0) and patch4 (s2 rank 1) both have s2_rank < 2
    assert target.candidate_count_s2 == 3  # F2_A_1_1, F2_A_2_2, F2_B_4_4
    # patch1 (s1 rank 0), patch3 (s1 rank 1), patch4 (s1 rank 1)
    assert target.candidate_count_s1 == 3


# --------------------------------------------------------------------------
# 2. The pairing guarantee: a patch whose S1 half is absent from the
#    archive is EXCLUDED from the manifest, even though folder-rank
#    predicted it as paired.
# --------------------------------------------------------------------------


def test_verify_pairing_excludes_patch_with_missing_s1_half(tmp_path):
    target = ben.compute_target_set(_scenario_metadata(), k=2)
    images_dir = tmp_path / "images"

    s2_path, s1_path = _build_scenario_archives(tmp_path, drop_patch4_s1=True)
    ben.stream_extract_archive("BigEarthNet-S2", str(s2_path), False, target.selected_s2_folders, images_dir)
    ben.stream_extract_archive("BigEarthNet-S1", str(s1_path), False, target.selected_s1_folders, images_dir)

    kept, dropped = ben.verify_pairing(target.target_rows, images_dir)

    kept_ids = {row.patch_id for row, _ in kept}
    assert kept_ids == {"F2_A_1_1"}
    assert len(dropped) == 1
    assert dropped[0]["patch_id"] == "F2_B_4_4"
    assert dropped[0]["missing_s1_bands"] == len(ben.S1_BANDS)
    assert dropped[0]["missing_s2_bands"] == 0  # its S2 half WAS extracted


# --------------------------------------------------------------------------
# 3. Early exit: stop reading after the last targeted folder rather than
#    draining the stream.
# --------------------------------------------------------------------------


def test_stream_extract_archive_stops_after_selected_folders(tmp_path):
    members: dict[str, bytes] = {}
    members.update(_s2_members("BigEarthNet-S2", "F2_A", "F2_A_1_1"))
    members.update(_s2_members("BigEarthNet-S2", "F2_B", "F2_B_4_4"))
    members.update(_s2_members("BigEarthNet-S2", "F2_C", "F2_C_5_5"))
    members.update(_s2_members("BigEarthNet-S2", "F2_D", "F2_D_3_3"))
    archive_path = _make_tar_zst(tmp_path, "BigEarthNet-S2", members)

    images_dir = tmp_path / "images"
    result = ben.stream_extract_archive(
        "BigEarthNet-S2", str(archive_path), False, {"F2_A", "F2_B"}, images_dir
    )

    assert result.stopped_early is True
    assert result.folders_seen == ["F2_A", "F2_B"]
    assert result.files_written == 2 * len(ben.S2_BANDS)

    # Folders past the cutoff must never have been written to disk.
    assert not (images_dir / "BigEarthNet-S2" / "F2_C").exists()
    assert not (images_dir / "BigEarthNet-S2" / "F2_D").exists()
    assert (images_dir / "BigEarthNet-S2" / "F2_A" / "F2_A_1_1").exists()
    assert (images_dir / "BigEarthNet-S2" / "F2_B" / "F2_B_4_4").exists()


# --------------------------------------------------------------------------
# 4 & 5. --dry-run writes no image files; end-to-end manifest round-trips
#         and its counts match what is actually on disk.
# --------------------------------------------------------------------------


def _patch_fetchers(monkeypatch, metadata_df, *, write_annotations=False):
    # The 5 GB production disk floor is meaningless for a fixture that writes a
    # few KB, and pytest's tmp_path lives under /tmp -- a ~6.8 GB tmpfs -- so
    # inheriting the real floor made these tests pass or fail on ambient /tmp
    # usage rather than on the code under test. Pin a tiny floor.
    monkeypatch.setattr(ben, "MIN_FREE_BYTES", 1024)
    monkeypatch.setattr(ben, "_fetch_metadata_df", lambda ben_dir: metadata_df)
    if write_annotations:

        def _fake_fetch_annotations(ben_dir: Path) -> Path:
            # Mirrors what the real fetcher does: lands the file at the
            # real destination path under ben_dir, just without any
            # network I/O.
            ben_dir.mkdir(parents=True, exist_ok=True)
            dest = ben_dir / "BigEarthNet.txt.parquet"
            pd.DataFrame({"question": ["q1", "q2"], "answer": ["a1", "a2"]}).to_parquet(dest)
            return dest

        monkeypatch.setattr(ben, "_fetch_annotations_file", _fake_fetch_annotations)
    else:

        def _boom(ben_dir):
            raise AssertionError("annotations should not be fetched in this test")

        monkeypatch.setattr(ben, "_fetch_annotations_file", _boom)


def test_dry_run_writes_no_image_files(tmp_path, monkeypatch):
    _patch_fetchers(monkeypatch, _scenario_metadata())

    def _boom(*args, **kwargs):
        raise AssertionError("stream_extract_archive must not run in --dry-run")

    monkeypatch.setattr(ben, "stream_extract_archive", _boom)

    data_root = tmp_path / "data"
    manifests_dir = tmp_path / "manifests"
    rc = ben.main(
        [
            "--folders",
            "2",
            "--data-root",
            str(data_root),
            "--manifests-dir",
            str(manifests_dir),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert not (data_root / "bigearthnet" / "images").exists()
    assert not (manifests_dir / "bigearthnet_s1s2_slice.json").exists()
    assert not (manifests_dir / "bigearthnet_annotations.json").exists()
    # The target list itself IS written (computing/reporting the target
    # set is part of what --dry-run does).
    assert (data_root / "bigearthnet" / "targets_k2_2.json").exists()


def test_full_pipeline_manifest_matches_disk(tmp_path, monkeypatch):
    metadata_df = _scenario_metadata()
    _patch_fetchers(monkeypatch, metadata_df, write_annotations=True)

    s2_fixture, s1_fixture = _build_scenario_archives(tmp_path / "archives", drop_patch4_s1=True)

    real_stream_extract = ben.stream_extract_archive

    def _fake_stream_extract(archive_root, source, is_url, selected_folders, images_dir, **kwargs):
        fixture = s2_fixture if archive_root == "BigEarthNet-S2" else s1_fixture
        return real_stream_extract(archive_root, str(fixture), False, selected_folders, images_dir, **kwargs)

    monkeypatch.setattr(ben, "stream_extract_archive", _fake_stream_extract)

    data_root = tmp_path / "data"
    manifests_dir = tmp_path / "manifests"
    rc = ben.main(
        [
            "--folders",
            "2",
            "--data-root",
            str(data_root),
            "--manifests-dir",
            str(manifests_dir),
            "--no-dry-run",
        ]
    )
    assert rc == 0

    images_dir = data_root / "bigearthnet" / "images"

    # -- image slice manifest --
    manifest = load_manifest("bigearthnet_s1s2_slice", manifests_dir)
    assert manifest.tier == Tier.LOCAL
    assert manifest.counts["k"] == 2
    assert manifest.counts["paired_patches"] == 1
    assert manifest.counts["dropped_unpaired"] == 1
    assert manifest.counts["classes_total"] >= manifest.counts["classes_covered"] > 0
    assert "4 European" not in manifest.notes  # no invented country count
    assert "archive PREFIX" in manifest.notes or "Archive PREFIX" in manifest.notes

    # Only patch1's files (12 S2 bands + 2 S1 bands) should be in the manifest.
    assert manifest.file_count == len(ben.S2_BANDS) + len(ben.S1_BANDS)
    for entry in manifest.files:
        on_disk = images_dir / entry.path
        assert on_disk.is_file(), f"manifest references missing file {entry.path}"
        assert on_disk.stat().st_size == entry.bytes
        assert sha256_file(on_disk) == entry.sha256

    # patch4's surviving S2 half (extracted, but dropped for missing S1)
    # must NOT appear in the manifest, even though its files exist on disk.
    manifest_paths = {e.path for e in manifest.files}
    assert not any("F2_B_4_4" in p for p in manifest_paths)
    assert (images_dir / "BigEarthNet-S2" / "F2_B" / "F2_B_4_4").exists()  # extracted...
    # ...but genuinely excluded from the manifest above, per the pairing guarantee.

    # patch2 (F2_A_2_2) sits in a selected S2 folder (F2_A) but is NOT in the
    # target list (its S1 half, F1_D_T1_2_2, is outside the selected S1
    # folders) -- it must never be written to disk at all, since extraction
    # is filtered to the target patch set, not "every patch in a selected
    # folder" (the latter would store ~3x more than the measured ~3.8 GB
    # for the real k=10 slice).
    assert not (images_dir / "BigEarthNet-S2" / "F2_A" / "F2_A_2_2").exists()

    # Folders beyond the k=2 cutoff were never extracted at all (early exit).
    assert not (images_dir / "BigEarthNet-S2" / "F2_C").exists()
    assert not (images_dir / "BigEarthNet-S2" / "F2_D").exists()
    assert not (images_dir / "BigEarthNet-S1" / "F1_C").exists()
    assert not (images_dir / "BigEarthNet-S1" / "F1_D").exists()

    # -- annotations manifest --
    ann_manifest = load_manifest("bigearthnet_annotations", manifests_dir)
    assert ann_manifest.counts["rows"] == 2
    assert ann_manifest.file_count == 1
    ann_entry = ann_manifest.files[0]
    assert (data_root / "bigearthnet" / ann_entry.path).stat().st_size == ann_entry.bytes

    # -- target list round-trips too --
    target_list = json.loads((data_root / "bigearthnet" / "targets_k2_2.json").read_text())
    assert target_list["k"] == 2
    assert target_list["predicted_paired_count"] == 2
    assert {p["patch_id"] for p in target_list["target_patches"]} == {"F2_A_1_1", "F2_B_4_4"}
