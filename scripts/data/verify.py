"""Re-validate local datasets under data/ against data/manifests/*.json.

For each manifest this checks, per file: exists, size matches, and
(unless --fast) sha256 matches. A dataset with zero of its files present
is reported as NOT_DOWNLOADED -- the normal state for most datasets for a
while (PLAN.md W1) -- distinct from FAIL, and does not fail the run.

Usage:
    uv run scripts/data/verify.py [--data-root data] [--manifests-dir data/manifests] [--fast]

Exit code is non-zero iff at least one dataset FAILs (some, but not all,
files present/correct, or a size/hash mismatch). All-PASS and all-
NOT_DOWNLOADED (and any mix of just those two) exit 0.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Run as a script (`python scripts/data/verify.py`) the repo root is not on
# sys.path, so the absolute `scripts.data.common` import below fails. Add it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.data.common import (  # noqa: E402
    DEFAULT_MANIFESTS_DIR,
    Manifest,
    iter_manifest_names,
    load_manifest,
    sha256_file,
)

PASS = "PASS"
FAIL = "FAIL"
NOT_DOWNLOADED = "NOT_DOWNLOADED"


@dataclass
class DatasetResult:
    name: str
    status: str
    present_bytes: int
    problems: list[str]


def _check_manifest_integrity(manifest: Manifest) -> list[str]:
    """Catch a hand-edited or buggy manifest before touching disk."""
    problems = []
    computed_bytes = sum(f.bytes for f in manifest.files)
    computed_count = len(manifest.files)
    if manifest.total_bytes != computed_bytes:
        problems.append(
            f"manifest total_bytes ({manifest.total_bytes}) != sum of file "
            f"entries ({computed_bytes})"
        )
    if manifest.file_count != computed_count:
        problems.append(
            f"manifest file_count ({manifest.file_count}) != number of file "
            f"entries ({computed_count})"
        )
    return problems


def verify_dataset(
    name: str,
    manifest: Manifest,
    data_root: Path,
    fast: bool = False,
) -> DatasetResult:
    problems = _check_manifest_integrity(manifest)
    root = data_root / manifest.root

    present = 0
    present_bytes = 0
    for entry in manifest.files:
        fpath = root / entry.path
        if not fpath.exists():
            continue
        present += 1
        actual_size = fpath.stat().st_size
        present_bytes += actual_size
        if actual_size != entry.bytes:
            problems.append(f"{entry.path}: size mismatch (expected {entry.bytes}, got {actual_size})")
            continue
        if not fast:
            actual_hash = sha256_file(fpath)
            if actual_hash != entry.sha256:
                problems.append(f"{entry.path}: sha256 mismatch (expected {entry.sha256}, got {actual_hash})")

    total_files = len(manifest.files)
    missing = total_files - present
    if present == 0 and total_files > 0:
        status = NOT_DOWNLOADED
        # Missing files aren't a "problem" in this state -- it's expected.
        problems = [p for p in problems if "size mismatch" not in p and "sha256 mismatch" not in p]
    elif missing > 0:
        status = FAIL
        problems.append(f"{missing} of {total_files} file(s) missing")
    elif problems:
        status = FAIL
    else:
        status = PASS

    return DatasetResult(name=name, status=status, present_bytes=present_bytes, problems=problems)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="Root directory datasets live under (default: data)")
    parser.add_argument(
        "--manifests-dir",
        default=str(DEFAULT_MANIFESTS_DIR),
        help=f"Directory of *.json manifests (default: {DEFAULT_MANIFESTS_DIR})",
    )
    parser.add_argument("--fast", action="store_true", help="Skip sha256 hashing; check existence/size only")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    manifests_dir = Path(args.manifests_dir)

    names = list(iter_manifest_names(manifests_dir))
    if not names:
        print(f"no manifests found under {manifests_dir}")
        return 0

    results: list[DatasetResult] = []
    for name in names:
        try:
            manifest = load_manifest(name, manifests_dir)
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the whole run
            results.append(DatasetResult(name=name, status=FAIL, present_bytes=0, problems=[f"unreadable manifest: {exc}"]))
            continue
        results.append(verify_dataset(name, manifest, data_root, fast=args.fast))

    total_bytes_on_disk = 0
    any_fail = False
    for r in results:
        total_bytes_on_disk += r.present_bytes
        print(f"[{r.status}] {r.name}")
        for problem in r.problems:
            print(f"    - {problem}")
        if r.status == FAIL:
            any_fail = True

    print(f"\ntotal verified data on disk: {total_bytes_on_disk} bytes ({total_bytes_on_disk / 1024**3:.3f} GB)")
    try:
        usage = shutil.disk_usage(data_root if data_root.exists() else Path("."))
        print(f"free space at {data_root}: {usage.free / 1024**3:.2f} GB")
    except OSError as exc:  # pragma: no cover -- best-effort diagnostic only
        print(f"could not stat free space at {data_root}: {exc}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
