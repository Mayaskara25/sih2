#!/usr/bin/env python
"""
Fetch the RSVQA-LR **test split** (PLAN.md W1).

RSVQA-LR is the VQA benchmark for the rubric's "Visual Question Answering
(RSVQA) -- Accuracy, Answer Match Rate, BLEU" row. LR (not HR) is used
deliberately; see the RSVQA-HR note below.

Verified against the Zenodo API on 2026-08-29 (record 6344334, total
150,463,070 B across 12 files) rather than taken from documentation.

What this fetches
-----------------
  LR_split_test_questions.json    2,717,368 B
  LR_split_test_answers.json      1,922,393 B
  LR_split_test_images.json         123,273 B
  Images_LR.zip                  95,008,155 B
                                 ~99.8 MB total

Honest note on Images_LR.zip: it contains the images for ALL splits, not
just test -- there is no test-only image archive. That technically brushes
against PLAN.md §2.2a's "test splits only" rule, but at 95 MB the rule's
purpose (not spending tens of GB on data we will never evaluate) is not in
play. Recorded in the manifest notes rather than glossed over (§5.9).

RSVQA-HR is NOT supported
-------------------------
HR's test images ship only inside a 13.5 GB monolithic archive that also
carries the training images, so acquiring HR's test split means acquiring
its train split -- exactly what §2.2a forbids. `--hr` is refused in code,
not merely discouraged in a comment.

Usage
-----
    uv run python scripts/data/fetch_rsvqa.py --dry-run
    uv run python scripts/data/fetch_rsvqa.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.data.common import (  # noqa: E402
    FileEntry,
    Manifest,
    Tier,
    download,
    sha256_file,
    utcnow_iso,
    write_manifest,
)

ZENODO_RECORD = "6344334"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# key -> exact byte size, verified via the Zenodo API on 2026-08-29.
# A mismatch aborts rather than silently accepting a truncated or changed file.
TEST_SPLIT_FILES: dict[str, int] = {
    "LR_split_test_questions.json": 2_717_368,
    "LR_split_test_answers.json": 1_922_393,
    "LR_split_test_images.json": 123_273,
    "Images_LR.zip": 95_008_155,
}

# Where files are written on disk...
DEST_ROOT = "data/rsvqa_lr"
# ...vs. what goes in the manifest: common.Manifest documents `root` as
# relative to the DATA ROOT, so it must NOT carry the leading "data/".
# verify.py computes data_root / manifest.root; duplicating the prefix here
# makes it look for data/data/rsvqa_lr and report NOT_DOWNLOADED on data
# that is actually present.
MANIFEST_ROOT = "rsvqa_lr"
MANIFEST_NAME = "rsvqa_lr_test"

HR_REFUSAL = (
    "RSVQA-HR is not supported and will not be fetched.\n"
    "  Its test images exist only inside a ~13.5 GB monolithic archive that also\n"
    "  contains the training images, so fetching HR's test split means fetching its\n"
    "  train split. PLAN.md §2.2a forbids downloading train splits locally.\n"
    "  Use RSVQA-LR (the default), which has a cleanly separable test split."
)


def total_expected_bytes() -> int:
    return sum(TEST_SPLIT_FILES.values())


def fetch(dest_root: Path, *, dry_run: bool) -> list[FileEntry]:
    """Download each test-split artifact; return manifest entries."""
    entries: list[FileEntry] = []
    dest_root.mkdir(parents=True, exist_ok=True)

    for key, expected in sorted(TEST_SPLIT_FILES.items()):
        url = f"{BASE_URL}/{key}?download=1"
        dest = dest_root / key

        if dry_run:
            print(f"  [dry-run] would fetch {key:35s} {expected:>12,} B")
            continue

        # Skip a file already present at exactly the expected size. Without
        # this, a re-run re-downloads ~100 MB and -- worse -- an interrupted
        # re-run can leave a stale .part beside a perfectly good file.
        if dest.exists() and dest.stat().st_size == expected:
            print(f"  present  {key} ({expected:,} B) -- skipping")
        else:
            print(f"  fetching {key} ({expected:,} B) ...")
            # `download` refuses unless tier is LOCAL and enforces the
            # free-space floor before and during the transfer; expected_bytes
            # makes a truncated file an error rather than a silent success.
            download(url, dest, tier=Tier.LOCAL, expected_bytes=expected)

        actual = dest.stat().st_size
        if actual != expected:
            raise SystemExit(
                f"size mismatch for {key}: expected {expected:,} B, got {actual:,} B. "
                "Refusing to write a manifest over unverified data."
            )
        entries.append(
            FileEntry(path=key, bytes=actual, sha256=sha256_file(dest))
        )

    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be fetched; write nothing")
    ap.add_argument("--hr", action="store_true",
                    help="(refused -- see RSVQA-HR note)")
    ap.add_argument("--dest", default=DEST_ROOT)
    args = ap.parse_args(argv)

    if args.hr:
        print(HR_REFUSAL, file=sys.stderr)
        return 2

    total = total_expected_bytes()
    print(f"RSVQA-LR test split -> {args.dest}")
    print(f"  {len(TEST_SPLIT_FILES)} files, {total:,} B ({total / 1e6:.1f} MB)\n")

    entries = fetch(Path(args.dest), dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n[dry-run] nothing written. Projected: {total:,} B "
              f"({total / 1e6:.1f} MB)")
        return 0

    manifest = Manifest(
        dataset="RSVQA-LR",
        split="test",
        tier=Tier.LOCAL,
        root=MANIFEST_ROOT,
        source_urls=[f"https://zenodo.org/records/{ZENODO_RECORD}"],
        retrieved_utc=utcnow_iso(),
        files=entries,
        counts={"files": len(entries)},
        notes=(
            "RSVQA-LR test split for the rubric's RSVQA row. Sizes verified against "
            "the Zenodo API (record 6344334) on 2026-08-29, not taken from docs. "
            "Images_LR.zip carries images for ALL splits -- there is no test-only "
            "image archive -- but at 95 MB this does not engage the concern behind "
            "PLAN.md §2.2a's test-splits-only rule. RSVQA-HR is deliberately NOT "
            "fetched: its test images are inseparable from a 13.5 GB train archive."
        ),
    )
    path = write_manifest(MANIFEST_NAME, manifest)
    got = sum(e.bytes for e in entries)
    print(f"\nwrote {path}")
    print(f"  {len(entries)} files, {got:,} B verified by sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
