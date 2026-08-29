"""Shared plumbing for every W1 dataset fetcher.

Two things dominate the design (PLAN.md S2.2a, W1 work order):

1. **Not blowing up the disk.**  ``require_free_space`` refuses BEFORE a
   download starts; ``guarded_download`` (what ``download`` calls) keeps
   checking free space *while streaming* and aborts if it drops below a
   floor, because a download that dies at 100% disk can wedge the machine.
2. **Reproducibility via a manifest.**  Every dataset that lands in
   ``data/`` gets a ``data/manifests/<name>.json`` recording exactly what
   was fetched (files, sizes, sha256, counts, source URLs) so someone
   without the data can verify a re-download matches, and ``verify.py``
   can re-validate a local copy at any time.

No dataset is ever fetched from this module directly at import time, and
this module never chooses a URL or a dataset name -- that is a fetcher's
job. This module only provides the primitives fetchers build on.

Tier discipline (PLAN.md S2.2a): BigEarthNet's S1/S2 archives, any
dataset's TRAIN split, and full extracted slices must NEVER be written to
this machine's disk -- they live in Drive/Kaggle or get streamed and
discarded. ``Tier`` + ``assert_local_allowed`` enforce that in code, not
in a comment: ``download``/``guarded_download`` take a required ``tier``
argument and refuse outright unless it is ``Tier.LOCAL``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "Tier",
    "TierViolation",
    "DiskBudgetExceeded",
    "DownloadError",
    "MIN_FREE_BYTES",
    "DEFAULT_MANIFESTS_DIR",
    "assert_local_allowed",
    "require_free_space",
    "download",
    "guarded_download",
    "sha256_file",
    "FileEntry",
    "Manifest",
    "write_manifest",
    "load_manifest",
    "manifest_path",
    "utcnow_iso",
]


# --------------------------------------------------------------------------
# Tier tagging (PLAN.md S2.2a)
# --------------------------------------------------------------------------


class Tier(str, Enum):
    """Where a dataset (or split of one) is allowed to live.

    LOCAL        -- may be written under data/ on this machine.
    CLOUD_ONLY   -- must live in Drive/Kaggle; never written to local disk
                    (e.g. train splits, the extracted BEN slice).
    STREAM_ONLY  -- passes through a pipe (curl | zstd -dc | tar -x) and is
                    never materialised anywhere as a whole archive
                    (e.g. the raw BEN S1/S2 .tar.zst archives).
    """

    LOCAL = "LOCAL"
    CLOUD_ONLY = "CLOUD_ONLY"
    STREAM_ONLY = "STREAM_ONLY"


class TierViolation(RuntimeError):
    """Raised when a fetcher tries to store a non-LOCAL-tier artifact on disk."""


def assert_local_allowed(tier: Tier) -> None:
    """Raise TierViolation unless ``tier`` is explicitly Tier.LOCAL.

    This is the mechanism, not a comment: any code path that writes bytes
    into ``data/`` must call this (directly, or via ``download`` /
    ``guarded_download`` which call it internally) before doing so.
    """
    if tier is not Tier.LOCAL:
        raise TierViolation(
            f"tier {tier!r} may not be stored locally under data/ -- "
            "PLAN.md S2.2a: only LOCAL-tier datasets/splits belong on this "
            "machine's disk. CLOUD_ONLY data goes to Drive/Kaggle; "
            "STREAM_ONLY data is piped through and discarded, never "
            "written as a whole file."
        )


# --------------------------------------------------------------------------
# Disk guard
# --------------------------------------------------------------------------

#: PLAN.md S2.2a: "If free space drops below 5 GB, stop and escalate rather
#: than deleting things ad hoc." This is the floor both the pre-check and
#: the mid-stream guard enforce.
MIN_FREE_BYTES = 5 * 1024**3


class DiskBudgetExceeded(RuntimeError):
    """Raised when a download would (or does) push free space below the floor."""


def _existing_ancestor(path: Path) -> Path:
    """Walk up from ``path`` to the nearest ancestor that actually exists.

    shutil.disk_usage requires an existing path; a dataset's target
    directory frequently does not exist yet on the very first download.
    """
    path = Path(path).resolve()
    while not path.exists():
        parent = path.parent
        if parent == path:
            break
        path = parent
    return path


def require_free_space(bytes_needed: int, path: str | Path = "data/") -> None:
    """Refuse BEFORE a download starts if free space is too tight.

    Checks actual free space via ``shutil.disk_usage`` on the nearest
    existing ancestor of ``path`` (the target directory need not exist
    yet). Raises DiskBudgetExceeded if fewer than ``bytes_needed`` bytes
    are free.
    """
    probe = _existing_ancestor(Path(path))
    usage = shutil.disk_usage(probe)
    if usage.free < bytes_needed:
        raise DiskBudgetExceeded(
            f"only {usage.free} bytes free at {probe} "
            f"({usage.free / 1024**3:.2f} GB); need {bytes_needed} bytes "
            f"({bytes_needed / 1024**3:.2f} GB)"
        )


# --------------------------------------------------------------------------
# Streaming download with resume
# --------------------------------------------------------------------------


class DownloadError(RuntimeError):
    """Raised for HTTP errors, size mismatches, or unrecoverable resume failures."""


_CONTENT_RANGE_RE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)")


def _parse_content_range(header: str) -> tuple[int, int, int | None]:
    m = _CONTENT_RANGE_RE.match(header.strip())
    if not m:
        raise DownloadError(f"unparseable Content-Range header: {header!r}")
    start, end, total = m.groups()
    return int(start), int(end), (None if total == "*" else int(total))


def _print_progress(written: int, total: int | None) -> None:
    if total:
        pct = min(100.0, 100.0 * written / total)
        sys.stderr.write(f"\r  {written / 1024**2:9.1f} MiB / {total / 1024**2:.1f} MiB ({pct:5.1f}%)")
    else:
        sys.stderr.write(f"\r  {written / 1024**2:9.1f} MiB")
    sys.stderr.flush()


def guarded_download(
    url: str,
    dest: str | Path,
    *,
    tier: Tier,
    expected_bytes: int | None = None,
    min_free_bytes: int = MIN_FREE_BYTES,
    chunk_size: int = 1024 * 1024,
    floor_check_every: int = 32 * 1024 * 1024,
    progress: bool = True,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> Path:
    """Stream ``url`` to ``dest``, resuming a partial ``.part`` file if present.

    Guarantees:
    - refuses outright unless ``tier is Tier.LOCAL`` (see assert_local_allowed);
    - refuses BEFORE starting if free space looks too tight for the whole
      download (when ``expected_bytes`` is known) or below the floor
      otherwise;
    - streams in ``chunk_size`` chunks, never loading the whole body into
      memory;
    - re-checks free space roughly every ``floor_check_every`` bytes and
      raises DiskBudgetExceeded mid-stream if it has dropped below
      ``min_free_bytes`` -- the floor matters more than the pre-check,
      because a download that dies at 100% disk can wedge the machine.
      The partial ``.part`` file is deliberately left in place so a retry
      can resume;
    - writes to ``dest`` + ``.part`` and only ``os.replace``s it onto
      ``dest`` after the transfer completes AND (if given) matches
      ``expected_bytes`` -- so an interrupted or truncated download can
      never masquerade as a complete file;
    - if the server honours ``Range`` (206 + Content-Range), resumes from
      the existing ``.part``'s size; if it ignores Range and replies 200
      with the full body, or replies 416, the partial file is discarded
      and the download restarts from zero rather than corrupting the file
      by appending a full body onto a partial one.
    """
    assert_local_allowed(tier)

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    offset = part.stat().st_size if part.exists() else 0

    # Pre-check: refuse before a single byte moves.
    if expected_bytes is not None:
        require_free_space(max(expected_bytes - offset, 0) + min_free_bytes, dest.parent)
    else:
        require_free_space(min_free_bytes, dest.parent)

    max_attempts = 2  # one restart-from-zero allowed if the server won't resume
    for attempt in range(max_attempts):
        headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
        request = urllib.request.Request(url, headers=headers)

        with urlopen(request) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()

            if status == 200:
                # Server ignored Range (or none was sent): full body from byte 0.
                if offset > 0:
                    offset = 0
                mode = "wb"
            elif status == 206:
                content_range = response.headers.get("Content-Range", "")
                start, _end, _total = _parse_content_range(content_range)
                if start != offset:
                    # Server resumed from somewhere we didn't ask for --
                    # do not trust it enough to append. Restart clean.
                    offset = 0
                    if attempt + 1 < max_attempts:
                        continue
                    raise DownloadError(
                        f"server returned Content-Range starting at {start}, "
                        f"expected {offset}, and no retries left for {url}"
                    )
                mode = "ab" if offset > 0 else "wb"
            elif status == 416:
                # Range not satisfiable. Complete already, or corrupt -- either way
                # we cannot append; verify or restart.
                if expected_bytes is not None and offset == expected_bytes:
                    os.replace(part, dest)
                    return dest
                offset = 0
                if attempt + 1 < max_attempts:
                    continue
                raise DownloadError(f"server returned 416 for {url} and retry exhausted")
            else:
                raise DownloadError(f"unexpected HTTP status {status} fetching {url}")

            if mode == "wb" and part.exists():
                part.unlink()

            written = offset
            since_check = 0
            with open(part, mode) as fh:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
                    since_check += len(chunk)
                    if since_check >= floor_check_every:
                        since_check = 0
                        fh.flush()
                        free = shutil.disk_usage(dest.parent).free
                        if free < min_free_bytes:
                            raise DiskBudgetExceeded(
                                f"free space dropped to {free} bytes "
                                f"(< floor {min_free_bytes}) while downloading "
                                f"{url}; partial file kept at {part} for resume"
                            )
                    if progress:
                        _print_progress(written, expected_bytes)
            if progress:
                sys.stderr.write("\n")
        break  # success -- exit the retry loop
    else:  # pragma: no cover -- defensive, loop always breaks or raises above
        raise DownloadError(f"failed to download {url} after {max_attempts} attempts")

    final_size = part.stat().st_size
    if expected_bytes is not None and final_size != expected_bytes:
        raise DownloadError(
            f"size mismatch downloading {url}: got {final_size} bytes, "
            f"expected {expected_bytes} (partial file kept at {part})"
        )

    os.replace(part, dest)
    return dest


def download(
    url: str,
    dest: str | Path,
    *,
    tier: Tier,
    expected_bytes: int | None = None,
    **kwargs: Any,
) -> Path:
    """Convenience entry point for fetchers -- always disk-guarded.

    Thin wrapper over ``guarded_download`` with the same defaults; kept as
    a separate name because "download" is what a fetcher script should
    call, while "guarded_download" documents that the guard is not
    optional.
    """
    return guarded_download(url, dest, tier=tier, expected_bytes=expected_bytes, **kwargs)


# --------------------------------------------------------------------------
# Checksums
# --------------------------------------------------------------------------


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the sha256 hex digest of a file, reading in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------

#: data/manifests/ is the one part of data/ that IS committed (see
#: .gitignore) -- it is what makes a subsampled slice reproducible for
#: someone who does not have the data.
DEFAULT_MANIFESTS_DIR = Path("data/manifests")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FileEntry:
    """One file tracked by a manifest, path relative to the manifest's `root`."""

    path: str
    bytes: int
    sha256: str

    def to_dict(self) -> dict:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, d: dict) -> "FileEntry":
        return cls(path=str(d["path"]), bytes=int(d["bytes"]), sha256=str(d["sha256"]))


@dataclass
class Manifest:
    """Everything verify.py needs to re-validate a local dataset copy.

    ``root`` is a path relative to the data root (e.g. "data/") under
    which this dataset's files live; each FileEntry.path is relative to
    ``root`` in turn. ``counts`` is free-form (e.g. {"images": 1000,
    "qa_pairs": 5000}) since different datasets count different things.
    ``notes`` exists specifically for recording substitutions (PLAN.md
    W1's Tier-3 abort trigger: "record the substitution ... in the
    manifest").
    """

    dataset: str
    split: str
    tier: Tier
    root: str
    source_urls: list[str]
    retrieved_utc: str
    files: list[FileEntry] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    notes: str = ""
    schema_version: int = 1

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes for f in self.files)

    @property
    def file_count(self) -> int:
        return len(self.files)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "split": self.split,
            "tier": Tier(self.tier).value,
            "root": self.root,
            "source_urls": list(self.source_urls),
            "retrieved_utc": self.retrieved_utc,
            "counts": dict(self.counts),
            "notes": self.notes,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "files": [f.to_dict() for f in self.files],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        required = ("dataset", "split", "tier", "root", "source_urls", "retrieved_utc")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"manifest missing required field(s): {missing}")
        return cls(
            dataset=str(d["dataset"]),
            split=str(d["split"]),
            tier=Tier(d["tier"]),
            root=str(d["root"]),
            source_urls=list(d["source_urls"]),
            retrieved_utc=str(d["retrieved_utc"]),
            files=[FileEntry.from_dict(f) for f in d.get("files", [])],
            counts=dict(d.get("counts", {})),
            notes=str(d.get("notes", "")),
            schema_version=int(d.get("schema_version", 1)),
        )


def manifest_path(name: str, manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR) -> Path:
    return Path(manifests_dir) / f"{name}.json"


def write_manifest(
    name: str,
    manifest: Manifest,
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
) -> Path:
    """Write ``manifest`` to ``<manifests_dir>/<name>.json``, atomically."""
    manifests_dir = Path(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    dest = manifest_path(name, manifests_dir)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, dest)
    return dest


def load_manifest(name: str, manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR) -> Manifest:
    """Read ``<manifests_dir>/<name>.json`` back into a Manifest."""
    path = manifest_path(name, manifests_dir)
    data = json.loads(path.read_text())
    return Manifest.from_dict(data)


def iter_manifest_names(manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR) -> Iterable[str]:
    """Yield dataset names (manifest file stems) found under ``manifests_dir``, sorted."""
    manifests_dir = Path(manifests_dir)
    if not manifests_dir.exists():
        return
    for path in sorted(manifests_dir.glob("*.json")):
        yield path.stem
