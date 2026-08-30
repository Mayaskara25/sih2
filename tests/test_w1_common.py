"""Tests for scripts/data/common.py and scripts/data/verify.py.

No real network: resume/streaming behaviour is exercised against a real
local HTTP server (stdlib http.server) bound to 127.0.0.1 on an
ephemeral port, so the actual status-code branching in
guarded_download() gets exercised rather than a monkeypatched stand-in.
Disk-space behaviour is exercised by monkeypatching shutil.disk_usage.
"""

from __future__ import annotations

import hashlib
import http.server
import re
import shutil
import threading
from collections import namedtuple
from pathlib import Path

import pytest

from scripts.data import verify
from scripts.data.common import (
    DiskBudgetExceeded,
    DownloadError,
    FileEntry,
    Manifest,
    Tier,
    TierViolation,
    assert_local_allowed,
    download,
    guarded_download,
    load_manifest,
    require_free_space,
    sha256_file,
    utcnow_iso,
    write_manifest,
)

_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])
GB = 1024**3


# --------------------------------------------------------------------------
# Local HTTP server fixtures
# --------------------------------------------------------------------------


def _make_range_handler(content: bytes):
    """A handler that honours Range requests with proper 206 responses."""

    class RangeHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 -- stdlib API name
            total = len(content)
            range_header = self.headers.get("Range")
            if range_header:
                m = re.match(r"bytes=(\d+)-", range_header)
                start = int(m.group(1)) if m else 0
                if start > total:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{total}")
                    self.end_headers()
                    return
                body = content[start:]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{total - 1}/{total}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(total))
                self.end_headers()
                self.wfile.write(content)

        def log_message(self, *args):  # silence stderr spam
            pass

    return RangeHandler


def _make_ignore_range_handler(content: bytes):
    """A handler that always replies 200 with the full body, Range or not."""

    class IgnoreRangeHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    return IgnoreRangeHandler


@pytest.fixture
def http_server():
    """Factory fixture: start_server(handler_cls) -> base_url. Cleaned up automatically."""
    servers = []

    def _start(handler_cls) -> str:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        port = server.server_address[1]
        return f"http://127.0.0.1:{port}/file"

    yield _start

    for server in servers:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------
# Disk guard
# --------------------------------------------------------------------------


def test_require_free_space_raises_when_short(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda path: _DiskUsage(100 * GB, 99 * GB, 1 * GB))
    with pytest.raises(DiskBudgetExceeded):
        require_free_space(10 * GB, tmp_path)


def test_require_free_space_passes_when_plenty(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda path: _DiskUsage(100 * GB, 1 * GB, 99 * GB))
    require_free_space(10 * GB, tmp_path)  # must not raise


def test_require_free_space_handles_nonexistent_target_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda path: _DiskUsage(100 * GB, 1 * GB, 99 * GB))
    target = tmp_path / "does" / "not" / "exist" / "yet"
    require_free_space(1 * GB, target)  # must not raise (walks up to tmp_path)


# --------------------------------------------------------------------------
# Tier guard
# --------------------------------------------------------------------------


def test_assert_local_allowed_ok_for_local():
    assert_local_allowed(Tier.LOCAL)  # must not raise


@pytest.mark.parametrize("tier", [Tier.CLOUD_ONLY, Tier.STREAM_ONLY])
def test_assert_local_allowed_raises_for_non_local(tier):
    with pytest.raises(TierViolation):
        assert_local_allowed(tier)


def test_download_refuses_non_local_tier_without_touching_network(tmp_path):
    # Bogus URL: if this were ever contacted the test would hang/error on
    # connection refused rather than raising TierViolation cleanly.
    with pytest.raises(TierViolation):
        download("http://127.0.0.1:1/nope", tmp_path / "out.bin", tier=Tier.CLOUD_ONLY, expected_bytes=10)


# --------------------------------------------------------------------------
# Streaming download + resume + atomic rename
# --------------------------------------------------------------------------


def test_full_download_succeeds_and_matches_content(tmp_path, http_server):
    content = b"abcdefgh" * 1024
    url = http_server(_make_range_handler(content))
    dest = tmp_path / "out.bin"

    result = download(url, dest, min_free_bytes=1024, tier=Tier.LOCAL, expected_bytes=len(content), progress=False)

    assert result == dest
    assert dest.exists()
    assert dest.read_bytes() == content
    assert not dest.with_name(dest.name + ".part").exists()


def test_resume_continues_partial_file_rather_than_restarting(tmp_path, http_server):
    content = bytes(range(256)) * 100  # 25600 bytes, deterministic
    url = http_server(_make_range_handler(content))
    dest = tmp_path / "out.bin"
    part = dest.with_name(dest.name + ".part")

    # Pretend a previous run got the first third down.
    prefix_len = len(content) // 3
    part.write_bytes(content[:prefix_len])

    result = download(url, dest, min_free_bytes=1024, tier=Tier.LOCAL, expected_bytes=len(content), progress=False)

    assert result == dest
    assert dest.read_bytes() == content, "resumed download must equal the full original content"
    assert not part.exists()


def test_server_ignoring_range_restarts_clean_not_corrupt(tmp_path, http_server):
    content = b"Z" * 5000
    url = http_server(_make_ignore_range_handler(content))
    dest = tmp_path / "out.bin"
    part = dest.with_name(dest.name + ".part")

    # Existing partial data the server will NOT be able to resume from
    # (this handler always answers 200 with the full body).
    part.write_bytes(b"garbage-prefix-that-does-not-belong")

    result = download(url, dest, min_free_bytes=1024, tier=Tier.LOCAL, expected_bytes=len(content), progress=False)

    assert result == dest
    # Must be exactly the server's content, not garbage-prefix + full body.
    assert dest.read_bytes() == content


def test_interrupted_download_leaves_no_file_that_looks_complete(tmp_path, http_server):
    content = b"short content, not what we expect"
    url = http_server(_make_range_handler(content))
    dest = tmp_path / "out.bin"
    part = dest.with_name(dest.name + ".part")

    with pytest.raises(DownloadError):
        # expected_bytes deliberately wrong -> size-mismatch guard fires
        # after the transfer, before the atomic rename.
        download(url, dest, min_free_bytes=1024, tier=Tier.LOCAL, expected_bytes=len(content) + 999, progress=False)

    assert not dest.exists(), "a size-mismatched transfer must never be renamed into the final path"
    assert part.exists(), "the partial file is kept so a retry can inspect/resume it"
    assert part.stat().st_size != len(content) + 999


def test_disk_floor_aborts_mid_stream_and_keeps_partial(tmp_path, http_server, monkeypatch):
    content = b"x" * 10_000
    url = http_server(_make_range_handler(content))
    dest = tmp_path / "out.bin"
    part = dest.with_name(dest.name + ".part")

    calls = {"n": 0}

    def fake_disk_usage(path):
        calls["n"] += 1
        # Call 1 is the pre-check: plenty of room. Later calls (mid-stream
        # floor checks) report the disk has filled up.
        if calls["n"] <= 1:
            return _DiskUsage(100 * GB, 1 * GB, 99 * GB)
        return _DiskUsage(100 * GB, 99 * GB, 1024)  # far below MIN_FREE_BYTES

    monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)

    with pytest.raises(DiskBudgetExceeded):
        # Exercise guarded_download directly (download() is a thin
        # pass-through to it) to make explicit that the guard itself, not
        # just its wrapper, aborts mid-stream.
        guarded_download(
            url,
            dest,
            tier=Tier.LOCAL,
            expected_bytes=len(content),
            chunk_size=1000,
            floor_check_every=1000,
            min_free_bytes=5 * GB,
            progress=False,
        )

    assert not dest.exists()
    assert part.exists(), "abort must leave the .part file in place for a later resume"
    assert 0 < part.stat().st_size < len(content)


# --------------------------------------------------------------------------
# Checksums
# --------------------------------------------------------------------------


def test_utcnow_iso_is_a_zulu_timestamp():
    stamp = utcnow_iso()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", stamp)


def test_sha256_file_matches_hashlib(tmp_path):
    data = b"the quick brown fox jumps over the lazy dog" * 1000
    f = tmp_path / "data.bin"
    f.write_bytes(data)

    assert sha256_file(f) == hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Manifest round-trip
# --------------------------------------------------------------------------


def test_manifest_round_trips(tmp_path):
    manifest = Manifest(
        dataset="rsvqa_lr",
        split="test",
        tier=Tier.LOCAL,
        root="rsvqa_lr/test",
        source_urls=["https://rsvqa.sylvainlobry.com/LR.zip"],
        retrieved_utc="2026-08-29T00:00:00Z",
        files=[
            FileEntry(path="images/0001.tif", bytes=1234, sha256="a" * 64),
            FileEntry(path="images/0002.tif", bytes=5678, sha256="b" * 64),
        ],
        counts={"images": 2, "qa_pairs": 10},
        notes="",
    )

    manifests_dir = tmp_path / "manifests"
    written_path = write_manifest("rsvqa_lr", manifest, manifests_dir=manifests_dir)

    assert written_path == manifests_dir / "rsvqa_lr.json"
    assert written_path.exists()

    loaded = load_manifest("rsvqa_lr", manifests_dir=manifests_dir)

    assert loaded.dataset == manifest.dataset
    assert loaded.split == manifest.split
    assert loaded.tier == Tier.LOCAL
    assert loaded.root == manifest.root
    assert loaded.source_urls == manifest.source_urls
    assert loaded.files == manifest.files
    assert loaded.counts == manifest.counts
    assert loaded.total_bytes == 1234 + 5678
    assert loaded.file_count == 2


def test_manifest_from_dict_rejects_missing_required_fields():
    with pytest.raises(ValueError):
        Manifest.from_dict({"dataset": "x"})


def test_manifest_from_dict_tolerates_unknown_keys():
    d = {
        "dataset": "x",
        "split": "test",
        "tier": "LOCAL",
        "root": "x/test",
        "source_urls": [],
        "retrieved_utc": "2026-08-29T00:00:00Z",
        "files": [],
        "from_the_future": "ignored",
    }
    m = Manifest.from_dict(d)
    assert m.dataset == "x"


# --------------------------------------------------------------------------
# verify.py
# --------------------------------------------------------------------------


def _write_dataset_files(root: Path, names_and_content: dict[str, bytes]) -> list[FileEntry]:
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, content in names_and_content.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append(FileEntry(path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    return entries


def _manifest_for(entries: list[FileEntry], dataset: str, root: str) -> Manifest:
    return Manifest(
        dataset=dataset,
        split="test",
        tier=Tier.LOCAL,
        root=root,
        source_urls=["https://example.invalid/data.zip"],
        retrieved_utc="2026-08-29T00:00:00Z",
        files=entries,
        counts={"images": len(entries)},
    )


def test_verify_passes_on_matching_dataset(tmp_path, capsys):
    data_root = tmp_path / "data"
    manifests_dir = tmp_path / "manifests"
    entries = _write_dataset_files(data_root / "good" / "test", {"a.bin": b"AAAA", "b.bin": b"BBBB"})
    write_manifest("good", _manifest_for(entries, "good", "good/test"), manifests_dir=manifests_dir)

    code = verify.main(["--data-root", str(data_root), "--manifests-dir", str(manifests_dir)])

    out = capsys.readouterr().out
    assert code == 0
    assert "[PASS] good" in out


def test_verify_reports_not_downloaded_distinctly_and_exits_zero(tmp_path, capsys):
    data_root = tmp_path / "data"  # deliberately never populated
    manifests_dir = tmp_path / "manifests"
    entries = [FileEntry(path="a.bin", bytes=4, sha256="0" * 64)]
    write_manifest("missing_entirely", _manifest_for(entries, "missing_entirely", "missing_entirely/test"), manifests_dir=manifests_dir)

    code = verify.main(["--data-root", str(data_root), "--manifests-dir", str(manifests_dir)])

    out = capsys.readouterr().out
    assert code == 0, "manifest exists but nothing downloaded yet must not fail the run"
    assert "[NOT_DOWNLOADED] missing_entirely" in out


def test_verify_detects_missing_file_as_fail_not_not_downloaded(tmp_path, capsys):
    data_root = tmp_path / "data"
    manifests_dir = tmp_path / "manifests"
    entries = _write_dataset_files(data_root / "partial" / "test", {"a.bin": b"AAAA"})
    # Manifest claims a second file that was never written.
    entries.append(FileEntry(path="b.bin", bytes=4, sha256="f" * 64))
    write_manifest("partial", _manifest_for(entries, "partial", "partial/test"), manifests_dir=manifests_dir)

    code = verify.main(["--data-root", str(data_root), "--manifests-dir", str(manifests_dir)])

    out = capsys.readouterr().out
    assert code != 0
    assert "[FAIL] partial" in out
    assert "missing" in out


def test_verify_detects_corrupted_file(tmp_path, capsys):
    data_root = tmp_path / "data"
    manifests_dir = tmp_path / "manifests"
    entries = _write_dataset_files(data_root / "corrupt" / "test", {"a.bin": b"AAAA"})
    write_manifest("corrupt", _manifest_for(entries, "corrupt", "corrupt/test"), manifests_dir=manifests_dir)

    # Corrupt the file after the manifest was written against the original content.
    (data_root / "corrupt" / "test" / "a.bin").write_bytes(b"AAAA-CORRUPTED")

    code = verify.main(["--data-root", str(data_root), "--manifests-dir", str(manifests_dir)])

    out = capsys.readouterr().out
    assert code != 0
    assert "[FAIL] corrupt" in out


def test_verify_detects_size_mismatch(tmp_path, capsys):
    data_root = tmp_path / "data"
    manifests_dir = tmp_path / "manifests"
    entries = _write_dataset_files(data_root / "sized" / "test", {"a.bin": b"AAAA"})
    # Hand-craft a manifest that disagrees with the real file size but
    # keep the file entry list internally consistent so we're testing the
    # on-disk-vs-manifest check specifically, not the integrity check.
    bad_entries = [FileEntry(path="a.bin", bytes=999, sha256=entries[0].sha256)]
    write_manifest("sized", _manifest_for(bad_entries, "sized", "sized/test"), manifests_dir=manifests_dir)

    code = verify.main(["--data-root", str(data_root), "--manifests-dir", str(manifests_dir)])

    out = capsys.readouterr().out
    assert code != 0
    assert "[FAIL] sized" in out
    assert "size mismatch" in out


def test_verify_fast_flag_skips_hashing_but_still_checks_size(tmp_path, capsys):
    data_root = tmp_path / "data"
    manifests_dir = tmp_path / "manifests"
    entries = _write_dataset_files(data_root / "fastcheck" / "test", {"a.bin": b"AAAA"})
    # Corrupt content but keep size identical -- --fast must NOT catch this,
    # proving it really skips hashing rather than hashing anyway.
    (data_root / "fastcheck" / "test" / "a.bin").write_bytes(b"ZZZZ")

    write_manifest("fastcheck", _manifest_for(entries, "fastcheck", "fastcheck/test"), manifests_dir=manifests_dir)

    code_fast = verify.main(["--data-root", str(data_root), "--manifests-dir", str(manifests_dir), "--fast"])
    assert code_fast == 0

    code_full = verify.main(["--data-root", str(data_root), "--manifests-dir", str(manifests_dir)])
    assert code_full != 0


def test_verify_exits_zero_when_mix_of_pass_and_not_downloaded(tmp_path, capsys):
    data_root = tmp_path / "data"
    manifests_dir = tmp_path / "manifests"
    good_entries = _write_dataset_files(data_root / "good" / "test", {"a.bin": b"AAAA"})
    write_manifest("good", _manifest_for(good_entries, "good", "good/test"), manifests_dir=manifests_dir)

    absent_entries = [FileEntry(path="a.bin", bytes=4, sha256="0" * 64)]
    write_manifest("absent", _manifest_for(absent_entries, "absent", "absent/test"), manifests_dir=manifests_dir)

    code = verify.main(["--data-root", str(data_root), "--manifests-dir", str(manifests_dir)])
    assert code == 0


def test_verify_no_manifests_exits_zero(tmp_path, capsys):
    code = verify.main(["--data-root", str(tmp_path / "data"), "--manifests-dir", str(tmp_path / "empty_manifests")])
    assert code == 0
    assert "no manifests found" in capsys.readouterr().out


def test_verify_cli_subprocess_exit_code(tmp_path):
    import subprocess
    import sys as _sys

    data_root = tmp_path / "data"
    manifests_dir = tmp_path / "manifests"
    entries = _write_dataset_files(data_root / "good" / "test", {"a.bin": b"AAAA"})
    write_manifest("good", _manifest_for(entries, "good", "good/test"), manifests_dir=manifests_dir)

    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [_sys.executable, "-m", "scripts.data.verify", "--data-root", str(data_root), "--manifests-dir", str(manifests_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
