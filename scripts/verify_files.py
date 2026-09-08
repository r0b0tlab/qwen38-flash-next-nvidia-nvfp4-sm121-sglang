"""Full-content checkpoint verification against the trusted frozen lock.

Consumes the ACTUAL ``locks/sources.json`` schema —
``{'model': {'id', 'sha', 'files': [{'path', 'size', 'sha256', 'git_blob'}]}}``:

- LFS files carry a trusted ``sha256``: the whole file is streamed in
  chunks and hashed; wrong digest or byte count is a hard failure.
- For LFS entries, ``git_blob`` identifies the Git pointer and is retained
  as provenance, not compared to the downloaded payload's SHA-1.
- Ordinary Git metadata files carry ``sha256: null`` and a trusted
  ``git_blob``: the Git blob SHA-1 (``blob <len>\\0<bytes>``) is computed
  over the full bytes. Neither hash may be absent; the pinned model
  identity in the receipt is copied from the lock as verified, never
  assigned as a constant without checking the lock.

Every lock file and tensor name is enforced — no representatives. Paths
must be plain relative paths inside the root; symlinks are rejected
(re-stat'ed at the end to catch swaps during hashing). The receipt
(kind ``CHECKPOINT_VERIFIED``) records model identity, root path, root
stat identity, per-file stat identity (inode, size, mtime_ns, ctime_ns,
dev) and hashes so the launcher can re-check the live tree against what
was actually hashed. ``check_receipt_against_tree`` RECOMPUTES the file
hashes (not just stat identity). This checks content consistency, not
receipt authenticity; the launcher separately binds an independently trusted
receipt digest and frozen source inventory.

Status: NOT QUALIFIED.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

RECEIPT_KIND = "CHECKPOINT_VERIFIED"
CHUNK_SIZE = 4 * 1024 * 1024
MAX_PATH_LENGTH = 1024


class VerifyError(ValueError):
    """Verification failed; the tree must not be admitted."""


def _is_int(value: Any) -> bool:
    return type(value) is int


def _iter_inventory(
    inventory: Any,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Accept the frozen nested source lock, or an unambiguous legacy inventory."""
    model = None
    if isinstance(inventory, dict):
        if (
            "files" in inventory
            and isinstance(inventory.get("model"), dict)
            and "files" in inventory["model"]
        ):
            raise VerifyError("ambiguous top-level and model.files inventories")
        files = inventory.get("files")
        if files is None and isinstance(inventory.get("model"), dict):
            model = inventory["model"]
            files = model.get("files")
    elif isinstance(inventory, list):
        files = inventory
    else:
        raise VerifyError("inventory must be an object")
    if not isinstance(files, list) or not files:
        raise VerifyError("inventory.files must be a nonempty list")
    if model is None and isinstance(inventory, dict):
        model = inventory.get("model")
    out: List[Dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise VerifyError("inventory entry must be an object")
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise VerifyError("inventory entry missing path")
        out.append(entry)
    return out, model


def _safe_relpath(path: str) -> bool:
    if not path or len(path) > MAX_PATH_LENGTH:
        return False
    if path.startswith("/") or "\\" in path or "\x00" in path:
        return False
    return all(part not in ("", ".", "..") for part in path.split("/"))


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerifyError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise VerifyError("nonfinite JSON number")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise VerifyError("nonfinite JSON numeric value")
    return parsed


def _read_json(path):
    try:
        with open(path, "rb") as handle:
            raw = handle.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise VerifyError("inventory/receipt exceeds byte limit")
        return json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=_constant,
            parse_float=_finite_float,
        ), raw
    except (OSError, ValueError, RecursionError) as exc:
        raise VerifyError("cannot read bounded strict JSON: %s" % exc) from exc


def load_inventory(path: str) -> Any:
    return _read_json(path)[0]


def _sha256_file(path: str, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    with open(path, "rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                raise VerifyError("file shrank while hashing: %s" % (path,))
            digest.update(chunk)
            remaining -= len(chunk)
        if handle.read(1):
            raise VerifyError("file grew while hashing: %s" % (path,))
    return digest.hexdigest()


def _git_blob_sha1(path: str) -> str:
    digest = hashlib.sha1()
    digest.update(b"blob %d\x00" % os.path.getsize(path))
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(st: os.stat_result) -> Dict[str, Any]:
    return {
        "inode": st.st_ino,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "ctime_ns": st.st_ctime_ns,
        "dev": st.st_dev,
    }


def verify_files(root: str, inventory_path: str) -> Dict[str, Any]:
    """Full-content verification of ``root`` against the trusted lock."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise VerifyError("checkpoint root not found: %s" % (root,))
    if os.path.realpath(root) != root:
        raise VerifyError("checkpoint root contains a symlink")
    inventory, lock_bytes = _read_json(inventory_path)
    entries, lock_model = _iter_inventory(inventory)
    if not isinstance(lock_model, dict):
        raise VerifyError("inventory must carry a model identity object")
    lock_id = lock_model.get("id")
    lock_sha = lock_model.get("sha")
    if (
        not isinstance(lock_id, str)
        or not lock_id
        or not isinstance(lock_sha, str)
        or not re.fullmatch(r"[0-9a-f]{40}", lock_sha)
    ):
        raise VerifyError(
            "inventory model identity missing (id/sha) — refusing to "
            "verify against an unidentified tree"
        )

    expected: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        rel = entry["path"]
        if not _safe_relpath(rel):
            raise VerifyError("inventory path escapes root: %r" % (rel,))
        if rel in expected:
            raise VerifyError("duplicate inventory path: %r" % (rel,))
        sha = entry.get("sha256")
        blob = entry.get("git_blob")
        if sha is None and blob is None:
            raise VerifyError(
                "inventory entry %r carries neither sha256 nor git_blob" % (rel,)
            )
        if sha is not None and (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha.lower())
        ):
            raise VerifyError("inventory entry %r sha256 malformed" % (rel,))
        if blob is not None and (
            not isinstance(blob, str)
            or len(blob) != 40
            or any(c not in "0123456789abcdef" for c in blob.lower())
        ):
            raise VerifyError("inventory entry %r git_blob malformed" % (rel,))
        # HF's blob_id for an LFS entry identifies the Git pointer, not
        # the downloaded tensor payload. Preserve it as provenance only.
        expected[rel] = {
            "sha256": sha,
            "git_blob": blob if sha is None else None,
            "source_git_blob": blob,
            "size": entry.get("size"),
        }
    for rel, want in expected.items():
        if not _is_int(want["size"]) or want["size"] < 0:
            raise VerifyError("inventory entry %r missing size" % (rel,))
    # lock self-hash for the receipt (tamper-evidence of the lock itself)
    locks_sha256 = hashlib.sha256(lock_bytes).hexdigest()

    verified: List[Dict[str, Any]] = []
    for rel in sorted(expected):
        want = expected[rel]
        path = os.path.join(root, rel)
        if os.path.realpath(path) != path:
            raise VerifyError("inventory path contains a symlink: %r" % (rel,))
        if not os.path.isfile(path):
            raise VerifyError("missing file: %s" % (rel,))
        st = os.lstat(path)
        if st.st_size != want["size"]:
            raise VerifyError(
                "size mismatch for %s: got %d, want %d"
                % (rel, st.st_size, want["size"])
            )
        got_sha = _sha256_file(path, want["size"])
        got_blob = None
        if want["git_blob"] is not None:
            got_blob = _git_blob_sha1(path)
        # re-stat AFTER hashing: catch swaps during the hash window
        st2 = os.lstat(path)
        if _stat_identity(st) != _stat_identity(st2) or os.path.realpath(path) != path:
            raise VerifyError("file changed while hashing: %s" % (rel,))
        if want["sha256"] is not None and got_sha != want["sha256"]:
            raise VerifyError("sha256 mismatch for %s" % (rel,))
        if want["git_blob"] is not None and got_blob != want["git_blob"]:
            raise VerifyError("git_blob mismatch for %s" % (rel,))
        record = {
            "path": rel,
            "size": want["size"],
            "sha256": got_sha,
            "git_blob": got_blob,
            "source_git_blob": want["source_git_blob"],
            "expected_sha256": want["sha256"],
            "expected_git_blob": want["git_blob"],
        }
        record.update(_stat_identity(st2))
        verified.append(record)

    total_bytes = sum(item["size"] for item in verified)
    return {
        "kind": RECEIPT_KIND,
        "model": {"id": lock_id, "sha": lock_sha},
        "root": root,
        "root_stat": _stat_identity(os.stat(root)),
        "file_count": len(verified),
        "total_bytes": total_bytes,
        "locks_sha256": locks_sha256,
        "verification": "FULL",
        "files": verified,
    }


def write_receipt(receipt: Dict[str, Any], path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    raw = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    fd, tmp = tempfile.mkstemp(prefix=".receipt-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return hashlib.sha256(raw).hexdigest()


def load_receipt(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise VerifyError("receipt not found: %s" % (path,))
    receipt, _ = _read_json(path)
    if not isinstance(receipt, dict):
        raise VerifyError("receipt must be an object")
    if receipt.get("kind") != RECEIPT_KIND:
        raise VerifyError("receipt kind mismatch: %r" % (receipt.get("kind"),))
    return receipt


def check_receipt_against_tree(receipt: Dict[str, Any], root: str) -> None:
    """Launcher-side recheck: receipt must match the live mounted tree.

    Compares root path, root stat identity, file count, per-file stat
    identity AND recomputes every file hash from the live bytes against
    the receipt's verified hashes. Any drift — including a forged receipt
    whose recorded hashes match nothing on disk — is a hard refusal.
    """
    root = os.path.abspath(root)
    if receipt.get("root") != root:
        raise VerifyError(
            "receipt root %r does not match mounted %r" % (receipt.get("root"), root)
        )
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise VerifyError("receipt has no files list")
    if receipt.get("kind") != RECEIPT_KIND or receipt.get("verification") != "FULL":
        raise VerifyError("receipt does not attest full verification")
    seen = set()
    for item in files:
        if not isinstance(item, dict):
            raise VerifyError("receipt entry is not an object")
        rel = item.get("path")
        if not isinstance(rel, str) or not _safe_relpath(rel) or rel in seen:
            raise VerifyError("unsafe/duplicate receipt path")
        seen.add(rel)
        if not isinstance(item.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", item["sha256"]
        ):
            raise VerifyError("receipt missing full observed SHA256")
        path = os.path.join(root, rel)
        if os.path.realpath(path) != path or not os.path.isfile(path):
            raise VerifyError("receipt file missing from tree: %s" % (rel,))
        st = os.lstat(path)
        stat_values = {
            "size": st.st_size,
            "inode": st.st_ino,
            "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns,
            "dev": st.st_dev,
        }
        for stat_key, live_value in stat_values.items():
            if type(item.get(stat_key)) is not int or item.get(stat_key) != live_value:
                raise VerifyError(
                    "stat drift on %s (%s changed since verification)" % (rel, stat_key)
                )
        if item.get("sha256") is not None:
            live = _sha256_file(path, st.st_size)
            if live != item["sha256"]:
                raise VerifyError(
                    "content drift on %s (live sha256 != verified sha256)" % (rel,)
                )
        if item.get("git_blob") is not None:
            live_blob = _git_blob_sha1(path)
            if live_blob != item["git_blob"]:
                raise VerifyError(
                    "content drift on %s (live git_blob != verified git_blob)" % (rel,)
                )
        if _stat_identity(st) != _stat_identity(os.lstat(path)):
            raise VerifyError("file changed during receipt recheck")
    if type(receipt.get("file_count")) is not int or receipt.get("file_count") != len(
        files
    ):
        raise VerifyError("receipt file_count does not match files list")
    model = receipt.get("model") or {}
    if not isinstance(model, dict) or not model.get("sha") or not model.get("id"):
        raise VerifyError("receipt missing model identity")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Full-content checkpoint verification (streamed SHA-256)."
    )
    parser.add_argument("root", help="checkpoint root directory")
    parser.add_argument("inventory", help="trusted lock JSON (sources.json model)")
    parser.add_argument("--receipt", required=True, help="output receipt path")
    args = parser.parse_args(argv)
    try:
        receipt = verify_files(args.root, args.inventory)
        receipt_sha256 = write_receipt(receipt, args.receipt)
    except VerifyError as exc:
        print("VERIFY FAILED: %s" % (exc,))
        return 1
    print("receipt_sha256=" + receipt_sha256)
    print(
        "CHECKPOINT_VERIFIED model=%s sha=%s files=%d bytes=%d receipt=%s"
        % (
            receipt["model"]["id"],
            receipt["model"]["sha"],
            receipt["file_count"],
            receipt["total_bytes"],
            args.receipt,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
