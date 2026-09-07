"""Full-content checkpoint verification against a pinned remote inventory.

Streams every file in SHA-256 chunks, compares size and digest against the
pinned inventory in ``locks/model.files.json`` (keys: path, size, sha256 —
or git_blob style with a sha256 member), and writes a verification receipt
to ``locks/receipt.json``. Missing inventory entries, size mismatch or
wrong bytes are hard failures — never a pass.

The receipt records model id/sha, root path, per-file counts and stat
identities (inode, mtime_ns, ctime_ns, size, sha256) so the launcher can
re-check the live tree against what was actually hashed.

Status: NOT QUALIFIED.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List

RECEIPT_KIND = "CHECKPOINT_VERIFIED"
CHUNK_SIZE = 4 * 1024 * 1024


class VerifyError(ValueError):
    """Verification failed; the tree must not be admitted."""


def _iter_inventory(inventory: Any) -> List[Dict[str, Any]]:
    if isinstance(inventory, dict):
        files = inventory.get("files")
        if not isinstance(files, list):
            raise VerifyError("inventory.files must be a list")
    elif isinstance(inventory, list):
        files = inventory
    else:
        raise VerifyError("inventory must be an object or list")
    out: List[Dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise VerifyError("inventory entry must be an object")
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise VerifyError("inventory entry missing path")
        out.append(entry)
    return out


def load_inventory(path: str) -> Any:
    if not os.path.isfile(path):
        raise VerifyError("inventory file not found: %s" % (path,))
    with open(path, "r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise VerifyError("inventory is not valid JSON: %s" % (exc,)) from exc


def _sha256_file(path: str, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    extra = b""
    with open(path, "rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                raise VerifyError("file shrank while hashing: %s" % (path,))
            digest.update(chunk)
            remaining -= len(chunk)
        extra = handle.read(1)
    if extra:
        raise VerifyError("file grew while hashing: %s" % (path,))
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
    """Full-content verification of ``root`` against ``inventory_path``."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise VerifyError("checkpoint root not found: %s" % (root,))
    inventory = load_inventory(inventory_path)
    entries = _iter_inventory(inventory)

    expected: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        rel = entry["path"]
        if rel.startswith("/") or ".." in rel.split("/") or not rel:
            raise VerifyError("inventory path escapes root: %r" % (rel,))
        if rel in expected:
            raise VerifyError("duplicate inventory path: %r" % (rel,))
        sha = entry.get("sha256")
        size = entry.get("size")
        if not isinstance(sha, str) or len(sha) != 64:
            raise VerifyError("inventory entry %r missing sha256" % (rel,))
        if not isinstance(size, int) or size < 0:
            raise VerifyError("inventory entry %r missing size" % (rel,))
        expected[rel] = {"sha256": sha, "size": size}

    verified: List[Dict[str, Any]] = []
    for rel in sorted(expected):
        want = expected[rel]
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            raise VerifyError("missing file: %s" % (rel,))
        st = os.stat(path)
        if st.st_size != want["size"]:
            raise VerifyError(
                "size mismatch for %s: got %d, want %d"
                % (rel, st.st_size, want["size"])
            )
        got = _sha256_file(path, want["size"])
        if got != want["sha256"]:
            raise VerifyError("sha256 mismatch for %s" % (rel,))
        record = {"path": rel, "sha256": got, "size": want["size"]}
        record.update(_stat_identity(st))
        verified.append(record)

    total_bytes = sum(item["size"] for item in verified)
    return {
        "kind": RECEIPT_KIND,
        "model": {
            "id": inventory.get("model", {}).get("id"),
            "sha": inventory.get("model", {}).get("sha"),
        },
        "root": root,
        "file_count": len(verified),
        "total_bytes": total_bytes,
        "files": verified,
    }


def write_receipt(receipt: Dict[str, Any], path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_receipt(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise VerifyError("receipt not found: %s" % (path,))
    with open(path, "r", encoding="utf-8") as handle:
        try:
            receipt = json.load(handle)
        except json.JSONDecodeError as exc:
            raise VerifyError("receipt is not valid JSON: %s" % (exc,)) from exc
    if not isinstance(receipt, dict):
        raise VerifyError("receipt must be an object")
    if receipt.get("kind") != RECEIPT_KIND:
        raise VerifyError("receipt kind mismatch: %r" % (receipt.get("kind"),))
    return receipt


def check_receipt_against_tree(receipt: Dict[str, Any], root: str) -> None:
    """Launcher-side recheck: receipt must match the live mounted tree.

    Compares root path, file count, per-file stat identity (inode, size,
    mtime_ns, ctime_ns) and the recorded full hash set. Any drift means
    the tree changed since verification — refuse to spawn.
    """
    root = os.path.abspath(root)
    if receipt.get("root") != root:
        raise VerifyError(
            "receipt root %r does not match mounted %r" % (receipt.get("root"), root)
        )
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise VerifyError("receipt has no files list")
    for item in files:
        rel = item.get("path")
        if not isinstance(rel, str):
            raise VerifyError("receipt entry missing path")
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            raise VerifyError("receipt file missing from tree: %s" % (rel,))
        st = os.stat(path)
        stat_values = {
            "size": st.st_size,
            "inode": st.st_ino,
            "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns,
        }
        for stat_key, live_value in stat_values.items():
            if item.get(stat_key) != live_value:
                raise VerifyError(
                    "stat drift on %s (%s changed since verification)"
                    % (rel, stat_key)
                )
    if receipt.get("file_count") != len(files):
        raise VerifyError("receipt file_count does not match files list")
    if not receipt.get("model", {}).get("sha"):
        raise VerifyError("receipt missing model.sha")


def main(argv: List[str] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Full-content checkpoint verification (streamed SHA-256)."
    )
    parser.add_argument("root", help="checkpoint root directory")
    parser.add_argument("inventory", help="pinned inventory JSON (model.files)")
    parser.add_argument("--receipt", required=True, help="output receipt path")
    args = parser.parse_args(argv)
    try:
        receipt = verify_files(args.root, args.inventory)
        write_receipt(receipt, args.receipt)
    except VerifyError as exc:
        print("VERIFY FAILED: %s" % (exc,))
        return 1
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
