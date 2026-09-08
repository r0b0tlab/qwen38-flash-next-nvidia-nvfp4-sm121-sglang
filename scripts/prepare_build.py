#!/usr/bin/env python3
"""Reconstruct a fresh, hash-verified build input tree; never build or serve."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def run(*args, cwd=None):
    return subprocess.check_output(list(args), cwd=cwd)


def prepare(root, upstream, wheel_dirs):
    root, upstream = root.resolve(strict=True), upstream.resolve(strict=True)
    runtime = json.loads((root / "locks/runtime.json").read_text())
    dependencies = json.loads((root / "locks/dependencies.json").read_text())
    source = runtime["sglang"]
    patch = root / "patches/sglang.patch"
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == source["patch_sha256"]
    destination = root / "build/sglang"
    if destination.exists():
        raise FileExistsError(
            "build/sglang already exists; use a fresh checkout/context"
        )
    wheels = []
    for entry in dependencies["wheels"]:
        choices = [
            directory / entry["file"]
            for directory in wheel_dirs
            if (directory / entry["file"]).is_file()
        ]
        if not choices:
            raise FileNotFoundError("missing pinned wheel: " + entry["file"])
        for choice in choices:
            if (
                choice.is_symlink()
                or hashlib.sha256(choice.read_bytes()).hexdigest() != entry["sha256"]
            ):
                raise ValueError("wheel identity mismatch: " + entry["file"])
        wheels.append((choices[0], entry))
    # Existence of the exact base object is required; no implicit ref repair/fetch.
    run("git", "cat-file", "-e", source["main"] + "^{commit}", cwd=upstream)
    archive = run("git", "archive", source["main"], cwd=upstream)
    (root / "build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="source-", dir=root / "build") as temp_name:
        temp = Path(temp_name)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(temp, filter="data")
        run("git", "init", "-q", cwd=temp)
        run("git", "add", "-f", "--all", cwd=temp)
        run("git", "apply", "--check", str(patch), cwd=temp)
        run("git", "apply", str(patch), cwd=temp)
        run("git", "add", "-f", "--all", cwd=temp)
        tree = run("git", "write-tree", cwd=temp).decode().strip()
        if tree != source["tree"]:
            raise ValueError("reconstructed tree differs from runtime lock")
        for relative, expected in source["python_files"].items():
            actual = temp / "python/sglang" / relative
            if hashlib.sha256(actual.read_bytes()).hexdigest() != expected:
                raise ValueError("source hash mismatch: " + relative)
        temp.rename(destination)
    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir(exist_ok=True)
    expected_names = {entry["file"] for _, entry in wheels}
    if any(path.name not in expected_names for path in wheelhouse.iterdir()):
        raise ValueError("wheelhouse contains an unlisted artifact")
    for path, entry in wheels:
        target = wheelhouse / entry["file"]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("existing wheelhouse artifact drift")
        else:
            shutil.copyfile(path, target)
            target.chmod(0o444)
    manifest = {
        "status": "BUILD_INPUTS_VERIFIED",
        "base": source["main"],
        "source_commit": source["commit"],
        "source_tree": tree,
        "source_archive_sha256": hashlib.sha256(archive).hexdigest(),
        "patch_sha256": source["patch_sha256"],
        "python_files": len(source["python_files"]),
        "wheels": len(wheels),
        "image_built": False,
        "gpu_qualification": "NOT_RUN",
    }
    (root / "build/preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        type=Path,
        required=True,
        help="existing read-only SGLang Git checkout containing the locked main",
    )
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        action="append",
        required=True,
        help="directory containing pinned wheels; repeatable",
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.upstream, args.wheel_dir), indent=2))


if __name__ == "__main__":
    main()
