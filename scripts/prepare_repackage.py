#!/usr/bin/env python3
"""Prepare a packaging-only retry from an independently retained native wheel.

The cold Dockerfile remains authoritative. This route changes only the source
of /tmp/sglang-wheels and inserts an exact artifact check; the unused builder
stage is not executed by BuildKit. No model/server/kernel code is substituted.
"""

import argparse
import email
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import struct
import zipfile


def render_recipe(original, wheel_sha):
    if not isinstance(wheel_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", wheel_sha):
        raise ValueError("exact wheel SHA256 required")
    anchor = "COPY --from=builder /wheels/ /tmp/sglang-wheels/"
    if original.count(anchor) != 1:
        raise ValueError("cold recipe wheel-source anchor changed")
    probe = (
        "import glob,hashlib; p=glob.glob('/tmp/sglang-wheels/*.whl'); "
        "assert len(p)==1; assert hashlib.sha256(open(p[0],'rb').read()).hexdigest()=='"
        + wheel_sha
        + "'"
    )
    replacement = (
        "COPY build/repackage/ /tmp/sglang-wheels/\nRUN python3 -c " + json.dumps(probe)
    )
    return original.replace(anchor, replacement)


def prepare(root, wheel, receipt):
    root = root.resolve(strict=True)
    if wheel.is_symlink() or not wheel.is_file():
        raise ValueError("wheel must be a regular non-symlink file")
    runtime = json.loads((root / "locks/runtime.json").read_text())
    source = runtime["sglang"]
    proof = json.loads(receipt.read_text())
    for key, value in (
        ("source_commit", source["commit"]),
        ("source_tree", source["tree"]),
        ("package_version", source["package_version"]),
    ):
        if proof.get(key) != value:
            raise ValueError("compiled artifact provenance mismatch: " + key)
    raw = wheel.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (
        wheel.name != proof.get("wheel_file")
        or len(raw) != proof.get("wheel_bytes")
        or digest != proof.get("wheel_sha256")
    ):
        raise ValueError("compiled wheel identity mismatch")
    spec = importlib.util.spec_from_file_location(
        "wheel_inventory", root / "docker/verify_environment.py"
    )
    assert spec and spec.loader
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    expected = checker.installed_source_inventory(source)
    with zipfile.ZipFile(wheel) as archive:
        actual = {
            p[len("sglang/") :]: p
            for p in archive.namelist()
            if p.startswith("sglang/") and p.endswith(".py")
        }
        if set(actual) != set(expected) | {"_version.py"}:
            raise ValueError("compiled wheel Python file set differs")
        for name, wanted in expected.items():
            if hashlib.sha256(archive.read(actual[name])).hexdigest() != wanted:
                raise ValueError("wheel source mismatch: " + name)
        metadata_names = [
            n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError("ambiguous wheel metadata")
        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
        if (
            metadata["Name"] != "sglang"
            or metadata["Version"] != source["package_version"]
        ):
            raise ValueError("wheel package version mismatch")
        native = [n for n in archive.namelist() if n.endswith(".so")]
        if len(native) != len(runtime["rust_modules"]):
            raise ValueError("native extension count differs")
        for module in runtime["rust_modules"]:
            candidates = [
                n for n in native if n.startswith(module.replace(".", "/") + ".")
            ]
            if len(candidates) != 1:
                raise ValueError("missing native module: " + module)
            header = archive.read(candidates[0])[:64]
            if (
                header[:6] != b"\x7fELF\x02\x01"
                or struct.unpack_from("<H", header, 18)[0] != 183
            ):
                raise ValueError("native extension is not AArch64 ELF")
    target = root / "build/repackage"
    target.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(wheel, target / wheel.name)
    (target / wheel.name).chmod(0o444)
    original = (root / "Dockerfile").read_text()
    recipe = render_recipe(original, digest)
    recipe_path = root / "build/Dockerfile.repackage"
    with recipe_path.open("x") as stream:
        stream.write(recipe)
    report = {
        "status": "PREBUILT_NATIVE_WHEEL_ADMITTED",
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "wheel_sha256": digest,
        "wheel_bytes": len(raw),
        "native_modules": runtime["rust_modules"],
        "installed_python_sources": len(expected),
        "cold_recipe_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "recipe_sha256": hashlib.sha256(recipe.encode()).hexdigest(),
        "recipe": str(recipe_path),
        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "gpu_qualification": "NOT_RUN",
    }
    with (root / "build/repackage.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.wheel, args.receipt), indent=2))


if __name__ == "__main__":
    main()
