"""Pinned dependency, source, ABI and default-user checks for image builds."""

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


WHEEL_OMISSIONS = (
    "multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py",
    "multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/diffusion_skill_env.py",
)


def installed_source_inventory(source):
    # The full source inventory remains intact for reconstruction. Upstream
    # wheel packaging omits exactly these two hidden developer-skill scripts.
    if source.get("wheel_omissions") != list(WHEEL_OMISSIONS):
        raise ValueError("unrecognized installed-wheel omission policy")
    if not set(WHEEL_OMISSIONS) <= source["python_files"].keys():
        raise ValueError("omitted developer files missing from full source inventory")
    return {
        name: sha
        for name, sha in source["python_files"].items()
        if name not in WHEEL_OMISSIONS
    }


def dependency_violations(packages, environment):
    versions = {canonicalize_name(row["name"]): row["version"] for row in packages}
    errors = []
    for row in packages:
        for text in row.get("requires", []):
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            found = versions.get(canonicalize_name(requirement.name))
            if found is None or not requirement.specifier.contains(
                found, prereleases=True
            ):
                errors.append((canonicalize_name(row["name"]), str(requirement), found))
    return errors


def verify(root, phase):
    dependencies = json.loads((root / "locks/dependencies.json").read_text())
    runtime = json.loads((root / "locks/runtime.json").read_text())
    packages = [
        {"name": d.metadata["Name"], "version": d.version, "requires": d.requires or []}
        for d in metadata.distributions()
    ]
    environment = {**default_environment(), "extra": ""}
    assert environment["platform_machine"] == "aarch64", "ARM64 image required"
    for name, wanted in dependencies["pins"].items():
        assert metadata.version(name) == wanted, (name, metadata.version(name), wanted)
    for text in dependencies["source_requirements"]:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        assert requirement.specifier.contains(
            metadata.version(requirement.name), prereleases=True
        ), text
    violations = dependency_violations(packages, environment)
    assert set(violations) == {
        tuple(row) for row in dependencies["metadata_exceptions"]
    }, violations
    checked = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    lines = (checked.stdout + checked.stderr).splitlines()
    assert checked.returncode == 1 and set(lines) == set(dependencies["pip_lines"]), (
        lines
    )
    special = dependencies["platform_exception"]
    distribution = metadata.distribution(special["name"])
    assert distribution.version == special["version"]
    assert "Tag: " + special["wheel_tag"] in (distribution.read_text("WHEEL") or "")
    libraries = []
    for item in distribution.files or []:
        if ".so" in str(item):
            path = Path(str(distribution.locate_file(item)))
            with path.open("rb") as stream:
                header = stream.read(64)
            assert header[:6] == b"\x7fELF\x02\x01"
            assert struct.unpack_from("<H", header, 18)[0] == special["elf_machine"]
            libraries.append(str(item))
    assert libraries, "no native library for SBSA attestation"
    version = None
    modules = []
    if phase == "runtime":
        assert (
            os.getuid() == runtime["runtime_uid"]
            and os.getgid() == runtime["runtime_gid"]
        )
        version = metadata.version("sglang")
        assert version == runtime["sglang"]["package_version"], version
        sglang = importlib.import_module("sglang")
        assert sglang.__file__ is not None
        installed = Path(sglang.__file__).resolve().parent
        assert "site-packages" in installed.parts and "/sgl-workspace/" not in str(
            installed
        ), str(installed)
        for relative, expected in installed_source_inventory(runtime["sglang"]).items():
            path = installed / relative
            assert (
                path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected
            ), relative
        for name in [
            "torch",
            "torchvision",
            "torchaudio",
            "cv2",
            "cutlass",
            "sglang.srt.models.qwen4_exp",
            "sglang.srt.models.qwen4_exp_mtp",
            "sglang.srt.multimodal.processors.qwen_vl",
            "sglang.benchmark.serving",
            *runtime["rust_modules"],
        ]:
            module = importlib.import_module(name)
            modules.append({"name": name, "path": getattr(module, "__file__", None)})
        for key in (
            "HOME",
            "HF_HOME",
            "SGLANG_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
        ):
            cache = Path(os.environ[key])
            assert cache.is_relative_to("/cache")
            cache.mkdir(parents=True, exist_ok=True)
            probe = cache / ".build-write-probe"
            with probe.open("xb") as stream:
                stream.write(b"owned cache probe")
            probe.unlink()
    return {
        "status": "ENVIRONMENT_CHECK_PASS",
        "phase": phase,
        "sglang_version": version,
        "source_commit": runtime["sglang"]["commit"],
        "source_tree": runtime["sglang"]["tree"],
        "metadata_exceptions": violations,
        "sbsa_libraries": libraries,
        "modules": modules,
        "gpu_qualification": "NOT_RUN",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prebuild", "runtime"), required=True)
    parser.add_argument("--root", type=Path, default=Path("/opt/r0b0tlab"))
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.phase), indent=2))


if __name__ == "__main__":
    main()
