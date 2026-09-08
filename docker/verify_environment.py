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


def verify_jit_temporary_directory(root):
    """Compile and dlopen a bounded native library in the runtime temp location."""
    import ctypes
    import tempfile

    with tempfile.TemporaryDirectory(prefix="jit-exec-probe-", dir=root) as directory:
        output = Path(directory) / "probe.so"
        subprocess.run(
            ["cc", "-shared", "-fPIC", "-x", "c", "-o", str(output), "-"],
            input="int r0b0tlab_jit_probe(void) { return 121; }",
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        library = ctypes.CDLL(str(output))
        assert library.r0b0tlab_jit_probe() == 121, "native JIT DSO execution failed"
    return "PASS"


def verify_ple_plan(root, runtime, core):
    """Attest image-owned plan bytes and validate them with the installed core."""
    with (root / "locks/ple.json").open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) > 1024 * 1024 or digest != runtime.get("ple_plan_sha256"):
        raise ValueError("image PLE plan differs from its runtime lock")
    plan = core.validate_plan(core._load_strict_json(raw, "image PLE plan"))
    with (root / "locks/sources.json").open("rb") as stream:
        source_raw = stream.read(4 * 1024 * 1024 + 1)
    if len(source_raw) > 4 * 1024 * 1024:
        raise ValueError("image source lock exceeds its size bound")
    sources = core._load_strict_json(source_raw, "image sources")
    model = sources.get("model") if isinstance(sources, dict) else None
    if (
        not isinstance(model, dict)
        or model.get("id") != plan["model"]["id"]
        or model.get("sha") != plan["model"]["revision"]
    ):
        raise ValueError("image PLE model binding mismatch")
    files = model.get("files")
    if not isinstance(files, list) or any(not isinstance(row, dict) for row in files):
        raise ValueError("image model inventory must be a list of objects")
    rows = [row for row in files if row.get("path") == plan["source"]["name"]]
    if (
        len(rows) != 1
        or type(rows[0].get("size")) is not int
        or rows[0]["size"] != plan["source"]["size"]
        or rows[0].get("sha256") != plan["source"]["sha256"]
    ):
        raise ValueError("image PLE source-file binding mismatch")
    return {
        "file_sha256": digest,
        "canonical_plan_sha256": core.plan_sha256(plan),
        "table_sha256": plan["table"]["sha256"],
    }


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
    ple_plan = None
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
            "TMPDIR",
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
        assert os.environ["TMPDIR"] == "/cache/tmp"
        verify_jit_temporary_directory(Path(os.environ["TMPDIR"]))
        core = importlib.import_module("sglang.srt.models.qwen4_exp_ple_cache")
        ple_plan = verify_ple_plan(root, runtime, core)
    return {
        "status": "ENVIRONMENT_CHECK_PASS",
        "phase": phase,
        "sglang_version": version,
        "source_commit": runtime["sglang"]["commit"],
        "source_tree": runtime["sglang"]["tree"],
        "metadata_exceptions": violations,
        "sbsa_libraries": libraries,
        "modules": modules,
        "ple_plan": ple_plan,
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
