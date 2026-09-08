"""Verified PLE preparation binding on the entrypoint exec path.

The wrapper reads the image-owned plan (``locks/ple.json``, resolved next to
the entrypoint package), independently binds it to the checked source lock,
lazily imports the installed core
``sglang.srt.models.qwen4_exp_ple_cache`` and validates its result receipt
before any server exec. Any failure refuses the entrypoint (nonzero, no
exec, no stale handoff environment).

The production core is installed only inside the packaged image; it does not
exist in this CPU test workspace. Unit tests therefore bind through the
import seam (``sys.modules``) using an explicit contract double. The double
proves the wrapper's own validation contract ONLY: it is not evidence about
the real model, the real cache directory, or the real image, and no test
here claims otherwise. The controller installs the real derived plan and
runs the real-core integration separately.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from runtime import ProfileError, profile_from_dict
from runtime import entrypoint as ep

SHA = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
PLE_SOURCE_NAME = "model-fp8-mtp-ple.safetensors"
PLE_SOURCE_SHA = "35" * 32
PLE_SOURCE_SIZE = 53717551730
TABLE_SHA = "b0" * 32
TABLE_SIZE = 51200245760
TENSOR_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
ROOT = Path(ep.__file__).resolve().parents[1]
CORE_MODULE = "sglang.srt.models.qwen4_exp_ple_cache"

SOURCES = {
    "model": {
        "id": ep.MODEL_ID,
        "sha": SHA,
        "files": [
            {
                "path": PLE_SOURCE_NAME,
                "size": PLE_SOURCE_SIZE,
                "sha256": PLE_SOURCE_SHA,
                "git_blob": "ab" * 20,
            }
        ],
    }
}

BASE_PROFILE = {
    "schema": 1,
    "mode": "ar",
    "context_length": 32768,
    "max_total_tokens": 32768,
    "vision": {"backend": "triton_attn", "cuda_graph": True},
}


def _hex64(seed):
    alphabet = "0123456789abcdef"
    return "".join(alphabet[(seed + i) % 16] for i in range(64))


def make_plan(**over):
    plan = {
        "schema": "qwen38fn.ple-plan.v1",
        "model": {"id": ep.MODEL_ID, "revision": SHA},
        "source": {
            "name": PLE_SOURCE_NAME,
            "size": PLE_SOURCE_SIZE,
            "sha256": PLE_SOURCE_SHA,
            "header_sha256": _hex64(1),
            "data_start": 416368,
        },
        "tensor_prefix": TENSOR_PREFIX,
        "dtype": "F8_E4M3",
        "shard_count": 128,
        "rows_per_shard": 2500012,
        "row_width": 160,
        "table": {
            "name": "synthetic_ple_table.bin",
            "shape": [320001536, 160],
            "size": TABLE_SIZE,
            "sha256": TABLE_SHA,
        },
        "scale": {
            "name": TENSOR_PREFIX + ".weight_scale",
            "offset": 416368,
            "size": 2,
            "sha256": _hex64(2),
        },
        "shards": [
            {
                "id": n,
                "name": "%s.shard_%d.weight" % (TENSOR_PREFIX, n),
                "offset": 0,
                "size": 1,
                "sha256": _hex64(3 + n),
            }
            for n in range(128)
        ],
    }
    for key, value in over.items():
        plan[key] = value
    return plan


def make_result(directory=None, mode="materialized"):
    directory = directory if directory is not None else ep.ple_dir(SHA)
    return {
        "receipt_path": directory + "/prepared.json",
        "receipt_sha256": TABLE_SHA,
        "mode": mode,
        "copied_bytes": 0 if mode == "reuse" else TABLE_SIZE,
        "elapsed_seconds": 0.5,
    }


def _write_plan(tmp_path, plan, name="ple.json"):
    path = tmp_path / name
    path.write_text(json.dumps(plan))
    return str(path)


@pytest.fixture
def fake_core(monkeypatch):
    """Contract double for the installed PLE core (NOT the real core)."""
    calls = []
    module = sys.modules.get(CORE_MODULE)
    module = type(sys)("fake_qwen4_exp_ple_cache")

    def prepare_cache(model_path, cache_dir, plan):
        calls.append({"model_path": model_path, "cache_dir": cache_dir})
        return make_result(cache_dir)

    module.prepare_cache = prepare_cache
    module.validate_plan = lambda plan: plan
    module.calls = calls
    monkeypatch.setitem(sys.modules, CORE_MODULE, module)
    return module


# ------------------------------------------------------------ env ownership


def test_build_env_clears_ambient_handoff_variables(monkeypatch):
    monkeypatch.setenv("R0B0TLAB_PLE_PREPARED_PATH", "/hostile/stale.json")
    monkeypatch.setenv("R0B0TLAB_PLE_PREPARED_SHA256", "deadbeef")
    env = ep.build_env(profile_from_dict(BASE_PROFILE), SOURCES)
    assert env["R0B0TLAB_PLE_PREPARED_PATH"] == ""
    assert env["R0B0TLAB_PLE_PREPARED_SHA256"] == ""


def test_ple_handoff_variables_are_owned_by_the_entrypoint():
    assert "R0B0TLAB_PLE_PREPARED_PATH" in ep.ENV_VARS_OWNED
    assert "R0B0TLAB_PLE_PREPARED_SHA256" in ep.ENV_VARS_OWNED


def test_pure_build_and_print_need_no_ple_plan_or_core(tmp_path, monkeypatch):
    plan_absent = tmp_path / "missing-plan.json"
    monkeypatch.setattr(ep, "PLE_PLAN_PATH", str(plan_absent))
    assert not plan_absent.exists()
    before = set(sys.modules)
    launch = ep.build_all(profile_from_dict(BASE_PROFILE), SOURCES)
    assert launch["env"]["R0B0TLAB_PLE_PREPARED_PATH"] == ""
    assert not any(name.startswith("sglang") for name in set(sys.modules) - before)


# ------------------------------------------------------------ plan reading


def test_plan_path_is_image_owned_locks_next_to_package():
    assert ep.PLE_PLAN_PATH == str(ROOT / "locks" / "ple.json")


def test_no_ple_plan_cli_override(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(BASE_PROFILE))
    with pytest.raises(SystemExit):
        ep.main(
            [
                "--profile",
                str(profile),
                "--sources",
                str(tmp_path / "missing.json"),
                "--ple-plan",
                str(tmp_path / "plan.json"),
            ]
        )


def test_read_ple_plan_happy_path(tmp_path):
    path = _write_plan(tmp_path, make_plan())
    assert ep.read_ple_plan(path)["tensor_prefix"] == TENSOR_PREFIX


def test_missing_plan_fails(tmp_path):
    with pytest.raises(ep.PLEPreparationError):
        ep.read_ple_plan(str(tmp_path / "absent.json"))


def test_overlong_plan_fails(tmp_path):
    path = tmp_path / "big.json"
    path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ep.PLEPreparationError):
        ep.read_ple_plan(str(path))


def test_duplicate_plan_key_fails(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text('{"schema":"qwen38fn.ple-plan.v1","schema":"qwen38fn.ple-plan.v1"}')
    with pytest.raises(ep.PLEPreparationError):
        ep.read_ple_plan(str(path))


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_nonfinite_plan_number_fails(tmp_path, token):
    path = tmp_path / "nonfinite.json"
    path.write_text('{"schema":"qwen38fn.ple-plan.v1","data_start":%s}' % token)
    with pytest.raises(ep.PLEPreparationError):
        ep.read_ple_plan(str(path))


def test_plan_not_an_object_fails(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[]")
    with pytest.raises(ep.PLEPreparationError):
        ep.read_ple_plan(str(path))


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema": "qwen38fn.ple-plan.v2"},
        {"dtype": "BF16"},
        {"model": {"id": "other/model", "revision": SHA}},
        {"model": {"id": ep.MODEL_ID, "revision": "fc694b54"}},
        {
            "tensor_prefix": "model.language_model.layers.2.ple.ple_embedding.ngram_embedding"
        },
        {"shard_count": 64},
        {"rows_per_shard": 2500011},
        {"row_width": 128},
        {
            "source": {
                "name": "model-fp8-mtp.safetensors",
                "size": PLE_SOURCE_SIZE,
                "sha256": PLE_SOURCE_SHA,
                "header_sha256": _hex64(1),
                "data_start": 416368,
            }
        },
        {
            "table": {
                "name": "t.bin",
                "shape": [320001536, 128],
                "size": TABLE_SIZE,
                "sha256": TABLE_SHA,
            }
        },
        {
            "table": {
                "name": "t.bin",
                "shape": [320001536, 160],
                "size": TABLE_SIZE - 1,
                "sha256": TABLE_SHA,
            }
        },
    ],
)
def test_plan_contract_mutations_fail(tmp_path, mutation):
    with pytest.raises(ep.PLEPreparationError):
        ep.bind_ple_plan(
            ep.read_ple_plan(_write_plan(tmp_path, make_plan(**mutation))), SOURCES
        )


def test_plan_duplicate_inventory_fails(tmp_path):
    plan = make_plan()
    plan["shards"][1] = dict(plan["shards"][0])
    with pytest.raises(ep.PLEPreparationError):
        ep.bind_ple_plan(ep.read_ple_plan(_write_plan(tmp_path, plan)), SOURCES)


def test_plan_truncated_shard_list_fails(tmp_path):
    plan = make_plan()
    plan["shards"] = plan["shards"][:127]
    with pytest.raises(ep.PLEPreparationError):
        ep.bind_ple_plan(ep.read_ple_plan(_write_plan(tmp_path, plan)), SOURCES)


def test_plan_bool_where_int_required_fails(tmp_path):
    plan = make_plan()
    plan["rows_per_shard"] = True
    with pytest.raises(ep.PLEPreparationError):
        ep.bind_ple_plan(ep.read_ple_plan(_write_plan(tmp_path, plan)), SOURCES)


def test_plan_float_where_int_required_fails(tmp_path):
    plan = make_plan()
    plan["source"]["data_start"] = 416368.0
    with pytest.raises(ep.PLEPreparationError):
        ep.bind_ple_plan(ep.read_ple_plan(_write_plan(tmp_path, plan)), SOURCES)


# ------------------------------------------------------- source binding


def test_binding_matches_plan_to_checked_source(fake_core, tmp_path):
    path = _write_plan(tmp_path, make_plan())
    monkey_patch = ep.PLE_PLAN_PATH
    ep.PLE_PLAN_PATH = path
    try:
        result = ep.prepare_ple_cache(SOURCES)
    finally:
        ep.PLE_PLAN_PATH = monkey_patch
    assert result["mode"] == "materialized"
    assert fake_core.calls == [
        {"model_path": ep.MODEL_PATH, "cache_dir": ep.ple_dir(SHA)}
    ]


@pytest.mark.parametrize(
    "sources, plan_over",
    [
        # wrong source file hash in the plan
        (
            SOURCES,
            {
                "source": {
                    "name": PLE_SOURCE_NAME,
                    "size": PLE_SOURCE_SIZE,
                    "sha256": _hex64(9),
                    "header_sha256": _hex64(1),
                    "data_start": 416368,
                }
            },
        ),
        # wrong source size in the plan
        (
            SOURCES,
            {
                "source": {
                    "name": PLE_SOURCE_NAME,
                    "size": PLE_SOURCE_SIZE + 1,
                    "sha256": PLE_SOURCE_SHA,
                    "header_sha256": _hex64(1),
                    "data_start": 416368,
                }
            },
        ),
        # plan revision differs from the frozen model revision
        (
            SOURCES,
            {
                "model": {
                    "id": ep.MODEL_ID,
                    "revision": "e" * 40,
                }
            },
        ),
        # sources lock without the PLE row entirely
        (
            {"model": {"id": ep.MODEL_ID, "sha": SHA, "files": []}},
            {},
        ),
        # duplicated inventory row for the same path
        (
            {
                "model": {
                    "id": ep.MODEL_ID,
                    "sha": SHA,
                    "files": [
                        SOURCES["model"]["files"][0],
                        SOURCES["model"]["files"][0],
                    ],
                }
            },
            {},
        ),
        # row path matches but hash differs from the plan
        (
            {
                "model": {
                    "id": ep.MODEL_ID,
                    "sha": SHA,
                    "files": [
                        {
                            "path": PLE_SOURCE_NAME,
                            "size": PLE_SOURCE_SIZE,
                            "sha256": _hex64(8),
                        }
                    ],
                }
            },
            {},
        ),
        # row path+hash match but size differs
        (
            {
                "model": {
                    "id": ep.MODEL_ID,
                    "sha": SHA,
                    "files": [
                        {
                            "path": PLE_SOURCE_NAME,
                            "size": PLE_SOURCE_SIZE - 1,
                            "sha256": PLE_SOURCE_SHA,
                        }
                    ],
                }
            },
            {},
        ),
    ],
)
def test_bad_model_source_binding_fails_before_core(
    fake_core, tmp_path, sources, plan_over
):
    path = _write_plan(tmp_path, make_plan(**plan_over))
    saved = ep.PLE_PLAN_PATH
    ep.PLE_PLAN_PATH = path
    try:
        with pytest.raises((ep.PLEPreparationError, ProfileError)):
            ep.prepare_ple_cache(sources)
    finally:
        ep.PLE_PLAN_PATH = saved
    assert fake_core.calls == [], "core must not run on a bad binding"


def test_sources_env_override_still_forbidden_on_ple_path(fake_core, tmp_path):
    path = _write_plan(tmp_path, make_plan())
    saved = ep.PLE_PLAN_PATH
    ep.PLE_PLAN_PATH = path
    try:
        with pytest.raises(ProfileError):
            ep.prepare_ple_cache({**SOURCES, "env": {"KEY": "1"}})
    finally:
        ep.PLE_PLAN_PATH = saved


# ----------------------------------------------------- core import seam


def test_missing_core_helper_fails(fake_core, tmp_path):
    del fake_core.prepare_cache
    path = _write_plan(tmp_path, make_plan())
    saved = ep.PLE_PLAN_PATH
    ep.PLE_PLAN_PATH = path
    try:
        with pytest.raises(ep.PLEPreparationError):
            ep.prepare_ple_cache(SOURCES)
    finally:
        ep.PLE_PLAN_PATH = saved


def test_unavailable_core_module_fails(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, CORE_MODULE, None)  # import -> ImportError
    path = _write_plan(tmp_path, make_plan())
    saved = ep.PLE_PLAN_PATH
    ep.PLE_PLAN_PATH = path
    try:
        with pytest.raises(ep.PLEPreparationError):
            ep.prepare_ple_cache(SOURCES)
    finally:
        ep.PLE_PLAN_PATH = saved


# ---------------------------------------------------- result validation


@pytest.mark.parametrize(
    "mutation",
    [
        {"receipt_path": "prepared.json"},
        {"receipt_path": ep.ple_dir(SHA) + "/PREPARED.json"},
        {"receipt_path": "/cache/ple/" + SHA + "/prepared.json.backup"},
        {"receipt_sha256": TABLE_SHA[:-1]},
        {"receipt_sha256": TABLE_SHA.upper()},
        {"receipt_sha256": "z" * 64},
        {"receipt_sha256": None},
        {"mode": "cached"},
        {"mode": "MATERIALIZED"},
        {"mode": None},
        {"copied_bytes": True},
        {"copied_bytes": "0"},
        {"copied_bytes": -1},
        {"copied_bytes": 1.0},
        {"elapsed_seconds": True},
        {"elapsed_seconds": -0.5},
        {"elapsed_seconds": "0.5"},
        {"elapsed_seconds": float("inf")},
        {"elapsed_seconds": float("nan")},
    ],
)
def test_invalid_preparer_result_fails(mutation):
    result = make_result()
    result.update(mutation)
    with pytest.raises(ep.PLEPreparationError):
        ep.validate_prepared_result(result, ep.ple_dir(SHA), TABLE_SIZE)


def test_reuse_result_must_copy_zero_bytes():
    result = make_result(mode="reuse")
    result["copied_bytes"] = TABLE_SIZE
    with pytest.raises(ep.PLEPreparationError):
        ep.validate_prepared_result(result, ep.ple_dir(SHA), TABLE_SIZE)


def test_cold_result_must_copy_exact_plan_size():
    result = make_result(mode="materialized")
    result["copied_bytes"] = 1
    with pytest.raises(ep.PLEPreparationError):
        ep.validate_prepared_result(result, ep.ple_dir(SHA), TABLE_SIZE)


def test_non_object_result_fails():
    with pytest.raises(ep.PLEPreparationError):
        ep.validate_prepared_result("ok", ep.ple_dir(SHA), TABLE_SIZE)


def test_missing_result_keys_fail():
    for key in (
        "receipt_path",
        "receipt_sha256",
        "mode",
        "copied_bytes",
        "elapsed_seconds",
    ):
        result = make_result()
        result.pop(key)
        with pytest.raises(ep.PLEPreparationError):
            ep.validate_prepared_result(result, ep.ple_dir(SHA), TABLE_SIZE)


# ------------------------------------------------------- exec ordering


def _write_launch_files(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(BASE_PROFILE))
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps(SOURCES))
    return ["--profile", str(profile), "--sources", str(sources)]


def test_main_orders_jit_then_ple_then_exec(fake_core, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("R0B0TLAB_PLE_PREPARED_PATH", "/hostile/stale.json")
    monkeypatch.setenv("R0B0TLAB_PLE_PREPARED_SHA256", "deadbeef")
    plan_path = _write_plan(tmp_path, make_plan())
    monkeypatch.setattr(ep, "PLE_PLAN_PATH", plan_path)
    order = []
    monkeypatch.setattr(ep, "prepare_jit_temp", lambda p: order.append(("jit", p)))
    real_prepare = ep.prepare_ple_cache

    def wrapped(sources):
        order.append(("ple", None))
        return real_prepare(sources)

    monkeypatch.setattr(ep, "prepare_ple_cache", wrapped)
    captured = {}

    def fake_exec(program, argv, environment):
        captured["program"] = program
        captured["argv"] = argv
        captured["env"] = environment

    monkeypatch.setattr(ep.os, "execvpe", fake_exec)
    ep.main(_write_launch_files(tmp_path))
    assert [name for name, _ in order] == ["jit", "ple"]
    assert captured["program"] == "sglang"
    environment = captured["env"]
    assert (
        environment["R0B0TLAB_PLE_PREPARED_PATH"] == ep.ple_dir(SHA) + "/prepared.json"
    )
    assert environment["R0B0TLAB_PLE_PREPARED_SHA256"] == TABLE_SHA
    metrics = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert metrics["ple"]["mode"] == "materialized"
    assert metrics["ple"]["receipt_path"] == ep.ple_dir(SHA) + "/prepared.json"
    assert metrics["ple"]["receipt_sha256"] == TABLE_SHA


def test_main_refuses_exec_on_invalid_preparation(
    fake_core, tmp_path, monkeypatch, capsys
):
    plan = make_plan()
    plan["table"]["size"] = TABLE_SIZE - 1  # plan contract mutation
    monkeypatch.setattr(ep, "PLE_PLAN_PATH", _write_plan(tmp_path, plan))
    executed = []
    monkeypatch.setattr(ep, "prepare_jit_temp", lambda p: None)
    monkeypatch.setattr(ep.os, "execvpe", lambda *a: executed.append(a))
    rc = ep.main(_write_launch_files(tmp_path))
    assert rc != 0
    assert executed == [], "must never exec on a refused preparation"
    assert "ENTRYPOINT REFUSED" in capsys.readouterr().err
    assert fake_core.calls == []


# ------------------------------------------------ real -m subprocess cases


def _write_fake_core_package(tmp_path, plan, behavior="ok"):
    """Fake installed core importable via PYTHONPATH; maps /cache/ple into tmp."""
    package = tmp_path / "pypkg" / "sglang" / "srt" / "models"
    package.mkdir(parents=True, exist_ok=True)
    for directory in (package, package.parent, package.parent.parent):
        (directory / "__init__.py").write_text("")
    module = package / "qwen4_exp_ple_cache.py"
    module.write_text(
        "import json, os, sys\n"
        "validated = False\n"
        "def validate_plan(plan):\n"
        "    global validated\n"
        "    validated = True\n"
        "    return plan\n"
        "def prepare_cache(model_path, cache_dir, plan):\n"
        "    assert validated, 'core validation must precede preparation'\n"
        "    mapped = os.environ['TEST_PLE_DIR_MAP']\n"
        "    receipt_name = 'prepared.json'\n"
        "    record = {'model_path': model_path, 'cache_dir': cache_dir,\n"
        "              'plan_revision': plan['model']['revision']}\n"
        "    with open(os.path.join(mapped, 'calls.json'), 'w') as handle:\n"
        "        json.dump(record, handle)\n"
        "    with open(os.path.join(mapped, receipt_name), 'w') as handle:\n"
        "        json.dump({'written': True}, handle)\n"
        "    if os.environ.get('TEST_PLE_CORE_BEHAVIOR') != 'ok':\n"
        "        return os.environ['TEST_PLE_CORE_BEHAVIOR']\n"
        "    return {'receipt_path': cache_dir + '/prepared.json',\n"
        "            'receipt_sha256': plan['table']['sha256'],\n"
        "            'mode': 'materialized',\n"
        "            'copied_bytes': plan['table']['size'],\n"
        "            'elapsed_seconds': 0.5}\n"
    )
    return module


def _write_mapped_startup(tmp_path, real_plan_path):
    """Extend the FS-only fixture: map the exact plan path onto a tmp copy."""
    startup = tmp_path / "sitecustomize.py"
    startup.write_text(
        "import os, builtins as _b\n"
        "_makedirs=os.makedirs\n"
        "_open=os.open\n"
        "_fopen=_b.open\n"
        "def mapped(p):\n"
        " return os.environ['TEST_JIT_TMP'] if str(p)=='/cache/tmp' else p\n"
        "os.makedirs=lambda p,*a,**k:_makedirs(mapped(p),*a,**k)\n"
        "os.open=lambda p,*a,**k:_open(mapped(p),*a,**k)\n"
        "def fopen(p,*a,**k):\n"
        " if str(p)==os.environ.get('TEST_PLE_PLAN_REAL'):\n"
        "  p=os.environ['TEST_PLE_PLAN_COPY']\n"
        " return _fopen(p,*a,**k)\n"
        "_b.open=fopen\n"
    )
    return startup


def _run_real_exec(tmp_path, plan, sources=None, core_behavior="ok"):
    mapped = tmp_path / "jit-temp"
    ple_mapped = tmp_path / "ple-cache"
    ple_mapped.mkdir()
    capture = tmp_path / "captured.json"
    executable = tmp_path / "sglang"
    executable.write_text(
        "#!"
        + sys.executable
        + '\nimport json, os, sys\nfrom pathlib import Path\nPath(os.environ["CAPTURE_PATH"]).write_text(json.dumps({"argv":sys.argv[1:],"env":{k:os.environ.get(k) for k in ["TMPDIR","R0B0TLAB_PLE_PREPARED_PATH","R0B0TLAB_PLE_PREPARED_SHA256","MAX_JOBS","FLASHINFER_NVCC_THREADS","SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK","SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH"]}}))\n'
    )
    executable.chmod(0o700)
    _write_fake_core_package(tmp_path, plan)
    _write_mapped_startup(tmp_path, ep.PLE_PLAN_PATH)
    plan_copy = tmp_path / "plan-copy.json"
    plan_copy.write_text(json.dumps(plan))
    sources = sources if sources is not None else SOURCES
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(BASE_PROFILE))
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(json.dumps(sources))
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path / "pypkg")
        + os.pathsep
        + str(tmp_path)
        + os.pathsep
        + str(Path(__file__).resolve().parents[1]),
        "TEST_JIT_TMP": str(mapped),
        "TEST_PLE_DIR_MAP": str(ple_mapped),
        "TEST_PLE_PLAN_REAL": ep.PLE_PLAN_PATH,
        "TEST_PLE_PLAN_COPY": str(plan_copy),
        "TEST_PLE_CORE_BEHAVIOR": core_behavior,
        "SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK": "1",
        "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH": "1",
        "TMPDIR": "/tmp/untrusted",
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "CAPTURE_PATH": str(capture),
        "R0B0TLAB_PLE_PREPARED_PATH": "/hostile/stale.json",
        "R0B0TLAB_PLE_PREPARED_SHA256": "deadbeef",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "runtime.entrypoint",
            "--profile",
            str(profile),
            "--sources",
            str(sources_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, capture, ple_mapped


def test_real_exec_hands_fresh_prepared_env_to_server(tmp_path):
    plan = make_plan()
    result, capture, ple_mapped = _run_real_exec(tmp_path, plan)
    assert result.returncode == 0, result.stderr
    assert capture.exists(), "module returned without execing the server"
    assert (ple_mapped / "calls.json").exists(), "fake core was never invoked"
    calls = json.loads((ple_mapped / "calls.json").read_text())
    assert calls["model_path"] == ep.MODEL_PATH
    assert calls["cache_dir"] == ep.ple_dir(SHA)
    assert calls["plan_revision"] == SHA
    data = json.loads(capture.read_text())
    # the hostile ambient handoff must lose to the freshly bound preparation
    assert data["env"]["R0B0TLAB_PLE_PREPARED_PATH"] == (
        ep.ple_dir(SHA) + "/prepared.json"
    )
    assert data["env"]["R0B0TLAB_PLE_PREPARED_SHA256"] == TABLE_SHA
    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["ple"]["mode"] == "materialized"
    assert metrics["ple"]["receipt_sha256"] == TABLE_SHA


def test_real_exec_refuses_on_bad_binding_without_exec_or_core(tmp_path):
    plan = make_plan()
    plan["source"]["sha256"] = _hex64(9)  # wrong source hash in the plan
    result, capture, ple_mapped = _run_real_exec(tmp_path, plan)
    assert result.returncode != 0
    assert "ENTRYPOINT REFUSED" in result.stderr
    assert not capture.exists(), "must not exec on a refused preparation"
    assert not (ple_mapped / "calls.json").exists(), "core must not run"


def test_real_exec_refuses_on_invalid_core_result(tmp_path):
    plan = make_plan()
    result, capture, ple_mapped = _run_real_exec(
        tmp_path, plan, core_behavior="not-a-dict"
    )
    assert result.returncode != 0
    assert "ENTRYPOINT REFUSED" in result.stderr
    assert not capture.exists()


def test_print_path_imports_no_core_and_needs_no_plan(tmp_path):
    """--print is pure: no PLE plan, no sglang import, no filesystem writes."""
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(BASE_PROFILE))
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(json.dumps(SOURCES))
    script = (
        "import sys, json\n"
        "sys.path.insert(0, r'%s')\n"
        "from runtime import entrypoint\n"
        "entrypoint.PLE_PLAN_PATH = 'missing-test-plan-that-is-never-read.json'\n"
        "rc = entrypoint.main(['--profile', r'%s', '--sources', r'%s', '--print'])\n"
        "assert rc == 0, rc\n"
        "print('PLE_PRINT_MARKER', 'sglang' in sys.modules)\n"
        % (ROOT, profile, sources_path)
    )
    env = {
        **os.environ,
        "R0B0TLAB_PLE_PREPARED_PATH": "/hostile/stale.json",
        "R0B0TLAB_PLE_PREPARED_SHA256": "deadbeef",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "PLE_PRINT_MARKER False" in result.stdout
    launch = json.loads(result.stdout.strip().splitlines()[0])
    assert launch["env"]["R0B0TLAB_PLE_PREPARED_PATH"] == ""
    assert launch["env"]["R0B0TLAB_PLE_PREPARED_SHA256"] == ""
    assert not (tmp_path / "missing-test-plan-that-is-never-read.json").exists()
