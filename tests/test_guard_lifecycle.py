"""The real run.sh -> guard.py -> Docker CLI subprocess path, CPU only.

A temporary sitecustomize replaces ONLY host probes. A temporary executable
replaces ONLY Docker; no endpoint, CUDA computation, or serving evidence is
fabricated. Existing guard argument parsing, transport, and lifecycle execute.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from scripts import guard
from tests.launcher_fixtures import make_env, argv, CID, IMAGE

FAKE_DOCKER = r"""
import json, os, pathlib, signal, sys
root = pathlib.Path(os.environ["TEST_DOCKER_ROOT"])
a = sys.argv[1:]
with (root / "calls.jsonl").open("a") as f: f.write(json.dumps(a) + "\n")
p = root / "container.json"
doc = json.loads(p.read_text()) if p.exists() else None
if a[:2] == ["image", "inspect"]:
    print(json.dumps([{"Id": a[2], "Architecture": "arm64", "Os": "linux"}]))
elif a[0] == "create":
    labels = dict(a[i+1].split("=",1) for i,v in enumerate(a) if v == "--label")
    doc = {"Id": "cd"*32, "Image": os.environ["TEST_IMAGE"], "Config": {"Labels": labels},
           "State": {"Running": False, "ExitCode": 0, "OOMKilled": False}, "polls": 0}
    pathlib.Path(a[a.index("--cidfile")+1]).write_text(doc["Id"])
    p.write_text(json.dumps(doc))
    print(doc["Id"])
elif a[0] == "start":
    doc["State"]["Running"] = True
    p.write_text(json.dumps(doc))
    if os.environ["TEST_SCENARIO"] == "term": os.kill(os.getppid(), signal.SIGTERM)
    print(doc["Id"])
elif a[:2] == ["container", "inspect"]:
    if doc is None: sys.exit(1)
    if doc["State"]["Running"] and os.environ["TEST_SCENARIO"] != "term":
        doc["polls"] += 1
        if doc["polls"] >= 2:
            doc["State"].update(Running=False, ExitCode=int(os.environ["TEST_SCENARIO"]))
        p.write_text(json.dumps(doc))
    print(json.dumps([doc]))
elif a[0] == "stop":
    doc["State"].update(Running=False, ExitCode=143)
    p.write_text(json.dumps(doc))
    print(doc["Id"])
elif a[0] == "logs": print("TEST-ONLY DOCKER TRANSPORT; NOT INFERENCE", file=sys.stderr)
else: sys.exit(2)
"""


@pytest.mark.parametrize("scenario,expected", [("0", 0), ("42", 42), ("term", 143)])
def test_real_shell_cli_lifecycle(tmp_path, scenario, expected):
    env = make_env(tmp_path)
    binary = tmp_path / "bin"
    binary.mkdir()
    fake = binary / "docker"
    fake.write_text("#!" + sys.executable + "\n" + FAKE_DOCKER)
    fake.chmod(0o700)
    patchdir = tmp_path / "probe-fixture"
    patchdir.mkdir()
    (patchdir / "sitecustomize.py").write_text(
        "from scripts import preflight\n"
        "preflight.run_preflight = lambda **kw: {'status':'PASS','test_only':True}\n"
        "preflight.read_meminfo = lambda: {'MemAvailable':16777216}\n"
    )
    execution = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.defpath,
        "PYTHONPATH": str(patchdir) + os.pathsep + str(guard.REPO),
        "PYTHON": sys.executable,
        "PYTHONDONTWRITEBYTECODE": "1",
        "TEST_DOCKER_ROOT": str(tmp_path),
        "TEST_IMAGE": IMAGE,
        "TEST_SCENARIO": scenario,
    }
    result = subprocess.run(
        ["sh", str(guard.REPO / "scripts/run.sh"), *argv(env)],
        cwd=tmp_path,
        env=execution,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == expected, result.stderr
    assert "Traceback" not in result.stderr
    doc = json.loads((tmp_path / "container.json").read_text())
    assert doc["State"]["Running"] is False
    final = json.loads((Path(env["state-dir"]) / "container.final.json").read_text())
    assert final["Id"] == CID
    assert (
        "TEST-ONLY DOCKER TRANSPORT"
        in (Path(env["state-dir"]) / "server.log").read_text()
    )
    calls = [json.loads(s) for s in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert sum(c[0] == "create" for c in calls) == 1
    assert sum(c[0] == "start" for c in calls) == 1
    assert sum(c[0] == "stop" for c in calls) == (1 if scenario == "term" else 0)
