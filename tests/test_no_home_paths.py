"""Guard: no absolute home paths may leak into tracked source files."""

from __future__ import annotations

import subprocess

LEAK = "/home/" + "r0b0tdgx"


def test_no_absolute_home_paths_in_tracked_source():
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    bad = []
    for f in out:
        if f.endswith((".py", ".sh", ".json", ".md", ".txt", ".yml", ".yaml")):
            try:
                content = open(f, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if LEAK in content:
                bad.append(f)
    assert bad == [], f"home paths leaked into tracked files: {bad}"
