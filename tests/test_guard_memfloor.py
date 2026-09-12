"""Guard watchdog floor: operator-visible relaxation via --mem-available-floor-gib.

The September 11 NIAH qualification (epoch niah-qual-20260911T1905Z) showed the
full-window 258,044-token prefill legitimately dips host MemAvailable to
~7.1 GiB for under 20 seconds. The hardcoded 8 GiB / 5-sample watchdog floor
killed an otherwise healthy serve (guard rc=8) mid-campaign. The floor stays a
release contract (default unchanged); an explicit operator override must be
bound into the launch record so it is claim-visible, never silent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import guard  # noqa: E402


class TestMemFloorArg:
    def test_default_floor_is_8_gib(self):
        argv = guard._parse_args(
            [
                "--image", "sha256:" + "a" * 64,
                "--profile", "/tmp/p.json",
                "--model-root", "/tmp/m",
                "--cache-dir", "/tmp/c",
                "--sources", "/tmp/s.json",
                "--receipt", "/tmp/r.json",
                "--receipt-sha256", "b" * 64,
                "--state-dir", "/tmp/st",
                "--print",
            ]
        )
        assert argv.mem_available_floor_gib == 8.0

    def test_floor_override_parsed(self):
        argv = guard._parse_args(
            [
                "--image", "sha256:" + "a" * 64,
                "--profile", "/tmp/p.json",
                "--model-root", "/tmp/m",
                "--cache-dir", "/tmp/c",
                "--sources", "/tmp/s.json",
                "--receipt", "/tmp/r.json",
                "--receipt-sha256", "b" * 64,
                "--state-dir", "/tmp/st",
                "--print",
                "--mem-available-floor-gib", "6.5",
            ]
        )
        assert argv.mem_available_floor_gib == 6.5

    def test_floor_override_rejected_below_hard_limit(self):
        import pytest

        with pytest.raises(SystemExit):
            guard._parse_args(
                [
                    "--image", "sha256:" + "a" * 64,
                    "--profile", "/tmp/p.json",
                    "--model-root", "/tmp/m",
                    "--cache-dir", "/tmp/c",
                    "--sources", "/tmp/s.json",
                    "--receipt", "/tmp/r.json",
                    "--receipt-sha256", "b" * 64,
                    "--state-dir", "/tmp/st",
                    "--print",
                    "--mem-available-floor-gib", "3.9",
                ]
            )

    def test_floor_kb_helper(self):
        assert guard._mem_floor_kb(8.0) == 8 * (1 << 30) // 1024
        assert guard._mem_floor_kb(6.5) == int(6.5 * (1 << 30)) // 1024
