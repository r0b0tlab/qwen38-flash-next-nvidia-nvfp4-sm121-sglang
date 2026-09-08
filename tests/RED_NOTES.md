"""Commit 1: adversarial public-CLI lifecycle tests (expected RED).

The old run() neither spawns nor watches (it hard-codes
['sglang','serve','--help'] and returns EXIT_SPAWN). These tests pin the
REAL create/start/watch/stop contract through guard.run(). They will
fail until guard.py implements the lifecycle; tests/testkit.py provides
the fake Docker transport (the only mocked irreversible boundary).
"""
