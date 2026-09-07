#!/bin/sh
# Owned single-GB10 launcher entry point (wrapper around scripts/guard.py).
#
# Installs signal traps BEFORE any spawn attempt so cleanup runs on
# TERM/INT through the spawn and inspect failure windows. Delegates all
# decisions (receipt gate, preflight, argv construction, watchdog) to
# scripts/guard.py. Never touches containers without the
# io.r0b0tlab.qwen38fn.owner label; never masks the child exit code with
# a cleanup failure (cleanup failures are reported on stderr and in the
# journal, and the ORIGINAL exit code is preserved).
#
# Status: NOT QUALIFIED — no performance claims.

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON="${PYTHON:-python3}"

LAUNCH_STATE_DIR="${QWEN38FN_STATE_DIR:-/var/lib/qwen38fn}"
CLEANUP_FAILED=0

note() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

cleanup() {
    # best-effort stop of only OUR container; scoped re-read inside guard.py
    if [ -n "${QWEN38FN_CID:-}" ]; then
        if ! "$PYTHON" "$SCRIPT_DIR/guard_stop.py" "$QWEN38FN_CID" \
            --state-dir "$LAUNCH_STATE_DIR" >/dev/null 2>&1; then
            CLEANUP_FAILED=1
            note "cleanup: scoped stop failed for ${QWEN38FN_CID}"
        fi
    fi
    if [ "$CLEANUP_FAILED" -ne 0 ]; then
        note "cleanup: incomplete (original exit code preserved)"
    fi
}

on_term() {
    note "caught TERM/INT; stopping owned container (if any)"
    cleanup
    exit 143
}

trap on_term TERM INT
trap cleanup EXIT

# preflight gate is inside guard.py; run() refuses on any FAIL/unknown
QWEN38FN_CID= exec "$PYTHON" "$SCRIPT_DIR/guard.py" "$@"
