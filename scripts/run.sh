#!/bin/sh
# Python owns signals and verified container cleanup; shell traps do not survive exec.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python3}" "$SCRIPT_DIR/guard.py" "$@"
