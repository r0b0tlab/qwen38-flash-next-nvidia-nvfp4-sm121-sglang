"""Scoped stop helper used by run.sh traps: stops ONLY a container whose
ownership labels (re-read at stop time) match this runtime's owner."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guard import EXIT_CLEANUP, EXIT_OK, LaunchError, stop_owned


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Ownership-scoped container stop (re-reads labels)."
    )
    parser.add_argument("cid")
    parser.add_argument("--state-dir", default="/var/lib/qwen38fn")
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        stop_owned(args.cid)
    except LaunchError as exc:
        print("STOP REFUSED/FAILED: %s" % (exc,), file=sys.stderr)
        return EXIT_CLEANUP
    print(
        json.dumps({"event": "container_stopped", "cid": args.cid})
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
