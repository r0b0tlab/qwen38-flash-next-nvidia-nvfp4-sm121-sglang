"""Stop exactly the container bound to an operator-selected launch record."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import guard


def main(argv=None, *, transport=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True)
    args = parser.parse_args(argv)
    try:
        record, _ = guard.read_json(guard.safe_path(args.record))
        guard.require_hex(record.get("cid"), 64, "container ID")
        guard.require_hex(record.get("nonce"), 32, "owner nonce")
        guard.require_hex(record.get("profile_sha256"), 64, "profile hash")
        if not isinstance(record.get("image"), str) or not record["image"].startswith(
            "sha256:"
        ):
            raise guard.LaunchError("image identity missing")
        guard.require_hex(record["image"][7:], 64, "image ID")
        guard.stop_owned(record["cid"], expect=record, transport=transport)
    except (OSError, ValueError, guard.TransportError) as error:
        print("STOP REFUSED/FAILED: " + str(error), file=sys.stderr)
        return guard.EXIT_CLEANUP
    print("STOPPED " + record["cid"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
