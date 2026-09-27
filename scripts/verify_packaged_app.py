"""Compatibility entry point for the current isolated desktop release verifier."""
import sys

from verify_release import main


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
