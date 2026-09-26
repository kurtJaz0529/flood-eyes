"""Build the green/portable ZIP for the v0.5.0 lite build.

Equivalent to build_app.ps1 step [4/4] (Compress-Archive -CompressionLevel Optimal),
implemented with stdlib zipfile so it can run without PowerShell.
"""
from __future__ import annotations

import os
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "dist", "慧眼识灾")
OUT = os.path.join(ROOT, "dist", "慧眼识灾_v0.5.0_lite.zip")


def main() -> int:
    if not os.path.isdir(SRC):
        print(f"missing source dir: {SRC}")
        return 2

    if os.path.exists(OUT):
        os.remove(OUT)

    total = 0
    files = 0
    t0 = time.time()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for base, _dirs, names in os.walk(SRC):
            for name in sorted(names):
                full = os.path.join(base, name)
                rel = os.path.relpath(full, os.path.dirname(SRC))
                zf.write(full, rel)
                files += 1
                total += os.path.getsize(full)
                if files % 1000 == 0:
                    print(f"  ... {files} files, {total / 1024 / 1024:.0f} MB raw", flush=True)

    size_mb = os.path.getsize(OUT) / 1024 / 1024
    print(f"files      : {files}")
    print(f"raw        : {total / 1024 / 1024:.1f} MB")
    print(f"zip        : {size_mb:.1f} MB")
    print(f"elapsed    : {time.time() - t0:.0f}s")
    print(f"output     : {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
