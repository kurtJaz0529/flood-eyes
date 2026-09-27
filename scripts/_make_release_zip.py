"""Build the release ZIP for the v0.5.0 lite/full build.

Replaces build_app.ps1 step [4/4] (Compress-Archive -CompressionLevel Optimal)
with a stdlib-only, scriptable entry point:

    python scripts/_make_release_zip.py --profile lite
    python scripts/_make_release_zip.py --profile full --source dist_full/慧眼识灾 --output out.zip

The archive keeps the top-level ``慧眼识灾`` directory. Runtime junk (root
``outputs``/``logs``/``__pycache__``/``.pytest_cache``, any ``__pycache__``,
``*.pyc`` and symlinks/reparse points) is excluded so a source tree that has
been run locally still produces a clean package. The ZIP is written to a temp
file in the output directory and atomically moved into place, so a failed run
never damages an existing archive.
"""
from __future__ import annotations

import argparse
import os
import re
import stat
import sys
import tempfile
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE_NAME = "慧眼识灾.exe"

# profile -> dist directory name (matches build/build_app.ps1)
PROFILE_DIST = {"lite": "dist", "full": "dist_full"}

# pruned at the source root only
ROOT_EXCLUDE_DIRS = {"outputs", "logs", "__pycache__", ".pytest_cache"}
# pruned at every level
ALL_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache"}
# skipped at every level
EXCLUDE_SUFFIXES = (".pyc",)


def default_source(profile: str) -> str:
    """Default source dir for a profile: <root>/<dist>/慧眼识灾."""
    return os.path.join(ROOT, PROFILE_DIST[profile], "慧眼识灾")


def read_version() -> str:
    """Read ``__version__`` from ``src/__init__.py`` by parsing, not importing.

    Importing ``src`` would pull in rasterio/torch; parsing keeps this script
    stdlib-only and usable without the runtime dependencies installed.
    """
    init = os.path.join(ROOT, "src", "__init__.py")
    with open(init, encoding="utf-8") as fh:
        text = fh.read()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise ValueError(f"无法从 {init} 解析 __version__")
    return match.group(1)


def default_output(profile: str, source: str, version: str) -> str:
    """Default zip path: 慧眼识灾_v<version>_<profile>.zip next to the source dir."""
    return os.path.join(os.path.dirname(os.path.abspath(source)),
                        f"慧眼识灾_v{version}_{profile}.zip")


def _is_link_or_reparse(path: str) -> bool:
    """True for symlinks and Windows reparse points (junctions)."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    attrs = getattr(st, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse and attrs & reparse)


def _assert_output_outside_source(source: str, output: str) -> None:
    src = os.path.realpath(source)
    out = os.path.realpath(output)
    try:
        common = os.path.commonpath([src, out])
    except ValueError:  # different drives -> cannot be nested
        return
    if os.path.normcase(common) == os.path.normcase(src):
        raise ValueError(f"输出不能位于源目录内: {output}")


def build_zip(source: str, output: str) -> dict:
    """Zip ``source`` into ``output``, preserving the top-level directory name.

    Returns stats ``{"files", "raw_bytes", "zip_bytes", "elapsed"}``.
    Raises ``FileNotFoundError``/``ValueError`` on invalid input; the existing
    ``output`` is left untouched on any failure.
    """
    source = os.path.abspath(source)
    output = os.path.abspath(output)
    if not os.path.isdir(source):
        raise FileNotFoundError(f"缺少源目录: {source}")
    if not os.path.isfile(os.path.join(source, EXE_NAME)):
        raise FileNotFoundError(f"源目录缺少 {EXE_NAME}: {source}")
    _assert_output_outside_source(source, output)
    if os.path.isdir(output):
        raise ValueError(f"输出路径是目录: {output}")

    out_dir = os.path.dirname(output) or "."
    os.makedirs(out_dir, exist_ok=True)

    parent = os.path.dirname(source)  # arcname base keeps the 慧眼识灾 top level
    files = 0
    raw = 0
    t0 = time.time()

    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".release_zip_", suffix=".part")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for base, dirs, names in os.walk(source, topdown=True):
                at_root = os.path.normcase(os.path.abspath(base)) == os.path.normcase(source)
                kept = []
                for name in dirs:
                    full = os.path.join(base, name)
                    if name in ALL_EXCLUDE_DIRS:
                        continue
                    if at_root and name in ROOT_EXCLUDE_DIRS:
                        continue
                    if _is_link_or_reparse(full):
                        continue
                    kept.append(name)
                dirs[:] = kept

                for name in sorted(names):
                    if name.endswith(EXCLUDE_SUFFIXES):
                        continue
                    full = os.path.join(base, name)
                    if _is_link_or_reparse(full) or not os.path.isfile(full):
                        continue
                    zf.write(full, os.path.relpath(full, parent))
                    files += 1
                    raw += os.path.getsize(full)
                    if files % 1000 == 0:
                        print(f"  ... {files} files, {raw / 1024 / 1024:.0f} MB raw",
                              flush=True)
        os.replace(tmp, output)
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    return {
        "files": files,
        "raw_bytes": raw,
        "zip_bytes": os.path.getsize(output),
        "elapsed": time.time() - t0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILE_DIST), default="lite")
    ap.add_argument("--source", default=None, help="源目录，默认 dist/慧眼识灾 或 dist_full/慧眼识灾")
    ap.add_argument("--output", default=None, help="输出 ZIP，默认 <源目录同级>/慧眼识灾_v<版本>_<profile>.zip")
    args = ap.parse_args(argv)

    source = os.path.abspath(args.source) if args.source else default_source(args.profile)

    if args.output:
        output = os.path.abspath(args.output)
    else:
        try:
            version = read_version()
        except (OSError, ValueError) as exc:
            print(f"读取版本失败: {exc}")
            return 2
        output = default_output(args.profile, source, version)

    try:
        stats = build_zip(source, output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误: {exc}")
        return 2
    except OSError as exc:
        print(f"打包失败: {type(exc).__name__}: {exc}")
        return 1

    print(f"source     : {source}")
    print(f"files      : {stats['files']}")
    print(f"raw        : {stats['raw_bytes'] / 1024 / 1024:.1f} MB")
    print(f"zip        : {stats['zip_bytes'] / 1024 / 1024:.1f} MB")
    print(f"elapsed    : {stats['elapsed']:.0f}s")
    print(f"output     : {output}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
