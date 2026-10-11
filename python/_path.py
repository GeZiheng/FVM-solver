"""Locate the built pyfvm extension and put its directory on sys.path.

The CMake target writes pyfvm to <build-dir>/python, so this scans
build/*/python under the repository root. Import this module before
`import pyfvm`:

    import _path  # noqa: F401
    import pyfvm as fvm

Selection rules:
- FVM_PYFVM_DIR, when set, must point at the directory holding the
  extension for the running interpreter.
- Otherwise only extensions matching the running interpreter tag are
  considered (a stale .pyd built for another CPython is never picked),
  and the newest match wins -- several build trees can coexist (e.g.
  build/release-py from a manual configure and build/Release-agent-py
  from the build-and-test MCP server).
"""

import glob
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V = sys.version_info
_TAG_PATTERNS = (f"cp{_V.major}{_V.minor}", f"cpython-{_V.major}{_V.minor}")


def _extension_files(directory):
    """Extension files in `directory` that match the running interpreter."""
    hits = glob.glob(os.path.join(directory, "pyfvm*.pyd"))
    hits += glob.glob(os.path.join(directory, "pyfvm*.so"))
    return [p for p in hits
            if any(pattern in os.path.basename(p) for pattern in _TAG_PATTERNS)]


def _find_build_dir():
    override = os.environ.get("FVM_PYFVM_DIR")
    if override:
        if not _extension_files(override):
            raise RuntimeError(
                f"FVM_PYFVM_DIR={override!r} holds no pyfvm extension for "
                f"this interpreter (tag patterns {_TAG_PATTERNS}).")
        return override

    candidates = sorted(glob.glob(os.path.join(_REPO_ROOT, "build", "*", "python")))
    best_dir, best_mtime = None, -1.0
    for cand in candidates:
        hits = _extension_files(cand)
        if hits:
            mtime = max(os.path.getmtime(p) for p in hits)
            if mtime > best_mtime:
                best_dir, best_mtime = cand, mtime
    if best_dir is None:
        scanned = candidates or ["<no build/*/python directories>"]
        raise RuntimeError(
            f"pyfvm extension not found for interpreter "
            f"{_V.major}.{_V.minor}.{_V.micro} (tag patterns {_TAG_PATTERNS}); "
            f"scanned: {scanned}. Configure with -DFVM_BUILD_PYTHON=ON and "
            "build the 'pyfvm' target first.")
    return best_dir


def _add_build_dir():
    cand = _find_build_dir()
    if cand not in sys.path:
        sys.path.insert(0, cand)
    return cand


_add_build_dir()
