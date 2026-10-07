"""Locate the built pyfvm extension and put its directory on sys.path.

The CMake target writes pyfvm to <build-dir>/python, so this scans
build/*/python under the repository root. Import this module before
`import pyfvm`:

    import _path  # noqa: F401
    import pyfvm as fvm
"""

import glob
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _add_build_dir():
    candidates = sorted(glob.glob(os.path.join(_REPO_ROOT, "build", "*", "python")))
    for cand in candidates:
        hit = glob.glob(os.path.join(cand, "pyfvm*.pyd")) or glob.glob(
            os.path.join(cand, "pyfvm*.so"))
        if hit:
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
    raise RuntimeError(
        "pyfvm extension not found. Configure with -DFVM_BUILD_PYTHON=ON "
        "and build the 'pyfvm' target first.")


_add_build_dir()
