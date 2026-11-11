"""Shared helpers for the case scripts in this directory.

Importing this module bootstraps sys.path (the repository python/
directory) and imports the pyfvm extension, so a case script's header
reduces to ``from _common import ...``.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _path  # noqa: F401,E402

import pyfvm as fvm


def write_results(mesh, velocity, pressure, case_dir, filename,
                  scalars=None, vectors=None):
    """Write VTK output under output/<case_dir>/ and print the path.

    The writer follows the filename extension: ".vtu" for unstructured
    meshes, ".vti" for structured ones.  scalars/vectors default to the
    pressure/velocity fields; pass them explicitly to override (an
    unused standard field then stays None).  Returns the output path.
    """
    output_dir = Path(__file__).resolve().parent / "output" / case_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    if scalars is None:
        scalars = {"pressure": pressure}
    if vectors is None:
        vectors = {"velocity": velocity}
    if filename.endswith(".vtu"):
        fvm.write_vtu(str(output_path), mesh, scalars, vectors)
    elif filename.endswith(".vti"):
        fvm.write_vti(str(output_path), mesh, scalars, vectors)
    else:
        raise ValueError(f"unsupported VTK extension: {filename}")
    print(f"Wrote {output_path}")
    return output_path


def frame_interval(steps, every=50, max_frames=50):
    """Step interval for frame output.

    Frames every `every` steps, stretched on long runs so the whole
    series stays within ~`max_frames` frames.
    """
    return max(every, -(-steps // max_frames))


def write_frame(mesh, velocity, pressure, case_dir, stem, index,
                scalars=None, vectors=None):
    """Write one VTK frame stem_{index:04d}.<ext> under output/<case_dir>/."""
    ext = "vtu" if isinstance(mesh, fvm.UnstructuredMesh) else "vti"
    return write_results(mesh, velocity, pressure, case_dir,
                         f"{stem}_{index:04d}.{ext}", scalars, vectors)


def read_su2_text(text, name="case.su2"):
    """Write SU2 text to a temporary file, read it back, return the mesh."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="ascii") as fh:
            fh.write(text)
        return fvm.read_su2(path)


def print_simple_status(label, result, suffix=""):
    """Print the shared SIMPLE convergence status line.

    `label` is the parenthesized context; "" prints no parentheses.
    `suffix` is appended verbatim after the residual (e.g. wall time).
    """
    state = "converged" if result.converged else "NOT converged"
    prefix = f" ({label})" if label else ""
    print(f"SIMPLE{prefix}: {state} in "
          f"{result.iterations} iterations, continuity residual "
          f"{result.history[-1].continuity:.3e}{suffix}")


def report_test(ok, detail):
    """Print the application-test verdict and exit non-zero on failure."""
    print(f"application test: {'PASS' if ok else 'FAIL'} {detail}")
    if not ok:
        sys.exit(1)
