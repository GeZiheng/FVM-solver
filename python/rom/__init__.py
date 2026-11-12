"""ROM (reduced-order model) toolkit: snapshots, POD and DMD.

The package core depends only on numpy and the standard library; it
never imports pyfvm.  Callers pass plain array-like fields (e.g. the
``to_numpy()`` arrays returned by the binding), so the algebraic parts
also work without the built extension.

Shape conventions (see the individual modules for full details):

- snapshot matrices: ``(S, N)`` with one snapshot per row, the layout
  of the ``.npz`` snapshot contract;
- POD modes (:mod:`rom.pod`) are stored as ROWS, shape ``(r, N)``;
- DMD modes (:mod:`rom.dmd`) are stored as COLUMNS, shape ``(N, r)``.

The two mode layouts each follow their own community convention; keep
the difference in mind when mixing POD and DMD results in one script.

:mod:`rom.plot` is an optional matplotlib diagnostic layer and is not
imported here (or by the test suite).
"""

from .snapshots import SnapshotCollector, SnapshotData, load_snapshots
from .pod import PodResult, pod_decompose
from .dmd import DmdResult, dmd

__version__ = "0.1.0"

__all__ = [
    "SnapshotCollector",
    "SnapshotData",
    "load_snapshots",
    "PodResult",
    "pod_decompose",
    "DmdResult",
    "dmd",
]
