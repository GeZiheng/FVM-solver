"""Snapshot collection and the versioned ``.npz`` snapshot contract.

This module is the data boundary between this repository and external
ROM consumers: it defines how flow-field snapshots are collected in
memory and written to / read from ``.npz`` files.  The package does not
import pyfvm; callers pass plain array-like fields (e.g. the
``to_numpy()`` arrays returned by the binding).

Shape conventions (contract ``format_version = 1``):

- ``format_version`` : scalar int64, always 1; the contract evolves by
  adding new keys only, never by changing existing ones.
- ``case``          : scalar str, case name.
- ``params``        : scalar str, case parameters as a JSON object.
- ``t``             : (S,) float64, snapshot times; the t=0 initial
  state is included and times are strictly increasing.
- ``u`` / ``v`` / ``p`` : (S, N) float64, rows = snapshots; N is the
  cell count and the cell order equals the pyfvm cell order
  (structured meshes ``j*nx + i``; SU2 unstructured = reader order).
- ``cx`` / ``cy``   : (N,) float64, optional cell-centre coordinates.

``cx``/``cy`` keys are omitted when no coordinates are supplied and
:func:`load_snapshots` returns ``None`` for absent keys.  Unknown extra
keys are ignored on load (additive-only contract evolution).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Contract version written by :meth:`SnapshotCollector.save`.
FORMAT_VERSION = 1

_REQUIRED_KEYS = ("format_version", "case", "params", "t", "u", "v", "p")


@dataclass
class SnapshotData:
    """Contents of a snapshot ``.npz`` file (see module docstring).

    Attributes:
        format_version: contract version, always 1 for this reader.
        case: case name (empty string when unset).
        params: case parameters parsed from the stored JSON string.
        t: (S,) float64 strictly increasing snapshot times.
        u, v, p: (S, N) float64 snapshot fields, rows = snapshots.
        cx, cy: (N,) float64 cell-centre coordinates, or None when the
            file carries no coordinate keys.
    """

    format_version: int
    case: str
    params: dict
    t: np.ndarray
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    cx: Optional[np.ndarray]
    cy: Optional[np.ndarray]


class SnapshotCollector:
    """Collect transient snapshots in memory for later ``.npz`` export.

    Fields handed to :meth:`append` are copied to float64 immediately
    (callers may pass live ``to_numpy()`` views; storing views is
    forbidden by the contract).  The read-only properties ``times``,
    ``u``, ``v`` and ``p`` stack the collected frames on access.
    """

    def __init__(self) -> None:
        self._times: list[float] = []
        self._u: list[np.ndarray] = []
        self._v: list[np.ndarray] = []
        self._p: list[np.ndarray] = []

    def __len__(self) -> int:
        """Number of collected snapshots."""
        return len(self._times)

    @staticmethod
    def _copy_field(field) -> np.ndarray:
        """Immediate float64 copy of an array-like field, flattened."""
        return np.array(field, dtype=np.float64, copy=True).reshape(-1)

    @staticmethod
    def _stack(name: str, parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            arr = np.zeros((0, 0), dtype=np.float64)
        else:
            try:
                arr = np.stack(parts)
            except ValueError as exc:
                raise ValueError(
                    f"inconsistent {name} field sizes across snapshots: {exc}"
                ) from exc
        arr.flags.writeable = False
        return arr

    def append(self, t: float, u, v, p) -> None:
        """Append one snapshot.

        Args:
            t: scalar snapshot time (float-convertible).
            u, v, p: array-like fields, flattened to 1-D and copied to
                float64 immediately.

        Raises:
            TypeError: if ``t`` is not scalar-convertible.
        """
        self._times.append(float(t))
        self._u.append(self._copy_field(u))
        self._v.append(self._copy_field(v))
        self._p.append(self._copy_field(p))

    @property
    def times(self) -> np.ndarray:
        """Snapshot times, read-only (S,) float64."""
        arr = np.array(self._times, dtype=np.float64)
        arr.flags.writeable = False
        return arr

    @property
    def u(self) -> np.ndarray:
        """u snapshots, read-only (S, N) float64 (rows = snapshots)."""
        return self._stack("u", self._u)

    @property
    def v(self) -> np.ndarray:
        """v snapshots, read-only (S, N) float64 (rows = snapshots)."""
        return self._stack("v", self._v)

    @property
    def p(self) -> np.ndarray:
        """p snapshots, read-only (S, N) float64 (rows = snapshots)."""
        return self._stack("p", self._p)

    @staticmethod
    def _cell_coords(name: str, coords, n: int) -> np.ndarray:
        arr = np.array(coords, dtype=np.float64, copy=True).reshape(-1)
        if arr.shape != (n,):
            raise ValueError(
                f"{name} must have shape ({n},) matching the field size, "
                f"got shape {arr.shape}")
        return arr

    def save(self, path, case: str = "", params: Optional[dict] = None,
             cx=None, cy=None) -> None:
        """Validate the collected stack and write a contract v1 ``.npz``.

        Args:
            path: output file path (``np.savez`` appends the ``.npz``
                suffix when missing).
            case: case name.
            params: case parameters, stored as a JSON string (``None``
                is stored as an empty JSON object).
            cx, cy: optional (N,) cell-centre coordinates.

        Raises:
            ValueError: if fewer than 2 snapshots were collected, the
                times are not strictly increasing, or the three fields
                (or cx/cy) have inconsistent shapes.
        """
        if len(self) < 2:
            raise ValueError(
                f"need at least 2 snapshots to save, got {len(self)}")
        t = np.array(self._times, dtype=np.float64)
        if not np.all(np.diff(t) > 0.0):
            raise ValueError("snapshot times must be strictly increasing")
        u = np.asarray(self.u)
        v = np.asarray(self.v)
        p = np.asarray(self.p)
        if not (u.shape == v.shape == p.shape):
            raise ValueError(
                f"field shapes differ: u {u.shape}, v {v.shape}, p {p.shape}")
        arrays = dict(
            format_version=np.int64(FORMAT_VERSION),
            case=str(case),
            params=json.dumps({} if params is None else params),
            t=t,
            u=u,
            v=v,
            p=p,
        )
        if cx is not None:
            arrays["cx"] = self._cell_coords("cx", cx, u.shape[1])
        if cy is not None:
            arrays["cy"] = self._cell_coords("cy", cy, u.shape[1])
        np.savez(path, **arrays)


def load_snapshots(path) -> SnapshotData:
    """Read a snapshot ``.npz`` file written per the v1 contract.

    Args:
        path: path to the ``.npz`` file.

    Returns:
        SnapshotData with the file contents (cx/cy are None when the
        coordinate keys are absent; unknown extra keys are ignored).

    Raises:
        ValueError: if a required key is missing, ``format_version``
            is not 1, or the stored shapes violate the contract.
    """
    with np.load(path) as npz:
        missing = [key for key in _REQUIRED_KEYS if key not in npz]
        if missing:
            raise ValueError(
                f"snapshot file {path!r} is missing required keys: {missing}")
        version = int(np.asarray(npz["format_version"]).item())
        if version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported snapshot format_version {version} "
                f"(expected {FORMAT_VERSION})")
        case = str(np.asarray(npz["case"]).item())
        params = json.loads(str(np.asarray(npz["params"]).item()))
        t = np.array(npz["t"], dtype=np.float64)
        u = np.array(npz["u"], dtype=np.float64)
        v = np.array(npz["v"], dtype=np.float64)
        p = np.array(npz["p"], dtype=np.float64)
        cx = (np.array(npz["cx"], dtype=np.float64)
              if "cx" in npz else None)
        cy = (np.array(npz["cy"], dtype=np.float64)
              if "cy" in npz else None)
    if u.ndim != 2 or not (u.shape == v.shape == p.shape):
        raise ValueError(
            f"fields must share one (S, N) shape, got u {u.shape}, "
            f"v {v.shape}, p {p.shape}")
    if t.shape != (u.shape[0],):
        raise ValueError(
            f"t must have shape ({u.shape[0]},), got {t.shape}")
    n = u.shape[1]
    if cx is not None and cx.shape != (n,):
        raise ValueError(f"cx must have shape ({n},), got {cx.shape}")
    if cy is not None and cy.shape != (n,):
        raise ValueError(f"cy must have shape ({n},), got {cy.shape}")
    return SnapshotData(format_version=version, case=case, params=params,
                        t=t, u=u, v=v, p=p, cx=cx, cy=cy)
