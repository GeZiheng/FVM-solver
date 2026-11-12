"""POD (proper orthogonal decomposition) of snapshot matrices.

Input convention: the snapshot matrix ``X`` has shape ``(S, N)`` with
one snapshot per row -- the layout of the snapshot ``.npz`` contract.

Shape conventions -- POD modes are ROWS:

- ``mean``              : (N,) time mean subtracted before the SVD
                          (all zeros when ``center=False``);
- ``modes``             : (r_full, N), one POD mode per ROW;
- ``singular_values``   : (r_full,) singular values of the centered
                          snapshot matrix;
- ``energy``            : (r_full,) normalized energies
                          ``e_k = sigma_k^2 / sum(sigma^2)``;
- ``cumulative_energy`` : (r_full,) cumulative sums of ``energy``.

NOTE: POD modes are stored as row vectors ``(r_full, N)``, following
the SVD convention ``Xc = U S V^T``.  DMD modes (:mod:`rom.dmd`) are
stored as COLUMNS ``(N, r)``; each follows its own community
convention, so keep the difference in mind when mixing the two
decompositions in one script.

The decomposition is an economic SVD of the (optionally mean-centered)
snapshot matrix; it is deterministic up to mode signs (the sign of a
singular vector is not fixed), so tests assert reconstruction
invariants rather than mode values.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np


@dataclass
class PodResult:
    """Result of :func:`pod_decompose` (see module docstring for shapes)."""

    mean: np.ndarray
    modes: np.ndarray
    singular_values: np.ndarray
    energy: np.ndarray
    cumulative_energy: np.ndarray

    def reconstruction_error(self, r: int) -> float:
        """Relative spectral error of the rank-r truncation.

        Computed from the singular-value spectrum alone as
        ``sqrt(sum_{k>r} sigma_k^2 / sum_k sigma_k^2)`` -- no matrix is
        rebuilt.  ``r`` may range from 0 (no modes) to r_full (exact
        reconstruction, error 0).
        """
        r = operator.index(r)
        s2 = self.singular_values * self.singular_values
        if r < 0 or r > s2.size:
            raise ValueError(
                f"r must be in [0, {s2.size}], got {r}")
        total = float(s2.sum())
        if total == 0.0:
            return 0.0
        return float(np.sqrt(s2[r:].sum() / total))

    def project(self, X) -> np.ndarray:
        """Project snapshots onto the POD modes.

        Args:
            X: (S, N) snapshot matrix (rows = snapshots).

        Returns:
            (S, r_full) temporal coefficients ``(X - mean) @ modes.T``.
        """
        X = np.asarray(X, dtype=np.float64)
        n = self.modes.shape[1]
        if X.ndim != 2 or X.shape[1] != n:
            raise ValueError(
                f"X must have shape (S, {n}) with one snapshot per row, "
                f"got {X.shape}")
        return (X - self.mean) @ self.modes.T

    def reconstruct(self, A) -> np.ndarray:
        """Reconstruct snapshots from temporal coefficients.

        Args:
            A: (S, r_full) coefficient matrix (e.g. from :meth:`project`).

        Returns:
            (S, N) snapshot matrix ``A @ modes + mean``.
        """
        A = np.asarray(A, dtype=np.float64)
        r_full = self.modes.shape[0]
        if A.ndim != 2 or A.shape[1] != r_full:
            raise ValueError(
                f"A must have shape (S, {r_full}), got {A.shape}")
        return A @ self.modes + self.mean


def pod_decompose(X, center: bool = True) -> PodResult:
    """POD of a snapshot matrix via an economic SVD.

    Args:
        X: (S, N) snapshot matrix, one snapshot per row.
        center: subtract the time mean (fluctuation POD) before the
            SVD; the mean field is returned so the full field is
            recovered by :meth:`PodResult.reconstruct`.  With
            ``center=False`` the mean is all zeros.

    Returns:
        PodResult with modes as ROWS (r_full, N); see the module
        docstring for all shape conventions.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D (S, N) matrix, got shape {X.shape}")
    if center:
        mean = X.mean(axis=0)
    else:
        mean = np.zeros(X.shape[1], dtype=np.float64)
    Xc = X - mean
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    total = float(np.sum(s * s))
    if total > 0.0:
        energy = (s * s) / total
    else:
        energy = np.zeros_like(s)
    cumulative_energy = np.cumsum(energy)
    return PodResult(mean=mean, modes=vt, singular_values=s, energy=energy,
                     cumulative_energy=cumulative_energy)
