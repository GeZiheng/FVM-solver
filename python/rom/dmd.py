"""Exact dynamic mode decomposition (DMD), Tu et al. (2014).

Input convention: the snapshot matrix ``X`` has shape ``(S, N)`` with
one snapshot per row, sampled at a uniform time step ``dt > 0`` -- the
layout of the snapshot ``.npz`` contract.

Shape conventions -- DMD modes are COLUMNS:

- ``modes``           : (N, r), one DMD mode per COLUMN;
- ``eigenvalues``     : (r,) complex discrete eigenvalues lambda;
- ``omega``           : (r,) complex continuous eigenvalues
                        ``omega = log(lambda) / dt``;
- ``amplitudes``      : (r,) complex modal amplitudes ``b`` fitting the
                        first snapshot, ``b = lstsq(Phi, x1)``;
- ``singular_values`` : full singular-value spectrum of ``X1`` (not
                        truncated to the rank);
- ``rank``, ``dt``    : truncation rank actually used and the time step.

NOTE: DMD modes are stored as column vectors ``(N, r)``, following the
DMD literature.  POD modes (:mod:`rom.pod`) are stored as ROWS
``(r, N)``; each follows its own community convention, so keep the
difference in mind when mixing the two decompositions in one script.

Algorithm (rank-truncated exact DMD):

1. ``X1 = X[:-1].T`` and ``X2 = X[1:].T`` (columns = snapshots, the
   literature layout);
2. economic SVD of ``X1``; the rank is either given (``r``) or chosen
   by the relative criterion ``sigma > sigma_max * rtol``;
3. ``Atilde = Ur^T X2 Vr Sigma_r^-1`` and its eigen-decomposition
   ``(lambda, W)``;
4. exact DMD modes ``Phi = X2 Vr Sigma_r^-1 W``;
5. ``omega = log(lambda) / dt`` and ``b = lstsq(Phi, x1)`` with ``x1``
   the first snapshot.

The dominant-frequency rule (:meth:`DmdResult.dominant_frequency`) is
deterministic: among the modes with ``|Im(omega)| > 0`` pick the one
with the largest amplitude magnitude ``|b|`` and return
``|Im(omega)| / (2 pi)``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np


@dataclass
class DmdResult:
    """Result of :func:`dmd` (see module docstring for shapes)."""

    modes: np.ndarray
    eigenvalues: np.ndarray
    omega: np.ndarray
    amplitudes: np.ndarray
    singular_values: np.ndarray
    rank: int
    dt: float

    def dominant_frequency(self) -> float:
        """Frequency of the dominant oscillating mode, in Hz.

        Deterministic rule: among the modes with a nonzero imaginary
        part of ``omega``, take the one with the largest amplitude
        magnitude ``|b|`` and return ``|Im(omega)| / (2 pi)``.

        Raises:
            ValueError: if no mode has ``|Im(omega)| > 0``.
        """
        imag = np.abs(self.omega.imag)
        candidates = imag > 0.0
        if not np.any(candidates):
            raise ValueError(
                "no DMD mode has a nonzero imaginary part of omega; "
                "dominant_frequency() needs an oscillating mode")
        magnitudes = np.where(candidates, np.abs(self.amplitudes), -1.0)
        idx = int(np.argmax(magnitudes))
        return float(imag[idx] / (2.0 * np.pi))

    def reconstruct(self, times) -> np.ndarray:
        """Reconstruct the field at arbitrary times.

        Evaluates ``x(t) = Re( sum_k b_k phi_k exp(omega_k t) )``.

        Args:
            times: array-like of query times (a scalar is accepted and
                returns a single row).

        Returns:
            (len(times), N) real-valued reconstruction.
        """
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        coeffs = (np.exp(times[:, None] * self.omega[None, :])
                  * self.amplitudes[None, :])
        return np.real(coeffs @ self.modes.T)


def dmd(X, dt, r=None, rtol: float = 1e-12) -> DmdResult:
    """Exact DMD of a uniformly sampled snapshot matrix.

    Args:
        X: (S, N) snapshot matrix, one snapshot per row.
        dt: uniform sampling interval, must be > 0.
        r: optional truncation rank; when given it is used as-is.
            Otherwise the rank is the relative truncation
            ``sigma > sigma_max * rtol`` of the X1 spectrum.
        rtol: relative singular-value threshold used only when ``r``
            is None; must be > 0.

    Returns:
        DmdResult with modes as COLUMNS (N, r); see the module docstring
        for all shape conventions.

    Raises:
        ValueError: if ``X`` is not 2-D, ``S < 2``, ``dt <= 0``, the
            given ``r`` is out of range, ``rtol <= 0``, or X1 has
            numerical rank 0 (an all-zero snapshot sequence).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D (S, N) matrix, got shape {X.shape}")
    if X.shape[0] < 2:
        raise ValueError(
            f"need at least 2 snapshots for DMD, got {X.shape[0]}")
    dt = float(dt)
    if not dt > 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    x1 = X[0]

    # Literature layout: columns = snapshots.
    x1_mat = X[:-1].T
    x2_mat = X[1:].T

    u, s, vt = np.linalg.svd(x1_mat, full_matrices=False)
    if r is None:
        rtol = float(rtol)
        if not rtol > 0.0:
            raise ValueError(f"rtol must be positive, got {rtol}")
        rank = int(np.sum(s > s[0] * rtol)) if s.size and s[0] > 0.0 else 0
    else:
        r = operator.index(r)
        if not 1 <= r <= s.size:
            raise ValueError(f"r must be in [1, {s.size}], got {r}")
        rank = r
    if rank == 0:
        raise ValueError(
            "numerical rank of X1 is zero; cannot decompose an "
            "all-zero snapshot sequence")

    u_r = u[:, :rank]
    s_r = s[:rank]
    v_r = vt[:rank].T
    inv_s = np.diag(1.0 / s_r)

    a_tilde = u_r.T @ x2_mat @ v_r @ inv_s
    eigenvalues, w = np.linalg.eig(a_tilde)
    modes = x2_mat @ v_r @ inv_s @ w

    eigenvalues = eigenvalues.astype(complex)
    omega = np.log(eigenvalues) / dt
    amplitudes = np.linalg.lstsq(modes, x1, rcond=None)[0].astype(complex)
    return DmdResult(modes=modes.astype(complex), eigenvalues=eigenvalues,
                     omega=omega, amplitudes=amplitudes,
                     singular_values=s, rank=rank, dt=dt)
