"""Offline-precomputed reduced operators for fixed-BC POD-Galerkin ROMs
(Phase 6.5b).

Replaces the per-stage FOM work of :class:`rom.galerkin.GalerkinRom`
(mass flux + momentum assemblies + pressure projection) with pure-numpy
small algebra.  The reduced right-hand side is decomposed as

    f(a) = f_np(a) + s_p(a)

with the joint coefficient state ``a = [a_u, a_v]``:

- ``f_np(a) = q + L a + Q[a, a]`` (per component) - the zero-pressure
  reduced right-hand side.  For a Central convection scheme this is
  *exactly* quadratic in ``a`` (the convection operator is linear in the
  mass flux, the mass flux is linear in the reconstructed velocity, and
  the fixed-BC boundary contributions are constants), so a least-squares
  fit recovers it to machine precision; the fit is verified on held-out
  states and raises if the quadratic assumption is broken.
- ``s_p(a) = h0 + H a`` - the pressure-gradient source closure.  Pressure
  is represented on its own POD basis ``p ~ pbar + Psi c`` (from the FOM
  pressure snapshots); each mode's reduced source ``g_k`` is isolated
  offline by the *b-difference trick* - the difference of two
  ``GalerkinRom.rhs`` evaluations with the same state but different
  pressure field content is exactly ``M_r^-1 Phi (-(grad psi_k) V)``
  because ``b^np`` is pressure-independent - and the coefficients are
  modelled by the affine map ``c(a) = K a + c0`` fitted to snapshot
  pairs by least squares (``H = G K``, ``h0 = g0 + G c0``; ``g0`` is the
  mean-pressure source).  This is a *data-driven closure*, not an
  operator projection; its error is quantified (fit diagnostics + the
  acceptance gates of the case drivers).

Everything is sampled in f-space (the rhs values already carry the
``M_r^-1`` factor), so the online model needs no mass matrix and no
pyfvm import.  Scope: fixed boundary conditions, plus a KNOWN boundary
parameter joined to the quadratic design (``parameter_samples``/
``parameter_bc`` offline, ``parameter_fn`` online - Phase 6 validation
cases, the D1 mechanism) and a live ``bc_refresh`` for the pressure
solves; arbitrary time-dependent BCs still need the boundary term
factored out (registered follow-up).  Non-Central convection makes
``f_np`` only piecewise quadratic (a warning is emitted).

Two online consumers (Phase 6.5b/6.5c):

- :class:`PrecomputedGalerkinRom` evaluates the quadratic part plus the
  affine pressure closure - pure numpy, but the closure's constant
  Jacobian extrapolates arbitrarily off the attractor and is documented
  UNSTABLE on the shedding limit cycle (diverges near t ~ 2.3; use for
  diagnostics/short horizons only).
- :class:`ReducedPressureRom` (6.5c M1/M2) replaces the closure with a
  per-stage pressure solve on exactly reconstructed Rhie-Chow cell fields
  (affine diagonal + quadratic steady residual + transient term), the
  solved pressure acting through ``g0 + G Psi^T (p - pbar)`` with ``Psi``
  built from projection-solution pressures - state dependent and stable.
  The default mode runs the FULL FOM solve; ``reduced_solve=True`` (M2)
  exports the assembled pressure equation and solves its Galerkin
  projection onto ``Psi`` instead (dense ``r_p x r_p``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .pod import pod_decompose


@dataclass
class QuadraticFit:
    """``f(a) = q + L a + a^T Q a`` coefficients (per output component:
    reduced f-space rows or cell-field rows).

    ``a`` is the joint state ``[a_u, a_v]``; ``Q`` is symmetric in its
    last two indices with off-diagonal pairs halved so ``a^T Q a`` counts
    each cross term once.  Evaluation goes through the flattened
    upper-triangle form (``quad`` over ``a_i a_j`` pairs) - a single
    GEMM instead of a per-output einsum over the (n_out, n_a, n_a)
    tensor, which is memory-bandwidth bound for cell-field fits.
    """

    q: np.ndarray  # (n_out,)
    L: np.ndarray  # (n_out, n_a)
    Q: np.ndarray  # (n_out, n_a, n_a)
    triu_i: np.ndarray = None  # (n_pairs,) upper-triangle indices
    triu_j: np.ndarray = None

    def __post_init__(self):
        if self.triu_i is None:
            iu, ju = np.triu_indices(self.Q.shape[1])
            self.triu_i, self.triu_j = iu, ju
        # Flattened feature x_p = a_i a_j counted once per pair; the
        # tensor contraction contributes Q[i,j] + Q[j,i] off the
        # diagonal and Q[i,i] on it (the stored off-diagonal entries are
        # pre-halved, the diagonal is not).
        self._quad_rows = self.Q[:, self.triu_i, self.triu_j] \
            + self.Q[:, self.triu_j, self.triu_i]
        self._quad_rows[:, self.triu_i == self.triu_j] *= 0.5

    def __call__(self, a: np.ndarray) -> np.ndarray:
        feats = a[self.triu_i] * a[self.triu_j]
        return self.q + self.L @ a + self._quad_rows @ feats


@dataclass
class ReducedOperatorFit:
    """Complete offline fit consumed by :class:`PrecomputedGalerkinRom`."""

    quad_u: QuadraticFit
    quad_v: QuadraticFit
    # Folded affine pressure closure per component: s_p(a) = h0 + H a.
    h0_u: np.ndarray  # (r,)
    h0_v: np.ndarray
    H_u: np.ndarray   # (r, n_a)
    H_v: np.ndarray
    # Raw closure ingredients (diagnostics / reuse).
    pressure_modes: np.ndarray        # Psi, (r_p, N)
    pressure_mean: np.ndarray         # pbar, (N,)
    source_modes_u: np.ndarray        # G, (r, r_p)
    source_modes_v: np.ndarray
    mean_source_u: np.ndarray         # g0 = source of pbar, (r,)
    mean_source_v: np.ndarray
    coefficient_map: np.ndarray       # K = [K | c0], (r_p, n_a + 1)
    # Rebuild inputs (M_r, state projection) and diagnostics.
    mass_diagonal: np.ndarray         # rho * V, (N,)
    modes_u: np.ndarray
    modes_v: np.ndarray
    mean_u: np.ndarray
    mean_v: np.ndarray
    rank_pressure: int = 0
    pressure_cumulative_energy: float = 0.0
    holdout_residual: float = np.inf  # max relative quadratic-fit error
    closure_train_residual: float = np.inf  # relative ||c_pred - c_snap||
    convection_is_central: bool = True


def _quadratic_design(samples: np.ndarray):
    """Design matrix ``[1, a, a_j a_k (j <= k)]`` for rows of samples."""
    n_s, n_a = samples.shape
    iu, ju = np.triu_indices(n_a)
    design = np.empty((n_s, 1 + n_a + iu.size), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1:1 + n_a] = samples
    design[:, 1 + n_a:] = samples[:, iu] * samples[:, ju]
    return design, iu, ju


def _fit_quadratic(samples: np.ndarray, values: np.ndarray) -> QuadraticFit:
    """Least-squares quadratic fit with column normalization.

    The normalization is folded into the returned coefficients, so
    evaluation takes raw features.
    """
    design, iu, ju = _quadratic_design(samples)
    scale = np.sqrt(np.mean(design * design, axis=0))
    scale[scale == 0.0] = 1.0
    coef, *_ = np.linalg.lstsq(design / scale, values, rcond=None)
    coef = coef / scale[:, None]  # (n_features, r)

    n_a = samples.shape[1]
    r = values.shape[1]
    q = coef[0]
    L = coef[1:1 + n_a].T
    Q = np.zeros((r, n_a, n_a), dtype=np.float64)
    quad = coef[1 + n_a:].T  # (r, n_pairs)
    Q[:, iu, ju] = quad
    Q[:, ju, iu] = quad
    off = iu != ju  # cross pairs appear twice in a^T Q a: halve them
    Q[:, iu[off], ju[off]] *= 0.5
    Q[:, ju[off], iu[off]] *= 0.5
    return QuadraticFit(q=q, L=L, Q=Q)


def make_fit_samples(a_u_snapshots, a_v_snapshots, parameter=None,
                     n_samples=None, rng=None):
    """Coefficient samples for the quadratic fit: snapshot rows plus
    GRADED small perturbations (0.1/0.2/0.3 per-coordinate sigma).

    The grading matters: the reduced rhs is exactly quadratic, but the
    snapshot coefficients lie on a low-dimensional attractor, so the
    joint quadratic design is underdetermined by on-manifold states
    alone.  Full-sigma perturbed copies overshoot (they anchor the fit
    far off the attractor and leave the near-manifold tube that the ROM
    actually explores unconstrained); graded small perturbations pin the
    quadratic form inside that tube (measured: machine-precision
    holdout vs ~30% error with full-sigma top-ups on the shedding
    window).

    Args:
        a_u_snapshots, a_v_snapshots: (S, r) coefficient matrices of the
            POD window (the attractor region the ROM will explore).
        parameter: optional (S,) known-parameter values aligned with the
            snapshot rows (Phase 6 validation cases: a time-dependent
            boundary value such as cos(omega t)).  The parameter is
            perturbed with the same graded scheme and returned as a
            third array; pass ``parameter_samples``/``parameter_bc`` to
            the fit functions to include it in the joint design.
        n_samples: total number of samples; default ``2 * n_features``
            of the joint quadratic design (>= that count is required by
            :func:`fit_reduced_operators`).
        rng: optional numpy Generator (deterministic callers pass one).
    """
    a_u = np.asarray(a_u_snapshots, dtype=np.float64)
    a_v = np.asarray(a_v_snapshots, dtype=np.float64)
    if a_u.ndim != 2 or a_v.ndim != 2 \
            or a_v.shape[0] != a_u.shape[0]:
        raise ValueError(
            f"a_u/a_v snapshots must share the row count, got "
            f"{a_u.shape} vs {a_v.shape}")
    rng = rng if rng is not None else np.random.default_rng()
    r_u, r_v = a_u.shape[1], a_v.shape[1]
    n_a = r_u + r_v + (0 if parameter is None else 1)
    if n_samples is None:
        n_samples = 2 * (1 + n_a + n_a * (n_a + 1) // 2)
    n_base = min(a_u.shape[0], max(1, n_samples // 2))
    pick = rng.choice(a_u.shape[0], size=n_base, replace=False)
    parts_u, parts_v = [a_u[pick]], [a_v[pick]]
    parts_p = [np.asarray(parameter, dtype=np.float64)[pick]] \
        if parameter is not None else []
    noise_u = np.maximum(a_u.std(axis=0), 1e-3)
    noise_v = np.maximum(a_v.std(axis=0), 1e-3)
    if parameter is not None:
        param = np.asarray(parameter, dtype=np.float64)
        if param.shape != (a_u.shape[0],):
            raise ValueError(
                f"parameter must have shape ({a_u.shape[0]},), got "
                f"{param.shape}")
        noise_p = max(float(param.std()), 1e-3)
    scales = (0.1, 0.2, 0.3)
    remaining = n_samples - n_base
    batch = 0
    while remaining > 0:
        take = min(remaining, max(1, a_u.shape[0]))
        more = rng.choice(a_u.shape[0], size=take, replace=False)
        scale = scales[min(batch, len(scales) - 1)]
        parts_u.append(a_u[more] + scale * noise_u
                       * rng.standard_normal((take, r_u)))
        parts_v.append(a_v[more] + scale * noise_v
                       * rng.standard_normal((take, r_v)))
        if parameter is not None:
            parts_p.append(param[more] + scale * noise_p
                           * rng.standard_normal(take))
        remaining -= take
        batch += 1
    out_u = np.vstack(parts_u)[:n_samples]
    out_v = np.vstack(parts_v)[:n_samples]
    if parameter is None:
        return out_u, out_v
    out_p = np.concatenate(parts_p)[:n_samples]
    return out_u, out_v, out_p


def projected_pressure_snapshots(mesh, mu, rho, pod_u, pod_v, bc_u, bc_v,
                                 bc_p, convection, a_u_snapshots,
                                 a_v_snapshots, dt_p, corrections=None,
                                 solver_config=None, parameter_samples=None,
                                 parameter_bc=None):
    """Per-state projection-solution pressures at snapshot coefficients
    (Phase 6.5c D3): the pressure basis must span the solutions the
    online ROM actually computes, and FOM (converged) pressure snapshots
    measurably do not (single-corrector projection solutions live
    partly outside their span).

    For every coefficient row, reconstructs the velocity, runs one plain
    ``pressure_projection(dt = dt_p)`` on scratch fields and records the
    solved absolute pressure.  Feed the result through
    :func:`rom.pod.pod_decompose` and pass ``(modes, mean)`` as
    ``pressure_basis`` to :func:`fit_reduced_operators`.

    ``parameter_samples``/``parameter_bc`` (Phase 6 validation cases):
    with a known boundary parameter (e.g. a time-dependent wall value),
    ``parameter_bc(value)`` is called before each row's projection so
    the sampled pressures see the same boundary state the online ROM
    will; the basis then spans the projection solutions across the
    parameter range.
    """
    import pyfvm as fvm  # lazy

    a_u_snapshots = np.asarray(a_u_snapshots, dtype=np.float64)
    a_v_snapshots = np.asarray(a_v_snapshots, dtype=np.float64)
    if a_u_snapshots.ndim != 2 or a_v_snapshots.ndim != 2 \
            or a_v_snapshots.shape[0] != a_u_snapshots.shape[0]:
        raise ValueError(
            f"a_u/a_v snapshots must share the row count, got "
            f"{a_u_snapshots.shape} vs {a_v_snapshots.shape}")
    if (parameter_samples is None) != (parameter_bc is None):
        raise ValueError("parameter_samples and parameter_bc must be "
                         "given together")
    if parameter_samples is not None:
        parameter_samples = np.asarray(parameter_samples, dtype=np.float64)
        if parameter_samples.shape != (a_u_snapshots.shape[0],):
            raise ValueError(
                f"parameter_samples must have shape "
                f"({a_u_snapshots.shape[0]},), got "
                f"{parameter_samples.shape}")
    modes_u = np.asarray(pod_u.modes, dtype=np.float64)
    n_cells = modes_u.shape[1]
    velocity = fvm.VectorField(mesh, "snap_velocity")
    pressure = fvm.ScalarField(mesh, "snap_pressure")
    flux = fvm.FaceFluxField(mesh, "snap_phi")
    out = np.empty((a_u_snapshots.shape[0], n_cells), dtype=np.float64)
    cfg = solver_config if solver_config is not None else fvm.SolverConfig()
    opts = corrections if corrections is not None else fvm.CorrectionOptions()
    for i, (a_u, a_v) in enumerate(zip(a_u_snapshots, a_v_snapshots)):
        if parameter_bc is not None:
            parameter_bc(parameter_samples[i])
        velocity.u.to_numpy()[:] = pod_u.mean + a_u @ modes_u
        velocity.v.to_numpy()[:] = pod_v.mean + a_v @ np.asarray(
            pod_v.modes, dtype=np.float64)
        fvm.compute_mass_flux(mesh, velocity, rho, bc_u, bc_v, flux, opts)
        pressure.to_numpy()[:] = 0.0
        fvm.pressure_projection(mesh, rho, mu, convection, bc_u, bc_v,
                                bc_p, velocity, pressure, flux,
                                solver_config=cfg, dt=dt_p,
                                corrections=opts)
        out[i] = pressure.to_numpy()
    return out


def fit_reduced_operators(mesh, mu, rho, pod_u, pod_v, bc_u, bc_v,
                          convection, a_u_samples, a_v_samples,
                          velocity_snapshots=None, pressure_snapshots=None,
                          pressure_rank=None, pressure_energy=0.9999,
                          pressure_basis=None, corrections=None,
                          holdout_samples=24, holdout_tolerance=1e-10,
                          rng=None, parameter_samples=None,
                          parameter_bc=None):
    """Fit the precomputed reduced operators on a fixed-BC case.

    Args:
        mesh..corrections: the GalerkinRom sampler configuration (a
            zero-pressure :class:`rom.galerkin.GalerkinRom` with an
            injectable pressure field is built internally; pyfvm is
            imported lazily exactly like ``rom.galerkin``).
        a_u_samples, a_v_samples: (S, r) coefficient samples covering
            the attractor (see :func:`make_fit_samples`); at least
            ``1 + n_a + n_a(n_a+1)/2`` rows with ``n_a = r_u + r_v``.
        velocity_snapshots, pressure_snapshots: paired (S, N) u/v/p
            snapshot matrices of the same window (the npz contract
            stores them); both are required for the affine pressure
            closure, None/None disables it (the fit then reproduces the
            zero-pressure ROM exactly).
        pressure_basis: optional ``(modes, mean)`` pair overriding the
            internal POD of ``pressure_snapshots`` (Phase 6.5c: the
            per-mode sources G are then built for THIS basis and the
            affine closure map is skipped - the closure is unused by
            :class:`ReducedPressureRom`, which needs only G/``Psi``).
        pressure_rank: pressure POD rank r_p; None picks the smallest
            rank reaching ``pressure_energy`` cumulative energy.
        holdout_samples: extra random on-attractor states verifying the
            quadratic exactness; the max relative error must stay below
            ``holdout_tolerance`` or RuntimeError is raised.
        rng: optional numpy Generator for the holdout draw.

    Returns:
        :class:`ReducedOperatorFit` with fit diagnostics attached.
    """
    import pyfvm as fvm  # lazy: rom.galerkin imports it the same way

    from rom.galerkin import GalerkinRom

    rng = rng if rng is not None else np.random.default_rng()

    central = convection is fvm.ConvectionScheme.Central
    if not central:
        warnings.warn(
            "precomputed reduced operators are exactly quadratic only "
            "for Central convection; other schemes make the convection "
            "coefficients piecewise linear in the mass flux and the fit "
            "becomes an approximation",
            stacklevel=2)

    modes_u = np.asarray(pod_u.modes, dtype=np.float64)
    modes_v = np.asarray(pod_v.modes, dtype=np.float64)
    r_u, n_cells = modes_u.shape
    r_v = modes_v.shape[0]
    a_u_samples = np.asarray(a_u_samples, dtype=np.float64)
    a_v_samples = np.asarray(a_v_samples, dtype=np.float64)
    if a_u_samples.ndim != 2 or a_u_samples.shape[1] != r_u \
            or a_v_samples.ndim != 2 \
            or a_v_samples.shape[0] != a_u_samples.shape[0] \
            or a_v_samples.shape[1] != r_v:
        raise ValueError(
            f"a_u/a_v samples must be row-aligned with widths "
            f"({r_u}, {r_v}), got {a_u_samples.shape} vs "
            f"{a_v_samples.shape}")
    if (parameter_samples is None) != (parameter_bc is None):
        raise ValueError("parameter_samples and parameter_bc must be "
                         "given together")
    if parameter_samples is not None:
        parameter_samples = np.asarray(parameter_samples,
                                       dtype=np.float64)
        if parameter_samples.shape != (a_u_samples.shape[0],):
            raise ValueError(
                f"parameter_samples must have shape "
                f"({a_u_samples.shape[0]},), got "
                f"{parameter_samples.shape}")
        if pressure_basis is None:
            raise ValueError(
                "the boundary parameter needs an external pressure_basis "
                "(the internal affine closure is fixed-BC only)")

    n_a = r_u + r_v + (0 if parameter_samples is None else 1)
    n_features = 1 + n_a + n_a * (n_a + 1) // 2
    if a_u_samples.shape[0] < n_features:
        raise ValueError(
            f"too few fit samples: {a_u_samples.shape[0]} < {n_features} "
            "(the joint quadratic design needs at least that many)")

    # Sampler: zero-pressure GalerkinRom with an injectable pressure
    # field (the caller-owned pattern of the oracle hook).  With a
    # boundary parameter, parameter_bc(value) sets the live BC fields
    # before each assembly - the sampled operator then varies with the
    # parameter exactly as the online ROM's C++ solves will.
    pressure_field = fvm.ScalarField(mesh, "fit_pressure")
    sampler = GalerkinRom(mesh, mu, rho, pod_u, pod_v, bc_u, bc_v,
                          convection, corrections=corrections,
                          pressure=pressure_field)

    def sample_all(u_samples, v_samples, params=None):
        f_u = np.empty((u_samples.shape[0], r_u))
        f_v = np.empty((v_samples.shape[0], r_v))
        pressure_field.to_numpy()[:] = 0.0
        for i, (a_u, a_v) in enumerate(zip(u_samples, v_samples)):
            if parameter_bc is not None:
                parameter_bc(params[i])
            f_u[i], f_v[i] = sampler.rhs(0.0, a_u, a_v)
        return f_u, f_v

    # -- quadratic (zero-pressure) part ----------------------------------
    if parameter_samples is None:
        joint_samples = np.hstack((a_u_samples, a_v_samples))
    else:
        joint_samples = np.hstack((a_u_samples, a_v_samples,
                                   parameter_samples[:, None]))
    f_u, f_v = sample_all(a_u_samples, a_v_samples, parameter_samples)
    quad_u = _fit_quadratic(joint_samples, f_u)
    quad_v = _fit_quadratic(joint_samples, f_v)

    # Hold-out exactness check on perturbed on-attractor states.
    h_base = min(a_u_samples.shape[0], holdout_samples)
    pick = rng.choice(a_u_samples.shape[0], size=h_base, replace=False)
    noise_u = np.maximum(a_u_samples.std(axis=0), 1e-3) * 0.25
    noise_v = np.maximum(a_v_samples.std(axis=0), 1e-3) * 0.25
    hold_u = a_u_samples[pick] + noise_u * rng.standard_normal((h_base, r_u))
    hold_v = a_v_samples[pick] + noise_v * rng.standard_normal((h_base, r_v))
    if parameter_samples is None:
        hold_p = None
    else:
        noise_p = max(float(parameter_samples.std()), 1e-3) * 0.25
        hold_p = parameter_samples[pick] + noise_p * rng.standard_normal(
            h_base)
    h_ref_u, h_ref_v = sample_all(hold_u, hold_v, hold_p)
    # Per-output reference scale over the holdout rows: a per-row norm
    # makes a vanishing component (e.g. the noise-level v of the Stokes
    # case) an arbitrarily small denominator and floors the check at its
    # roundoff; the max-over-rows scale keeps the machine-precision
    # non-quadraticity detector without that artifact.
    scale_u = max(float(np.linalg.norm(h_ref_u, axis=1).max()), 1e-30)
    scale_v = max(float(np.linalg.norm(h_ref_v, axis=1).max()), 1e-30)
    holdout_residual = 0.0
    for i in range(h_base):
        a = np.concatenate((hold_u[i], hold_v[i]))
        if hold_p is not None:
            a = np.append(a, hold_p[i])
        for quad, ref, scale in ((quad_u, h_ref_u[i], scale_u),
                                 (quad_v, h_ref_v[i], scale_v)):
            holdout_residual = max(
                holdout_residual,
                float(np.linalg.norm(quad(a) - ref)) / scale)
    if holdout_residual > holdout_tolerance:
        raise RuntimeError(
            "quadratic fit failed the held-out exactness check "
            f"(max relative error {holdout_residual:.3e} > "
            f"{holdout_tolerance:.1e}): the zero-pressure reduced rhs is "
            "not quadratic in the coefficients for this configuration")

    # -- pressure basis and closure --------------------------------------
    h0_u = np.zeros(r_u)
    h0_v = np.zeros(r_v)
    H_u = np.zeros((r_u, n_a))
    H_v = np.zeros((r_v, n_a))
    K = np.zeros((0, n_a + 1))
    closure_residual = 0.0

    external_basis = pressure_basis is not None
    if external_basis:
        if pressure_snapshots is not None:
            raise ValueError(
                "pressure_basis and pressure_snapshots are alternatives "
                "(an external basis skips the internal POD and the "
                "affine closure map)")
        pressure_modes = np.asarray(pressure_basis[0], dtype=np.float64)
        pressure_mean = np.asarray(pressure_basis[1], dtype=np.float64)
        if pressure_modes.ndim != 2 or pressure_modes.shape[1] != n_cells \
                or pressure_mean.shape != (n_cells,):
            raise ValueError(
                f"pressure_basis must be (modes (r_p, {n_cells}), mean "
                f"({n_cells},)), got {pressure_modes.shape} / "
                f"{pressure_mean.shape}")
        rank_p = pressure_modes.shape[0]
        if rank_p == 0:
            raise ValueError(
                "an external pressure basis must carry at least one mode "
                "(rank 0 silently degenerates to the zero-pressure rhs)")
        energy_p = float("nan")
    else:
        has_closure = pressure_snapshots is not None
        if has_closure != (velocity_snapshots is not None):
            raise ValueError(
                "the affine pressure closure needs paired velocity AND "
                "pressure snapshots (pass both or neither)")
        if has_closure:
            u_snap = np.asarray(velocity_snapshots[0], dtype=np.float64)
            v_snap = np.asarray(velocity_snapshots[1], dtype=np.float64)
            p_snap = np.asarray(pressure_snapshots, dtype=np.float64)
            if u_snap.ndim != 2 or u_snap.shape[1] != n_cells \
                    or v_snap.shape != u_snap.shape \
                    or p_snap.shape != u_snap.shape:
                raise ValueError(
                    "velocity/pressure snapshots must share shape "
                    f"(S, {n_cells}), got u {u_snap.shape}, v "
                    f"{v_snap.shape}, p {p_snap.shape}")
            pod_p = pod_decompose(p_snap)
            if pressure_rank is None:
                reached = np.nonzero(
                    pod_p.cumulative_energy >= pressure_energy)[0]
                rank_p = int(reached[0]) + 1 if reached.size \
                    else pod_p.modes.shape[0]
            else:
                rank_p = min(int(pressure_rank), pod_p.modes.shape[0])
            pressure_modes = pod_p.modes[:rank_p]
            pressure_mean = pod_p.mean
            energy_p = float(pod_p.cumulative_energy[rank_p - 1])
        else:
            pressure_modes = np.zeros((0, n_cells))
            pressure_mean = np.zeros(n_cells)
            rank_p = 0
            energy_p = 0.0

    G_u = np.zeros((r_u, rank_p))
    G_v = np.zeros((r_v, rank_p))
    g0_u = np.zeros(r_u)
    g0_v = np.zeros(r_v)
    if rank_p > 0:
        # Per-mode reduced sources via the b-difference trick, evaluated
        # at the mean-field state a_ref = 0 (any state gives the same
        # difference up to rounding: b^np is pressure-independent).  With
        # a boundary parameter the difference is parameter-independent
        # too (b^np's parameter dependence cancels), so one fixed
        # parameter value pins the BC state for the whole block.
        a_ref_u = np.zeros(r_u)
        a_ref_v = np.zeros(r_v)
        if parameter_bc is not None:
            parameter_bc(parameter_samples[0])
        pressure_field.to_numpy()[:] = 0.0
        ref_u, ref_v = sampler.rhs(0.0, a_ref_u, a_ref_v)
        for k in range(rank_p):
            pressure_field.to_numpy()[:] = pressure_modes[k]
            f_u_k, f_v_k = sampler.rhs(0.0, a_ref_u, a_ref_v)
            G_u[:, k] = f_u_k - ref_u
            G_v[:, k] = f_v_k - ref_v
        pressure_field.to_numpy()[:] = pressure_mean
        f_mean_u, f_mean_v = sampler.rhs(0.0, a_ref_u, a_ref_v)
        g0_u = f_mean_u - ref_u
        g0_v = f_mean_v - ref_v

    if not external_basis and rank_p > 0:
        # Affine coefficient map c(a) = K a + c0 from snapshot pairs
        # (lstsq returns (n_a + 1, r_p); transpose to (r_p, n_a + 1)).
        a_u_snap = (u_snap - pod_u.mean) @ modes_u.T
        a_v_snap = (v_snap - pod_v.mean) @ modes_v.T
        c_snap = (p_snap - pressure_mean) @ pressure_modes.T  # (S, r_p)
        design = np.hstack((np.hstack((a_u_snap, a_v_snap)),
                            np.ones((u_snap.shape[0], 1))))
        coef, *_ = np.linalg.lstsq(design, c_snap, rcond=None)
        pred = design @ coef
        closure_residual = float(
            np.linalg.norm(pred - c_snap) / np.linalg.norm(c_snap))
        K = coef.T
        h0_u = g0_u + G_u @ K[:, -1]
        h0_v = g0_v + G_v @ K[:, -1]
        H_u = G_u @ K[:, :-1]
        H_v = G_v @ K[:, :-1]

    volumes = np.empty(n_cells, dtype=np.float64)
    for cell in range(n_cells):
        volumes[cell] = mesh.cell_volume(cell)

    return ReducedOperatorFit(
        quad_u=quad_u,
        quad_v=quad_v,
        h0_u=h0_u,
        h0_v=h0_v,
        H_u=H_u,
        H_v=H_v,
        pressure_modes=pressure_modes,
        pressure_mean=pressure_mean,
        source_modes_u=G_u,
        source_modes_v=G_v,
        mean_source_u=g0_u,
        mean_source_v=g0_v,
        coefficient_map=K,
        mass_diagonal=rho * volumes,
        modes_u=modes_u,
        modes_v=modes_v,
        mean_u=np.asarray(pod_u.mean, dtype=np.float64),
        mean_v=np.asarray(pod_v.mean, dtype=np.float64),
        rank_pressure=rank_p,
        pressure_cumulative_energy=energy_p,
        holdout_residual=holdout_residual,
        closure_train_residual=closure_residual,
        convection_is_central=central,
    )


@dataclass
class RhieChowFieldFit:
    """Cell-space coefficient fields reconstructing the Rhie-Chow data
    exactly online (Phase 6.5c D2):

    ``diag(a) = a_P(a) + rho V / dt_p`` with ``a_P`` affine in ``a`` and
    ``uHat(a) = u(a) + r_np(a) / diag(a)`` with ``r_np = b^np - A u``
    quadratic in ``a`` (steady assembly, relaxation = 1, dt = 0, zero
    pressure; the transient-diagonal ddt terms cancel algebraically in
    the Euler theta = 1 form).  ``triuI/triuJ`` map the quadratic
    coefficient rows onto ``a_i a_j`` pairs.
    """

    volumes: np.ndarray             # (N,) cell volumes
    diag_u: QuadraticFit            # a_P of the u equation (Q ~ 0)
    diag_v: QuadraticFit
    residual_u: QuadraticFit        # r_np of the u equation
    residual_v: QuadraticFit
    holdout_residual: float = np.inf  # max per-component-scaled field error


def fit_rhie_chow_fields(mesh, mu, rho, pod_u, pod_v, bc_u, bc_v,
                         convection, a_u_samples, a_v_samples,
                         corrections=None, holdout_samples=8,
                         holdout_tolerance=1e-10, rng=None,
                         parameter_samples=None, parameter_bc=None):
    """Fit the Rhie-Chow cell-field coefficients on a fixed-BC case.

    Samples the public assembly chain (``compute_mass_flux`` +
    ``assemble_momentum`` at relaxation = 1, dt = 0, zero pressure) at the
    same graded coefficient samples the reduced fit uses; the per-cell
    diagonal is affine and the per-cell steady residual quadratic, both
    recovered by least squares and verified on perturbed held-out states
    (per-component-scaled relative error below ``holdout_tolerance``
    or RuntimeError; the scale is each component's peak over the
    holdout set).  ``r_np`` needs the full joint quadratic sample count,
    which :func:`make_fit_samples` produces by default.

    ``parameter_samples``/``parameter_bc``: with a known boundary
    parameter, ``parameter_bc(value)`` is called before each sampled
    assembly and the parameter joins the quadratic design (the diagonal
    stays affine, the residual stays quadratic in the augmented vector).
    """
    import pyfvm as fvm  # lazy, like every sampler path in this module

    rng = rng if rng is not None else np.random.default_rng()
    modes_u = np.asarray(pod_u.modes, dtype=np.float64)
    modes_v = np.asarray(pod_v.modes, dtype=np.float64)
    n_cells = modes_u.shape[1]
    a_u_samples = np.asarray(a_u_samples, dtype=np.float64)
    a_v_samples = np.asarray(a_v_samples, dtype=np.float64)
    if (parameter_samples is None) != (parameter_bc is None):
        raise ValueError("parameter_samples and parameter_bc must be "
                         "given together")
    if parameter_samples is not None:
        parameter_samples = np.asarray(parameter_samples,
                                       dtype=np.float64)
        if parameter_samples.shape != (a_u_samples.shape[0],):
            raise ValueError(
                f"parameter_samples must have shape "
                f"({a_u_samples.shape[0]},), got "
                f"{parameter_samples.shape}")
    n_a = a_u_samples.shape[1] + modes_v.shape[0] \
        + (0 if parameter_samples is None else 1)
    n_features = 1 + n_a + n_a * (n_a + 1) // 2
    if a_u_samples.shape[0] < n_features:
        raise ValueError(
            f"too few field-fit samples: {a_u_samples.shape[0]} < "
            f"{n_features} (the joint quadratic design in cell space)")

    velocity = fvm.VectorField(mesh, "fit_velocity")
    flux = fvm.FaceFluxField(mesh, "phi")
    zero_pressure = fvm.ScalarField(mesh, "fit_zero_pressure")

    def sample_fields(u_samples, v_samples, params=None):
        n = u_samples.shape[0]
        diag = [np.empty((n, n_cells)), np.empty((n, n_cells))]
        residual = [np.empty((n, n_cells)), np.empty((n, n_cells))]
        for i, (a_u, a_v) in enumerate(zip(u_samples, v_samples)):
            if parameter_bc is not None:
                parameter_bc(params[i])
            u = pod_u.mean + a_u @ modes_u
            v = pod_v.mean + a_v @ modes_v
            velocity.u.to_numpy()[:] = u
            velocity.v.to_numpy()[:] = v
            fvm.compute_mass_flux(mesh, velocity, rho, bc_u, bc_v, flux)
            for component, bc, field in ((0, bc_u, u), (1, bc_v, v)):
                s = fvm.assemble_momentum(
                    mesh, flux, velocity, mu, convection, bc,
                    component=component, pressure=zero_pressure,
                    bc_p=fvm.BoundaryField(mesh))
                au = np.bincount(s.rows, weights=s.vals * field[s.cols],
                                 minlength=n_cells)
                diag[component][i] = s.diag
                residual[component][i] = s.rhs_no_pressure - au
        return diag, residual

    if parameter_samples is None:
        joint = np.hstack((a_u_samples, a_v_samples))
    else:
        joint = np.hstack((a_u_samples, a_v_samples,
                           parameter_samples[:, None]))
    diag_vals, res_vals = sample_fields(a_u_samples, a_v_samples,
                                        parameter_samples)
    diag_fits = [_fit_quadratic(joint, diag_vals[c]) for c in (0, 1)]
    res_fits = [_fit_quadratic(joint, res_vals[c]) for c in (0, 1)]

    # Held-out per-cell verification on perturbed states.
    h_base = min(a_u_samples.shape[0], holdout_samples)
    pick = rng.choice(a_u_samples.shape[0], size=h_base, replace=False)
    noise_u = np.maximum(a_u_samples.std(axis=0), 1e-3) * 0.25
    noise_v = np.maximum(a_v_samples.std(axis=0), 1e-3) * 0.25
    r_u_s, r_v_s = a_u_samples.shape[1], a_v_samples.shape[1]
    hold_u = a_u_samples[pick] + noise_u * rng.standard_normal((h_base,
                                                                r_u_s))
    hold_v = a_v_samples[pick] + noise_v * rng.standard_normal((h_base,
                                                                r_v_s))
    if parameter_samples is None:
        hold_p = None
    else:
        noise_p = max(float(parameter_samples.std()), 1e-3) * 0.25
        hold_p = parameter_samples[pick] + noise_p * rng.standard_normal(
            h_base)
    d_ref, r_ref = sample_fields(hold_u, hold_v, hold_p)
    # Per-component reference scale (max over cells AND holdout rows):
    # a per-cell denominator lets a noise component with no physical
    # dynamics (e.g. the Stokes v) floor the machine-precision detector
    # at arbitrary relative error; the per-component scale keeps exact
    # fits at machine precision (their local errors already are) while
    # a vanishing component divides by its own peak.
    d_scale = [max(float(np.abs(d_ref[c]).max()), 1e-30) for c in (0, 1)]
    r_scale = [max(float(np.abs(r_ref[c]).max()), 1e-30) for c in (0, 1)]
    holdout_residual = 0.0
    for i in range(h_base):
        a = np.concatenate((hold_u[i], hold_v[i]))
        if hold_p is not None:
            a = np.append(a, hold_p[i])
        for fitter, ref, scale in ((diag_fits[0], d_ref[0][i], d_scale[0]),
                                   (diag_fits[1], d_ref[1][i], d_scale[1]),
                                   (res_fits[0], r_ref[0][i], r_scale[0]),
                                   (res_fits[1], r_ref[1][i], r_scale[1])):
            holdout_residual = max(
                holdout_residual,
                float(np.abs(fitter(a) - ref).max()) / scale)
    if holdout_residual > holdout_tolerance:
        raise RuntimeError(
            "Rhie-Chow field fit failed the held-out exactness check "
            f"(max per-cell relative error {holdout_residual:.3e} > "
            f"{holdout_tolerance:.1e})")

    volumes = np.empty(n_cells, dtype=np.float64)
    for cell in range(n_cells):
        volumes[cell] = mesh.cell_volume(cell)

    return RhieChowFieldFit(volumes=volumes, diag_u=diag_fits[0],
                            diag_v=diag_fits[1], residual_u=res_fits[0],
                            residual_v=res_fits[1],
                            holdout_residual=holdout_residual)


class PrecomputedGalerkinRom:
    """Online POD-Galerkin ROM evaluating only offline-fitted operators.

    The state and POD machinery mirror :class:`rom.galerkin.GalerkinRom`;
    the right-hand side is the fitted quadratic + affine pressure
    closure, so one RK4 stage costs O(r^2)/O(r^3) numpy work with no
    FOM calls.  Fixed-BC operators unless ``parameter_fn`` is given: a
    known boundary parameter (a time-dependent wall value) is evaluated
    at every RK4 stage time and appended to the coefficient vector the
    fitted operators consume (Phase 6 validation cases; the fits must
    have been built with ``parameter_samples``/``parameter_bc``).
    ``advance`` also accepts ``bc_refresh`` so the live BC fields track
    the parameter - :class:`ReducedPressureRom` pressure solves read
    them.
    """

    def __init__(self, fit: ReducedOperatorFit, parameter_fn=None):
        self.fit = fit
        self.modes_u = fit.modes_u
        self.modes_v = fit.modes_v
        self.mean_u = fit.mean_u
        self.mean_v = fit.mean_v
        self.a_u = np.zeros(fit.modes_u.shape[0])
        self.a_v = np.zeros(fit.modes_v.shape[0])
        self._parameter_fn = parameter_fn

    # -- state ----------------------------------------------------------------

    def set_state_from_fields(self, velocity) -> None:
        """Project a full-order velocity field onto the POD bases."""
        self.a_u = ((np.asarray(velocity.u.to_numpy(), dtype=np.float64)
                     - self.mean_u) @ self.modes_u.T)
        self.a_v = ((np.asarray(velocity.v.to_numpy(), dtype=np.float64)
                     - self.mean_v) @ self.modes_v.T)

    def reconstruct_arrays(self):
        """Current reduced state as full-order (u, v) arrays."""
        u = self.mean_u + self.a_u @ self.modes_u
        v = self.mean_v + self.a_v @ self.modes_v
        return u, v

    # -- time integration -----------------------------------------------------

    def _augmented_state(self, a_u, a_v, parameter):
        """Coefficient vector the fitted operators consume: the joint
        state plus the stage-time boundary parameter when the ROM was
        built with ``parameter_fn`` (dimension-checked against the
        fit)."""
        a = np.concatenate((a_u, a_v))
        if parameter is None:
            return a
        a = np.append(a, parameter)
        n_a = self.fit.quad_u.L.shape[1]
        if a.size != n_a:
            raise ValueError(
                f"the fit consumes {n_a} coefficients but got {a.size} "
                "(parameter_fn given for a fit built without a "
                "parameter column?)")
        return a

    def rhs(self, a_u, a_v, parameter=None):
        """Fitted reduced right-hand side (stage-time parameter aware)."""
        a = self._augmented_state(a_u, a_v, parameter)
        f_u = self.fit.quad_u(a) + self.fit.h0_u + self.fit.H_u @ a
        f_v = self.fit.quad_v(a) + self.fit.h0_v + self.fit.H_v @ a
        return f_u, f_v

    def _rk4_step(self, dt, t=None, bc_refresh=None):
        def stage(fraction):
            t_stage = t if t is None else t + fraction * dt
            if bc_refresh is not None:
                bc_refresh(t_stage)
            if self._parameter_fn is None:
                return None
            return self._parameter_fn(t_stage)

        a_u, a_v = self.a_u, self.a_v
        k1u, k1v = self.rhs(a_u, a_v, stage(0.0))
        k2u, k2v = self.rhs(a_u + 0.5 * dt * k1u, a_v + 0.5 * dt * k1v,
                            stage(0.5))
        k3u, k3v = self.rhs(a_u + 0.5 * dt * k2u, a_v + 0.5 * dt * k2v,
                            stage(0.5))
        k4u, k4v = self.rhs(a_u + dt * k3u, a_v + dt * k3v, stage(1.0))
        self.a_u = a_u + dt / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        self.a_v = a_v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

    def advance(self, t0, dt, n_steps, bc_refresh=None, record_every=0):
        """Advance ``n_steps`` explicit RK4 steps (GalerkinRom semantics).

        ``bc_refresh(t)`` runs at every RK4 stage time so the live BC
        fields track the boundary parameter; with ``parameter_fn`` set
        the fitted operators additionally consume the parameter value
        (the fits must have been built for it).  Without either hook
        the operators are the fixed-BC Phase 6.5b scope.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        times, hist_u, hist_v = [], [], []
        for step in range(n_steps):
            self._rk4_step(dt, t0 + step * dt, bc_refresh)
            if record_every > 0 and (step + 1) % record_every == 0:
                times.append(t0 + (step + 1) * dt)
                hist_u.append(self.a_u.copy())
                hist_v.append(self.a_v.copy())
        if record_every > 0:
            if not times or times[-1] < t0 + n_steps * dt:
                times.append(t0 + n_steps * dt)
                hist_u.append(self.a_u.copy())
                hist_v.append(self.a_v.copy())
            return (np.asarray(times), np.asarray(hist_u),
                    np.asarray(hist_v))
        return None


class ReducedPressureRom(PrecomputedGalerkinRom):
    """Precomputed ROM whose per-stage pressure comes from a pressure
    solve on exactly reconstructed Rhie-Chow fields (Phase 6.5c) - the
    6.4 projection mechanism with the momentum assemblies replaced by
    offline-fitted cell-field coefficients.

    Per stage: reconstruct ``diag(a) = a_P(a) + rho V/dt_p`` (affine
    steady diagonal + transient term) and ``uHat(a) = u(a) +
    r_np(a)/diag(a)`` (quadratic steady residual; the ddt terms cancel
    algebraically in the Euler theta = 1 form), then solve the pressure
    equation and apply it through the precomputed per-mode sources
    ``s_p = g_0 + G Psi^T (p - pbar)``.  The pressure response is
    state-dependent and elliptic (no off-manifold extrapolation), unlike
    the affine closure of a pressure-fitted
    :class:`PrecomputedGalerkinRom`, which is documented unstable on the
    shedding limit cycle.

    Two solve modes: the default (M1) runs a full FOM pressure solve via
    ``pyfvm.pressure_projection_from_fields``; ``reduced_solve=True``
    (M2) exports the assembled pressure equation and solves its Galerkin
    projection onto the pressure basis instead,
    ``A_r = Psi A_p Psi^T``, ``b_r = Psi (b_p - A_p pbar)``, a dense
    ``r_p x r_p`` system.  The equation is still solved (the Jacobian
    stays state-dependent through the per-stage operator); only the
    solution space shrinks to ``pbar + span(Psi)``, so the deviation
    from M1 is the projection-solution POD truncation.

    ``dt_p`` mirrors ``GalerkinRom(projection_dt=...)``: the transient
    diagonal ``a_P + rho V/dt_p`` is REQUIRED wherever the steady Central
    diagonal is indefinite (cell Peclet >> 2; must be positive).
    """

    def __init__(self, fit, field_fit, mesh, rho, bc_u, bc_v, bc_p,
                 dt_p, solver_config=None, corrections=None,
                 reduced_solve=False, parameter_fn=None):
        import pyfvm as fvm  # lazy: the online path needs the binding

        super().__init__(fit, parameter_fn=parameter_fn)
        if dt_p is None or dt_p <= 0.0:
            raise ValueError(
                f"dt_p must be positive (transient diagonal), got {dt_p}")
        if reduced_solve and fit.pressure_modes.shape[0] == 0:
            raise ValueError(
                "reduced_solve requires a non-empty pressure basis "
                "(fit.pressure_modes)")
        self._fvm = fvm
        self._mesh = mesh
        self._rho = float(rho)
        self._bc_u = bc_u
        self._bc_v = bc_v
        self._bc_p = bc_p
        self._dt_p = float(dt_p)
        self._corrections = (corrections if corrections is not None
                             else fvm.CorrectionOptions())
        self._solver_cfg = (solver_config if solver_config is not None
                            else fvm.SolverConfig())
        self._field_fit = field_fit
        self._transient = (self._rho * field_fit.volumes / self._dt_p)
        self._velocity = fvm.VectorField(mesh, "rom_velocity")
        self._pressure = fvm.ScalarField(mesh, "rom_pressure")
        self._d_u = fvm.ScalarField(mesh, "rom_d_u")
        self._d_v = fvm.ScalarField(mesh, "rom_d_v")
        self._uhat_u = fvm.ScalarField(mesh, "rom_uhat_u")
        self._uhat_v = fvm.ScalarField(mesh, "rom_uhat_v")
        self._reduced_solve = bool(reduced_solve)
        self._projection_cache = None
        self.last_projection = None

    def rhs(self, a_u, a_v, parameter=None):
        """Reduced right-hand side with a per-stage pressure solve (full
        FOM solve by default; reduced basis projection with
        ``reduced_solve=True``); the optional stage-time boundary
        parameter is appended to the coefficient vector the fitted
        operators consume."""
        fvm = self._fvm
        fields = self._field_fit
        u = self.mean_u + a_u @ self.modes_u
        v = self.mean_v + a_v @ self.modes_v
        a = self._augmented_state(a_u, a_v, parameter)
        diag_u = fields.diag_u(a) + self._transient
        diag_v = fields.diag_v(a) + self._transient
        r_u = fields.residual_u(a)
        r_v = fields.residual_v(a)
        self._velocity.u.to_numpy()[:] = u
        self._velocity.v.to_numpy()[:] = v
        self._d_u.to_numpy()[:] = fields.volumes / diag_u
        self._d_v.to_numpy()[:] = fields.volumes / diag_v
        self._uhat_u.to_numpy()[:] = u + r_u / diag_u
        self._uhat_v.to_numpy()[:] = v + r_v / diag_v
        if self._reduced_solve:
            c = self._reduced_pressure_solve()
        else:
            result = fvm.pressure_projection_from_fields(
                self._mesh, self._rho, self._bc_u, self._bc_v, self._bc_p,
                self._d_u, self._d_v, self._uhat_u, self._uhat_v,
                self._pressure, solver_config=self._solver_cfg,
                corrections=self._corrections)
            self.last_projection = (result.max_imbalance, result.du_max,
                                    result.dv_max, result.dp_max)
            c = self.fit.pressure_modes @ (self._pressure.to_numpy()
                                           - self.fit.pressure_mean)
        f_u = (self.fit.quad_u(a) + self.fit.mean_source_u
               + self.fit.source_modes_u @ c)
        f_v = (self.fit.quad_v(a) + self.fit.mean_source_v
               + self.fit.source_modes_v @ c)
        return f_u, f_v

    def _reduced_pressure_solve(self):
        """Galerkin projection of the assembled pressure equation onto the
        pressure basis (Phase 6.5c M2).

        ``pressure_projection_from_fields(..., return_systems=True)``
        exports the *uneliminated* system, so a pure-Neumann domain works
        unchanged: the mean-centred POD modes are orthogonal to the
        constants and the projection removes the null space on its own,
        while a constant offset stays invisible to ``c`` (``Psi 1 = 0``).
        The exported RHS is the first pass (explicit cross term evaluated
        on the incoming pressure) - the same equation the M1 full solve
        assembles for ``nNonOrthogonalCorrectors = 0``, before its
        reference-cell elimination.
        """
        fvm = self._fvm
        psi = self.fit.pressure_modes
        pbar = self.fit.pressure_mean
        result, system = fvm.pressure_projection_from_fields(
            self._mesh, self._rho, self._bc_u, self._bc_v, self._bc_p,
            self._d_u, self._d_v, self._uhat_u, self._uhat_v,
            self._pressure, solver_config=self._solver_cfg,
            corrections=self._corrections, return_systems=True)
        self.last_projection = (result.max_imbalance, 0.0, 0.0, 0.0)
        n = int(system.n)
        r_p = psi.shape[0]
        rows = np.asarray(system.rows)
        cols = np.asarray(system.cols)
        vals = np.asarray(system.vals)
        b = np.asarray(system.b)
        # The COO pattern is fixed across stages for a given mesh and BC
        # family (the export never eliminates and no coefficient hits an
        # exact zero), so the flattened index map, the gathered Psi^T
        # columns and the pbar gather are cached after the first stage;
        # only the value-dependent multiply and the projections run per
        # stage.  The pattern equality check guards BC or mesh changes.
        cache = self._projection_cache
        if (cache is None or cache[0] != n or cache[1] != rows.size
                or not np.array_equal(cache[2], rows)
                or not np.array_equal(cache[3], cols)):
            psi_t_cols = np.ascontiguousarray(psi.T)[cols]
            idx = (rows[:, None] * r_p + np.arange(r_p)).ravel()
            pbar_cols = pbar[cols]
            weights_buf = np.empty((rows.size, r_p))
            cache = (n, rows.size, rows, cols, idx, psi_t_cols, pbar_cols,
                     weights_buf)
            self._projection_cache = cache
        _, _, _, _, idx, psi_t_cols, pbar_cols, weights_buf = cache
        # The assembled equation is for the ABSOLUTE pressure, so the mean
        # decomposition p = pbar + Psi^T c reaches the RHS as -A_p pbar
        # (one sparse matvec via bincount).
        rhs = psi @ (b - np.bincount(rows, weights=vals * pbar_cols,
                                     minlength=n))
        # Y = A_p Psi^T through a flattened bincount (O(nnz r_p)), then
        # A_r = Psi Y (O(N r_p^2)); the dense r_p x r_p solve replaces the
        # full sparse CG solve of M1.
        np.multiply(vals[:, None], psi_t_cols, out=weights_buf)
        y = np.bincount(idx, weights=weights_buf.ravel(),
                        minlength=n * r_p)
        a_r = psi @ y.reshape(n, r_p)
        c = np.linalg.solve(a_r, rhs)
        self._pressure.to_numpy()[:] = pbar + c @ psi
        return c
