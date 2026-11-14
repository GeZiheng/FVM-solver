"""Velocity-only intrusive POD-Galerkin ROM (Phase 6.3).

Unlike the rest of the ``rom`` package this module drives the FOM
through the pyfvm extension, so ``pyfvm`` is imported lazily inside
:class:`GalerkinRom` -- importing ``rom.galerkin`` alone still works
without the built extension.

Model (see docs/rom.md and the 6.2 operator identity):

- u = mean_u + a_u @ Phi_u, v = mean_v + a_v @ Phi_v with the POD modes
  of :func:`rom.pod.pod_decompose` (row-orthonormal, (r, N) rows);
- the semi-discrete FOM is ``M du/dt = -A_sp(t, u) u + b_sp(t)`` with the
  diagonal mass matrix ``M = rho * diag(V)`` (assembleMomentum theta-split
  identity) and the steady operators exported by ``pyfvm.assemble_momentum``
  (dt = 0, zero pressure field, Neumann pressure BC -- no pressure source);
- projection gives the reduced ODE
  ``M_r da/dt = Phi (b - A u)^T``,  ``M_r = Phi diag(rho V) Phi^T``.

With ``pressure_projection=True`` (Phase 6.4) every RK4 stage first runs
one steady FOM pressure projection on the frozen reconstruction (the
pressure half-step of a single PISO corrector, ``pyfvm.pressure_projection``)
and the residual assembly reads the projected pressure instead of the zero
field - the quantified candidate against the structural pressure leakage.

``M_r`` is a small dense SPD matrix (POD modes are NOT mass-orthogonal);
it is cached once and solved with ``numpy.linalg.solve`` per residual
evaluation.

The residual reconstruction cannot satisfy the boundary values exactly
(the POD subspace does not contain the boundary correction); boundary
effects enter through the assembled RHS, the standard Galerkin-ROM
behaviour.  Time-dependent boundary conditions are refreshed at every
RK4 stage time through a caller-supplied ``bc_refresh(t)`` callback.
"""

from __future__ import annotations

import numpy as np


class GalerkinRom:
    """Online velocity-only POD-Galerkin ROM advanced with explicit RK4.

    Args:
        mesh: pyfvm mesh (any ``Mesh`` binding).
        mu: dynamic viscosity (matches ``assemble_momentum``).
        rho: density (mass matrix weight; the steady assembly itself is
            rho-independent).
        pod_u, pod_v: :class:`rom.pod.PodResult` for the u/v snapshots
            (use all modes; truncate by slicing ``result.modes[:r]`` and
            passing a lightweight substitute, or just build the ROM from
            truncated results via ``pod_decompose`` on the same data).
        bc_u, bc_v: pyfvm ``BoundaryField`` for the two momentum
            components (time-dependent values may be updated by
            ``bc_refresh``).
        convection: pyfvm ``ConvectionScheme`` value (passed through to
            the assembly).
        corrections: optional pyfvm ``CorrectionOptions``.
        pressure: optional pyfvm ``ScalarField`` read by every residual
            evaluation (oracle/diagnostic hook: a velocity-only ROM
            leaves it at zero and absorbs the resulting O(t) pressure
            leakage; supplying a pressure field -- e.g. the exact one on
            manufactured cases -- isolates and quantifies that leakage).
            The field is read live, so a ``bc_refresh`` callback may
            update it at stage times.  Mutually exclusive with
            ``pressure_projection``.
        pressure_projection: when True, every stage runs one predictive
            FOM pressure projection on the frozen reconstruction before
            the assembly (Phase 6.4 quantified candidate): the projected
            pressure is written into the internal pressure field and read
            by that stage's residual, mirroring the oracle timing.  The
            corrected velocity is discarded (the ROM state stays the
            coefficient vector).  Cannot be combined with an injected
            ``pressure`` field.
        pressure_solver_config: optional pyfvm ``SolverConfig`` for the
            per-stage pressure solve (None -> the binding default).
        bc_p: optional pyfvm ``BoundaryField`` for the pressure equation
            (None -> a fresh all-Neumann field, the previous behaviour;
            used by the projection and by every residual assembly).
        projection_dt: optional positive float. None (default) keeps the
            steady projection operator (the validated Taylor-Green path).
            A positive dt assembles the projection momentum with the
            transient diagonal a_P + rho*V/dt and the frozen state as the
            old velocity, so uHat = u_stage + r_s/(a_P + rho*V/dt) - a
            damped residual step. Use it where the steady operator
            diagonal is indefinite (cell Peclet >> 2 with a Central
            scheme, e.g. the coarse shedding mesh).

    Attributes:
        last_projection: ``(max_imbalance, du_max, dv_max, dp_max)`` of the
            most recent per-stage projection (None until one has run;
            diagnostic visibility only).
    """

    def __init__(self, mesh, mu, rho, pod_u, pod_v, bc_u, bc_v,
                 convection, corrections=None, pressure=None,
                 pressure_projection=False, pressure_solver_config=None,
                 bc_p=None, projection_dt=None):
        import pyfvm as fvm  # lazy: keeps `import rom.galerkin` extension-free

        if corrections is None:
            corrections = fvm.CorrectionOptions()

        if pod_u.modes.shape[1] != pod_v.modes.shape[1]:
            raise ValueError(
                "pod_u and pod_v must share the mesh (mode width "
                f"{pod_u.modes.shape[1]} vs {pod_v.modes.shape[1]})")
        if mesh.cell_count() != pod_u.modes.shape[1]:
            raise ValueError(
                "POD mode width must equal the mesh cell count "
                f"({pod_u.modes.shape[1]} vs {mesh.cell_count()})")
        if pressure_projection and pressure is not None:
            raise ValueError(
                "pressure_projection and an injected pressure field are "
                "mutually exclusive: the projection computes the per-stage "
                "pressure itself, an oracle hook would be overwritten")
        if projection_dt is not None and projection_dt <= 0.0:
            raise ValueError(
                f"projection_dt must be positive, got {projection_dt}")
        self._fvm = fvm
        self._mesh = mesh
        self._mu = float(mu)
        self._rho = float(rho)
        self._convection = convection
        self._corrections = corrections
        self._bc_u = bc_u
        self._bc_v = bc_v
        self._bc_refresh = None

        self.modes_u = np.asarray(pod_u.modes, dtype=np.float64)
        self.modes_v = np.asarray(pod_v.modes, dtype=np.float64)
        self.mean_u = np.asarray(pod_u.mean, dtype=np.float64)
        self.mean_v = np.asarray(pod_v.mean, dtype=np.float64)
        self.n_cells = self.modes_u.shape[1]

        # FOM working state: velocity, pressure (zero unless injected),
        # persistent flux.  With an injected pressure field the caller
        # owns it; the assembly still uses the configured bc_p.  In
        # projection mode the internal pressure field receives the
        # per-stage projection instead.
        self._velocity = fvm.VectorField(mesh, "rom_velocity")
        self._pressure = (pressure if pressure is not None
                          else fvm.ScalarField(mesh, "rom_pressure"))
        self._flux = fvm.FaceFluxField(mesh, "rom_phi")
        self._bc_p = bc_p if bc_p is not None else fvm.BoundaryField(mesh)
        self._projection = bool(pressure_projection)
        self._pressure_solver_config = pressure_solver_config
        self._projection_dt = projection_dt
        self._scratch = fvm.VectorField(mesh, "rom_projection_scratch")
        self.last_projection = None

        # Diagonal mass matrix and the reduced mass matrices (dense SPD).
        volumes = np.empty(self.n_cells, dtype=np.float64)
        for cell in range(self.n_cells):
            volumes[cell] = mesh.cell_volume(cell)
        self._mass_diagonal = self._rho * volumes
        self._mass_u = (self.modes_u * self._mass_diagonal) @ self.modes_u.T
        self._mass_v = (self.modes_v * self._mass_diagonal) @ self.modes_v.T

        self.a_u = np.zeros(self.modes_u.shape[0], dtype=np.float64)
        self.a_v = np.zeros(self.modes_v.shape[0], dtype=np.float64)

    # -- state ----------------------------------------------------------------

    def set_state_from_fields(self, velocity) -> None:
        """Project a full-order velocity field onto the POD bases.

        Least squares with row-orthonormal modes reduces to the plain
        inner products ``a = (u - mean) @ modes.T`` (exact round-trip for
        a full-rank basis).
        """
        self.a_u = ((np.asarray(velocity.u.to_numpy(), dtype=np.float64)
                     - self.mean_u) @ self.modes_u.T)
        self.a_v = ((np.asarray(velocity.v.to_numpy(), dtype=np.float64)
                     - self.mean_v) @ self.modes_v.T)

    def reconstruct_arrays(self):
        """Current reduced state as full-order (u, v) arrays."""
        u = self.mean_u + self.a_u @ self.modes_u
        v = self.mean_v + self.a_v @ self.modes_v
        return u, v

    def write_state(self, velocity) -> None:
        """Write the current reduced state into a pyfvm VectorField."""
        u, v = self.reconstruct_arrays()
        velocity.u.to_numpy()[:] = u
        velocity.v.to_numpy()[:] = v

    # -- residual and time integration ----------------------------------------

    def _write_reconstruction(self) -> None:
        u, v = self.reconstruct_arrays()
        self._velocity.u.to_numpy()[:] = u
        self._velocity.v.to_numpy()[:] = v

    def _assemble_residual(self, system, field, modes):
        """Phi (b - A u)^T for one component via a COO mat-vec."""
        residual = (system.b
                    - np.bincount(system.rows,
                                  weights=system.vals * field[system.cols],
                                  minlength=self.n_cells))
        return modes @ residual

    def rhs(self, t, a_u, a_v):
        """Reduced right-hand side f(t, a) (one FOM residual evaluation).

        Reconstructs the field, refreshes time-dependent BCs, recomputes
        the mass flux and reassembles both steady momentum systems.

        With ``pressure_projection=True`` the stage first runs one
        predictive FOM pressure projection on the frozen reconstruction
        (mirroring the oracle timing: after the mass flux, before the
        assembly), so the assembly reads the projected pressure.  The
        corrected velocity is written to a scratch field and discarded;
        the flux is then restored to the reconstruction's own mass flux,
        so the only difference vs the zero/oracle pathways is the
        pressure field content.  With ``projection_dt`` set the
        projection assembles the transient diagonal a_P + rho*V/dt
        (frozen state as u_old, damped residual step) instead of the
        steady operator - required where the steady diagonal is
        indefinite (cell Peclet >> 2).  Each projection costs two
        momentum assemblies plus one pressure solve per stage;
        ``last_projection`` keeps the diagnostics of the most recent one
        (the last stage of an RK4 step).

        Note: stores ``a_u``/``a_v`` on the object (the reconstruction is
        written from them); ``advance`` leaves the accepted end state in
        place, but calling ``rhs`` directly with different arguments
        clobbers ``self.a_u``/``self.a_v``.
        """
        self.a_u, self.a_v = a_u, a_v
        self._write_reconstruction()
        if self._bc_refresh is not None:
            self._bc_refresh(t)
        fvm = self._fvm
        fvm.compute_mass_flux(self._mesh, self._velocity, self._rho,
                              self._bc_u, self._bc_v, self._flux,
                              self._corrections)
        if self._projection:
            # scratch copy of the reconstruction; the corrected velocity is
            # discarded (the ROM state must stay the coefficient vector).
            self._scratch.u.to_numpy()[:] = self._velocity.u.to_numpy()
            self._scratch.v.to_numpy()[:] = self._velocity.v.to_numpy()
            # The binding takes SolverConfig by value, so a None config
            # must be resolved here (passing None would raise TypeError).
            solver_cfg = (self._pressure_solver_config
                          if self._pressure_solver_config is not None
                          else fvm.SolverConfig())
            projection_dt = (0.0 if self._projection_dt is None
                             else self._projection_dt)
            result = fvm.pressure_projection(
                self._mesh, self._rho, self._mu, self._convection,
                self._bc_u, self._bc_v, self._bc_p, self._scratch,
                self._pressure, self._flux, solver_config=solver_cfg,
                dt=projection_dt,
                corrections=self._corrections)
            self.last_projection = (result.max_imbalance, result.du_max,
                                    result.dv_max, result.dp_max)
            # Restore the reconstruction's mass flux so the only difference
            # vs the oracle pathway is the pressure field content (plan D1
            # step 3).
            fvm.compute_mass_flux(self._mesh, self._velocity, self._rho,
                                  self._bc_u, self._bc_v, self._flux,
                                  self._corrections)
        u = self._velocity.u.to_numpy()
        v = self._velocity.v.to_numpy()
        kwargs = dict(mu=self._mu, scheme=self._convection,
                      pressure=self._pressure, bc_p=self._bc_p,
                      rho=self._rho, corrections=self._corrections)
        sys_u = fvm.assemble_momentum(self._mesh, self._flux,
                                      self._velocity, component=0,
                                      bc=self._bc_u, **kwargs)
        sys_v = fvm.assemble_momentum(self._mesh, self._flux,
                                      self._velocity, component=1,
                                      bc=self._bc_v, **kwargs)
        f_u = np.linalg.solve(self._mass_u,
                              self._assemble_residual(sys_u, u, self.modes_u))
        f_v = np.linalg.solve(self._mass_v,
                              self._assemble_residual(sys_v, v, self.modes_v))
        return f_u, f_v

    def _rk4_step(self, t, dt):
        a_u, a_v = self.a_u, self.a_v
        k1u, k1v = self.rhs(t, a_u, a_v)
        k2u, k2v = self.rhs(t + 0.5 * dt, a_u + 0.5 * dt * k1u,
                            a_v + 0.5 * dt * k1v)
        k3u, k3v = self.rhs(t + 0.5 * dt, a_u + 0.5 * dt * k2u,
                            a_v + 0.5 * dt * k2v)
        k4u, k4v = self.rhs(t + dt, a_u + dt * k3u, a_v + dt * k3v)
        self.a_u = a_u + dt / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        self.a_v = a_v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

    def advance(self, t0, dt, n_steps, bc_refresh=None, record_every=0):
        """Advance ``n_steps`` explicit RK4 steps of size ``dt``.

        Args:
            t0: start time (BC callback argument).
            dt: reduced time step (independent of the FOM dt).
            n_steps: number of RK4 steps.
            bc_refresh: optional ``bc_refresh(t)`` called at every stage
                time before each residual evaluation (mirrors the
                per-step BC refresh of the transient case scripts).
            record_every: when > 0, record ``(times, A_u, A_v)`` every
                ``record_every`` steps (plus the final state); otherwise
                return None.

        In projection mode every RK4 stage additionally performs one FOM
        pressure projection (four per accepted step); see :meth:`rhs`.

        Returns:
            ``(times, a_u_history, a_v_history)`` or None.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        self._bc_refresh = bc_refresh
        times, hist_u, hist_v = [], [], []
        for step in range(n_steps):
            self._rk4_step(t0 + step * dt, dt)
            if record_every > 0 and (step + 1) % record_every == 0:
                times.append(t0 + (step + 1) * dt)
                hist_u.append(self.a_u.copy())
                hist_v.append(self.a_v.copy())
        self._bc_refresh = None
        if record_every > 0:
            if not times or times[-1] < t0 + n_steps * dt:
                times.append(t0 + n_steps * dt)
                hist_u.append(self.a_u.copy())
                hist_v.append(self.a_v.copy())
            return (np.asarray(times), np.asarray(hist_u),
                    np.asarray(hist_v))
        return None
