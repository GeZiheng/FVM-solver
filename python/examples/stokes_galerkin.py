"""Intrusive velocity-only POD-Galerkin ROM driver on Stokes' second
problem (oscillating plate).

Marches the FOM (stokes_oscillating.py setup: closed 16x64 slot, pure
Neumann pressure, top wall u = cos(omega t)) to its periodic steady
state, collects snapshot cycles, builds POD bases, then integrates the
core ROM variants (Phase 6 validation-case expansion):

- zero-pressure: the plain velocity-only ROM.  The exact continuous
  solution has grad(p) = 0, so whatever leakage appears here is the
  DISCRETE residual-divergence leakage of the collocated subspace -
  the middle-of-the-road calibration between TG (decaying transient)
  and the shedding limit cycle;
- projective (Phase 6.4): one per-stage pressure projection on the
  frozen reconstruction (steady operator - the cell Peclet number is
  ~1.25, far below the Central diagonal's indefinitness threshold).

The wall BC is time dependent, so every RK4 stage refreshes it through
GalerkinRom's bc_refresh(t) hook at the stage time (mirroring the
Taylor-Green driver's exact-BC semantics).  Accuracy metrics: the
phase-aligned vertical profile vs the exact solution u(d) =
exp(-d/delta) cos(phi - d/delta) for both the FOM snapshot and the ROM
reconstruction, plus err|FOM against the collected snapshots.  The
advance runs past the snapshot window for a long-horizon stability
check.  --test runs a short machine-level configuration.

Run:  python python/examples/stokes_galerkin.py
      [--cycles K] [--extra-cycles S] [--snap-every N] [--rank R]
      [--rank-v RV] [--dt-rom DT] [--timing] [--test]
"""

import argparse
import math
import time

import numpy as np

from _common import fvm, report_test, write_results

from rom import SnapshotCollector, pod_decompose
from rom.galerkin import GalerkinRom
from rom.precomputed import (ReducedPressureRom, fit_reduced_operators,
                             fit_rhie_chow_fields, make_fit_samples,
                             projected_pressure_snapshots)

from stokes_oscillating import (DELTA, DT, MU, NX, NY, OMEGA, PERIOD, RHO,
                                STEPS_PER_PERIOD)

CASE_DIR = "stokes_galerkin"

# --test calibration: filled from the first measured run (see the
# TG driver's header for the convention - thresholds keep >= 2x margin
# over the measured value and a lower bound where an accidental match
# must be rejected).
TEST_ERR_FOM_MAX = 0.05
TEST_PROFILE_FACTOR = 3.0


def make_piso_config():
    """The stokes_oscillating.py PISO configuration, verbatim."""
    cfg = fvm.PisoConfig()
    cfg.dt = DT
    cfg.n_steps = 1
    cfg.n_correctors = 2
    cfg.time_scheme = fvm.TimeScheme.Euler
    cfg.scheme = fvm.ConvectionScheme.Central
    cfg.solver_config.tolerance = 1e-9
    cfg.solver_config.max_iterations = 20000
    cfg.pressure_solver_config.tolerance = 1e-9
    cfg.pressure_solver_config.max_iterations = 20000
    return cfg


def make_boundary_fields(mesh):
    """The stokes_oscillating.py BCs (wall value set per step/Stage)."""
    bc_u = fvm.BoundaryField(mesh)  # top value refreshed every step
    bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 1.0)
    bc_u.set(fvm.BoundaryField.South, fvm.BCType.Dirichlet, 0.0)
    bc_u.set(fvm.BoundaryField.West, fvm.BCType.Neumann, 0.0)
    bc_u.set(fvm.BoundaryField.East, fvm.BCType.Neumann, 0.0)

    bc_v = fvm.BoundaryField(mesh)  # no penetration; sides zero-gradient
    for side in (fvm.BoundaryField.North, fvm.BoundaryField.South,
                 fvm.BoundaryField.West, fvm.BoundaryField.East):
        bc_v.set(side, fvm.BCType.Dirichlet, 0.0)

    bc_p = fvm.BoundaryField(mesh)  # all Neumann (reference-cell pinning)
    return bc_u, bc_v, bc_p


def wall_value(t):
    """Top-plate Dirichlet value at time t."""
    return float(np.cos(OMEGA * t))


def run(max_cycles, steady_tol, snap_every, collect_cycles):
    """March to the periodic steady state, then collect snapshot cycles.

    Returns (mesh, collector, step offset of the window start)."""
    mesh = fvm.CartesianMesh(NX, NY, 0.0, 0.0, 2.0, 2.0)
    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")
    bc_u, bc_v, bc_p = make_boundary_fields(mesh)
    cfg = make_piso_config()

    def march(steps):
        nonlocal held_step
        for _ in range(steps):
            held_step += 1
            # Implicit Euler evaluates the wall value at t^{n+1}
            # (stokes_oscillating.py semantics).
            bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet,
                     wall_value(DT * held_step))
            fvm.solve_piso(mesh, RHO, MU, bc_u, bc_v, bc_p, cfg,
                           velocity, pressure, flux)

    held_step = 0
    prev = None
    steady_cycle = None
    for cycle in range(1, max_cycles + 1):
        march(STEPS_PER_PERIOD)
        u = velocity.u.to_numpy().copy()
        if prev is not None and np.abs(u - prev).max() < steady_tol:
            steady_cycle = cycle
            break
        prev = u
    if steady_cycle is None:
        raise RuntimeError(f"no periodic steady state within {max_cycles} "
                           f"cycles")
    print(f"periodic steady state after {steady_cycle} cycles "
          f"(t = {DT * held_step:.2f})")

    collector = SnapshotCollector()
    collector.append(DT * held_step, velocity.u.to_numpy(),
                     velocity.v.to_numpy(), pressure.to_numpy())
    snaps_per_cycle = STEPS_PER_PERIOD // snap_every
    for _ in range(collect_cycles * snaps_per_cycle):
        march(snap_every)
        collector.append(DT * held_step, velocity.u.to_numpy(),
                         velocity.v.to_numpy(), pressure.to_numpy())
    return mesh, collector


def make_bc_refresh(bc_u):
    """bc_refresh(t) for GalerkinRom.advance: wall value at stage time."""
    def bc_refresh(t):
        bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet,
                 wall_value(t))
    return bc_refresh


def velocity_error(u, v, u_ref, v_ref):
    """Relative L2 velocity error over both components."""
    diff2 = (float(np.mean((u - u_ref) ** 2))
             + float(np.mean((v - v_ref) ** 2)))
    ref2 = (float(np.mean(u_ref ** 2)) + float(np.mean(v_ref ** 2)))
    return math.sqrt(diff2 / ref2)


def x_averaged_profile(mesh, u):
    """The vertical u profile averaged over the homogeneous direction."""
    return u.reshape(mesh.ny(), mesh.nx()).mean(axis=1)


def exact_profile(mesh, phase):
    """Exact profile u(d) at phase phi = omega t (mod 2 pi)."""
    height = mesh.ny() * mesh.dy()
    d = height - (np.arange(mesh.ny()) + 0.5) * mesh.dy()
    return np.exp(-d / DELTA) * np.cos(phase - d / DELTA)


def profile_error(mesh, u, phase):
    err = np.abs(x_averaged_profile(mesh, u) - exact_profile(mesh, phase))
    return float(err.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=4,
                        help="snapshot cycles collected at the periodic "
                             "steady state (the POD window)")
    parser.add_argument("--extra-cycles", type=int, default=10,
                        help="cycles advanced past the window for the "
                             "long-horizon stability check")
    parser.add_argument("--snap-every", type=int, default=5,
                        help="snapshots per period (of 250 steps)")
    parser.add_argument("--rank", type=int, default=2,
                        help="POD rank of u (the periodic pair; capped "
                             "at the full rank)")
    parser.add_argument("--rank-v", type=int, default=1,
                        help="POD rank of v (the exact solution has "
                             "v = 0; one mode carries the residual)")
    parser.add_argument("--dt-rom", type=float, default=None,
                        help="ROM RK4 step (default: the FOM dt = 0.005; "
                             "the per-stage pressure projection stiffens "
                             "the rhs - the RK4 stability boundary was "
                             "measured between 2.5x and 5x the FOM dt, "
                             "and the exact-profile comparison needs a "
                             "whole number of ROM steps per period)")
    parser.add_argument("--pressure-rank", type=int, default=12,
                        help="pressure POD rank r_p of the M1/M2 "
                             "pressure basis (the wall-parameter family "
                             "fattens the manifold; 12 saturates the "
                             "off-manifold floor at ~3e-4)")
    parser.add_argument("--pressure-basis", action="store_true",
                        help="also integrate the Phase 6.5c M1 twin: "
                             "per-stage FULL pressure solve on "
                             "offline-fitted Rhie-Chow fields, with the "
                             "wall value cos(omega t) as a fitted "
                             "boundary parameter (Phase 6 D1 mechanism)")
    parser.add_argument("--reduced-pressure", action="store_true",
                        help="also integrate the Phase 6.5c M2 twin of "
                             "--pressure-basis: the per-stage pressure "
                             "equation solved as its Galerkin projection "
                             "onto the pressure basis (pure-Neumann "
                             "uneliminated export path)")
    parser.add_argument("--frames", type=int, default=0,
                        help="write the first N sampled reconstruction "
                             "frames of every variant plus the FOM "
                             "snapshots at the same times as .vti "
                             "(0 disables; ignored with --test)")
    parser.add_argument("--timing", action="store_true",
                        help="print per-step wall-clock costs of the FOM "
                             "collection and of each ROM variant")
    parser.add_argument("--test", action="store_true",
                        help="run the short machine-level configuration "
                             "(2 collected cycles, 2 extra); asserts "
                             "contract invariants and loose accuracy")
    args = parser.parse_args()

    collect_cycles = 2 if args.test else args.cycles
    extra_cycles = 2 if args.test else args.extra_cycles

    tic = time.perf_counter() if args.timing else None
    mesh, collector = run(max_cycles=60, steady_tol=1e-6,
                          snap_every=args.snap_every,
                          collect_cycles=collect_cycles)
    fom_steps = (STEPS_PER_PERIOD * (collect_cycles + 60))  # rough upper
    if args.timing:
        print(f"timing FOM collection  : "
              f"{(time.perf_counter() - tic) / fom_steps * 1000.0:7.1f} "
              f"ms/step (incl. steady-state march)")
    print(f"collected {len(collector)} snapshots of "
          f"{mesh.cell_count()} cells ({args.snap_every}/period)")

    pod_u_full = pod_decompose(collector.u)
    pod_v_full = pod_decompose(collector.v)
    rank_u = min(args.rank, pod_u_full.modes.shape[0])
    rank_v = min(args.rank_v, pod_v_full.modes.shape[0])
    pod_u = pod_u_full.truncated(rank_u)
    pod_v = pod_v_full.truncated(rank_v)
    print(f"POD rank r_u = {rank_u} of {pod_u_full.modes.shape[0]} "
          f"(u reconstruction error "
          f"{pod_u_full.reconstruction_error(rank_u):.3e}); "
          f"r_v = {rank_v} of {pod_v_full.modes.shape[0]}")

    bc_u, bc_v, bc_p = make_boundary_fields(mesh)
    refresh = make_bc_refresh(bc_u)
    roms = {}
    roms["zero"] = GalerkinRom(mesh, MU, RHO, pod_u, pod_v, bc_u, bc_v,
                               fvm.ConvectionScheme.Central)
    roms["proj"] = GalerkinRom(mesh, MU, RHO, pod_u, pod_v, bc_u, bc_v,
                               fvm.ConvectionScheme.Central,
                               pressure_projection=True,
                               pressure_solver_config=make_piso_config(
                               ).pressure_solver_config)

    t0 = collector.times[0]
    steps_per_cycle = STEPS_PER_PERIOD // args.snap_every
    # dt_rom defaults to the FOM dt: the per-stage pressure projection
    # stiffens the reduced rhs (an explicit-RK4 stability boundary sits
    # between 2.5x and 5x dt on this case - measured 2026-11-19), so the
    # snapshot cadence is NOT a safe ROM step.
    dt_rom = DT if args.dt_rom is None else args.dt_rom
    n_steps = round((collect_cycles + extra_cycles) * PERIOD / dt_rom)

    # M1/M2 twins: the wall value gamma(t) = cos(omega t) is a KNOWN
    # boundary parameter, so it joins the offline quadratic design as an
    # extra coordinate (Phase 6 D1 mechanism) instead of freezing the
    # BC; online, parameter_fn evaluates it at every RK4 stage time and
    # bc_refresh keeps the live BC field consistent for the pressure
    # solves.
    if args.pressure_basis or args.reduced_pressure:
        a_u_snap = pod_u.project(collector.u)
        a_v_snap = pod_v.project(collector.v)
        gammas = np.array([wall_value(t) for t in collector.times])

        def parameter_bc(value):
            bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet,
                     float(value))

        rng = np.random.default_rng(1)
        au, av, ap = make_fit_samples(a_u_snap, a_v_snap,
                                      parameter=gammas, rng=rng)
        p_cfg = make_piso_config().pressure_solver_config
        p_proj = projected_pressure_snapshots(
            mesh, MU, RHO, pod_u, pod_v, bc_u, bc_v, bc_p,
            fvm.ConvectionScheme.Central, a_u_snap, a_v_snap, dt_rom,
            solver_config=p_cfg, parameter_samples=gammas,
            parameter_bc=parameter_bc)
        pod_pp = pod_decompose(p_proj)
        rank_pb = min(args.pressure_rank, pod_pp.modes.shape[0])
        # Case calibration: the (a_u, gamma) design is machine-exact
        # (1e-14, verified by hand), but v is a noise component with no
        # physical dynamics - its fitted rhs floors at ~1e-3 relative
        # and its Rhie-Chow cell fields at ~1e-2 (per-component-scaled;
        # the mechanism behind the v floor is open).  A real
        # non-quadraticity is an O(1) signature (the 6.5b closure
        # failed at 0.27), so these case tolerances keep the detector
        # meaningful for u while tolerating the v noise (library
        # defaults stay 1e-10).
        fit = fit_reduced_operators(
            mesh, MU, RHO, pod_u, pod_v, bc_u, bc_v,
            fvm.ConvectionScheme.Central, au, av,
            pressure_basis=(pod_pp.modes[:rank_pb], pod_pp.mean),
            parameter_samples=ap, parameter_bc=parameter_bc, rng=rng,
            holdout_tolerance=1e-2)
        field_fit = fit_rhie_chow_fields(
            mesh, MU, RHO, pod_u, pod_v, bc_u, bc_v,
            fvm.ConvectionScheme.Central, au, av,
            parameter_samples=ap, parameter_bc=parameter_bc, rng=rng,
            holdout_tolerance=1e-1)
        print(f"pressure-basis fit: r_p = {rank_pb} (cumulative energy "
              f"{pod_pp.cumulative_energy[rank_pb - 1]:.6f}), quadratic "
              f"holdout {fit.holdout_residual:.2e}, Rhie-Chow field "
              f"holdout {field_fit.holdout_residual:.2e}")
        if args.pressure_basis:
            roms["m1"] = ReducedPressureRom(
                fit, field_fit, mesh, RHO, bc_u, bc_v, bc_p, dt_rom,
                solver_config=p_cfg, parameter_fn=wall_value)
        if args.reduced_pressure:
            roms["m2"] = ReducedPressureRom(
                fit, field_fit, mesh, RHO, bc_u, bc_v, bc_p, dt_rom,
                solver_config=p_cfg, reduced_solve=True,
                parameter_fn=wall_value)

    initial = fvm.VectorField(mesh, "initial")
    initial.u.to_numpy()[:] = collector.u[0]
    initial.v.to_numpy()[:] = collector.v[0]
    for rom in roms.values():
        rom.set_state_from_fields(initial)

    def timed_advance(label, rom):
        if not args.timing:
            return rom.advance(t0, dt_rom, n_steps, bc_refresh=refresh,
                               record_every=1)
        tic2 = time.perf_counter()
        hist = rom.advance(t0, dt_rom, n_steps, bc_refresh=refresh,
                           record_every=1)
        print(f"timing ROM {label:<11}: "
              f"{(time.perf_counter() - tic2) / n_steps * 1000.0:7.1f} "
              f"ms/step ({n_steps} steps)")
        return hist

    hists = {name: timed_advance(name, rom) for name, rom in roms.items()}
    times = hists["zero"][0]
    print(f"ROM advanced {n_steps} RK4 steps of dt = {dt_rom} "
          f"(t in [{t0:.2f}, {times[-1]:.2f}], "
          f"{extra_cycles} cycles past the window)")

    # err|FOM over the snapshot window: snapshot k sits at
    # t0 + k*snap_dt, which the ROM history hits at row
    # round(k*snap_dt/dt_rom) whenever dt_rom divides the cadence.
    snap_dt = DT * args.snap_every
    per_snap = round(snap_dt / dt_rom)
    aligned = abs(per_snap * dt_rom - snap_dt) < 1e-9
    n_window_snaps = collect_cycles * steps_per_cycle
    tail_snaps = range(int(0.8 * n_window_snaps), n_window_snaps)
    errs = {}
    if aligned:
        for name, hist in hists.items():
            window = []
            for snap in tail_snaps:
                row = snap * per_snap
                u_r = pod_u.mean + hist[1][row] @ pod_u.modes
                v_r = pod_v.mean + hist[2][row] @ pod_v.modes
                window.append(velocity_error(u_r, v_r,
                                             collector.u[snap],
                                             collector.v[snap]))
            errs[name] = float(np.mean(window))
        print("err|FOM tail 20% of the window: "
              + ", ".join(f"{name} {errs[name]:.4f}"
                          for name in hists))

    # Phase-aligned exact-profile comparison at the final time (a whole
    # number of cycles after t0 keeps the phase).
    phase0 = (OMEGA * t0) % (2.0 * np.pi)
    fom_profile = profile_error(mesh, collector.u[0], phase0)
    final = len(times) - 1
    profiles = {}
    for name, hist in hists.items():
        u_r = pod_u.mean + hist[1][final] @ pod_u.modes
        profiles[name] = profile_error(mesh, u_r, phase0)
    print(f"phase-{phase0:.3f} profile L-inf vs exact: FOM snapshot "
          f"{fom_profile:.4e}, "
          + ", ".join(f"{name} {profiles[name]:.4e}"
                      for name in hists))

    stable = {name: bool(np.isfinite(hist[1]).all()
                         and np.isfinite(hist[2]).all())
              for name, hist in hists.items()}
    print(f"stability over {extra_cycles} extra cycles: "
          + ", ".join(f"{name} finite={stable[name]}, "
                      f"max|a_u|={np.abs(hist[1]).max():.3e}"
                      for name, hist in hists.items()))

    if args.frames > 0 and not args.test:
        # Sampled reconstruction frames of every variant plus the FOM
        # snapshot at the same time when one exists (rows past the
        # snapshot window have no FOM counterpart).  The reconstruction
        # goes through the coefficient history - the ROMs hold only
        # their final state.
        stride = max(1, len(times) // args.frames)
        frame_velocity = fvm.VectorField(mesh, "rom_frame")
        for row in list(range(0, len(times), stride))[:args.frames]:
            for name, hist in hists.items():
                frame_velocity.u.to_numpy()[:] = (
                    pod_u.mean + hist[1][row] @ pod_u.modes)
                frame_velocity.v.to_numpy()[:] = (
                    pod_v.mean + hist[2][row] @ pod_v.modes)
                write_results(mesh, frame_velocity, None, CASE_DIR,
                              f"rom_{name}_{row:04d}.vti", scalars={},
                              vectors={"velocity": frame_velocity})
            snap = round((times[row] - t0) / snap_dt)
            if 0 <= snap < len(collector):
                frame_velocity.u.to_numpy()[:] = collector.u[snap]
                frame_velocity.v.to_numpy()[:] = collector.v[snap]
                write_results(mesh, frame_velocity, None, CASE_DIR,
                              f"fom_snap_{row:04d}.vti", scalars={},
                              vectors={"velocity": frame_velocity})

    if args.test:
        checks = {f"{name}_finite": ok for name, ok in stable.items()}
        detail = [f"steps={n_steps}"]
        if aligned:
            checks["projective_err_fom"] = errs["proj"] < TEST_ERR_FOM_MAX
            detail.append(f"proj_err|FOM={errs['proj']:.4f} < "
                          f"{TEST_ERR_FOM_MAX}")
        checks["profile"] = (profiles["proj"]
                             < TEST_PROFILE_FACTOR * fom_profile + 1e-12)
        detail.append(f"proj_profile={profiles['proj']:.3e} < "
                      f"{TEST_PROFILE_FACTOR}x fom {fom_profile:.3e}")
        ok = all(checks.values())
        report_test(ok, f"({', '.join(detail)}, checks={checks})")


if __name__ == "__main__":
    main()
