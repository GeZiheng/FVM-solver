"""POD-Galerkin ROM driver on the backward-facing-step STARTUP
transient (Phase 6 validation-case expansion).

Marches the FOM from rest (parabolic Armaly-convention inlet at
Re = 389, mu = 2/389, Central, PISO Crank-Nicolson) until the
reattachment length plateaus, collecting snapshots along the way.
The startup trajectory decays to a FIXED POINT - the third attractor
type after TG's decaying transient and the shedding limit cycle - on
an L-shaped SU2 mesh with a Dirichlet-p outlet (the BC family TG
lacks, complementing the shedding case on different physics).

Four core variants are integrated from the POD window start:

- zero-pressure GalerkinRom (the leakage baseline);
- projective (Phase 6.4) with the transient diagonal projection_dt =
  dt_rom - the steady Central operator is indefinite here (cell Peclet
  >> 2, the shedding configuration);
- pressure-basis M1 (Phase 6.5c) and its reduced solve M2 - the
  per-stage pressure equation on offline-fitted Rhie-Chow fields.

Metrics: err|FOM along the startup (ROM vs the collected snapshots),
the reattachment length of each variant's final reconstruction vs the
FOM's own, and long-tail stability.  The evidence grid is n = 8 (4480
cells, the shedding-case scale); the literature x_r anchoring was
calibrated at n >= 16, so the accuracy anchor here is the FOM
reference itself.  --test runs a short machine-level configuration.

Run:  python python/examples/bfs_galerkin.py
      [--n N] [--dt DT] [--t-min T] [--t-max T] [--snap-every K]
      [--rank R] [--pressure-rank RP] [--window-frac F] [--timing]
      [--test]
"""

import argparse
import math
import time

import numpy as np

from _common import fvm, read_su2_text, report_test, write_results

from mesh_su2 import l_shaped_bfs_grid
from rom import SnapshotCollector, pod_decompose
from rom.galerkin import GalerkinRom
from rom.precomputed import (ReducedPressureRom, fit_reduced_operators,
                             fit_rhie_chow_fields, make_fit_samples,
                             projected_pressure_snapshots)

from backward_facing_step import (configure_boundary_conditions,
                                  reattachment_length)

RE = 389
MU = 2.0 / RE                # Armaly convention (parabolic inlet)
RHO = 1.0
CASE_DIR = "bfs_galerkin"
XR_CHECK_EVERY = 50          # steps between reattachment checks
XR_PLATEAU_REL = 0.01        # relative drift tolerated on the plateau
XR_PLATEAU_CHECKS = 3        # consecutive checks inside the tolerance

TEST_ERR_FOM_MAX = 0.10      # --test calibration (filled from measured)


def make_piso_config(dt):
    cfg = fvm.PisoConfig()
    cfg.dt = dt
    cfg.n_steps = 1
    cfg.n_correctors = 2
    cfg.time_scheme = fvm.TimeScheme.CrankNicolson
    cfg.scheme = fvm.ConvectionScheme.Central
    cfg.solver_config.tolerance = 1e-9
    cfg.solver_config.max_iterations = 20000
    cfg.pressure_solver_config.tolerance = 1e-9
    cfg.pressure_solver_config.max_iterations = 20000
    return cfg


def run(n, dt, t_min, t_max, snap_every):
    """March the startup transient from rest until x_r plateaus.

    Returns (mesh, collector, xr_fom, t_end)."""
    mesh = read_su2_text(l_shaped_bfs_grid(n),
                         name="backward_facing_step_rom.su2")
    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")
    velocity.u.set_constant(0.0)
    velocity.v.set_constant(0.0)
    pressure.set_constant(0.0)
    flux.set_zero()
    bc_u, bc_v, bc_p = configure_boundary_conditions(mesh, RE)
    cfg = make_piso_config(dt)

    collector = SnapshotCollector()
    collector.append(0.0, velocity.u.to_numpy(), velocity.v.to_numpy(),
                     pressure.to_numpy())

    steps = round(t_max / dt)
    xr_history = []
    plateau_step = None
    for step in range(1, steps + 1):
        fvm.solve_piso(mesh, RHO, MU, bc_u, bc_v, bc_p, cfg,
                       velocity, pressure, flux)
        if step % snap_every == 0:
            collector.append(step * dt, velocity.u.to_numpy(),
                             velocity.v.to_numpy(),
                             pressure.to_numpy())
        if step % XR_CHECK_EVERY == 0:
            try:
                xr = reattachment_length(mesh, velocity)
            except RuntimeError:
                xr = None  # no recirculation yet during the initial jet
            xr_history.append((step, xr))
            if (xr is not None and step * dt >= t_min
                    and len(xr_history) >= XR_PLATEAU_CHECKS):
                recent = [x for _, x in xr_history[-XR_PLATEAU_CHECKS:]
                          if x is not None]
                if (len(recent) == XR_PLATEAU_CHECKS
                        and max(recent) - min(recent)
                        < XR_PLATEAU_REL * abs(recent[-1])):
                    plateau_step = step
                    break
        if step % 1000 == 0:
            xr = xr_history[-1][1] if xr_history else None
            print(f"t = {step * dt:7.2f}  ({step}/{steps})"
                  f"  x_r/h = {xr if xr is None else round(xr, 3)}",
                  flush=True)
    if plateau_step is None:
        raise RuntimeError(f"no x_r plateau within t_max = {t_max}")
    xr_fom = reattachment_length(mesh, velocity)
    print(f"startup plateau at t = {plateau_step * dt:.2f} "
          f"(x_r/h = {xr_fom:.4f}, {len(collector)} snapshots)")
    return mesh, collector, xr_fom


def velocity_error(u, v, u_ref, v_ref):
    """Relative L2 velocity error over both components."""
    diff2 = (float(np.mean((u - u_ref) ** 2))
             + float(np.mean((v - v_ref) ** 2)))
    ref2 = (float(np.mean(u_ref ** 2)) + float(np.mean(v_ref ** 2)))
    return math.sqrt(diff2 / ref2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8,
                        help="grid resolution parameter (n = 8 -> 4480 "
                             "cells, the shedding-case scale)")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--t-min", type=float, default=60.0,
                        help="minimum march time before the plateau "
                             "criterion can fire")
    parser.add_argument("--t-max", type=float, default=200.0,
                        help="march cap (RuntimeError if no plateau)")
    parser.add_argument("--snap-every", type=int, default=10)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--pressure-rank", type=int, default=16)
    parser.add_argument("--window-frac", type=float, default=0.1,
                        help="fraction of the trajectory skipped before "
                             "the POD window (the initial jet)")
    parser.add_argument("--frames", type=int, default=0,
                        help="write the first N sampled reconstruction "
                             "frames of every variant plus the FOM "
                             "snapshots at the same times as .vtu "
                             "(0 disables; ignored with --test)")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--test", action="store_true",
                        help="short machine-level configuration "
                             "(n = 8, t_min = 15, t_max = 40); asserts "
                             "contract invariants and loose accuracy")
    args = parser.parse_args()

    t_min = 15.0 if args.test else args.t_min
    t_max = 40.0 if args.test else args.t_max
    rank = 8 if args.test else args.rank

    tic = time.perf_counter() if args.timing else None
    mesh, collector, xr_fom = run(args.n, args.dt, t_min, t_max,
                                  args.snap_every)
    if args.timing:
        fom_steps = round(collector.times[-1] / args.dt)
        print(f"timing FOM collection  : "
              f"{(time.perf_counter() - tic) / fom_steps * 1000.0:7.1f} "
              f"ms/step ({fom_steps} steps)")

    first = int(args.window_frac * len(collector))
    times = collector.times
    print(f"POD window: t in [{times[first]:.2f}, {times[-1]:.2f}] "
          f"({len(collector) - first} snapshots)")

    pod_u_full = pod_decompose(collector.u[first:])
    pod_v_full = pod_decompose(collector.v[first:])
    pod_u = pod_u_full.truncated(rank)
    pod_v = pod_v_full.truncated(rank)
    print(f"POD rank r = {rank} of {pod_u_full.modes.shape[0]} "
          f"(u reconstruction error "
          f"{pod_u_full.reconstruction_error(rank):.3e})")

    bc_u, bc_v, bc_p = configure_boundary_conditions(mesh, RE)
    p_cfg = make_piso_config(args.dt).pressure_solver_config
    dt_rom = args.dt
    t0 = times[first]
    n_steps = round((times[-1] - t0) / dt_rom)

    initial = fvm.VectorField(mesh, "initial")
    initial.u.to_numpy()[:] = collector.u[first]
    initial.v.to_numpy()[:] = collector.v[first]

    roms = {}
    roms["zero"] = GalerkinRom(mesh, MU, RHO, pod_u, pod_v, bc_u, bc_v,
                               fvm.ConvectionScheme.Central)
    roms["proj"] = GalerkinRom(mesh, MU, RHO, pod_u, pod_v, bc_u, bc_v,
                               fvm.ConvectionScheme.Central,
                               pressure_projection=True,
                               pressure_solver_config=p_cfg,
                               projection_dt=dt_rom)

    # Offline fits for M1/M2 (static BCs: no parameter coordinate here).
    rng = np.random.default_rng(1)
    au_samples, av_samples = make_fit_samples(
        pod_u.project(collector.u[first:]),
        pod_v.project(collector.v[first:]), rng=rng)
    p_proj = projected_pressure_snapshots(
        mesh, MU, RHO, pod_u, pod_v, bc_u, bc_v, bc_p,
        fvm.ConvectionScheme.Central,
        pod_u.project(collector.u[first:]),
        pod_v.project(collector.v[first:]), dt_rom,
        solver_config=p_cfg)
    pod_pp = pod_decompose(p_proj)
    rank_pb = min(args.pressure_rank, pod_pp.modes.shape[0])
    fit = fit_reduced_operators(
        mesh, MU, RHO, pod_u, pod_v, bc_u, bc_v,
        fvm.ConvectionScheme.Central, au_samples, av_samples,
        pressure_basis=(pod_pp.modes[:rank_pb], pod_pp.mean), rng=rng)
    field_fit = fit_rhie_chow_fields(
        mesh, MU, RHO, pod_u, pod_v, bc_u, bc_v,
        fvm.ConvectionScheme.Central, au_samples, av_samples, rng=rng)
    print(f"pressure-basis fit: r_p = {rank_pb} (cumulative energy "
          f"{pod_pp.cumulative_energy[rank_pb - 1]:.6f}), quadratic "
          f"holdout {fit.holdout_residual:.2e}, Rhie-Chow field holdout "
          f"{field_fit.holdout_residual:.2e}")
    roms["m1"] = ReducedPressureRom(fit, field_fit, mesh, RHO, bc_u, bc_v,
                                    bc_p, dt_rom, solver_config=p_cfg)
    roms["m2"] = ReducedPressureRom(fit, field_fit, mesh, RHO, bc_u, bc_v,
                                    bc_p, dt_rom, solver_config=p_cfg,
                                    reduced_solve=True)

    snap_dt = args.dt * args.snap_every
    per_snap = round(snap_dt / dt_rom)

    def timed_advance(label, rom):
        rom.set_state_from_fields(initial)
        if not args.timing:
            return rom.advance(t0, dt_rom, n_steps, record_every=1)
        tic2 = time.perf_counter()
        hist = rom.advance(t0, dt_rom, n_steps, record_every=1)
        print(f"timing ROM {label:<7}: "
              f"{(time.perf_counter() - tic2) / n_steps * 1000.0:7.1f} "
              f"ms/step ({n_steps} steps)")
        return hist

    hists = {name: timed_advance(name, rom) for name, rom in roms.items()}
    print(f"ROMs advanced {n_steps} RK4 steps of dt = {dt_rom} "
          f"(t in [{t0:.2f}, {t0 + n_steps * dt_rom:.2f}])")

    # err|FOM along the window (ROM rows hit snapshot times exactly:
    # dt_rom divides the snapshot cadence).
    n_window = (len(collector) - 1) - first
    tail0 = first + int(0.8 * n_window)
    print(f"{'variant':>8} {'err|FOM tail':>13} {'x_r/h':>9} "
          f"{'xr dev':>8} {'finite':>7}")
    results = {}
    for name, hist in hists.items():
        errs = []
        for snap in range(tail0, len(collector)):
            # The final snapshot sits exactly at the advance end time;
            # the history records rows 0..n_steps-1, so clamp the row.
            row = min((snap - first) * per_snap, hist[1].shape[0] - 1)
            u_r = pod_u.mean + hist[1][row] @ pod_u.modes
            v_r = pod_v.mean + hist[2][row] @ pod_v.modes
            errs.append(velocity_error(u_r, v_r, collector.u[snap],
                                       collector.v[snap]))
        err_tail = float(np.mean(errs))
        final_field = fvm.VectorField(mesh, f"final_{name}")
        u_fin, v_fin = roms[name].reconstruct_arrays()
        final_field.u.to_numpy()[:] = u_fin
        final_field.v.to_numpy()[:] = v_fin
        try:
            xr = reattachment_length(mesh, final_field)
            xr_dev = abs(xr - xr_fom) / xr_fom
        except RuntimeError:
            xr, xr_dev = None, float("inf")
        finite = bool(np.isfinite(hist[1]).all()
                      and np.isfinite(hist[2]).all())
        results[name] = (err_tail, xr, xr_dev, finite)
        print(f"{name:>8} {err_tail:13.4f} "
              f"{xr if xr is None else format(xr, '9.4f'):>9} "
              f"{xr_dev:8.4f} {str(finite):>7}")
    print(f"FOM reference x_r/h = {xr_fom:.4f}")

    if args.frames > 0 and not args.test:
        # Sampled reconstruction frames of every variant plus the FOM
        # snapshot at the same time (the SU2 mesh writes .vtu).  The
        # reconstruction goes through the coefficient history - the
        # ROMs hold only their final state.
        rom_times = hists["zero"][0]
        stride = max(1, len(rom_times) // args.frames)
        frame_velocity = fvm.VectorField(mesh, "rom_frame")
        for row in list(range(0, len(rom_times), stride))[:args.frames]:
            for name, hist in hists.items():
                frame_velocity.u.to_numpy()[:] = (
                    pod_u.mean + hist[1][row] @ pod_u.modes)
                frame_velocity.v.to_numpy()[:] = (
                    pod_v.mean + hist[2][row] @ pod_v.modes)
                write_results(mesh, frame_velocity, None, CASE_DIR,
                              f"rom_{name}_{row:04d}.vtu", scalars={},
                              vectors={"velocity": frame_velocity})
            snap = first + round((rom_times[row] - t0) / snap_dt)
            if 0 <= snap < len(collector):
                frame_velocity.u.to_numpy()[:] = collector.u[snap]
                frame_velocity.v.to_numpy()[:] = collector.v[snap]
                write_results(mesh, frame_velocity, None, CASE_DIR,
                              f"fom_snap_{row:04d}.vtu", scalars={},
                              vectors={"velocity": frame_velocity})

    if args.test:
        checks = {f"{name}_finite": res[3] for name, res in results.items()}
        checks["proj_err"] = results["proj"][0] < TEST_ERR_FOM_MAX
        checks["m2_err"] = results["m2"][0] < TEST_ERR_FOM_MAX
        detail = [f"proj_err={results['proj'][0]:.4f} "
                  f"m2_err={results['m2'][0]:.4f} < {TEST_ERR_FOM_MAX}",
                  f"x_r informational only at the short --test rank "
                  f"(POD floor {pod_u_full.reconstruction_error(rank):.3f})"]
        ok = all(checks.values())
        report_test(ok, f"({', '.join(detail)}, checks={checks})")


if __name__ == "__main__":
    main()
