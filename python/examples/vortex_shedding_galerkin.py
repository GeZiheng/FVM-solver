"""Velocity-only POD-Galerkin ROM diagnostic on cylinder vortex shedding.

Elastic target of phase 6.3: diagnostics only, NO physical assertions
(a pure-velocity ROM has no pressure constraint, so limit-cycle
preservation is not guaranteed -- plan risk 4).

Runs the same coarse Re = 100 shedding FOM as vortex_shedding_rom.py
(make_mesh grid, cylinder far-field static BCs, deterministic
asymmetric seed, u/v/p snapshots every --snap-every steps), takes a
windowed POD over the last --window-frac fraction, and integrates the
Galerkin ROM from the window start with rom.galerkin.GalerkinRom
(explicit RK4, static BCs so no bc_refresh).  The oscillation frequency
of the first two POD coefficients (a_1, a_2) is extracted with an FFT
of the mean-removed a_2 record (a_2 has no DC offset issues on the
window; method noted because peak-counting would work equally well)
and converted into a Strouhal number St = f * D / U_inf (D = 1, U = 1),
printed against the FOM St from the force-coefficient history and the
literature band 0.164-0.167.  --projection additionally integrates a
second ROM on the same windowed basis with a per-stage predictive FOM
pressure projection (Phase 6.4) and prints the zero-pressure vs
projective comparison (St, coefficient std, relative error vs the FOM
snapshots) -- still diagnostics only, NO physical assertions.  --test
runs a short machine-level configuration (advance completes,
coefficients finite, projection round-trip) and asserts nothing
physical.

Run:  python python/examples/vortex_shedding_galerkin.py
      [--grid {coarse,medium,fine}] [--dt DT] [--end-time T] [--re RE]
      [--snap-every K] [--window-frac F] [--rank R] [--dt-rom DT]
      [--projection] [--precomputed] [--pressure-basis]
      [--pressure-rank RP] [--reduced-pressure] [--timing]
      [--test]
"""

import argparse
import math
import time

import numpy as np

from _common import fvm, report_test, write_results

from rom import SnapshotCollector, pod_decompose
from rom.galerkin import GalerkinRom
from rom.precomputed import (PrecomputedGalerkinRom, ReducedPressureRom,
                             fit_reduced_operators, fit_rhie_chow_fields,
                             make_fit_samples,
                             projected_pressure_snapshots)

from cylinder import (GRIDS, configure_boundary_conditions,
                      force_coefficients)
from vortex_shedding import DIAMETER, RE, make_mesh, periodic_metrics

# U_inf = 1 (inflow Dirichlet u = 1, force_coefficients default u_in).
U_INF = 1.0
LITERATURE_ST = (0.164, 0.167)
CASE_DIR = "vortex_shedding_galerkin"


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


def run(re, grid, dt, end_time, snap_every, sample):
    """FOM collection, mirroring vortex_shedding_rom.run (same seed);
    returns (mesh, collector, history=[(t, Cd, Cl)])."""
    mesh = make_mesh(grid)
    rho = 1.0
    mu = rho / re

    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")
    velocity.u.set_constant(1.0)
    # Deterministic asymmetric seed, identical to
    # vortex_shedding.run_case / vortex_shedding_rom.run.
    v_view = velocity.v.to_numpy()
    for cell in range(mesh.cell_count()):
        x, y = mesh.cell_centroid(cell)
        v_view[cell] = (1.0e-3 * math.sin(math.pi * x / 6.0)
                        * math.exp(-(x * x + y * y)))
    pressure.set_constant(0.0)
    flux.set_zero()

    bc_u, bc_v, bc_p = configure_boundary_conditions(mesh)
    cfg = make_piso_config(dt)

    steps = round(end_time / dt)
    collector = SnapshotCollector()
    collector.append(0.0, velocity.u.to_numpy(), velocity.v.to_numpy(),
                     pressure.to_numpy())
    history = []
    for step in range(steps):
        fvm.solve_piso(mesh, rho, mu, bc_u, bc_v, bc_p, cfg,
                       velocity, pressure, flux)
        if (step + 1) % snap_every == 0:
            collector.append((step + 1) * dt, velocity.u.to_numpy(),
                             velocity.v.to_numpy(),
                             pressure.to_numpy())
        if (step + 1) % sample == 0:
            cd, cl = force_coefficients(mesh, velocity, pressure,
                                        rho=rho, mu=mu)
            history.append(((step + 1) * dt, cd, cl))
        if (step + 1) % 500 == 0:
            print(f"t = {(step + 1) * dt:7.2f}  ({step + 1}/{steps})",
                  flush=True)
    return mesh, collector, history, (bc_u, bc_v, bc_p)


def relative_velocity_error(u, v, u_ref, v_ref):
    """Relative L2 velocity error over both components."""
    diff2 = (float(np.mean((u - u_ref) ** 2))
             + float(np.mean((v - v_ref) ** 2)))
    ref2 = (float(np.mean(u_ref ** 2)) + float(np.mean(v_ref ** 2)))
    return math.sqrt(diff2 / ref2)


def coefficient_strouhal(times, a_record):
    """Strouhal number of a mean-removed coefficient record via FFT.

    The dominant nonzero rfft bin of the detrended signal is the
    oscillation frequency; St = f * D / U with D = U = 1.  FFT is used
    over peak-counting because it stays well-defined when the record
    is too short for clean peaks (a pure diagnostic either way).
    """
    signal = np.asarray(a_record, dtype=np.float64)
    signal = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(signal))
    dt = times[1] - times[0]
    freqs = np.fft.rfftfreq(signal.size, d=dt)
    dominant = int(np.argmax(spectrum[1:])) + 1  # skip the DC bin
    return freqs[dominant] * DIAMETER / U_INF


def window_start(count, window_frac):
    """First index of the periodic window (last window_frac fraction)."""
    return int(count * (1.0 - window_frac))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=sorted(GRIDS), default="coarse")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--end-time", type=float, default=250.0)
    parser.add_argument("--re", type=float, default=RE)
    parser.add_argument("--snap-every", type=int, default=5)
    parser.add_argument("--window-frac", type=float, default=0.6)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dt-rom", type=float, default=None,
                        help="ROM RK4 step (default: the FOM dt)")
    parser.add_argument("--projection", action="store_true",
                        help="also integrate a per-stage FOM pressure "
                             "projection ROM (Phase 6.4) on the same "
                             "windowed basis and print the comparison "
                             "(diagnostics only)")
    parser.add_argument("--precomputed", action="store_true",
                        help="also integrate the offline-precomputed ROM "
                             "(Phase 6.5b: fitted quadratic operators + "
                             "pressure closure, no per-stage FOM work) "
                             "and print the comparison")
    parser.add_argument("--pressure-rank", type=int, default=16,
                        help="pressure POD rank r_p of the precomputed "
                             "closure / pressure basis (default 16)")
    parser.add_argument("--pressure-basis", action="store_true",
                        help="also integrate the Phase 6.5c M1 ROM: "
                             "precomputed quadratic operators + a "
                             "per-stage full pressure solve on "
                             "offline-fitted Rhie-Chow fields (pressure "
                             "applied through a projection-solution "
                             "basis)")
    parser.add_argument("--reduced-pressure", action="store_true",
                        help="also integrate the Phase 6.5c M2 twin of "
                             "--pressure-basis: the per-stage pressure "
                             "equation is exported and solved as its "
                             "Galerkin projection onto the pressure basis "
                             "(dense r_p x r_p, no sparse solve; requires "
                             "--pressure-basis)")
    parser.add_argument("--timing", action="store_true",
                        help="print per-step wall-clock costs of the FOM "
                             "snapshot collection and of each ROM "
                             "variant's advance")
    parser.add_argument("--frames", type=int, default=0,
                        help="write the first N ROM reconstruction frames "
                             "(both variants with --projection) plus the "
                             "FOM snapshots at the same times as .vtu "
                             "(0 disables; ignored with --test)")
    parser.add_argument("--test", action="store_true",
                        help="run the short machine-level configuration "
                             "(coarse, dt=0.01, T=20, rank 8); "
                             "asserts contract invariants only")
    args = parser.parse_args()
    if not 0.0 < args.window_frac <= 1.0:
        parser.error("--window-frac must be in (0, 1]")
    if args.reduced_pressure and not args.pressure_basis:
        parser.error("--reduced-pressure requires --pressure-basis "
                     "(it reuses that variant's fitted operators)")

    end_time = 20.0 if args.test else args.end_time
    rank = 8 if args.test else args.rank
    dt_rom = args.dt if args.dt_rom is None else args.dt_rom

    fom_steps = round(end_time / args.dt)
    tic = time.perf_counter() if args.timing else None
    mesh, collector, history, (bc_u, bc_v, bc_p) = run(
        args.re, args.grid, args.dt, end_time, args.snap_every,
        sample=10)
    if args.timing:
        print(f"timing FOM collection  : "
              f"{(time.perf_counter() - tic) / fom_steps * 1000.0:7.1f} "
              f"ms/step ({fom_steps} steps, mesh/BC setup included)")
    print(f"collected {len(collector)} snapshots of "
          f"{mesh.cell_count()} cells (snap-every {args.snap_every})")

    first = window_start(len(collector), args.window_frac)
    times = collector.times
    window_len = len(collector) - first
    print(f"periodic window: t in [{times[first]:.2f}, "
          f"{times[-1]:.2f}] ({window_len} snapshots)")

    pod_u_full = pod_decompose(collector.u[first:])
    pod_v_full = pod_decompose(collector.v[first:])
    pod_u = pod_u_full.truncated(rank)
    pod_v = pod_v_full.truncated(rank)
    print(f"windowed POD rank r = {pod_u.modes.shape[0]} of "
          f"{pod_u_full.modes.shape[0]} (u reconstruction error "
          f"{pod_u_full.reconstruction_error(pod_u.modes.shape[0]):.3e})")

    # ROM from the window start: initial state = the first window
    # snapshot; static BCs, so no bc_refresh.
    rom = GalerkinRom(mesh, 1.0 / args.re, 1.0, pod_u, pod_v,
                      bc_u, bc_v, fvm.ConvectionScheme.Central)
    initial = fvm.VectorField(mesh, "rom_initial")
    initial.u.to_numpy()[:] = collector.u[first]
    initial.v.to_numpy()[:] = collector.v[first]
    rom.set_state_from_fields(initial)

    # Optional projective twin (Phase 6.4, revised D1): same windowed
    # basis and initial state, but every RK4 stage first runs one FOM
    # pressure projection (case pressure BCs and the FOM pressure solver
    # precision; the pressure outflow patch is Dirichlet, so the
    # pressure equation takes its well-posed path).  The projection uses
    # the transient diagonal (projection_dt = the ROM stage dt): the
    # steady Central operator is indefinite on this mesh at Re = 100
    # (Pe >> 2; 60/4224 cells with a_P <= 0, min -0.0303, d min -671,
    # CG/BiCGSTAB both fail to converge) while the FOM is immune through
    # its transient rho*V/dt diagonal (min +0.1997) - see
    # .agents/notes/phase6-4-g3.md for the measurements.
    rom_projective = None
    if args.projection:
        rom_projective = GalerkinRom(
            mesh, 1.0 / args.re, 1.0, pod_u, pod_v, bc_u, bc_v,
            fvm.ConvectionScheme.Central, pressure_projection=True,
            pressure_solver_config=make_piso_config(
                args.dt).pressure_solver_config,
            bc_p=bc_p, projection_dt=dt_rom)
        rom_projective.set_state_from_fields(initial)

    # Optional precomputed twin (Phase 6.5b): the quadratic operators are
    # sampled offline on the same windowed basis (zero-pressure GalerkinRom
    # rhs at snapshot coefficients + perturbations); the pressure closure
    # is fitted from the paired windowed velocity/pressure snapshots.  The
    # online advance is pure numpy - no per-stage FOM calls.
    rom_precomputed = None
    if args.precomputed:
        rng = np.random.default_rng(0)
        au_samples, av_samples = make_fit_samples(
            pod_u.project(collector.u[first:]),
            pod_v.project(collector.v[first:]), rng=rng)
        fit = fit_reduced_operators(
            mesh, 1.0 / args.re, 1.0, pod_u, pod_v, bc_u, bc_v,
            fvm.ConvectionScheme.Central, au_samples, av_samples,
            velocity_snapshots=(collector.u[first:], collector.v[first:]),
            pressure_snapshots=collector.p[first:],
            pressure_rank=args.pressure_rank, rng=rng)
        rom_precomputed = PrecomputedGalerkinRom(fit)
        rom_precomputed.set_state_from_fields(initial)
        print(f"precomputed fit: r_p = {fit.rank_pressure} "
              f"(cumulative energy {fit.pressure_cumulative_energy:.6f}), "
              f"quadratic holdout residual {fit.holdout_residual:.2e}, "
              f"closure train residual {fit.closure_train_residual:.2e}")

    # Optional pressure-basis twin (Phase 6.5c M1): precomputed quadratic
    # operators + per-stage FULL pressure solve on offline-fitted
    # Rhie-Chow fields (transient diagonal dt_p = dt_rom, the projection
    # twin's configuration); the solved pressure acts through a basis
    # built from projection-solution pressures (FOM pressure snapshots
    # measurably do not span them).
    rom_pressure_basis = None
    if args.pressure_basis:
        rng = np.random.default_rng(1)
        au_samples, av_samples = make_fit_samples(
            pod_u.project(collector.u[first:]),
            pod_v.project(collector.v[first:]), rng=rng)
        p_cfg = make_piso_config(args.dt).pressure_solver_config
        p_proj = projected_pressure_snapshots(
            mesh, 1.0 / args.re, 1.0, pod_u, pod_v, bc_u, bc_v, bc_p,
            fvm.ConvectionScheme.Central,
            pod_u.project(collector.u[first:]),
            pod_v.project(collector.v[first:]), dt_rom,
            solver_config=p_cfg)
        pod_pp = pod_decompose(p_proj)
        rank_pb = min(args.pressure_rank, pod_pp.modes.shape[0])
        fit_pb = fit_reduced_operators(
            mesh, 1.0 / args.re, 1.0, pod_u, pod_v, bc_u, bc_v,
            fvm.ConvectionScheme.Central, au_samples, av_samples,
            pressure_basis=(pod_pp.modes[:rank_pb], pod_pp.mean), rng=rng)
        field_fit = fit_rhie_chow_fields(
            mesh, 1.0 / args.re, 1.0, pod_u, pod_v, bc_u, bc_v,
            fvm.ConvectionScheme.Central, au_samples, av_samples, rng=rng)
        rom_pressure_basis = ReducedPressureRom(
            fit_pb, field_fit, mesh, 1.0, bc_u, bc_v, bc_p, dt_rom,
            solver_config=p_cfg)
        rom_pressure_basis.set_state_from_fields(initial)
        print(f"pressure-basis fit: r_p = {rank_pb} "
              f"(cumulative energy "
              f"{pod_pp.cumulative_energy[rank_pb - 1]:.6f}), "
              f"quadratic holdout {fit_pb.holdout_residual:.2e}, "
              f"Rhie-Chow field holdout "
              f"{field_fit.holdout_residual:.2e}")

    # Optional reduced-solve twin (Phase 6.5c M2): the same pressure-basis
    # fit, but the per-stage pressure equation is exported uneliminated and
    # solved as its Galerkin projection onto the pressure basis
    # (A_r = Psi A_p Psi^T, dense r_p x r_p) - the M1 mechanism with the
    # sparse solve replaced by the reduced one.
    rom_pressure_reduced = None
    if args.reduced_pressure:
        rom_pressure_reduced = ReducedPressureRom(
            fit_pb, field_fit, mesh, 1.0, bc_u, bc_v, bc_p, dt_rom,
            solver_config=p_cfg, reduced_solve=True)
        rom_pressure_reduced.set_state_from_fields(initial)

    t0 = times[first]
    duration = times[-1] - t0
    n_steps = round(duration / dt_rom)

    def timed_advance(label, rom_obj):
        if not args.timing:
            return rom_obj.advance(t0, dt_rom, n_steps, record_every=1)
        tic = time.perf_counter()
        hist = rom_obj.advance(t0, dt_rom, n_steps, record_every=1)
        print(f"timing ROM {label:<11}: "
              f"{(time.perf_counter() - tic) / n_steps * 1000.0:7.1f} "
              f"ms/step ({n_steps} steps)")
        return hist

    hist = timed_advance("zero-pressure", rom)
    rec_times, a_u_hist, a_v_hist = hist
    a_up_hist = a_vp_hist = None
    if rom_projective is not None:
        _, a_up_hist, a_vp_hist = timed_advance("projection",
                                                rom_projective)
    a_pc_u_hist = a_pc_v_hist = None
    if rom_precomputed is not None:
        _, a_pc_u_hist, a_pc_v_hist = timed_advance("precomputed",
                                                    rom_precomputed)
    a_pb_u_hist = a_pb_v_hist = None
    if rom_pressure_basis is not None:
        _, a_pb_u_hist, a_pb_v_hist = timed_advance("pressure-basis",
                                                    rom_pressure_basis)
    a_pr_u_hist = a_pr_v_hist = None
    if rom_pressure_reduced is not None:
        _, a_pr_u_hist, a_pr_v_hist = timed_advance("pressure-reduced",
                                                    rom_pressure_reduced)
    print(f"ROM advanced {n_steps} RK4 steps of dt = {dt_rom} "
          f"(t in [{t0:.2f}, {rec_times[-1]:.2f}])")

    # Diagnostics: ROM St from the a_1/a_2 oscillation, FOM St from
    # the force-coefficient history, literature band for context.
    st_a1 = coefficient_strouhal(rec_times, a_u_hist[:, 0])
    st_a2 = coefficient_strouhal(rec_times, a_v_hist[:, 0])
    print(f"ROM St (a_1 FFT) = {st_a1:.4f}   ROM St (a_2 FFT) = "
          f"{st_a2:.4f}   (literature Re=100 band "
          f"{LITERATURE_ST[0]:.3f}-{LITERATURE_ST[1]:.3f}, "
          f"diagnostic only)")
    try:
        st, cd_mean, cl_amp, peaks = periodic_metrics(history)
        print(f"FOM St = {st:.4f} (Cd_mean={cd_mean:.4f}, "
              f"Cl_amp={cl_amp:.4f}, {peaks} Cl peaks)")
    except RuntimeError as exc:
        print(f"FOM St: not established ({exc})")
    # Coefficient-boundedness drift diagnostic (no assertion).
    span = (float(np.abs(a_u_hist).max()), float(np.abs(a_v_hist).max()))
    print(f"coefficient envelope max|a_u| = {span[0]:.3e}, "
          f"max|a_v| = {span[1]:.3e} (limit-cycle preservation is NOT "
          f"guaranteed for a velocity-only ROM)")

    twins = []
    if rom_projective is not None:
        twins.append(("projective", a_up_hist, a_vp_hist))
    if rom_precomputed is not None:
        twins.append(("precomputed", a_pc_u_hist, a_pc_v_hist))
    if rom_pressure_basis is not None:
        twins.append(("pressure-basis", a_pb_u_hist, a_pb_v_hist))
    if rom_pressure_reduced is not None:
        twins.append(("pressure-reduced", a_pr_u_hist, a_pr_v_hist))
    if twins:
        # Variant comparison (diagnostics only; the std rows are the
        # persistent-amplitude metric behind the documented limit-cycle
        # collapse, err|FOM is the mean relative L2 velocity error
        # against the nearest FOM snapshot over the last 20% of the
        # record).
        snap_dt = args.dt * args.snap_every
        tail0 = int(0.8 * rec_times.size)

        def tail_err(a_uh, a_vh):
            errs = []
            for row in range(tail0, rec_times.size):
                idx = min(first + round((rec_times[row] - t0) / snap_dt),
                          len(collector) - 1)
                errs.append(relative_velocity_error(
                    pod_u.mean + a_uh[row] @ pod_u.modes,
                    pod_v.mean + a_vh[row] @ pod_v.modes,
                    collector.u[idx], collector.v[idx]))
            return float(np.mean(errs))

        print("ROM variant comparison (no physical assertions):")
        print(f"{'':>14} {'zero':>12} "
              + " ".join(f"{name:>12}" for name, _, _ in twins))
        print(f"{'St(a1 FFT)':>14} {st_a1:12.4f} "
              + " ".join(f"{coefficient_strouhal(rec_times, hu[:, 0]):12.4f}"
                         for _, hu, _ in twins))
        print(f"{'St(a2 FFT)':>14} {st_a2:12.4f} "
              + " ".join(f"{coefficient_strouhal(rec_times, hv[:, 0]):12.4f}"
                         for _, _, hv in twins))
        print(f"{'std(a1)':>14} {np.std(a_u_hist[:, 0]):12.3e} "
              + " ".join(f"{np.std(hu[:, 0]):12.3e}" for _, hu, _ in twins))
        print(f"{'std(a2)':>14} {np.std(a_v_hist[:, 0]):12.3e} "
              + " ".join(f"{np.std(hv[:, 0]):12.3e}" for _, _, hv in twins))
        print(f"{'err|FOM(1)':>14} {tail_err(a_u_hist, a_v_hist):12.4f} "
              + " ".join(f"{tail_err(hu, hv):12.4f}" for _, hu, hv in twins)
              + "   (mean over the last 20% of the record, vs nearest "
              "FOM snapshot)")
        if rom_projective is not None:
            print(f"projective last-stage projection (imb, du, dv, dp): "
                  f"{rom_projective.last_projection}")
    if rom_pressure_basis is not None and rom_pressure_reduced is not None:
        # D9 gate: the reduced solve must track the M1 full solve's
        # trajectory (relative history deviation; the gap is the pressure
        # POD truncation of the projection solutions).
        rel_u = (np.linalg.norm(a_pr_u_hist - a_pb_u_hist)
                 / np.linalg.norm(a_pb_u_hist))
        rel_v = (np.linalg.norm(a_pr_v_hist - a_pb_v_hist)
                 / np.linalg.norm(a_pb_v_hist))
        print(f"M2 vs M1 trajectory rel dev: u {rel_u:.3e}, v {rel_v:.3e} "
              f"(gate 1e-3)")

    if args.frames > 0 and not args.test:
        # ROM reconstructions at recorded rows plus the matching FOM
        # snapshots (same times) for side-by-side comparison.  The FOM
        # snapshot is shared between the variants; each ROM writes its
        # own frame series (rom_recon = zero-pressure, rom_projective =
        # the projection twin, --projection only).
        snap_dt = args.dt * args.snap_every
        stride = max(1, len(rec_times) // args.frames)
        frame_velocity = fvm.VectorField(mesh, "rom_frame")
        fom_velocity = fvm.VectorField(mesh, "fom_frame")
        frame_roms = [("rom_recon", rom, a_u_hist, a_v_hist)]
        if rom_projective is not None:
            frame_roms.append(("rom_projective", rom_projective,
                               a_up_hist, a_vp_hist))
        for row in list(range(0, len(rec_times), stride))[:args.frames]:
            for name, frame_rom, au_hist, av_hist in frame_roms:
                frame_rom.a_u = au_hist[row].copy()
                frame_rom.a_v = av_hist[row].copy()
                frame_rom.write_state(frame_velocity)
                write_results(mesh, frame_velocity, None, CASE_DIR,
                              f"{name}_{row:04d}.vtu", scalars={},
                              vectors={"velocity": frame_velocity})
            idx = min(first + round((rec_times[row] - t0) / snap_dt),
                      len(collector) - 1)
            fom_velocity.u.to_numpy()[:] = collector.u[idx]
            fom_velocity.v.to_numpy()[:] = collector.v[idx]
            write_results(mesh, fom_velocity, None, CASE_DIR,
                          f"fom_snap_{row:04d}.vtu", scalars={},
                          vectors={"velocity": fom_velocity})

    if args.test:
        # Machine-level assertions only: the advance completed, all
        # recorded coefficients are finite, and the projection onto the
        # row-orthonormal basis is idempotent (project of a
        # reconstruction reproduces the coefficients exactly).
        checks = {}
        detail = []
        checks["advance_completed"] = (
            rec_times.size == n_steps
            and abs(rec_times[-1] - (t0 + n_steps * dt_rom)) < 1e-9)
        checks["coefficients_finite"] = bool(
            np.isfinite(a_u_hist).all() and np.isfinite(a_v_hist).all())
        detail.append(f"steps={rec_times.size}, max|a_u|={span[0]:.3e}")
        if rom_projective is not None:
            # machine-level only: the projection-mode twin completed its
            # advance with finite coefficients (same record schedule).
            checks["projective_finite"] = bool(
                np.isfinite(a_up_hist).all() and np.isfinite(a_vp_hist).all())
        if rom_precomputed is not None:
            # machine-level only: same for the precomputed twin.
            checks["precomputed_finite"] = bool(
                np.isfinite(a_pc_u_hist).all()
                and np.isfinite(a_pc_v_hist).all())
        if rom_pressure_basis is not None:
            # machine-level only: same for the pressure-basis twin.
            checks["pressure_basis_finite"] = bool(
                np.isfinite(a_pb_u_hist).all()
                and np.isfinite(a_pb_v_hist).all())
        if rom_pressure_reduced is not None:
            # machine-level only: same for the reduced-solve twin.
            checks["pressure_reduced_finite"] = bool(
                np.isfinite(a_pr_u_hist).all()
                and np.isfinite(a_pr_v_hist).all())
        rom.write_state(initial)
        rom.set_state_from_fields(initial)
        np.testing.assert_allclose(rom.a_u, a_u_hist[-1], atol=1e-12)
        np.testing.assert_allclose(rom.a_v, a_v_hist[-1], atol=1e-12)
        checks["projection_roundtrip"] = True
        ok = all(checks.values())
        report_test(ok, f"({', '.join(detail)}, checks={checks})")


if __name__ == "__main__":
    main()
