"""ROM exploration driver: POD/DMD of cylinder vortex-shedding snapshots.

Runs the same coarse-mesh Re = 100 shedding integration as
vortex_shedding.py (rectangular O-4 grid via make_mesh, cylinder.py far
field BCs, deterministic asymmetric seed) while collecting u/v/p
snapshots through the npz contract.  It prints the POD energy spectrum
and reconstruction errors for the full run and for a periodic window
(the last --window-frac fraction, transient discarded, mirroring
periodic_metrics' measurement window), then applies exact DMD to the
window and converts the dominant DMD frequency into a Strouhal number
St = f * D / U_inf (D = 1 from CYLINDER_RADIUS = 0.5, U_inf = 1 from
the inflow BC and force_coefficients' u_in default).  The St value is
printed against the literature band 0.164-0.167 for Re = 100 as a
diagnostic only; nothing about the shedding physics is asserted.  When
the Cl history has enough peaks, the force-based FOM Strouhal number
(vortex_shedding.periodic_metrics) is printed for comparison.  No
matplotlib: table output only (rom.plot.py remains for ad-hoc use).

Snapshots land on multiples of --snap-every only, so the spacing stays
uniform (a DMD prerequisite); a final partial interval is dropped.

Run:  python python/examples/vortex_shedding_rom.py
      [--grid {coarse,medium,fine}] [--dt DT] [--end-time T] [--re RE]
      [--sample N] [--snap-every K] [--window-frac F]
      [--ranks "1,2,4,..."] [--modes N] [--test]
"""

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from _common import fvm, report_test, write_results

from rom import SnapshotCollector, dmd, load_snapshots, pod_decompose

from cylinder import (GRIDS, configure_boundary_conditions,
                      force_coefficients)
from vortex_shedding import DIAMETER, RE, make_mesh, periodic_metrics

CASE_DIR = "vortex_shedding_rom"
TABLE_MODES = 16
# U_inf = 1: cylinder.configure_boundary_conditions sets Dirichlet
# u = 1 on the inflow/far-field patches and force_coefficients defaults
# u_in = 1, so St = f * D / U with D = DIAMETER = 1 and U = 1.
U_INF = 1.0
LITERATURE_ST = (0.164, 0.167)


def parse_ranks(text):
    return [int(tok) for tok in text.split(",") if tok.strip()]


def print_energy_table(result, rows, title):
    print(title)
    print(f"{'mode':>4} {'sigma':>14} {'energy':>12} {'cumulative':>12}")
    for k in range(min(rows, result.singular_values.size)):
        print(f"{k + 1:>4} {result.singular_values[k]:14.6e} "
              f"{result.energy[k]:12.6e} "
              f"{result.cumulative_energy[k]:12.8f}")


def print_error_table(result, ranks, title):
    print(title)
    r_full = result.singular_values.size
    for r in ranks:
        if r > r_full:
            # err(r) is exactly 0 for any r >= r_full (all modes kept).
            print(f"  err({r:>3}) = 0.0 (r >= r_full = {r_full})")
        else:
            print(f"  err({r:>3}) = {result.reconstruction_error(r):.6e}")


def print_dmd_spectrum(result, rows, dt_snap):
    """Print the DMD spectrum sorted by amplitude magnitude."""
    order = np.argsort(np.abs(result.amplitudes))[::-1]
    print(f"DMD spectrum (rank {result.rank}, dt_snap {dt_snap}):")
    print(f"{'mode':>4} {'Re(lambda)':>12} {'Im(lambda)':>12} "
          f"{'|lambda|':>10} {'f=|Im(w)|/2pi':>14} {'|b|':>12}")
    for idx in order[:rows]:
        lam = result.eigenvalues[idx]
        freq = abs(result.omega.imag[idx]) / (2.0 * math.pi)
        print(f"{idx + 1:>4} {lam.real:12.5f} {lam.imag:+12.5f} "
              f"{abs(lam):10.5f} {freq:14.6f} "
              f"{abs(result.amplitudes[idx]):12.5e}")


def run(re, grid, dt, end_time, snap_every, sample):
    """Integrate and collect snapshots; returns (mesh, collector,
    history) with history = [(t, Cd, Cl)] sampled every ``sample``
    steps."""
    mesh = make_mesh(grid)
    rho = 1.0
    mu = rho / re

    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")
    velocity.u.set_constant(1.0)
    # Deterministic asymmetric seed, identical to vortex_shedding.run_case
    # (copied verbatim so this driver reproduces the same trajectory;
    # run_case is not factored out for reuse).
    v_view = velocity.v.to_numpy()
    for cell in range(mesh.cell_count()):
        x, y = mesh.cell_centroid(cell)
        v_view[cell] = (1.0e-3 * math.sin(math.pi * x / 6.0)
                        * math.exp(-(x * x + y * y)))
    pressure.set_constant(0.0)
    flux.set_zero()

    bc_u, bc_v, bc_p = configure_boundary_conditions(mesh)
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

    steps = round(end_time / dt)
    collector = SnapshotCollector()
    collector.append(0.0, velocity.u.to_numpy(), velocity.v.to_numpy(),
                     pressure.to_numpy())
    history = []
    for step in range(steps):
        fvm.solve_piso(mesh, rho, mu, bc_u, bc_v, bc_p, cfg,
                       velocity, pressure, flux)
        # Multiples of snap_every only: uniform snapshot spacing (a
        # final partial interval is dropped, see module docstring).
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
    return mesh, collector, history


def window_slice(count, window_frac):
    """First index of the periodic window (last window_frac fraction)."""
    return int(count * (1.0 - window_frac))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=sorted(GRIDS), default="coarse")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--end-time", type=float, default=250.0)
    parser.add_argument("--re", type=float, default=RE)
    parser.add_argument("--sample", type=int, default=10,
                        help="Cd/Cl sampling interval in steps (FOM "
                             "Strouhal diagnostic)")
    parser.add_argument("--snap-every", type=int, default=5)
    parser.add_argument("--window-frac", type=float, default=0.6,
                        help="fraction of the run kept as the periodic "
                             "window (from the end; 1.0 keeps the whole "
                             "run including the t=0 transient)")
    parser.add_argument("--ranks", default="1,2,4,8,16,32",
                        help="comma-separated ranks for the err(r) table")
    parser.add_argument("--modes", type=int, default=0,
                        help="write the mean field and the first N "
                             "windowed POD modes of u as .vtu "
                             "(0 disables)")
    parser.add_argument("--test", action="store_true",
                        help="run the short machine-level configuration "
                             "(coarse, dt=0.01, T=20) and assert contract "
                             "invariants only")
    args = parser.parse_args()
    if not 0.0 < args.window_frac <= 1.0:
        parser.error("--window-frac must be in (0, 1]")

    if args.test:
        end_time = 20.0
    else:
        end_time = args.end_time

    mesh, collector, history = run(args.re, args.grid, args.dt, end_time,
                                   args.snap_every, args.sample)
    print(f"collected {len(collector)} snapshots of "
          f"{mesh.cell_count()} cells (snap-every {args.snap_every})")

    ranks = parse_ranks(args.ranks)
    full = pod_decompose(collector.u)
    print_energy_table(full, TABLE_MODES,
                       f"POD energy spectrum, u, full run "
                       f"({len(collector)} snapshots):")
    print_error_table(full, ranks, "POD reconstruction error, full run:")

    first = window_slice(len(collector), args.window_frac)
    times = collector.times
    windowed = pod_decompose(collector.u[first:])
    print_energy_table(windowed, TABLE_MODES,
                       f"POD energy spectrum, u, window t >= {times[first]:.2f} "
                       f"({len(collector) - first} snapshots):")
    print_error_table(windowed, ranks, "POD reconstruction error, window:")

    dt_snap = args.snap_every * args.dt
    result = dmd(collector.u[first:], dt_snap)
    print_dmd_spectrum(result, TABLE_MODES, dt_snap)

    # DMD Strouhal number: diagnostic only, never asserted.
    try:
        strouhal = (result.dominant_frequency() * DIAMETER / U_INF)
        print(f"DMD St = {strouhal:.4f}  (literature Re=100 band "
              f"{LITERATURE_ST[0]:.3f}-{LITERATURE_ST[1]:.3f}, "
              f"diagnostic only)")
    except ValueError as exc:
        print(f"DMD St: no oscillating mode ({exc})")

    # Force-based FOM Strouhal from the same run, when shedding has
    # established enough peaks (mirrors vortex_shedding.py metrics).
    try:
        st, cd_mean, cl_amp, peaks = periodic_metrics(history)
        print(f"FOM St = {st:.4f} (Cd_mean={cd_mean:.4f}, "
              f"Cl_amp={cl_amp:.4f}, {peaks} Cl peaks)")
    except RuntimeError as exc:
        print(f"FOM St: not established ({exc})")

    params = {"re": args.re, "grid": args.grid, "dt": args.dt,
              "end_time": end_time, "snap_every": args.snap_every,
              "window_frac": args.window_frac, "rho": 1.0}

    if args.test:
        checks = {}
        detail_values = []
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vortex_shedding_rom.npz")
            collector.save(path, case=CASE_DIR, params=params)
            data = load_snapshots(path)
        checks["format_version"] = data.format_version == 1
        detail_values.append(f"format_version={data.format_version}")
        shapes_ok = all(
            getattr(data, field).shape == (len(collector),
                                           mesh.cell_count())
            for field in ("u", "v", "p"))
        checks["shapes"] = shapes_ok
        checks["times_increasing"] = bool(
            np.all(np.diff(data.t) > 0.0))
        r_full = full.modes.shape[0]
        err = full.reconstruction_error(r_full)
        checks["full_rank_error"] = err < 1e-12
        detail_values.append(f"err(r_full={r_full})={err:.2e}")
        checks["dmd_finite"] = bool(
            np.isfinite(result.eigenvalues).all()
            and np.isfinite(result.omega).all()
            and np.isfinite(result.modes).all()
            and result.rank >= 1)
        detail_values.append(f"dmd_rank={result.rank}")
        ok = all(checks.values())
        report_test(ok, f"({', '.join(detail_values)}, "
                        f"checks={checks})")
        return

    output_dir = (Path(__file__).resolve().parent / "output" / CASE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = mesh.cell_count()
    cx = np.empty(count)
    cy = np.empty(count)
    for cell in range(count):
        cx[cell], cy[cell] = mesh.cell_centroid(cell)
    path = output_dir / f"vortex_re{int(round(args.re))}_{args.grid}.npz"
    collector.save(str(path), case=CASE_DIR, params=params, cx=cx, cy=cy)
    print(f"Wrote {path}")
    if args.modes > 0:
        name = "pod_u_mean"
        field = fvm.ScalarField(mesh, name)
        field.to_numpy()[:] = windowed.mean
        write_results(mesh, None, None, CASE_DIR, f"{name}.vtu",
                      scalars={name: field}, vectors={})
        for k in range(min(args.modes, windowed.modes.shape[0])):
            name = f"pod_u_{k + 1:03d}"
            field = fvm.ScalarField(mesh, name)
            field.to_numpy()[:] = windowed.modes[k]
            write_results(mesh, None, None, CASE_DIR, f"{name}.vtu",
                          scalars={name: field}, vectors={})


if __name__ == "__main__":
    main()
