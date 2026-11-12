"""ROM exploration driver: POD of decaying Taylor-Green snapshots.

Runs the same PISO Taylor-Green integration as taylor_green.py (exact
per-face Dirichlet BCs refreshed at t^{n+1} before every single-step
solve_piso call), collects u/v/p snapshots with rom.SnapshotCollector
(including the t = 0 initial state) and writes them through the npz
contract, then prints the POD energy spectrum and reconstruction-error
tables.  Optionally writes the mean field and the first N POD modes of
u as .vti files.  No matplotlib: everything is table output
(rom.plot.py remains available for ad-hoc exploration).

Default: 64^2, dt = 0.0025, T = 2.0, ~200 snapshots (--snap-every
defaults to steps/200 rounded; an explicit value is used as-is).
Snapshots land on multiples of --snap-every only, so the snapshot
spacing is uniform (a DMD prerequisite); a final partial interval is
dropped.

Run:  python python/examples/taylor_green_rom.py
      [--nx N] [--dt DT] [--end-time T] [--snap-every K]
      [--ranks "1,2,4,..."] [--modes N] [--test]
"""

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from _common import fvm, report_test, write_results

from rom import SnapshotCollector, load_snapshots, pod_decompose

from taylor_green import NU, RHO, exact_fields, refresh_boundary_conditions

CASE_DIR = "taylor_green_rom"
TARGET_SNAPSHOTS = 200
TABLE_MODES = 16
# --test band (32^2, dt = 0.005, T = 0.5, snap-every 2): calibrated
# 2026-11-12, same configuration as test_rom.py (measured there
# cumulative_energy[1] = 0.9999928; here 0.9999925).  Threshold 0.9999
# leaves a ~14x margin on the observed energy deficiency (~7e-6).
TEST_ENERGY_THRESHOLD = 0.9999


def parse_ranks(text):
    return [int(tok) for tok in text.split(",") if tok.strip()]


def cell_centres(mesh):
    """(cx, cy) cell-centre coordinate arrays in pyfvm cell order."""
    count = mesh.cell_count()
    cx = np.empty(count)
    cy = np.empty(count)
    for cell in range(count):
        cx[cell], cy[cell] = mesh.cell_centroid(cell)
    return cx, cy


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


def write_pod_modes(mesh, result, count, prefix="pod_u"):
    """Write the mean field and the first ``count`` modes as .vti."""
    name = f"{prefix}_mean"
    field = fvm.ScalarField(mesh, name)
    field.to_numpy()[:] = result.mean
    write_results(mesh, None, None, CASE_DIR, f"{name}.vti",
                  scalars={name: field}, vectors={})
    for k in range(min(count, result.modes.shape[0])):
        name = f"{prefix}_{k + 1:03d}"
        field = fvm.ScalarField(mesh, name)
        field.to_numpy()[:] = result.modes[k]
        write_results(mesh, None, None, CASE_DIR, f"{name}.vti",
                      scalars={name: field}, vectors={})


def run(nx, dt, end_time, snap_every):
    """Integrate and collect snapshots; returns (mesh, collector, K)."""
    mesh = fvm.CartesianMesh(nx, nx, 0.0, 0.0, math.pi, math.pi)
    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")

    u0, v0, p0 = exact_fields(mesh, 0.0)
    velocity.u.to_numpy()[:] = u0
    velocity.v.to_numpy()[:] = v0
    pressure.to_numpy()[:] = p0

    bc_u = fvm.BoundaryField(mesh)
    bc_v = fvm.BoundaryField(mesh)
    bc_p = fvm.BoundaryField(mesh)

    cfg = fvm.PisoConfig()
    cfg.dt = dt
    cfg.n_steps = 1
    cfg.n_correctors = 2
    cfg.time_scheme = fvm.TimeScheme.CrankNicolson
    cfg.scheme = fvm.ConvectionScheme.Central
    cfg.solver_config.tolerance = 1e-10
    cfg.solver_config.max_iterations = 20000
    cfg.pressure_solver_config.tolerance = 1e-10
    cfg.pressure_solver_config.max_iterations = 20000

    steps = round(end_time / dt)
    if snap_every is None:
        snap_every = max(1, round(steps / TARGET_SNAPSHOTS))
    collector = SnapshotCollector()
    collector.append(0.0, velocity.u.to_numpy(), velocity.v.to_numpy(),
                     pressure.to_numpy())
    for step in range(steps):
        # Exact BCs at the end-of-step level of the implicit operator
        # (same pattern as taylor_green.run_level).
        refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p,
                                    (step + 1) * dt)
        fvm.solve_piso(mesh, RHO, NU, bc_u, bc_v, bc_p, cfg,
                       velocity, pressure, flux)
        # Multiples of snap_every only: uniform snapshot spacing (a
        # final partial interval is dropped, see module docstring).
        if (step + 1) % snap_every == 0:
            collector.append((step + 1) * dt, velocity.u.to_numpy(),
                             velocity.v.to_numpy(),
                             pressure.to_numpy())
    return mesh, collector, snap_every


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--end-time", type=float, default=2.0)
    parser.add_argument("--snap-every", type=int, default=None,
                        help="store a snapshot every K steps (default: K "
                             "chosen for ~%d snapshots)"
                             % TARGET_SNAPSHOTS)
    parser.add_argument("--ranks", default="1,2,4,8,16,32",
                        help="comma-separated ranks for the err(r) table")
    parser.add_argument("--modes", type=int, default=0,
                        help="write the mean field and the first N POD "
                             "modes of u as .vti (0 disables)")
    parser.add_argument("--test", action="store_true",
                        help="run the short asserted configuration "
                             "(32^2, dt=0.005, T=0.5, snap-every 2)")
    args = parser.parse_args()

    if args.test:
        nx, dt, end_time, snap_every = 32, 0.005, 0.5, 2
    else:
        nx, dt, end_time, snap_every = (args.nx, args.dt, args.end_time,
                                        args.snap_every)

    mesh, collector, snap_every = run(nx, dt, end_time, snap_every)
    print(f"collected {len(collector)} snapshots of "
          f"{mesh.cell_count()} cells (snap-every {snap_every})")

    result = pod_decompose(collector.u)
    print_energy_table(result, TABLE_MODES,
                       f"POD energy spectrum, u ({len(collector)} snapshots):")
    print_error_table(result, parse_ranks(args.ranks),
                      "POD reconstruction error:")

    params = {"nx": nx, "dt": dt, "nu": NU, "rho": RHO,
              "end_time": end_time, "snap_every": snap_every}

    if args.test:
        checks = {}
        detail_values = []
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "taylor_green_rom.npz")
            collector.save(path, case=CASE_DIR, params=params,
                           cx=None, cy=None)
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
        r_full = result.modes.shape[0]
        err = result.reconstruction_error(r_full)
        checks["full_rank_error"] = err < 1e-12
        detail_values.append(f"err(r_full={r_full})={err:.2e}")
        energy2 = float(result.cumulative_energy[1])
        checks["energy_2"] = energy2 >= TEST_ENERGY_THRESHOLD
        detail_values.append(
            f"cum_energy[2]={energy2:.7f}>={TEST_ENERGY_THRESHOLD}")
        ok = all(checks.values())
        report_test(ok, f"({', '.join(detail_values)}, "
                        f"checks={checks})")
        return

    output_dir = (Path(__file__).resolve().parent / "output" / CASE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    cx, cy = cell_centres(mesh)
    path = output_dir / f"taylor_green_nx{nx}.npz"
    collector.save(str(path), case=CASE_DIR, params=params, cx=cx, cy=cy)
    print(f"Wrote {path}")
    if args.modes > 0:
        write_pod_modes(mesh, result, args.modes)


if __name__ == "__main__":
    main()
