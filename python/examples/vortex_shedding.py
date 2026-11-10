"""Karman vortex shedding past a circular cylinder at Re=100 via PISO.

The rectangular-domain O-4 mesh of cylinder.py carries the same
far-field boundary conditions as the steady case; at Re=100 the wake
is unsteady and sheds periodically.  The defaults are the CI short
version (coarse grid, dt = 0.01, T = 250); the full study runs the
medium preset with --grid medium --dt 0.005 --end-time 400 (any
cylinder.py grid preset is accepted).  The --test bands are
calibrated at Re=100 only; other --re values are study-only.  A small
deterministic asymmetric
perturbation seeds the instability (the mesh and BCs are exactly
symmetric, so round-off alone would be the only symmetry breaker).
Cd(t) and Cl(t) are integrated on the cylinder patch every --sample
steps; the periodic window (the last 60% of the run) yields the
Strouhal number from the mean Cl peak spacing, the mean drag and the
Cl amplitude.

Literature at Re=100: St ~ 0.164-0.167 (experiments 0.164-0.166),
Cd_mean ~ 1.33 (2D solutions 1.32-1.40), Cl amplitude ~ 0.33
(2D solutions 0.30-0.38).

Run:  python python/examples/vortex_shedding.py
      [--grid {coarse,medium}] [--dt DT] [--end-time T]
      [--sample N] [--re RE] [--test]
"""

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _path  # noqa: F401,E402

import pyfvm as fvm

from mesh_su2 import cylinder_rectangle_grid
from cylinder import (CYLINDER_RADIUS, DOMAIN, GRIDS, SQUARE_HALF,
                      configure_boundary_conditions, force_coefficients)

# Calibration 2026-11-10:
#   coarse  (dt=0.01, T=250, 24 Cl peaks): St=0.1593, Cd_mean=1.3689,
#           Cl_amp=0.3505  [CI entry, ~11 min solo]
#   medium  (dt=0.005, T=300, 30 peaks): St=0.1651, Cd_mean=1.3796,
#           Cl_amp=0.3476  [full study - lands mid-band]
# Literature at Re=100: St 0.164-0.167, Cd_mean 1.32-1.40, Cl amp
# 0.30-0.38.  Coarse->medium drift 0.006/0.011/0.003; bands keep >2x
# the drift around the CI values and cover the 2D literature clusters.
STROUHAL_BAND = (0.145, 0.185)
CD_MEAN_BAND = (1.20, 1.55)
CL_AMPLITUDE_BAND = (0.25, 0.45)

RE = 100.0
# Must stay in sync with force_coefficients' defaults (u_in = diameter
# = 1); St = f * D / U with U = 1.
DIAMETER = 2.0 * CYLINDER_RADIUS
PERIODIC_WINDOW = 0.6  # last 60% of the samples form the measurement window


def make_mesh(grid):
    n_theta, n_r, n_w, n_e, n_n, n_s = GRIDS[grid]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cylinder.su2")
        with open(path, "w", encoding="ascii") as fh:
            fh.write(cylinder_rectangle_grid(
                n_theta=n_theta, n_r=n_r, radius=CYLINDER_RADIUS,
                half=SQUARE_HALF, domain=DOMAIN,
                n_w=n_w, n_e=n_e, n_n=n_n, n_s=n_s))
        return fvm.read_su2(path)


def run_case(re, grid, dt, end_time, sample):
    """Integrate the shedding flow and return the coefficient history."""
    mesh = make_mesh(grid)
    rho = 1.0
    mu = rho / re

    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")
    velocity.u.set_constant(1.0)
    # Deterministic asymmetric seed for the shedding instability.
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
    history = []
    for step in range(steps):
        fvm.solve_piso(mesh, rho, mu, bc_u, bc_v, bc_p, cfg,
                       velocity, pressure, flux)
        if (step + 1) % sample == 0:
            cd, cl = force_coefficients(mesh, velocity, pressure,
                                        rho=rho, mu=mu)
            history.append(((step + 1) * dt, cd, cl))
            print(f"t={(step + 1) * dt:7.2f}  Cd={cd:+.6f}  Cl={cl:+.6f}",
                  flush=True)
    return mesh, velocity, pressure, history


def periodic_metrics(history):
    """Return (St, Cd_mean, Cl_amplitude) from the periodic window."""
    first = int(len(history) * (1.0 - PERIODIC_WINDOW))
    window = history[first:]
    cl_values = [cl for _, _, cl in window]
    cd_values = [cd for _, cd, _ in window]
    times = [t for t, _, _ in window]

    peaks = []
    for i in range(1, len(cl_values) - 1):
        if cl_values[i] > cl_values[i - 1] and cl_values[i] > cl_values[i + 1]:
            peaks.append(i)
    if len(peaks) < 3:
        raise RuntimeError("shedding did not establish: fewer than 3 Cl "
                           f"peaks in the window ({len(peaks)})")
    periods = [times[peaks[i + 1]] - times[peaks[i]]
               for i in range(len(peaks) - 1)]
    mean_period = sum(periods) / len(periods)
    strouhal = DIAMETER / mean_period  # f * D / U with U = 1

    cd_mean = sum(cd_values) / len(cd_values)
    cl_amplitude = 0.5 * (max(cl_values) - min(cl_values))
    return strouhal, cd_mean, cl_amplitude, len(peaks)


def write_result(mesh, velocity, pressure, grid):
    output_dir = Path(__file__).resolve().parent / "output" / "vortex"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"vortex_re100_{grid}.vtu"
    fvm.write_vtu(str(output_path), mesh,
                  {"pressure": pressure}, {"velocity": velocity})
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=sorted(GRIDS), default="coarse")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--end-time", type=float, default=250.0)
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--re", type=float, default=RE)
    parser.add_argument("--test", action="store_true",
                        help="assert the shedding checks and exit non-zero "
                             "on failure")
    args = parser.parse_args()

    mesh, velocity, pressure, history = run_case(
        args.re, args.grid, args.dt, args.end_time, args.sample)
    strouhal, cd_mean, cl_amplitude, peaks = periodic_metrics(history)
    print(f"grid={args.grid} dt={args.dt} T={args.end_time} peaks={peaks}: "
          f"St={strouhal:.4f} Cd_mean={cd_mean:.4f} "
          f"Cl_amp={cl_amplitude:.4f}")
    write_result(mesh, velocity, pressure, args.grid)

    if args.test:
        checks = {
            "St": STROUHAL_BAND[0] <= strouhal <= STROUHAL_BAND[1],
            "Cd_mean": CD_MEAN_BAND[0] <= cd_mean <= CD_MEAN_BAND[1],
            "Cl_amp": (CL_AMPLITUDE_BAND[0] <= cl_amplitude
                       <= CL_AMPLITUDE_BAND[1]),
        }
        detail = ", ".join(
            f"{name}={value:.4f} in {band}={ok}"
            for (name, value, band), ok in zip(
                (("St", strouhal, STROUHAL_BAND),
                 ("Cd", cd_mean, CD_MEAN_BAND),
                 ("Cl_amp", cl_amplitude, CL_AMPLITUDE_BAND)),
                checks.values()))
        ok = all(checks.values())
        print(f"application test: {'PASS' if ok else 'FAIL'} ({detail})")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
