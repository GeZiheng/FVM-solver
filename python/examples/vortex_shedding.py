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

from _common import (fvm, frame_interval, read_su2_text, report_test,
                     write_frame, write_results)

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
    return read_su2_text(cylinder_rectangle_grid(
        n_theta=n_theta, n_r=n_r, radius=CYLINDER_RADIUS,
        half=SQUARE_HALF, domain=DOMAIN,
        n_w=n_w, n_e=n_e, n_n=n_n, n_s=n_s), name="cylinder.su2")


def run_case(re, grid, dt, end_time, sample, frame_every=None):
    """Integrate the shedding flow and return the coefficient history.

    With `frame_every` set, frames are written every `frame_interval`
    steps (stretched so long runs stay within ~50 frames) plus one
    final frame.
    """
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
    interval = frame_interval(steps) if frame_every is not None else None
    history = []
    frame_index = 0
    last_frame_step = None
    if interval is not None:
        # Frame 0 is the t = 0 initial field (steps counted from 0).
        print(f"frame {frame_index}: t = {0.0:.2f}", flush=True)
        write_frame(mesh, velocity, pressure, "vortex", "vortex_shedding",
                    frame_index)
        frame_index += 1
        last_frame_step = 0
    for step in range(steps):
        fvm.solve_piso(mesh, rho, mu, bc_u, bc_v, bc_p, cfg,
                       velocity, pressure, flux)
        if interval is not None and (step + 1) % interval == 0:
            print(f"frame {frame_index}: t = {(step + 1) * dt:.2f}",
                  flush=True)
            write_frame(mesh, velocity, pressure, "vortex",
                        "vortex_shedding", frame_index)
            frame_index += 1
            last_frame_step = step + 1
        if (step + 1) % sample == 0:
            cd, cl = force_coefficients(mesh, velocity, pressure,
                                        rho=rho, mu=mu)
            history.append(((step + 1) * dt, cd, cl))
            print(f"t={(step + 1) * dt:7.2f}  Cd={cd:+.6f}  Cl={cl:+.6f}",
                  flush=True)
    # The series always ends at the final state; skip if already written.
    if interval is not None and last_frame_step != steps:
        print(f"frame {frame_index}: t = {steps * dt:.2f}", flush=True)
        write_frame(mesh, velocity, pressure, "vortex", "vortex_shedding",
                    frame_index)
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
    parser.add_argument("--frames-every", type=int, default=50,
                        help="write a VTK frame every K steps (0 disables)")
    args = parser.parse_args()

    frame_every = None if args.test else (args.frames_every or None)
    mesh, velocity, pressure, history = run_case(
        args.re, args.grid, args.dt, args.end_time, args.sample,
        frame_every=frame_every)
    strouhal, cd_mean, cl_amplitude, peaks = periodic_metrics(history)
    print(f"grid={args.grid} dt={args.dt} T={args.end_time} peaks={peaks}: "
          f"St={strouhal:.4f} Cd_mean={cd_mean:.4f} "
          f"Cl_amp={cl_amplitude:.4f}")
    if frame_every is None:
        write_results(mesh, velocity, pressure, "vortex",
                      f"vortex_re100_{args.grid}.vtu")

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
        report_test(ok, f"({detail})")


if __name__ == "__main__":
    main()
