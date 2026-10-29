"""Stokes' second problem (oscillating plate) via PISO, compared against
the exact semi-infinite solution.

The top plate of a closed-sided slot oscillates as u_wall(t) = cos(w t);
with nu = 0.1 and a period of exactly 1.25 (dt = 0.005, 250 steps/period)
the penetration depth is delta = sqrt(2 nu / w) ~ 0.199 and the opposite
wall at H = 2 sees e^{-H/delta} ~ 4e-5 of the amplitude.  The wall value is
time-dependent but spatially uniform, so it is refreshed from Python before
every single-step solve_piso call.  After the periodic steady state is
reached, the profile at phase zero is compared against
u(y) = U exp(-y/delta) cos(y/delta).

--test additionally reruns impulsive Couette with one 100-step call and 100
single-step calls and asserts the two final states agree (stepwise calling
semantics).

Run:  python python/examples/stokes_oscillating.py [--test]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _path  # noqa: F401,E402

import numpy as np
import pyfvm as fvm

RHO, MU = 1.0, 0.1          # nu = 0.1
DT = 0.005
PERIOD = 1.25               # exactly 250 steps per period
STEPS_PER_PERIOD = round(PERIOD / DT)
OMEGA = 2.0 * np.pi / PERIOD
DELTA = np.sqrt(2.0 * (MU / RHO) / OMEGA)
NX, NY = 16, 64


def make_fields(nx, ny, lx, ly):
    mesh = fvm.CartesianMesh(nx, ny, 0.0, 0.0, lx, ly)
    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")
    return mesh, velocity, pressure, flux


def oscillating_run(max_cycles=40, steady_tol=1e-6):
    """March to the periodic steady state; return the final fields."""
    nx, ny = NX, NY
    mesh, velocity, pressure, flux = make_fields(nx, ny, 2.0, 2.0)

    bc_u = fvm.BoundaryField(mesh)  # top value refreshed every step
    bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 1.0)
    bc_u.set(fvm.BoundaryField.South, fvm.BCType.Dirichlet, 0.0)
    bc_u.set(fvm.BoundaryField.West, fvm.BCType.Neumann, 0.0)
    bc_u.set(fvm.BoundaryField.East, fvm.BCType.Neumann, 0.0)

    bc_v = fvm.BoundaryField(mesh)  # no penetration; sides zero-gradient
    bc_v.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 0.0)
    bc_v.set(fvm.BoundaryField.South, fvm.BCType.Dirichlet, 0.0)
    bc_v.set(fvm.BoundaryField.West, fvm.BCType.Neumann, 0.0)
    bc_v.set(fvm.BoundaryField.East, fvm.BCType.Neumann, 0.0)

    bc_p = fvm.BoundaryField(mesh)  # all Neumann (reference-cell pinning)

    cfg = fvm.PisoConfig()
    cfg.dt = DT
    cfg.n_steps = 1
    cfg.n_correctors = 2
    cfg.time_scheme = fvm.TimeScheme.Euler
    cfg.solver_config.tolerance = 1e-9
    cfg.solver_config.max_iterations = 20000

    step = 0
    prev = None
    steady_cycle = None
    for cycle in range(1, max_cycles + 1):
        for _ in range(STEPS_PER_PERIOD):
            step += 1
            # Implicit Euler evaluates the wall value at t^{n+1}.
            bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet,
                     np.cos(OMEGA * DT * step))
            fvm.solve_piso(mesh, RHO, MU, bc_u, bc_v, bc_p, cfg,
                           velocity, pressure, flux)
        u = velocity.u.to_numpy().reshape(ny, nx).copy()
        if prev is not None and np.abs(u - prev).max() < steady_tol:
            print(f"periodic steady state after {cycle} cycles "
                  f"(t = {DT * step:.2f})")
            steady_cycle = cycle
            break
        prev = u
    if steady_cycle is None:
        raise RuntimeError("no periodic steady state within "
                           f"{max_cycles} cycles")
    # Advance two more periods at the steady state before sampling.
    for _ in range(2 * STEPS_PER_PERIOD):
        step += 1
        bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet,
                 np.cos(OMEGA * DT * step))
        fvm.solve_piso(mesh, RHO, MU, bc_u, bc_v, bc_p, cfg,
                       velocity, pressure, flux)
    return mesh, velocity, pressure


def oscillating_metrics(mesh, velocity):
    """L-inf / L2 error of the phase-zero profile vs the exact solution."""
    height = mesh.ny() * mesh.dy()
    # Distance from the oscillating (top) plate, evaluated at cell centres.
    d = height - (np.arange(mesh.ny()) + 0.5) * mesh.dy()
    u_num = velocity.u.to_numpy().reshape(mesh.ny(), mesh.nx())
    # The field is x-uniform; average over the homogeneous direction.
    profile = u_num.mean(axis=1)
    # Phase zero (omega*t = 2 pi k): u = U exp(-d/delta) cos(d/delta).
    u_exact = np.exp(-d / DELTA) * np.cos(d / DELTA)
    err = np.abs(profile - u_exact)
    metrics = {"linf": err.max(), "l2": np.sqrt(np.mean(err ** 2))}
    print(f"phase-0 profile: max |u - exact| = {metrics['linf']:.4e}, "
          f"L2 = {metrics['l2']:.4e}")
    return metrics


def couette_stepwise_sanity():
    """One 100-step call vs 100 single-step calls must agree."""
    def setup():
        n = 32
        mesh, velocity, pressure, flux = make_fields(n, n, 1.0, 1.0)
        bc_u = fvm.BoundaryField(mesh)
        bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 1.0)
        bc_u.set(fvm.BoundaryField.South, fvm.BCType.Dirichlet, 0.0)
        bc_u.set(fvm.BoundaryField.West, fvm.BCType.Neumann, 0.0)
        bc_u.set(fvm.BoundaryField.East, fvm.BCType.Neumann, 0.0)
        bc_v = fvm.BoundaryField(mesh)
        for side in (fvm.BoundaryField.North, fvm.BoundaryField.South,
                     fvm.BoundaryField.West, fvm.BoundaryField.East):
            bc_v.set(side, fvm.BCType.Dirichlet, 0.0)
        bc_p = fvm.BoundaryField(mesh)
        cfg = fvm.PisoConfig()
        cfg.dt = 0.005
        cfg.n_correctors = 2
        cfg.time_scheme = fvm.TimeScheme.Euler
        cfg.solver_config.tolerance = 1e-9
        cfg.solver_config.max_iterations = 20000
        return mesh, velocity, pressure, flux, bc_u, bc_v, bc_p, cfg

    mesh, velocity, pressure, flux, bc_u, bc_v, bc_p, cfg = setup()
    cfg.n_steps = 100
    fvm.solve_piso(mesh, RHO, MU, bc_u, bc_v, bc_p, cfg,
                   velocity, pressure, flux)
    u_long = velocity.u.to_numpy().copy()

    mesh, velocity, pressure, flux, bc_u, bc_v, bc_p, cfg = setup()
    cfg.n_steps = 1
    for _ in range(100):
        fvm.solve_piso(mesh, RHO, MU, bc_u, bc_v, bc_p, cfg,
                       velocity, pressure, flux)
    u_stepwise = velocity.u.to_numpy().copy()

    diff = np.abs(u_long - u_stepwise).max()
    print(f"Couette stepwise sanity: max |u(100x1) - u(1x100)| = {diff:.3e}")
    return diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true",
                        help="assert the exact-solution and stepwise-calling "
                             "checks and exit non-zero on failure")
    args = parser.parse_args()

    mesh, velocity, pressure = oscillating_run()
    metrics = oscillating_metrics(mesh, velocity)

    output_dir = Path(__file__).resolve().parent / "output" / "stokes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "oscillating.vti"
    fvm.write_vti(str(output_path), mesh,
                  {"pressure": pressure}, {"velocity": velocity})
    print(f"Wrote {output_path}")

    if args.test:
        # Calibrated in Phase 2: observed 2.8e-3 (mixed first-order Euler
        # temporal error and near-plate discretisation); 1.4x headroom.
        ok = metrics["linf"] <= 4.0e-3
        print(f"application test: {'PASS' if ok else 'FAIL'} "
              f"(profile L-inf {metrics['linf']:.4e} <= 4.0e-3)")
        if not ok:
            sys.exit(1)
        diff = couette_stepwise_sanity()
        ok = diff <= 1.0e-9
        print(f"application test: {'PASS' if ok else 'FAIL'} "
              f"(stepwise difference {diff:.3e} <= 1e-9)")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
