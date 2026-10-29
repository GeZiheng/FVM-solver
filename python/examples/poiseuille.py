"""Developing plane Poiseuille flow via SIMPLE, compared against the exact
fully developed solution.

A unit bulk-velocity stream enters a channel of L = 10, H = 1 at Re = 20.
The profile is compared against the parabola u = 1.5 U (1 - ((y-0.5)/0.5)^2)
and the pressure slope against dp/dx = -12 mu U / H^2 in the developed
region x in [7, 9].  --test asserts both and exits non-zero on failure.

Run:  python python/examples/poiseuille.py [--test]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _path  # noqa: F401,E402

import numpy as np
import pyfvm as fvm


def run():
    """Solve the channel and return the fields plus metric inputs."""
    nx, ny = 320, 32
    length, height = 10.0, 1.0
    rho, mu, u_bulk = 1.0, 0.05, 1.0  # Re = 20

    mesh = fvm.CartesianMesh(nx, ny, 0.0, 0.0, length, height)
    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")

    bc_u = fvm.BoundaryField(mesh)  # uniform inlet, zero-gradient outlet
    bc_u.set(fvm.BoundaryField.West, fvm.BCType.Dirichlet, u_bulk)
    bc_u.set(fvm.BoundaryField.East, fvm.BCType.Neumann, 0.0)
    bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 0.0)
    bc_u.set(fvm.BoundaryField.South, fvm.BCType.Dirichlet, 0.0)

    bc_v = fvm.BoundaryField(mesh)  # no penetration; outlet zero-gradient
    bc_v.set(fvm.BoundaryField.West, fvm.BCType.Dirichlet, 0.0)
    bc_v.set(fvm.BoundaryField.East, fvm.BCType.Neumann, 0.0)
    bc_v.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 0.0)
    bc_v.set(fvm.BoundaryField.South, fvm.BCType.Dirichlet, 0.0)

    bc_p = fvm.BoundaryField(mesh)  # zero-gradient except outlet p = 0
    bc_p.set(fvm.BoundaryField.East, fvm.BCType.Dirichlet, 0.0)

    cfg = fvm.SimpleConfig()
    cfg.tolerance = 1e-6
    cfg.max_iterations = 2000
    cfg.relaxation_u = 0.7
    cfg.relaxation_p = 0.3
    cfg.scheme = fvm.ConvectionScheme.Central
    cfg.solver_config.tolerance = 1e-9
    cfg.solver_config.max_iterations = 2000

    result = fvm.solve_simple(mesh, rho, mu, bc_u, bc_v, bc_p, cfg,
                              velocity, pressure, flux)
    state = "converged" if result.converged else "NOT converged"
    print(f"SIMPLE: {state} in {result.iterations} iterations, continuity "
          f"residual {result.history[-1].continuity:.3e}")
    if not result.converged:
        raise RuntimeError("SIMPLE did not converge")

    return mesh, velocity, pressure, u_bulk, mu


def channel_metrics(mesh, velocity, pressure, u_bulk, mu):
    """Profile and pressure-slope errors in the developed region."""
    ny = mesh.ny()
    height = ny * mesh.dy()
    y = (np.arange(ny) + 0.5) * mesh.dy()
    x = (np.arange(mesh.nx()) + 0.5) * mesh.dx()
    u = velocity.u.to_numpy().reshape(mesh.ny(), mesh.nx())
    p = pressure.to_numpy().reshape(mesh.ny(), mesh.nx())

    exact = 1.5 * u_bulk * (1.0 - ((y - 0.5 * height) / (0.5 * height)) ** 2)
    dev = (x >= 7.0) & (x <= 9.0)
    profile_err = np.abs(u[:, dev] - exact[:, None]).max()

    p_mean = p[:, dev].mean(axis=0)
    slope = np.polyfit(x[dev], p_mean, 1.0)[0]
    slope_exact = -12.0 * mu * u_bulk / height ** 2
    slope_rel = abs(slope - slope_exact) / abs(slope_exact)

    print(f"developed profile: max |u - parabola| = {profile_err:.4e}")
    print(f"pressure slope: {slope:+.6f} vs {slope_exact:+.6f} "
          f"(rel {slope_rel:.4e})")
    return {"profile_linf": profile_err, "slope_rel": slope_rel}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true",
                        help="assert the analytical checks and exit non-zero "
                             "on failure")
    args = parser.parse_args()

    mesh, velocity, pressure, u_bulk, mu = run()
    metrics = channel_metrics(mesh, velocity, pressure, u_bulk, mu)

    output_dir = Path(__file__).resolve().parent / "output" / "poiseuille"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "poiseuille.vti"
    fvm.write_vti(str(output_path), mesh,
                  {"pressure": pressure}, {"velocity": velocity})
    print(f"Wrote {output_path}")

    if args.test:
        profile_ok = metrics["profile_linf"] <= 3.0e-3
        slope_ok = metrics["slope_rel"] <= 1.0e-2
        ok = profile_ok and slope_ok
        print(f"application test: {'PASS' if ok else 'FAIL'} "
              f"(profile {metrics['profile_linf']:.4e} <= 3.0e-3: "
              f"{profile_ok}; slope rel {metrics['slope_rel']:.4e} "
              f"<= 1e-2: {slope_ok})")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
