"""Lid-driven cavity at Re = 100 via SIMPLE (same case as the SIMPLE test
in tests/test_simple.cpp, at 64x64).

Run:  python python/examples/cavity.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _path  # noqa: F401,E402

import numpy as np
import pyfvm as fvm


def main():
    n = 64
    mesh = fvm.CartesianMesh(n, n, 0.0, 0.0, 1.0, 1.0)

    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")

    bc_u = fvm.BoundaryField()  # lid moves with U = 1, other walls no-slip
    bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 1.0)
    bc_u.set(fvm.BoundaryField.East, fvm.BCType.Dirichlet, 0.0)
    bc_u.set(fvm.BoundaryField.West, fvm.BCType.Dirichlet, 0.0)
    bc_u.set(fvm.BoundaryField.South, fvm.BCType.Dirichlet, 0.0)

    bc_v = fvm.BoundaryField()  # no penetration on all walls
    for side in (fvm.BoundaryField.North, fvm.BoundaryField.East,
                 fvm.BoundaryField.West, fvm.BoundaryField.South):
        bc_v.set(side, fvm.BCType.Dirichlet, 0.0)

    bc_p = fvm.BoundaryField()  # zero-gradient pressure on all walls (default)

    cfg = fvm.SimpleConfig()
    cfg.tolerance = 1e-6
    cfg.max_iterations = 3000
    cfg.relaxation_u = 0.7
    cfg.relaxation_p = 0.3
    cfg.solver_config.tolerance = 1e-9
    cfg.solver_config.max_iterations = 2000

    result = fvm.solve_simple(mesh, 1.0, 0.01, bc_u, bc_v, bc_p, cfg,
                              velocity, pressure, flux)
    state = "converged" if result.converged else "NOT converged"
    print(f"SIMPLE: {state} in {result.iterations} iterations, "
          f"continuity residual {result.history[-1].continuity:.3e}")

    speed = np.hypot(velocity.u.to_numpy(), velocity.v.to_numpy())
    print(f"max speed: {speed.max():.4f}")

    fvm.write_vti("cavity.vti", mesh,
                  {"pressure": pressure}, {"velocity": velocity})
    print("Wrote cavity.vti")


if __name__ == "__main__":
    main()
