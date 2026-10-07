"""Impulsive start of Couette flow via PISO (translation of the case in
tests/test_piso.cpp).

Bottom wall (y = 0) fixed, top wall (y = 1) impulsively moved to u = 1 at
t = 0. Compares against the exact Fourier-series solution at t = 0.5.

Run:  python python/examples/impulsive_couette.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _path  # noqa: F401,E402

import numpy as np
import pyfvm as fvm


def exact_couette(y, t, nu):
    """Exact series solution, vectorised over y."""
    n = np.arange(1, 201)
    sign = np.where(n % 2 == 0, 1.0, -1.0)
    terms = (sign / n)[:, None] * np.sin(np.outer(n, np.pi * y)) \
        * np.exp(-n * n * np.pi * np.pi * nu * t)[:, None]
    return y + (2.0 / np.pi) * terms.sum(axis=0)


def main():
    n = 32
    mesh = fvm.CartesianMesh(n, n, 0.0, 0.0, 1.0, 1.0)
    rho, mu = 1.0, 0.1  # nu = 0.1

    bc_u = fvm.BoundaryField()  # top lid drives; west/east zero-gradient
    bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 1.0)
    bc_u.set(fvm.BoundaryField.South, fvm.BCType.Dirichlet, 0.0)

    bc_v = fvm.BoundaryField()  # no penetration on all sides
    for side in (fvm.BoundaryField.North, fvm.BoundaryField.East,
                 fvm.BoundaryField.West, fvm.BoundaryField.South):
        bc_v.set(side, fvm.BCType.Dirichlet, 0.0)

    bc_p = fvm.BoundaryField()  # zero-gradient everywhere (closed domain)

    velocity = fvm.VectorField(mesh, "u")
    pressure = fvm.ScalarField(mesh, "p")
    flux = fvm.FaceFluxField(mesh, "phi")

    cfg = fvm.PisoConfig()
    cfg.dt = 0.005
    cfg.n_steps = 100  # t = 0.5
    cfg.n_correctors = 2
    cfg.time_scheme = fvm.TimeScheme.Euler
    cfg.solver_config.tolerance = 1e-9
    cfg.solver_config.max_iterations = 20000

    result = fvm.solve_piso(mesh, rho, mu, bc_u, bc_v, bc_p, cfg,
                            velocity, pressure, flux)

    t_end = cfg.dt * cfg.n_steps
    y = (np.arange(n) + 0.5) * mesh.dy()
    u_exact = exact_couette(y, t_end, mu / rho)
    u_num = velocity.u.to_numpy().reshape(n, n)  # row j = profile at y_j
    max_err = np.abs(u_num - u_exact[:, None]).max()

    print(f"PISO: {result.steps} steps to t = {t_end}, "
          f"max |u - u_exact| = {max_err:.4f}")
    print(f"final continuity residual: {result.history[-1].continuity:.3e}")

    fvm.write_vti("couette.vti", mesh,
                  {"pressure": pressure}, {"velocity": velocity})
    print("Wrote couette.vti")


if __name__ == "__main__":
    main()
