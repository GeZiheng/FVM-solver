"""Impulsive start of Couette flow via PISO (translation of the case in
tests/test_piso.cpp).

Bottom wall (y = 0) fixed, top wall (y = 1) impulsively moved to u = 1 at
t = 0. Compares against the exact Fourier-series solution at t = 0.5.

Run:  python python/examples/impulsive_couette.py
"""

import numpy as np

from _common import fvm, frame_interval, write_frame


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

    bc_u = fvm.BoundaryField(mesh)  # top lid drives; west/east zero-gradient
    bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 1.0)
    bc_u.set(fvm.BoundaryField.South, fvm.BCType.Dirichlet, 0.0)

    bc_v = fvm.BoundaryField(mesh)  # no penetration on all sides
    for side in (fvm.BoundaryField.North, fvm.BoundaryField.East,
                 fvm.BoundaryField.West, fvm.BoundaryField.South):
        bc_v.set(side, fvm.BCType.Dirichlet, 0.0)

    bc_p = fvm.BoundaryField(mesh)  # zero-gradient everywhere (closed domain)

    velocity = fvm.VectorField(mesh, "u")
    pressure = fvm.ScalarField(mesh, "p")
    flux = fvm.FaceFluxField(mesh, "phi")

    cfg = fvm.PisoConfig()
    cfg.dt = 0.005
    cfg.n_steps = 1  # per-step calls; the march below writes frames
    cfg.n_correctors = 2
    cfg.time_scheme = fvm.TimeScheme.Euler
    cfg.solver_config.tolerance = 1e-9
    cfg.solver_config.max_iterations = 20000
    cfg.pressure_solver_config.tolerance = 1e-9
    cfg.pressure_solver_config.max_iterations = 20000

    steps = 100  # t = 0.5
    interval = frame_interval(steps)
    frame_index = 0
    last_frame_step = None
    # Frame 0 is the t = 0 initial field (steps counted from 0).
    print(f"frame {frame_index}: t = {0.0:.3f}")
    write_frame(mesh, velocity, pressure, "impulsive_couette",
                "impulsive_couette", frame_index)
    frame_index += 1
    last_frame_step = 0
    for step in range(steps):
        result = fvm.solve_piso(mesh, rho, mu, bc_u, bc_v, bc_p, cfg,
                                velocity, pressure, flux)
        if (step + 1) % interval == 0:
            print(f"frame {frame_index}: t = {(step + 1) * cfg.dt:.3f}")
            write_frame(mesh, velocity, pressure, "impulsive_couette",
                        "impulsive_couette", frame_index)
            frame_index += 1
            last_frame_step = step + 1
    # The series always ends at the final state; skip if already written.
    if last_frame_step != steps:
        print(f"frame {frame_index}: t = {steps * cfg.dt:.3f}")
        write_frame(mesh, velocity, pressure, "impulsive_couette",
                    "impulsive_couette", frame_index)

    t_end = cfg.dt * steps
    y = (np.arange(n) + 0.5) * mesh.dy()
    u_exact = exact_couette(y, t_end, mu / rho)
    u_num = velocity.u.to_numpy().reshape(n, n)  # row j = profile at y_j
    max_err = np.abs(u_num - u_exact[:, None]).max()

    print(f"PISO: {steps} steps to t = {t_end}, "
          f"max |u - u_exact| = {max_err:.4f}")
    print(f"final continuity residual: {result.history[-1].continuity:.3e}")


if __name__ == "__main__":
    main()
