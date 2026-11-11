"""Steady convection-diffusion in a unit square.

Recirculating divergence-free velocity from psi = sin(pi x) sin(pi y);
west wall hot (T = 1), east wall cold (T = 0), north/south adiabatic.
Run:  python python/examples/convection_diffusion.py
"""

import numpy as np

from _common import fvm, write_results


def main():
    nx = ny = 64
    mesh = fvm.CartesianMesh(nx, ny, 0.0, 0.0, 1.0, 1.0)

    rho = 1.0
    gamma = fvm.ScalarField(mesh, "gamma")
    gamma.set_constant(0.01)  # Pe ~ 100

    # Cell centres in the field's storage order (cell index = j * nx + i).
    x = (np.arange(nx) + 0.5) * mesh.dx()
    y = (np.arange(ny) + 0.5) * mesh.dy()
    X, Y = np.meshgrid(x, y)  # shape (ny, nx); ravel() matches cell order

    velocity = fvm.VectorField(mesh, "velocity")
    velocity.u.to_numpy()[:] = np.ravel(np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y))
    velocity.v.to_numpy()[:] = np.ravel(-np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y))

    bc = fvm.BoundaryField(mesh)  # north/south default to zero-flux (adiabatic)
    bc.set(fvm.BoundaryField.West, fvm.BCType.Dirichlet, 1.0)
    bc.set(fvm.BoundaryField.East, fvm.BCType.Dirichlet, 0.0)

    solver_cfg = fvm.SolverConfig()
    solver_cfg.tolerance = 1e-10
    solver_cfg.max_iterations = 2000

    flux = fvm.interpolate_cell_velocity_flux(mesh, velocity, rho)
    sol = fvm.solve_transport(mesh, flux, gamma, fvm.ConvectionScheme.Upwind,
                              bc, solver_cfg)

    temperature = fvm.ScalarField(mesh, "temperature")
    temperature.to_numpy()[:] = sol

    write_results(mesh, velocity, None, "convection_diffusion",
                  "convection_diffusion.vti",
                  scalars={"temperature": temperature})
    print(f"T range: [{sol.min():.6f}, {sol.max():.6f}]")


if __name__ == "__main__":
    main()
