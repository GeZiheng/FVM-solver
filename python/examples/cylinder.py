"""Steady flow past a circular cylinder in a rectangular domain via SIMPLE.

The O-4 multiblock grid (see mesh_su2.cylinder_rectangle_grid) covers the
domain [-10, 25] x [-10, 10] (blockage 1/20) around a cylinder of
diameter D = 1.  Drag is evaluated from the cylinder patch with the
project's owner-relative boundary-face convention: the area vector points
from the adjacent fluid cell into the cylinder.

Run:  python python/examples/cylinder.py
      [--re RE] [--grid {coarse,medium,fine}] [--test]
"""

import argparse
import math

from _common import (fvm, print_simple_status, read_su2_text, report_test,
                     write_results)

from mesh_su2 import cylinder_rectangle_grid

# Calibration 2026-11-10 (Central, rectangular domain [-10,25]x[-10,10],
# far-field Dirichlet; full study in docs/numerical.md):
#   Re=40: Cd = 1.6007 (coarse) / 1.6064 (medium) / 1.6066 (fine, outer
#          refinement); core-Richardson ~1.611 (a converged 128x56 core
#          gives 1.6096).  L_w/D = 1.750 / 1.953 / 1.953, core trend ~2.2.
#   Re=20: Cd = 2.1547 / 2.1638 / 2.1639.  L_w/D = 0.848 / 0.870 / 0.870.
# Literature anchors: Fornberg 1980 Cd = 1.550 (Re=40), ~2.00 (Re=20);
# Dennis & Chang 1970 Cd = 1.522, L_w/D ~ 2.28 (Re=40).  The measured Cd
# sits ~4% above the anchors: the finite domain (blockage 1/20) with
# far-field velocity Dirichlet sides; a symmetry-side variant lowers Cd
# by only 0.25%, and the old r=15 O-grid's semicircular p=0 outflow
# (which cut through the streamlines) reads ~1.562.
# Band margins vs the CI medium values: Cd(40) 27x/20x and Cd(20)
# 23x/15x the coarse->medium drift; L_w lower bound keeps 0.15 on the
# medium value (the refinement direction lengthens the wake).
CD_BANDS = {20: (1.95, 2.30), 40: (1.45, 1.72)}
WAKE_LENGTH_BAND = (1.80, 2.65)
WAKE_TOLERANCE = 0.25

CYLINDER_RADIUS = 0.5
SQUARE_HALF = 3.0
DOMAIN = (-10.0, 25.0, -10.0, 10.0)

# (n_theta, n_r, n_w, n_e, n_n, n_s); n_theta must be a multiple of 8.
# The grid study refines the core (coarse -> medium) and then the outer
# blocks (medium -> fine): refining the polar core beyond 96x40 at
# relaxation 0.7/0.3 (and at any tested lower relaxation) destabilizes
# the steady SIMPLE outer loop at Re=20 -- see docs/numerical.md for the
# isolation evidence.
GRIDS = {
    "coarse": (64, 28, 16, 24, 16, 16),
    "medium": (96, 40, 24, 36, 24, 24),
    "fine": (96, 40, 32, 48, 32, 32),
}


def configure_boundary_conditions(mesh):
    """Return velocity and pressure BCs; patch order follows the markers."""
    bc_u = fvm.BoundaryField(mesh)
    bc_v = fvm.BoundaryField(mesh)
    bc_p = fvm.BoundaryField(mesh)

    # Patch 0: no-slip cylinder.
    bc_u.set(0, fvm.BCType.Dirichlet, 0.0)
    bc_v.set(0, fvm.BCType.Dirichlet, 0.0)
    bc_p.set(0, fvm.BCType.Neumann, 0.0)

    # Patch 1: uniform far-field inflow on the west side.
    bc_u.set(1, fvm.BCType.Dirichlet, 1.0)
    bc_v.set(1, fvm.BCType.Dirichlet, 0.0)
    bc_p.set(1, fvm.BCType.Neumann, 0.0)

    # Patch 2: pressure outflow on the east side.
    bc_u.set(2, fvm.BCType.Neumann, 0.0)
    bc_v.set(2, fvm.BCType.Neumann, 0.0)
    bc_p.set(2, fvm.BCType.Dirichlet, 0.0)

    # Patches 3/4: far-field velocity on the top/bottom sides.
    for patch in (3, 4):
        bc_u.set(patch, fvm.BCType.Dirichlet, 1.0)
        bc_v.set(patch, fvm.BCType.Dirichlet, 0.0)
        bc_p.set(patch, fvm.BCType.Neumann, 0.0)

    return bc_u, bc_v, bc_p


def solve_case(mesh, re):
    """Solve one Reynolds number and return the fields and SIMPLE result."""
    rho = 1.0
    mu = 1.0 / re

    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")
    velocity.u.set_constant(1.0)
    velocity.v.set_constant(0.0)
    pressure.set_constant(0.0)
    flux.set_zero()

    bc_u, bc_v, bc_p = configure_boundary_conditions(mesh)
    cfg = fvm.SimpleConfig()
    cfg.scheme = fvm.ConvectionScheme.Central
    cfg.tolerance = 1e-6
    cfg.max_iterations = 5000
    cfg.relaxation_u = 0.7
    cfg.relaxation_p = 0.3
    cfg.solver_config.tolerance = 1e-9
    cfg.solver_config.max_iterations = 20000
    cfg.pressure_solver_config.tolerance = 1e-9
    cfg.pressure_solver_config.max_iterations = 20000

    result = fvm.solve_simple(mesh, rho, mu, bc_u, bc_v, bc_p, cfg,
                              velocity, pressure, flux)
    print_simple_status(f"Re = {re}, cells = {mesh.cell_count()}", result)
    if not result.converged:
        raise RuntimeError(f"SIMPLE did not converge at Re = {re}")
    return velocity, pressure, result


def force_coefficients(mesh, velocity, pressure, rho, mu,
                       u_in=1.0, diameter=1.0):
    """Integrate pressure and wall shear on the cylinder patch.

    Returns the drag and lift coefficients; the lift is a symmetry
    diagnostic only and carries no assertion.
    """
    u = velocity.u.to_numpy()
    v = velocity.v.to_numpy()
    p = pressure.to_numpy()
    force_x = 0.0
    force_y = 0.0
    for face in mesh.patch_faces(0):
        area_x, area_y = mesh.face_area_vector(face)
        area = math.hypot(area_x, area_y)
        normal_x, normal_y = area_x / area, area_y / area
        tangent_x, tangent_y = -normal_y, normal_x

        owner = mesh.face_owner(face)
        force_x += p[owner] * area_x
        force_y += p[owner] * area_y

        cell_x, cell_y = mesh.cell_centroid(owner)
        face_x, face_y = mesh.face_centroid(face)
        normal_distance = -((cell_x - face_x) * normal_x
                            + (cell_y - face_y) * normal_y)
        u_tangent = u[owner] * tangent_x + v[owner] * tangent_y
        shear = mu * u_tangent / normal_distance
        force_x += shear * tangent_x * area
        force_y += shear * tangent_y * area

    dynamic_scale = rho * u_in ** 2 * diameter
    return 2.0 * force_x / dynamic_scale, 2.0 * force_y / dynamic_scale


def wake_length(mesh, velocity, tolerance=WAKE_TOLERANCE):
    """Find the furthest reversed-flow cell near the wake centreline.

    A cell counts when its centroid satisfies x > r (downstream of the
    rear stagnation point), |y| <= tolerance (roughly two thirds of the
    Re=40 recirculation-bubble half-width, one to two cell rows on the
    default grids) and u < 0.  The length is measured from the cylinder
    rear (x = r).
    """
    u = velocity.u.to_numpy()
    furthest_x = None
    for cell in range(mesh.cell_count()):
        x, y = mesh.cell_centroid(cell)
        if x > CYLINDER_RADIUS and abs(y) <= tolerance and u[cell] < 0.0:
            furthest_x = x if furthest_x is None else max(furthest_x, x)
    if furthest_x is None:
        return None
    return furthest_x - CYLINDER_RADIUS


def run_case(re, grid):
    """Generate, solve, post-process, and write one case."""
    n_theta, n_r, n_w, n_e, n_n, n_s = GRIDS[grid]
    mesh = read_su2_text(cylinder_rectangle_grid(
        n_theta=n_theta, n_r=n_r, radius=CYLINDER_RADIUS,
        half=SQUARE_HALF, domain=DOMAIN,
        n_w=n_w, n_e=n_e, n_n=n_n, n_s=n_s), name="cylinder.su2")

    velocity, pressure, result = solve_case(mesh, re)
    cd, cl = force_coefficients(mesh, velocity, pressure,
                                rho=1.0, mu=1.0 / re)
    wake = wake_length(mesh, velocity)
    wake_text = "nan" if wake is None else f"{wake:.6f}"
    print(f"Re={re} grid={grid} ntheta={n_theta}xnr={n_r} "
          f"cells={mesh.cell_count()} iterations={result.iterations} "
          f"converged={result.converged} Cd={cd:.6f} Cl={cl:+.6f} "
          f"L_w/D={wake_text}")
    write_results(mesh, velocity, pressure, "cylinder",
                  f"cylinder_re{re}_{grid}.vtu")
    return cd, cl, wake


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--re", type=int, default=40)
    parser.add_argument("--grid", choices=sorted(GRIDS), default="medium")
    parser.add_argument("--test", action="store_true",
                        help="assert the benchmark checks and exit non-zero "
                             "on failure")
    args = parser.parse_args()

    cd, cl, wake = run_case(args.re, args.grid)
    if args.test:
        band = CD_BANDS.get(args.re)
        if args.re == 40:
            cd_ok = band is not None and band[0] <= cd <= band[1]
            wake_ok = (wake is not None
                       and WAKE_LENGTH_BAND[0] <= wake <= WAKE_LENGTH_BAND[1])
            ok = cd_ok and wake_ok
            detail = f"Cd in range={cd_ok}, L_w/D in range={wake_ok}"
        elif args.re == 20:
            ok = band is not None and band[0] <= cd <= band[1]
            detail = f"Cd in range={ok}"
        else:
            ok = False
            detail = "no thresholds for this Reynolds number"
        report_test(ok, f"({detail})")


if __name__ == "__main__":
    main()
