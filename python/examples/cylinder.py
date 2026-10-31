"""Two-dimensional circular-cylinder flow via SIMPLE.

The O-grid has diameter D = 1 and an outer radius of 15 (blockage ratio
1/30).  Drag is evaluated from the cylinder patch with the project's
owner-relative boundary-face convention: the area vector points from the
adjacent fluid cell into the cylinder.

Run:  python python/examples/cylinder.py
      [--re RE] [--ntheta N] [--nr N] [--test]
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

from mesh_su2 import polar_ogrid

# Calibration 2026-10-31 (Central, O-grid 96x64 -> 128x80):
#   Re=40: Cd 1.5592 -> 1.5618 (grid drift 0.17%), L_w/D 2.028 -> 2.069
#   Re=20: Cd 2.0901 -> 2.0944 (grid drift 0.21%)
# Literature anchors: Fornberg 1980 Cd = 1.550 (Re=40) and ~2.00 (Re=20);
# Dennis & Chang 1.522 (Re=40).  The bands keep >50x the observed drift.
CD_BANDS = {20: (1.80, 2.30), 40: (1.40, 1.72)}
WAKE_LENGTH_BAND = (1.8, 2.8)
CYLINDER_RADIUS = 0.5
OUTER_RADIUS = 15.0


def configure_boundary_conditions(mesh):
    """Return velocity and pressure BCs; patch order follows the markers."""
    bc_u = fvm.BoundaryField(mesh)
    bc_v = fvm.BoundaryField(mesh)
    bc_p = fvm.BoundaryField(mesh)

    # Patch 0: no-slip cylinder.
    bc_u.set(0, fvm.BCType.Dirichlet, 0.0)
    bc_v.set(0, fvm.BCType.Dirichlet, 0.0)
    bc_p.set(0, fvm.BCType.Neumann, 0.0)

    # Patch 1: uniform inflow on the upstream semicircle.
    bc_u.set(1, fvm.BCType.Dirichlet, 1.0)
    bc_v.set(1, fvm.BCType.Dirichlet, 0.0)
    bc_p.set(1, fvm.BCType.Neumann, 0.0)

    # Patch 2: pressure outflow on the downstream semicircle.
    bc_u.set(2, fvm.BCType.Neumann, 0.0)
    bc_v.set(2, fvm.BCType.Neumann, 0.0)
    bc_p.set(2, fvm.BCType.Dirichlet, 0.0)

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

    result = fvm.solve_simple(mesh, rho, mu, bc_u, bc_v, bc_p, cfg,
                              velocity, pressure, flux)
    state = "converged" if result.converged else "NOT converged"
    print(f"SIMPLE (Re = {re}, cells = {mesh.cell_count()}): {state} in "
          f"{result.iterations} iterations, continuity residual "
          f"{result.history[-1].continuity:.3e}")
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


def wake_length(mesh, velocity, n_theta, cylinder_radius=CYLINDER_RADIUS):
    """Find the furthest reversed-flow cell near the wake centreline.

    The near-centreline tolerance is the local azimuthal cell spacing
    (``r_c * 2*pi/n_theta``), scaled by 0.75; the O-grid geometry is known
    here because this is post-processing only.
    """
    u = velocity.u.to_numpy()
    furthest_x = None
    angular_spacing = 2.0 * math.pi / n_theta
    for cell in range(mesh.cell_count()):
        x, y = mesh.cell_centroid(cell)
        radius = math.hypot(x, y)
        if (x > cylinder_radius and abs(y) < 0.75 * radius * angular_spacing
                and u[cell] < 0.0):
            furthest_x = x if furthest_x is None else max(furthest_x, x)
    if furthest_x is None:
        return None
    return furthest_x - cylinder_radius


def write_result(mesh, velocity, pressure, re):
    output_dir = Path(__file__).resolve().parent / "output" / "cylinder"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cylinder_re{re}.vtu"
    fvm.write_vtu(str(output_path), mesh,
                  {"pressure": pressure}, {"velocity": velocity})
    print(f"Wrote {output_path}")


def run_case(re, n_theta, n_r):
    """Generate, solve, post-process, and write one case."""
    cylinder_radius, outer_radius = CYLINDER_RADIUS, OUTER_RADIUS
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cylinder.su2")
        with open(path, "w", encoding="ascii") as fh:
            fh.write(polar_ogrid(n_theta, n_r, cylinder_radius, outer_radius))
        mesh = fvm.read_su2(path)

    velocity, pressure, result = solve_case(mesh, re)
    cd, cl = force_coefficients(mesh, velocity, pressure,
                                rho=1.0, mu=1.0 / re)
    wake = wake_length(mesh, velocity, n_theta)
    wake_text = "nan" if wake is None else f"{wake:.6f}"
    print(f"Re={re} ntheta={n_theta}xnr={n_r} cells={mesh.cell_count()} "
          f"iterations={result.iterations} converged={result.converged} "
          f"Cd={cd:.6f} Cl={cl:+.6f} L_w/D={wake_text}")
    write_result(mesh, velocity, pressure, re)
    return cd, cl, wake


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--re", type=int, default=40)
    parser.add_argument("--ntheta", type=int, default=96)
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--test", action="store_true",
                        help="assert the benchmark checks and exit non-zero "
                             "on failure")
    args = parser.parse_args()

    cd, cl, wake = run_case(args.re, args.ntheta, args.nr)
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
        print(f"application test: {'PASS' if ok else 'FAIL'} ({detail})")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
