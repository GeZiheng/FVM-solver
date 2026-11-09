"""Steady Poiseuille flow in a sheared channel with per-face velocities.

The sheared constant-section quad mesh (the end faces tilt by
atan(0.3) ~ 16.7 degrees) exercises the non-orthogonal corrections,
while per-face Dirichlet values on BOTH slanted end faces prescribe the
exact developed profile u = 4*y*(1-y), v = 0 (Phase 5.1 per-face BC
API).  Both end fluxes cancel exactly in the discrete quadrature, so
the pressure stays pure-Neumann with the analytic normal derivatives
dp/dn = +/- 8/sqrt(1+s^2) on the ends and zero on the walls, pinned by
the reference cell.  This deliberately avoids Dirichlet p on slanted
faces: without an explicit boundary completion term (registered future
work in docs/numerical.md) its boundary coefficient C_b carries a
shear-proportional, grid-independent offset that biases the profile by
~7% at shear = 0.3.  The remaining error is the O(h) tangential
truncation of Dirichlet u on slanted faces, with a small coefficient.
SIMPLE needs n_non_orthogonal_correctors = 1 (or gentler relaxation) to
converge at shear = 0.3 on finer grids.

Run:  python python/examples/skewed_poiseuille.py [--n N] [--test]
"""

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _path  # noqa: F401,E402

import numpy as np
import pyfvm as fvm

from mesh_su2 import sheared_channel

SHEAR = 0.3

# Calibration 2026-11-03 (Upwind, SIMPLE, mu = rho = 1, nc = 1):
#   n=16: profile err 1.412e-2, |v| 4.864e-3, flux mismatch 0 (exact)
#   n=32: profile err 6.784e-3, |v| 2.262e-3 (O(h), ratio ~2.1)
# The profile/v bands keep a 2x margin over the n=16 calibration; the
# quadrature band bounds the deterministic n=16 midpoint-rule value.
PROFILE_BAND = 3.0e-2
V_BAND = 1.2e-2
QUADRATURE_BAND = 2.0e-3

def parabola(y):
    return 4.0 * y * (1.0 - y)


def configure_boundary_conditions(mesh):
    """Per-face parabolic velocities on both slanted ends; Neumann p."""
    bc_u = fvm.BoundaryField(mesh)
    bc_v = fvm.BoundaryField(mesh)
    bc_p = fvm.BoundaryField(mesh)

    # Patches 0 (south) and 2 (north): no-slip walls.
    for wall in (0, 2):
        bc_u.set(wall, fvm.BCType.Dirichlet, 0.0)
        bc_v.set(wall, fvm.BCType.Dirichlet, 0.0)
        bc_p.set(wall, fvm.BCType.Neumann, 0.0)

    # The exact developed profile on both slanted ends (west = patch 3 is
    # the inlet, east = patch 1 the outlet).  Both discrete end fluxes
    # cancel exactly, which keeps the pure-Neumann pressure problem
    # compatible (reference-cell pinning supplies the gauge).
    dpdn = 8.0 / math.sqrt(1.0 + SHEAR * SHEAR)
    for end, sign in ((3, 1.0), (1, -1.0)):
        values = []
        for face in mesh.patch_faces(end):
            _, y = mesh.face_centroid(face)
            values.append(parabola(y))
        bc_u.set_values(end, fvm.BCType.Dirichlet, np.array(values))
        bc_v.set(end, fvm.BCType.Dirichlet, 0.0)
        # Analytic pressure normal derivative on the slanted end.
        bc_p.set(end, fvm.BCType.Neumann, sign * dpdn)

    return bc_u, bc_v, bc_p


def solve_case(mesh):
    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")
    # Start from the developed guess; the pressure stays at zero gauge.
    u = velocity.u.to_numpy()
    for cell in range(mesh.cell_count()):
        _, y = mesh.cell_centroid(cell)
        u[cell] = parabola(y)

    bc_u, bc_v, bc_p = configure_boundary_conditions(mesh)
    cfg = fvm.SimpleConfig()
    cfg.scheme = fvm.ConvectionScheme.Upwind
    cfg.tolerance = 1e-9
    cfg.max_iterations = 5000
    cfg.relaxation_u = 0.7
    cfg.relaxation_p = 0.3
    # One pressure sub-iteration stabilizes shear = 0.3 on finer grids.
    cfg.corrections.n_non_orthogonal_correctors = 1
    cfg.solver_config.tolerance = 1e-9
    cfg.solver_config.max_iterations = 20000
    cfg.pressure_solver_config.tolerance = 1e-9
    cfg.pressure_solver_config.max_iterations = 20000

    result = fvm.solve_simple(mesh, 1.0, 1.0, bc_u, bc_v, bc_p, cfg,
                              velocity, pressure, flux)
    state = "converged" if result.converged else "NOT converged"
    print(f"SIMPLE ({mesh.cell_count()} cells): {state} in "
          f"{result.iterations} iterations, continuity residual "
          f"{result.history[-1].continuity:.3e}")
    if not result.converged:
        raise RuntimeError("SIMPLE did not converge")
    return velocity, pressure, flux


def diagnostics(mesh, velocity, flux, n):
    """Return the inlet-flux and downstream-profile error measures."""
    phi = flux.to_numpy()
    u = velocity.u.to_numpy()
    v = velocity.v.to_numpy()

    # The west faces partition the slanted edge uniformly, so each face
    # carries Sx = -1/n exactly; the discrete analytic inlet flux is the
    # midpoint quadrature of the parabola (continuous value 2/3).
    discrete_flux = 0.0
    analytic_terms = []
    for face in mesh.patch_faces(3):
        _, y = mesh.face_centroid(face)
        discrete_flux += phi[face]
        analytic_terms.append(parabola(y))
    analytic_sum = -sum(analytic_terms) / n
    quadrature_gap = abs(analytic_sum + 2.0 / 3.0)
    flux_mismatch = abs(discrete_flux - analytic_sum)

    # Downstream section: the developed solution reproduces u(y) with
    # v -> 0; both errors are O(h) (Dirichlet-u tangential truncation on
    # the slanted end faces).
    profile_error = 0.0
    v_error = 0.0
    worst = (0.0, 0.0, 0.0)
    for cell in range(mesh.cell_count()):
        x, y = mesh.cell_centroid(cell)
        if x > 0.75:
            err = abs(u[cell] - parabola(y))
            if err > profile_error:
                profile_error = err
                worst = (x, y, u[cell])
            v_error = max(v_error, abs(v[cell]))

    print(f"inlet flux: solver={discrete_flux:.12f} "
          f"analytic(discrete)={analytic_sum:.12f} "
          f"mismatch={flux_mismatch:.3e}")
    print(f"quadrature gap to 2/3: {quadrature_gap:.3e}")
    print(f"downstream profile error: {profile_error:.3e}, "
          f"max|v|: {v_error:.3e}")
    print(f"worst profile cell: x={worst[0]:.3f} y={worst[1]:.3f} "
          f"u={worst[2]:.4f}")
    return flux_mismatch, quadrature_gap, profile_error, v_error


def write_result(mesh, velocity, pressure, n):
    output_dir = Path(__file__).resolve().parent / "output" / \
        "skewed_poiseuille"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"skewed_poiseuille_n{n}.vtu"
    fvm.write_vtu(str(output_path), mesh,
                  {"pressure": pressure}, {"velocity": velocity})
    print(f"Wrote {output_path}")


def run_case(n, write=True):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sheared_channel.su2")
        with open(path, "w", encoding="ascii") as fh:
            fh.write(sheared_channel(n, SHEAR))
        mesh = fvm.read_su2(path)

    velocity, pressure, flux = solve_case(mesh)
    measures = diagnostics(mesh, velocity, flux, n)
    if write:
        write_result(mesh, velocity, pressure, n)
    return measures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=16,
                        help="cells per edge (default: 16)")
    parser.add_argument("--test", action="store_true",
                        help="assert the analytic-profile checks and exit "
                         "non-zero on failure; bands assume --n 16")
    args = parser.parse_args()

    flux_mismatch, quadrature_gap, profile_error, v_error = run_case(
        args.n, write=not args.test)

    if args.test:
        checks = {
            "inlet_flux": flux_mismatch <= 1.0e-9,
            "quadrature": quadrature_gap <= QUADRATURE_BAND,
            "profile": profile_error <= PROFILE_BAND,
            "cross_flow": v_error <= V_BAND,
        }
        ok = all(checks.values())
        print(f"application test: {'PASS' if ok else 'FAIL'} {checks}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()


