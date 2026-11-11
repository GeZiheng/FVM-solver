"""Two-dimensional backward-facing step flow via SIMPLE.

The inlet section is 10 step-heights long and the downstream section is
30 step-heights long (expansion ratio 2).  The reattachment length is
measured from the downstream bottom-wall row: the first sign change of the
cell-centred streamwise velocity from negative to positive, with the zero
obtained by linear interpolation between the two cell centres.

Run:  python python/examples/backward_facing_step.py
      [--n N] [--re RE] [--test]
"""

import argparse

import numpy as np

from _common import (fvm, print_simple_status, read_su2_text, report_test,
                     write_results)

from mesh_su2 import l_shaped_bfs_grid

# Literature magnitude guard for the reattachment length of the 2:1
# expansion (Armaly et al. 1983 experiments / Erturk 2008 2D solutions;
# Reynolds-number definitions vary across sources, so the band is wide).
# Calibration 2026-10-31 (Upwind, uniform inlet 10h upstream):
#   Re= 50: x_r/h = 2.363 (n=8), 2.616 (n=16), 2.711 (n=24)
#   Re=100: x_r/h = 3.841 (n=8), 4.374 (n=16), 4.569 (n=24)
# Grid-to-grid drift n=16 -> n=24 is 3.6%/4.5% (apparent order ~2.5,
# Richardson extrapolation x_r(Re=100) ~ 4.7, inside the literature
# cluster ~4.5-5); the trend ratio is 1.67 against the 1.5 limit.
# Calibration 2026-11-10 (Central, fully developed parabolic inlet,
# Armaly-convention Re=389, Phase 5.3-3): x_r/h = 8.1025 (n=16),
# 8.0949 (n=24) -- drift 0.09%, grid-converged.  Armaly et al. 1983
# measure x_r/h ~ 7.9 at Re=389 (their Re = U_bulk*2h/nu, hence
# nu = 2/389 here); 2D solutions sit in the 8.2-8.6 cluster (the
# experiment turns 3D above Re ~ 400).  Band margins ~100x the drift.
REATTACHMENT_BANDS = {50: (2.0, 4.0), 100: (3.0, 7.0), 389: (7.2, 9.0)}

# Reynolds numbers using the fully developed parabolic inlet (bulk
# velocity one over the unit-height inlet channel, so mu = 1/re keeps
# the step-height Reynolds definition); lower ones stay uniform-inlet.
PARABOLIC_INLET_RES = (389,)


def configure_boundary_conditions(mesh, re):
    """Return velocity and pressure BCs; patch order follows the markers."""
    bc_u = fvm.BoundaryField(mesh)
    bc_v = fvm.BoundaryField(mesh)
    bc_p = fvm.BoundaryField(mesh)

    # Patch 0: uniform inlet (Re 50/100 historical calibration) or the
    # fully developed profile u = 6 s (1 - s), s = y - 1 on the unit
    # inlet channel (Phase 5.3-3).
    if re in PARABOLIC_INLET_RES:
        values = []
        for face in mesh.patch_faces(0):
            _, y = mesh.face_centroid(face)
            s = y - 1.0
            values.append(6.0 * s * (1.0 - s))
        bc_u.set_values(0, fvm.BCType.Dirichlet, np.array(values))
    else:
        bc_u.set(0, fvm.BCType.Dirichlet, 1.0)
    bc_v.set(0, fvm.BCType.Dirichlet, 0.0)
    bc_p.set(0, fvm.BCType.Neumann, 0.0)

    # Patch 1: pressure outlet.
    bc_u.set(1, fvm.BCType.Neumann, 0.0)
    bc_v.set(1, fvm.BCType.Neumann, 0.0)
    bc_p.set(1, fvm.BCType.Dirichlet, 0.0)

    # Patches 2 and 3: top wall and the composite bottom/step wall.
    for wall in (2, 3):
        bc_u.set(wall, fvm.BCType.Dirichlet, 0.0)
        bc_v.set(wall, fvm.BCType.Dirichlet, 0.0)
        bc_p.set(wall, fvm.BCType.Neumann, 0.0)

    return bc_u, bc_v, bc_p


def solve_case(mesh, re):
    """Solve one Reynolds number and return the fields and SIMPLE result."""
    rho = 1.0
    # Armaly et al. (1983) define Re = U_bulk * D_h / nu with the inlet
    # hydraulic diameter D_h = 2h (2D channel) and the parabolic-profile
    # bulk velocity; with U_bulk = 1 and h = 1 the Armaly-Re benchmark
    # point is nu = 2/re.  The legacy Re=50/100 uniform-inlet entries
    # keep the step-height convention (U_bulk * h / nu = 1/re) they were
    # calibrated under.
    mu = (2.0 / re) if re in PARABOLIC_INLET_RES else (1.0 / re)

    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")
    velocity.u.set_constant(1.0)
    velocity.v.set_constant(0.0)
    pressure.set_constant(0.0)
    flux.set_zero()

    bc_u, bc_v, bc_p = configure_boundary_conditions(mesh, re)
    cfg = fvm.SimpleConfig()
    # Re=389 uses Central (the long shear layer suffers under Upwind's
    # numerical diffusion); the low-Re entries keep their calibrated
    # Upwind setup.
    if re in PARABOLIC_INLET_RES:
        cfg.scheme = fvm.ConvectionScheme.Central
        cfg.max_iterations = 8000
    else:
        cfg.scheme = fvm.ConvectionScheme.Upwind
        cfg.max_iterations = 3000
    cfg.tolerance = 1e-6
    cfg.relaxation_u = 0.7
    cfg.relaxation_p = 0.3
    cfg.solver_config.tolerance = 1e-9
    cfg.solver_config.max_iterations = 20000
    cfg.pressure_solver_config.tolerance = 1e-9
    cfg.pressure_solver_config.max_iterations = 20000

    result = fvm.solve_simple(mesh, rho, mu, bc_u, bc_v, bc_p, cfg,
                              velocity, pressure, flux)
    print_simple_status(f"Re = {re}, n = {mesh.cell_count()} cells", result)
    if not result.converged:
        raise RuntimeError(f"SIMPLE did not converge at Re = {re}")
    return velocity, pressure, result


def reattachment_length(mesh, velocity):
    """Locate the first downstream negative-to-positive u crossing."""
    u = velocity.u.to_numpy()
    wall_cells = []
    for face in mesh.patch_faces(3):
        x, _ = mesh.face_centroid(face)
        if x > 0.0:  # The downstream floor; x = 0 is the step face.
            owner = mesh.face_owner(face)
            x_cell, _ = mesh.cell_centroid(owner)
            wall_cells.append((x_cell, float(u[owner])))
    wall_cells.sort(key=lambda item: item[0])

    for (x0, u0), (x1, u1) in zip(wall_cells, wall_cells[1:]):
        if u0 < 0.0 and u1 > 0.0:
            fraction = -u0 / (u1 - u0)
            return x0 + fraction * (x1 - x0)

    raise RuntimeError("no reattachment point found on the downstream floor")


def run_case(re, n):
    """Generate, solve, post-process, and write one case."""
    mesh = read_su2_text(l_shaped_bfs_grid(n), name="backward_facing_step.su2")

    velocity, pressure, result = solve_case(mesh, re)
    x_r = reattachment_length(mesh, velocity)
    print(f"Re={re} n={n} iterations={result.iterations} "
          f"converged={result.converged} x_r/h={x_r:.6f}")
    write_results(mesh, velocity, pressure, "backward_facing_step",
                  f"bfs_re{re}.vtu")
    return x_r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=16,
                        help="cells per step height (default: 16)")
    parser.add_argument("--re", type=int, default=None,
                        help="one Reynolds number; by default run 50 and 100")
    parser.add_argument("--test", action="store_true",
                        help="assert the reattachment-length checks and exit "
                             "non-zero on failure")
    args = parser.parse_args()

    re_values = (50, 100) if args.re is None else (args.re,)
    reattachment = {re: run_case(re, args.n) for re in re_values}

    if args.test:
        range_ok = {}
        for re, x_r in reattachment.items():
            band = REATTACHMENT_BANDS.get(re)
            range_ok[re] = band is not None and band[0] <= x_r <= band[1]
        trend_ok = True
        if len(reattachment) == 2:
            trend_ok = reattachment[100] >= 1.5 * reattachment[50]
        ok = all(range_ok.values()) and trend_ok
        report_test(ok, f"(range={range_ok}, trend={trend_ok})")


if __name__ == "__main__":
    main()
