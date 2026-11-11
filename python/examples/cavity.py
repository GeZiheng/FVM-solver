"""Lid-driven cavity via SIMPLE, compared against the Ghia et al. (1982)
benchmark on the centrelines.

The cavity is a unit square with a unit-speed lid; with rho = 1 the Reynolds
number is Re = 1/mu.  Run with --re {100,400,1000,3200}; --test asserts the
centreline errors against the tabulated reference data and exits non-zero on
failure.  Re = 3200 uses a 256x256 grid (Phase 5.3-1); the linear solvers can
be swapped with --backend/--method (momentum) and --p-backend/--p-method
(pressure) for the AMG vs BiCGSTAB-ILUT showcase recorded in
docs/numerical.md.

Run:  python python/examples/cavity.py [--re {100,400,1000,3200}] [--test]
      python python/examples/cavity.py --re 3200 --backend amgcl
"""

import argparse
import time

import numpy as np

from _common import (fvm, print_simple_status, report_test, write_results)

# Ghia, Ghia & Shin (1982), Tables I/II: centreline velocity profiles on a
# 129x129 grid.  Columns are Re = 100 / 400 / 1000.
GHIA_Y = [
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
    0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000,
]
GHIA_U = {
    100: [1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
          -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
          -0.04775, -0.04192, -0.03717, 0.00000],
    400: [1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256,
          0.02135, -0.11477, -0.17119, -0.32726, -0.24299, -0.14612,
          -0.10338, -0.09266, -0.08186, 0.00000],
    1000: [1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719,
           0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730,
           -0.22220, -0.20196, -0.18109, 0.00000],
    # (y=0.4531, Re=3200): the original prints -0.86636, a widely reproduced
    # erroneous outlier (non-physical between the -0.04272/-0.24427
    # neighbours, breaks the monotone cross-Re trend -0.10648(1000) ->
    # -0.07404(5000); flagged e.g. by Tomczak 2021, arXiv:2108.13241).
    # Corrected to -0.07585 from our converged 256x256 solution (128x128
    # probes: -0.07802/-0.07316 at relax_u 0.7/0.5); the station is excluded
    # from the u error metrics (ANOMALOUS_STATIONS_U) since the correction
    # is ours, not an independent reference.
    3200: [1.00000, 0.53236, 0.48296, 0.46547, 0.46101, 0.34682, 0.19791,
           0.07156, -0.04272, -0.07585, -0.24427, -0.34323, -0.41933,
           -0.37827, -0.35344, -0.32407, 0.00000],
}
GHIA_X = [
    1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
    0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000,
]
GHIA_V = {
    100: [0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
          -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317,
          0.10890, 0.10091, 0.09233, 0.00000],
    # (x=0.9063, Re=400): the original prints -0.23827, a widely reproduced
    # typo (non-monotone between -0.22847 and -0.44993); corrected with the
    # mean of two independent high-accuracy solutions (Wan, -0.37828;
    # Chen et al., -0.38825).
    400: [0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.38327,
          -0.44993, -0.38598, 0.05188, 0.30174, 0.30203, 0.28124, 0.22965,
          0.20920, 0.19713, 0.18360, 0.00000],
    1000: [0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51500,
           -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627,
           0.30353, 0.29012, 0.27485, 0.00000],
    3200: [0.00000, -0.39017, -0.47425, -0.52357, -0.54053, -0.44307,
           -0.37401, -0.31184, 0.00999, 0.28188, 0.29030, 0.37119, 0.42768,
           0.41906, 0.40917, 0.39560, 0.00000],
}

# Station indices (into GHIA_Y/GHIA_X) excluded from the error metrics:
# entries whose printed reference value is a known transcription artifact
# corrected above from our own solution -- keeping them would make the
# comparison self-referential. The artifact is in the u table only; the
# v-table value at the same station is a legitimate reference point.
ANOMALOUS_STATIONS_U = {3200: (9,)}
ANOMALOUS_STATIONS_V = {}


def run(re, momentum_solver=None, pressure_solver=None):
    """Solve the cavity at the given Re and return the centrelines.

    momentum_solver / pressure_solver override the linear-solver
    (backend, method) pair of the momentum and pressure equations; None
    keeps the historical defaults (eigen BiCGSTAB / eigen CG).
    """
    n = 256 if re == 3200 else (128 if re == 1000 else 64)
    mesh = fvm.CartesianMesh(n, n, 0.0, 0.0, 1.0, 1.0)
    rho, mu = 1.0, 1.0 / re

    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")

    bc_u = fvm.BoundaryField(mesh)  # lid moves with U = 1, other walls no-slip
    bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 1.0)
    bc_u.set(fvm.BoundaryField.East, fvm.BCType.Dirichlet, 0.0)
    bc_u.set(fvm.BoundaryField.West, fvm.BCType.Dirichlet, 0.0)
    bc_u.set(fvm.BoundaryField.South, fvm.BCType.Dirichlet, 0.0)

    bc_v = fvm.BoundaryField(mesh)  # no penetration on all walls
    for side in (fvm.BoundaryField.North, fvm.BoundaryField.East,
                 fvm.BoundaryField.West, fvm.BoundaryField.South):
        bc_v.set(side, fvm.BCType.Dirichlet, 0.0)

    bc_p = fvm.BoundaryField(mesh)  # zero-gradient pressure on all walls

    cfg = fvm.SimpleConfig()
    # Re = 3200 (Phase 5.3-1): the outer criterion is loosened to 2e-5. At
    # 256x256 the scaled u/v residuals stall in the 1e-6..1e-5 band for
    # thousands of iterations (corner slow modes) while the centreline
    # profiles are already Ghia-close (calibration: the residual crosses
    # 2e-5 around iteration ~2560, profiles recorded in docs/numerical.md).
    cfg.tolerance = 2e-5 if re == 3200 else 1e-6
    cfg.max_iterations = 3000 if re == 3200 else (
        6000 if re == 1000 else 3000)
    cfg.relaxation_u = 0.6 if re == 1000 else 0.7
    cfg.relaxation_p = 0.3
    cfg.scheme = fvm.ConvectionScheme.Central
    cfg.solver_config.tolerance = 1e-9
    cfg.solver_config.max_iterations = 2000
    cfg.pressure_solver_config.tolerance = 1e-9
    cfg.pressure_solver_config.max_iterations = 2000
    if momentum_solver:
        cfg.solver_config.backend, cfg.solver_config.method = momentum_solver
    if pressure_solver:
        cfg.pressure_solver_config.backend, \
            cfg.pressure_solver_config.method = pressure_solver

    t0 = time.perf_counter()
    result = fvm.solve_simple(mesh, rho, mu, bc_u, bc_v, bc_p, cfg,
                              velocity, pressure, flux)
    wall = time.perf_counter() - t0
    tag = ""
    if momentum_solver:
        tag = f", {momentum_solver[0]}/{momentum_solver[1]}"
    if pressure_solver:
        tag += f", p {pressure_solver[0]}/{pressure_solver[1]}"
    print_simple_status(f"Re = {re}, {n}x{n}{tag}", result,
                        suffix=f", {wall:.1f} s wall time")

    if not result.converged:
        raise RuntimeError(f"SIMPLE did not converge at Re = {re}")

    # Cell-centred coordinates; row j = y_j, column i = x_i.
    x = (np.arange(n) + 0.5) * mesh.dx()
    y = (np.arange(n) + 0.5) * mesh.dy()
    u = velocity.u.to_numpy().reshape(n, n)
    v = velocity.v.to_numpy().reshape(n, n)

    u_center = 0.5 * (u[:, n // 2 - 1] + u[:, n // 2])      # u(x = 0.5)
    v_center = 0.5 * (v[n // 2 - 1, :] + v[n // 2, :])      # v(y = 0.5)

    return mesh, velocity, pressure, (x, u_center, y, v_center)


def profile_errors(re, x, u_center, y, v_center):
    """L-inf / L2 errors of the centreline profiles vs the Ghia tables."""
    # Extend the cell-centred profiles with the exactly imposed wall values
    # (u = 1 on the lid, 0 elsewhere; v = 0 on all walls): without them
    # np.interp would clamp the table's wall points to the last cell centre.
    y_full = np.concatenate(([0.0], y, [1.0]))
    u_full = np.concatenate(([0.0], u_center, [1.0]))
    x_full = np.concatenate(([0.0], x, [1.0]))
    v_full = np.concatenate(([0.0], v_center, [0.0]))
    u_num = np.interp(GHIA_Y, y_full, u_full)
    v_num = np.interp(GHIA_X, x_full, v_full)
    keep_u = [k for k in range(len(GHIA_Y))
              if k not in ANOMALOUS_STATIONS_U.get(re, ())]
    keep_v = [k for k in range(len(GHIA_X))
              if k not in ANOMALOUS_STATIONS_V.get(re, ())]
    u_err = np.abs(u_num[keep_u] - np.array(GHIA_U[re])[keep_u])
    v_err = np.abs(v_num[keep_v] - np.array(GHIA_V[re])[keep_v])
    metrics = {
        "u_linf": u_err.max(),
        "u_l2": np.sqrt(np.mean(u_err ** 2)),
        "v_linf": v_err.max(),
        "v_l2": np.sqrt(np.mean(v_err ** 2)),
    }
    print(f"Re = {re}: |u - Ghia|  L-inf {metrics['u_linf']:.4e}  "
          f"L2 {metrics['u_l2']:.4e}   |v - Ghia|  L-inf "
          f"{metrics['v_linf']:.4e}  L2 {metrics['v_l2']:.4e}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--re", type=int, choices=(100, 400, 1000, 3200),
                        default=100)
    parser.add_argument("--test", action="store_true",
                        help="assert the Ghia centrelines and exit non-zero "
                             "on failure")
    parser.add_argument("--backend", choices=("eigen", "amgcl"),
                        help="momentum linear-solver backend (default eigen)")
    parser.add_argument("--method",
                        choices=("bicgstab", "cg", "bicgstab_ilut"),
                        help="momentum linear-solver method (default "
                             "bicgstab)")
    parser.add_argument("--p-backend", dest="p_backend",
                        choices=("eigen", "amgcl"),
                        help="pressure linear-solver backend (default eigen)")
    parser.add_argument("--p-method", dest="p_method",
                        choices=("cg", "bicgstab", "bicgstab_ilut"),
                        help="pressure linear-solver method (default cg)")
    args = parser.parse_args()

    mom_solver = None
    if args.backend or args.method:
        mom_solver = (args.backend or "eigen", args.method or "bicgstab")
    p_solver = None
    if args.p_backend or args.p_method:
        p_solver = (args.p_backend or "eigen", args.p_method or "cg")

    mesh, velocity, pressure, (x, u_center, y, v_center) = run(
        args.re, mom_solver, p_solver)
    metrics = profile_errors(args.re, x, u_center, y, v_center)

    write_results(mesh, velocity, pressure, "cavity",
                  f"cavity_re{args.re}.vti")

    if args.test:
        # Calibrated in Phase 2 (Central scheme); >= 2x headroom over the
        # observed maxima (L-inf: 8.6e-3 / 8.7e-3 / 1.2e-2, L2 half that).
        # Re = 3200: observed at the 256x256 3000-iteration calibration run
        # (worst L-inf 3.7e-2, L2 1.6e-2), times a 1.2 lag allowance for the
        # earlier (~iteration 2560) tolerance-2e-5 stopping state, times 2
        # (1.55e-2 * 1.2 * 2 = 3.7e-2).
        thresholds = {100: (2.0e-2, 1.0e-2), 400: (2.0e-2, 1.2e-2),
                      1000: (3.0e-2, 1.5e-2), 3200: (9.0e-2, 3.7e-2)}
        limit, limit_l2 = thresholds[args.re]
        worst = max(metrics["u_linf"], metrics["v_linf"])
        worst_l2 = max(metrics["u_l2"], metrics["v_l2"])
        ok = worst <= limit and worst_l2 <= limit_l2
        report_test(ok, f"(max profile L-inf {worst:.4e} <= {limit:.1e}, "
                        f"L2 {worst_l2:.4e} <= {limit_l2:.1e})")


if __name__ == "__main__":
    main()
