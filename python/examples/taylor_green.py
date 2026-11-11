"""Decaying Taylor-Green vortices via PISO against the exact solution.

The single-mode 2D Taylor-Green solution on [0, pi]^2 with rho = 1 and
nu = 0.1:

    u = -A exp(-2 nu t) cos(x) sin(y)
    v = +A exp(-2 nu t) sin(x) cos(y)
    p = -(A^2 / 4) exp(-4 nu t) (cos(2x) + cos(2y))

is an exact unsteady Navier-Stokes solution.  All four sides carry the
exact per-face Dirichlet values of u, v and p at the end-of-step time
level t^(n+1) (the level of the implicit operator), refreshed before
every single-step solve_piso call (the Stokes-oscillating-plate
pattern).  The final state at T = 1 is compared in the discrete L2
norm.  The grid study 64^2 -> 128^2 -> 256^2 at one small shared dt
verifies the spatial second-order convergence; with dt proportional to
h a mixed dt^2/h-like transient term degrades the apparent order to
~1.4, so temporal refinement is not asserted (evidence in
docs/numerical.md).

Run:  python python/examples/taylor_green.py
      [--full] [--test]
"""

import argparse
import math

import numpy as np

from _common import (fvm, frame_interval, report_test, write_frame,
                     write_results)

RHO = 1.0
NU = 0.1
END_TIME = 1.0

# Calibration 2026-11-10 (CrankNicolson, n_correctors=2, BCs at t^{n+1}):
#   L2(u) at fixed dt = 0.00125: 64^2 1.927e-4, 128^2 3.884e-5
#   (ratio 4.96 -> spatial order ~2.3), 256^2 1.875e-5 (ratio 2.07 - a
#   first-order-in-dt mixed term, ~dt/h^2-like, catches up with the
#   spatial error there: halving dt at 256^2 halves the error to 9.21e-6).
#   The test asserts the clean 64 -> 128 spatial pair only; see
#   docs/numerical.md for the decomposition evidence.
ORDER_BAND = (1.7, 2.5)
U_L2_CAP_64 = 4.0e-4  # ~2x the measured 1.927e-4

# (nx, dt): one shared small dt so the spatial convergence dominates
# (with dt proportional to h the total apparent order drops to ~1.4
# from a mixed dt^2/h-like transient term).  The test path runs the
# first two; --full adds 256^2.
LEVELS = ((64, 0.00125), (128, 0.00125), (256, 0.00125))


def exact_fields(mesh, t):
    """Return (u, v, p) cell-centred arrays of the exact solution."""
    u = np.empty(mesh.cell_count())
    v = np.empty(mesh.cell_count())
    p = np.empty(mesh.cell_count())
    decay_u = math.exp(-2.0 * NU * t)
    decay_p = math.exp(-4.0 * NU * t)
    for cell in range(mesh.cell_count()):
        x, y = mesh.cell_centroid(cell)
        u[cell] = -decay_u * math.cos(x) * math.sin(y)
        v[cell] = decay_u * math.sin(x) * math.cos(y)
        p[cell] = -0.25 * decay_p * (math.cos(2.0 * x) + math.cos(2.0 * y))
    return u, v, p


def refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p, t):
    """Impose the exact per-face Dirichlet values at time ``t``."""
    for patch in range(mesh.patch_count()):
        u_faces, v_faces, p_faces = [], [], []
        for face in mesh.patch_faces(patch):
            x, y = mesh.face_centroid(face)
            decay_u = math.exp(-2.0 * NU * t)
            decay_p = math.exp(-4.0 * NU * t)
            u_faces.append(-decay_u * math.cos(x) * math.sin(y))
            v_faces.append(decay_u * math.sin(x) * math.cos(y))
            p_faces.append(-0.25 * decay_p
                           * (math.cos(2.0 * x) + math.cos(2.0 * y)))
        bc_u.set_values(patch, fvm.BCType.Dirichlet, np.array(u_faces))
        bc_v.set_values(patch, fvm.BCType.Dirichlet, np.array(v_faces))
        bc_p.set_values(patch, fvm.BCType.Dirichlet, np.array(p_faces))


def l2_error(computed, exact):
    """Discrete L2 norm of the difference."""
    diff = np.asarray(computed) - np.asarray(exact)
    return math.sqrt(float(np.mean(diff * diff)))


def run_level(nx, dt, frame_every=None):
    """Integrate one grid level to END_TIME; return (errors, fields).

    With `frame_every` set, frames are written every `frame_interval`
    steps (stretched so long levels stay within ~50 frames) plus one
    final frame.
    """
    mesh = fvm.CartesianMesh(nx, nx, 0.0, 0.0, math.pi, math.pi)
    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")

    u0, v0, p0 = exact_fields(mesh, 0.0)
    velocity.u.to_numpy()[:] = u0
    velocity.v.to_numpy()[:] = v0
    pressure.to_numpy()[:] = p0

    bc_u = fvm.BoundaryField(mesh)
    bc_v = fvm.BoundaryField(mesh)
    bc_p = fvm.BoundaryField(mesh)

    cfg = fvm.PisoConfig()
    cfg.dt = dt
    cfg.n_steps = 1
    cfg.n_correctors = 2
    cfg.time_scheme = fvm.TimeScheme.CrankNicolson
    cfg.scheme = fvm.ConvectionScheme.Central
    cfg.solver_config.tolerance = 1e-10
    cfg.solver_config.max_iterations = 20000
    cfg.pressure_solver_config.tolerance = 1e-10
    cfg.pressure_solver_config.max_iterations = 20000

    steps = round(END_TIME / dt)
    interval = frame_interval(steps) if frame_every is not None else None
    frame_index = 0
    last_frame_step = None
    if interval is not None:
        # Frame 0 is the t = 0 initial field (steps counted from 0).
        print(f"frame {frame_index}: t = {0.0:.4f}")
        write_frame(mesh, velocity, pressure, "taylor_green",
                    f"taylor_green_{nx}", frame_index)
        frame_index += 1
        last_frame_step = 0
    for step in range(steps):
        # BCs at the end-of-step level of the implicit operator
        # (empirically the most accurate choice; see module docstring).
        refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p,
                                    (step + 1) * dt)
        fvm.solve_piso(mesh, RHO, NU, bc_u, bc_v, bc_p, cfg,
                       velocity, pressure, flux)
        if interval is not None and (step + 1) % interval == 0:
            print(f"frame {frame_index}: t = {(step + 1) * dt:.4f}",
                  flush=True)
            write_frame(mesh, velocity, pressure, "taylor_green",
                        f"taylor_green_{nx}", frame_index)
            frame_index += 1
            last_frame_step = step + 1
    # The series always ends at the final state; skip if already written.
    if interval is not None and last_frame_step != steps:
        print(f"frame {frame_index}: t = {steps * dt:.4f}", flush=True)
        write_frame(mesh, velocity, pressure, "taylor_green",
                    f"taylor_green_{nx}", frame_index)

    u_exact, v_exact, p_exact = exact_fields(mesh, END_TIME)
    errors = (l2_error(velocity.u.to_numpy(), u_exact),
              l2_error(velocity.v.to_numpy(), v_exact),
              l2_error(pressure.to_numpy(), p_exact))
    print(f"nx={nx} dt={dt} steps={steps}: "
          f"L2(u)={errors[0]:.6e} L2(v)={errors[1]:.6e} "
          f"L2(p)={errors[2]:.6e}", flush=True)
    return errors, mesh, velocity, pressure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="run all three levels including 256^2")
    parser.add_argument("--test", action="store_true",
                        help="assert the convergence checks and exit "
                             "non-zero on failure")
    parser.add_argument("--frames-every", type=int, default=50,
                        help="write a VTK frame every K steps on the first "
                             "grid level (0 disables)")
    args = parser.parse_args()

    frame_every = None if args.test else (args.frames_every or None)
    levels = LEVELS if args.full else LEVELS[:2]
    results = [run_level(nx, dt, frame_every=frame_every if i == 0 else None)
               for i, (nx, dt) in enumerate(levels)]
    errors = [result[0] for result in results]

    if not args.test:
        if frame_every is None:
            errors_0, mesh, velocity, pressure = results[0]
            write_results(mesh, velocity, pressure, "taylor_green",
                          f"taylor_green_{levels[0][0]}.vti")
        return

    order = math.log(errors[0][0] / errors[1][0]) / math.log(2.0)
    order_ok = ORDER_BAND[0] <= order <= ORDER_BAND[1]
    cap_ok = errors[0][0] <= U_L2_CAP_64
    detail = (f"order={order:.3f} in {ORDER_BAND}, "
              f"L2(u,64)={errors[0][0]:.3e} <= {U_L2_CAP_64:.1e}")
    ok = order_ok and cap_ok
    if args.full:
        # The 128->256 ratio stalls at ~2 (measured 2.07) at this dt;
        # reported for the record but not asserted - see the calibration
        # comment and docs/numerical.md for the discriminator evidence.
        order2 = math.log(errors[1][0] / errors[2][0]) / math.log(2.0)
        detail += f", order(128->256)={order2:.3f} (reported, unasserted)"
    report_test(ok, f"({detail})")


if __name__ == "__main__":
    main()
