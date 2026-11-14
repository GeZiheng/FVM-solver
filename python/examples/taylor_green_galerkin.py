"""Intrusive velocity-only POD-Galerkin ROM driver on Taylor-Green.

Collects PISO snapshots exactly like taylor_green_rom.py (exact per-face
Dirichlet BCs refreshed at t^{n+1}, one snapshot per step), builds
centered POD bases for u and v, then integrates THREE Galerkin ROM
variants from t = 0 with rom.galerkin.GalerkinRom (explicit RK4):

- zero-pressure: the plain velocity-only ROM; its error growth
  quantifies the structural pressure leakage of a collocated
  non-divergence-free POD subspace without the grad(p) term;
- oracle: the residual evaluation is fed the EXACT pressure field at
  every RK4 stage time (through the GalerkinRom pressure hook and a
  bc_refresh that also refreshes the exact BCs); it isolates the
  machinery error from the leakage (oracle error settles at the FOM
  discretization level);
- projective (Phase 6.4): every RK4 stage first runs one predictive FOM
  pressure projection on the frozen reconstruction (the pressure
  half-step of a single PISO corrector) and the residual reads the
  projected pressure -- the quantified candidate against the leakage.

Output is an err(t) table (all variants vs the exact solution and vs
the collected FOM trajectory) plus a single-step consistency check
(one ROM RK4 step vs one FOM PISO step at the same dt -- different
integrators, so "small, not equal").  --frames optionally writes the
reconstructed ROM velocity as .vti frames.  --test runs a short
asserted configuration and writes no files.

Run:  python python/examples/taylor_green_galerkin.py
      [--nx N] [--dt DT] [--end-time T] [--rank R] [--dt-rom DT]
      [--frames N] [--test]
"""

import argparse
import math

import numpy as np

from _common import fvm, report_test, write_results

from rom import SnapshotCollector, pod_decompose
from rom.galerkin import GalerkinRom

from taylor_green import NU, RHO, exact_fields, refresh_boundary_conditions

CASE_DIR = "taylor_green_galerkin"
TABLE_ROWS = 20

# --test calibration (2026-11-13, parent-session measurements on the
# 16^2 / dt = 0.01 / T = 0.3 / full-rank configuration below; recorded
# in .agents/plans/phase6-3-galerkin-velocity-rom.md):
#   single-step relative difference 0.006  -> threshold 0.02 (~3x margin);
#   zero-pressure T = 0.3 error     0.176  -> band [0.09, 0.35] (the
#     lower bound rejects an accidental oracle match, the upper bound
#     caps the leakage band);
#   oracle T = 0.3 error 0.0091 (combined u+v relative L2, at the FOM
#     discretization floor: oracle|FOM = 0.0065) -> threshold 0.02
#     (~2.2x margin over both the measured value and the floor).
# Projective (Phase 6.4): hard ceiling 0.05 per plan D4 (below half the
# zero-pressure leakage; a measured value above it is a stop-and-report).
# Measured T = 0.3 error 0.0229 (between the oracle floor 0.0091 and the
# zero-pressure leakage 0.182).
TEST_SINGLE_STEP_MAX = 0.02
TEST_ZERO_PRESSURE_BAND = (0.09, 0.35)
TEST_ORACLE_MAX = 0.02
TEST_PROJECTIVE_MAX = 0.05


def make_piso_config(dt):
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
    return cfg


def run(nx, dt, end_time):
    """FOM snapshot collection, mirroring taylor_green_rom.run with
    snap_every = 1 (one snapshot per step, including t = 0)."""
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
    cfg = make_piso_config(dt)

    steps = round(end_time / dt)
    collector = SnapshotCollector()
    collector.append(0.0, velocity.u.to_numpy(), velocity.v.to_numpy(),
                     pressure.to_numpy())
    for step in range(steps):
        refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p,
                                    (step + 1) * dt)
        fvm.solve_piso(mesh, RHO, NU, bc_u, bc_v, bc_p, cfg,
                       velocity, pressure, flux)
        collector.append((step + 1) * dt, velocity.u.to_numpy(),
                         velocity.v.to_numpy(),
                         pressure.to_numpy())
    return mesh, collector, (bc_u, bc_v, bc_p)


def make_bc_refresh(mesh, bc_u, bc_v, bc_p, pressure=None):
    """bc_refresh(t) for GalerkinRom.advance: exact velocity BCs at the
    stage time; with an injected pressure field, also overwrite it with
    the exact pressure at the stage time (the oracle variant)."""
    def bc_refresh(t):
        refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p, t)
        if pressure is not None:
            _, _, p = exact_fields(mesh, t)
            pressure.to_numpy()[:] = p
    return bc_refresh


def build_rom(mesh, pod_u, pod_v, bc_u, bc_v, pressure=None,
              pressure_projection=False, pressure_solver_config=None,
              projection_dt=None):
    return GalerkinRom(mesh, NU, RHO, pod_u, pod_v, bc_u, bc_v,
                       fvm.ConvectionScheme.Central, pressure=pressure,
                       pressure_projection=pressure_projection,
                       pressure_solver_config=pressure_solver_config,
                       projection_dt=projection_dt)


def relative_error(computed, reference):
    diff = np.asarray(computed) - np.asarray(reference)
    return math.sqrt(float(np.mean(diff * diff))
                     / float(np.mean(np.asarray(reference) ** 2)))


def velocity_error(u, v, u_ref, v_ref):
    """Relative L2 velocity error over both components."""
    diff2 = (float(np.mean((u - u_ref) ** 2))
             + float(np.mean((v - v_ref) ** 2)))
    ref2 = (float(np.mean(u_ref ** 2)) + float(np.mean(v_ref ** 2)))
    return math.sqrt(diff2 / ref2)


def single_step_consistency(mesh, dt, pod_u, pod_v, bc_u, bc_v, bc_p):
    """One ROM RK4 step vs one FOM PISO step from the same t = 0 state.

    Returns the relative L2 velocity difference of the two end states
    (different integrators: small, not equal)."""
    u0, v0, _ = exact_fields(mesh, 0.0)

    fom = fvm.VectorField(mesh, "fom")
    fom.u.to_numpy()[:] = u0
    fom.v.to_numpy()[:] = v0
    fom_p = fvm.ScalarField(mesh, "fom_p")
    fom_flux = fvm.FaceFluxField(mesh, "fom_phi")
    refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p, dt)
    fvm.solve_piso(mesh, RHO, NU, bc_u, bc_v, bc_p, make_piso_config(dt),
                   fom, fom_p, fom_flux)

    rom = build_rom(mesh, pod_u, pod_v, bc_u, bc_v)
    initial = fvm.VectorField(mesh, "initial")
    initial.u.to_numpy()[:] = u0
    initial.v.to_numpy()[:] = v0
    rom.set_state_from_fields(initial)
    rom.advance(0.0, dt, 1,
                bc_refresh=make_bc_refresh(mesh, bc_u, bc_v, bc_p))
    u_rom, v_rom = rom.reconstruct_arrays()
    return velocity_error(u_rom, v_rom,
                          fom.u.to_numpy(), fom.v.to_numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--end-time", type=float, default=2.0)
    parser.add_argument("--rank", type=int, default=8,
                        help="POD rank r shared by u and v (capped at "
                             "the full rank)")
    parser.add_argument("--dt-rom", type=float, default=None,
                        help="ROM RK4 step (default: the FOM dt)")
    parser.add_argument("--projection-dt", type=float, default=None,
                        help="transient-diagonal projection (Phase 6.4 "
                             "revised D1): assemble the per-stage pressure "
                             "projection with a_P + rho V/dt (damped "
                             "residual step) instead of the steady "
                             "operator; None keeps the steady operator")
    parser.add_argument("--frames", type=int, default=0,
                        help="write the first N reconstructed velocity "
                             "frames of both variants as .vti (0 "
                             "disables; ignored with --test)")
    parser.add_argument("--test", action="store_true",
                        help="run the short asserted configuration "
                             "(16^2, dt=0.01, T=0.3, full rank); "
                             "writes no files")
    args = parser.parse_args()

    if args.test:
        nx, dt, end_time, rank = 16, 0.01, 0.3, None  # None = full rank
    else:
        nx, dt, end_time, rank = args.nx, args.dt, args.end_time, args.rank

    dt_rom = dt if (args.dt_rom is None or args.test) else args.dt_rom
    mesh, collector, (bc_u, bc_v, bc_p) = run(nx, dt, end_time)
    print(f"collected {len(collector)} snapshots of "
          f"{mesh.cell_count()} cells (one per step)")

    pod_u_full = pod_decompose(collector.u)
    pod_v_full = pod_decompose(collector.v)
    if rank is None:
        rank = pod_u_full.modes.shape[0]
    pod_u = pod_u_full.truncated(rank)
    pod_v = pod_v_full.truncated(rank)
    print(f"POD rank r = {rank} of {pod_u_full.modes.shape[0]} "
          f"(u reconstruction error "
          f"{pod_u_full.reconstruction_error(rank):.3e})")

    single_step = single_step_consistency(mesh, dt, pod_u, pod_v,
                                          bc_u, bc_v, bc_p)
    print(f"single-step consistency (1 ROM RK4 vs 1 FOM PISO, dt={dt}): "
          f"relative difference {single_step:.4f}")

    # Three variants: zero pressure (leakage), oracle (exact pressure
    # injected at every RK4 stage time) and projective (Phase 6.4: one
    # FOM pressure projection per stage; the pressure solver mirrors the
    # FOM pressure precision).
    oracle_pressure = fvm.ScalarField(mesh, "oracle_pressure")
    fom_pressure_cfg = make_piso_config(dt).pressure_solver_config
    rom_zero = build_rom(mesh, pod_u, pod_v, bc_u, bc_v)
    rom_oracle = build_rom(mesh, pod_u, pod_v, bc_u, bc_v,
                           pressure=oracle_pressure)
    rom_projective = build_rom(mesh, pod_u, pod_v, bc_u, bc_v,
                               pressure_projection=True,
                               pressure_solver_config=fom_pressure_cfg,
                               projection_dt=args.projection_dt)
    initial = fvm.VectorField(mesh, "initial")
    initial.u.to_numpy()[:] = collector.u[0]
    initial.v.to_numpy()[:] = collector.v[0]
    rom_zero.set_state_from_fields(initial)
    rom_oracle.set_state_from_fields(initial)
    rom_projective.set_state_from_fields(initial)

    n_steps = round(end_time / dt_rom)
    refresh_plain = make_bc_refresh(mesh, bc_u, bc_v, bc_p)
    refresh_oracle = make_bc_refresh(mesh, bc_u, bc_v, bc_p,
                                     pressure=oracle_pressure)
    hist_zero = rom_zero.advance(0.0, dt_rom, n_steps,
                                 bc_refresh=refresh_plain,
                                 record_every=1)
    hist_oracle = rom_oracle.advance(0.0, dt_rom, n_steps,
                                     bc_refresh=refresh_oracle,
                                     record_every=1)
    hist_projective = rom_projective.advance(0.0, dt_rom, n_steps,
                                             bc_refresh=refresh_plain,
                                             record_every=1)
    print(f"projective last-stage projection (imb, du, dv, dp): "
          f"{rom_projective.last_projection}")

    # err(t) table: all variants against the exact solution and the
    # FOM trajectory (FOM index = round(t / dt); exact matches because
    # dt_rom defaults to dt and record_every = 1).
    print(f"{'t':>7} {'zero|exact':>12} {'oracle|exact':>13} "
          f"{'proj|exact':>12} {'zero|FOM':>12} {'oracle|FOM':>12} "
          f"{'proj|FOM':>12}")
    times = hist_zero[0]
    stride = max(1, len(times) // TABLE_ROWS)
    rows = list(range(0, len(times), stride))
    if rows[-1] != len(times) - 1:
        rows.append(len(times) - 1)
    finals = {}
    for row in rows:
        t = times[row]
        a_u, a_v = hist_zero[1][row], hist_zero[2][row]
        u_zero = pod_u.mean + a_u @ pod_u.modes
        v_zero = pod_v.mean + a_v @ pod_v.modes
        a_u, a_v = hist_oracle[1][row], hist_oracle[2][row]
        u_orc = pod_u.mean + a_u @ pod_u.modes
        v_orc = pod_v.mean + a_v @ pod_v.modes
        a_u, a_v = hist_projective[1][row], hist_projective[2][row]
        u_prj = pod_u.mean + a_u @ pod_u.modes
        v_prj = pod_v.mean + a_v @ pod_v.modes
        u_ex, v_ex, _ = exact_fields(mesh, t)
        idx = min(round(t / dt), len(collector.u) - 1)
        u_fom, v_fom = collector.u[idx], collector.v[idx]
        errs = (velocity_error(u_zero, v_zero, u_ex, v_ex),
                velocity_error(u_orc, v_orc, u_ex, v_ex),
                velocity_error(u_prj, v_prj, u_ex, v_ex),
                velocity_error(u_zero, v_zero, u_fom, v_fom),
                velocity_error(u_orc, v_orc, u_fom, v_fom),
                velocity_error(u_prj, v_prj, u_fom, v_fom))
        finals[t] = errs
        print(f"{t:7.3f} {errs[0]:12.5f} {errs[1]:13.5f} "
              f"{errs[2]:12.5f} {errs[3]:12.5f} {errs[4]:12.5f} "
              f"{errs[5]:12.5f}")

    if args.frames > 0 and not args.test:
        stride = max(1, len(times) // args.frames)
        frame_velocity = fvm.VectorField(mesh, "rom_velocity")
        variants = (("rom_zero", rom_zero, hist_zero),
                    ("rom_oracle", rom_oracle, hist_oracle))
        for row in list(range(0, len(times), stride))[:args.frames]:
            for name, rom, history in variants:
                # the ROM holds its final state; restore the recorded
                # coefficient row before writing the frame.
                rom.a_u = history[1][row].copy()
                rom.a_v = history[2][row].copy()
                rom.write_state(frame_velocity)
                write_results(mesh, frame_velocity, None, CASE_DIR,
                              f"{name}_{row:04d}.vti", scalars={},
                              vectors={"velocity": frame_velocity})

    if args.test:
        checks = {}
        detail = [f"single_step={single_step:.4f}"]
        checks["single_step"] = single_step < TEST_SINGLE_STEP_MAX
        checks["single_step_nonzero"] = single_step > 0.0
        t_end = max(finals)
        zero_err = finals[t_end][0]
        oracle_err = finals[t_end][1]
        checks["zero_pressure_band"] = (
            TEST_ZERO_PRESSURE_BAND[0] <= zero_err
            <= TEST_ZERO_PRESSURE_BAND[1])
        detail.append(f"zero_T={t_end:.2f}_err={zero_err:.4f} in "
                      f"{TEST_ZERO_PRESSURE_BAND}")
        checks["oracle"] = oracle_err <= TEST_ORACLE_MAX
        detail.append(f"oracle_T={t_end:.2f}_err={oracle_err:.4f} <= "
                      f"{TEST_ORACLE_MAX}")
        projective_err = finals[t_end][2]
        checks["projective"] = projective_err <= TEST_PROJECTIVE_MAX
        detail.append(f"projective_T={t_end:.2f}_err={projective_err:.4f} "
                      f"<= {TEST_PROJECTIVE_MAX}")
        ok = all(checks.values())
        report_test(ok, f"({', '.join(detail)}, checks={checks})")
        return


if __name__ == "__main__":
    main()
