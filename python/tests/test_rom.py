"""ROM toolkit tests: snapshot npz contract, POD/DMD algebra, and a
coarse Taylor-Green end-to-end smoke.

Layers (plan phase 6.1, test table):
- pure-algebra unit tests (POD exact reconstruction, POD centering,
  DMD synthetic signal) -- numpy only, they never import pyfvm;
- npz contract tests (roundtrip plus error paths) -- numpy only;
- one coarse-grid Taylor-Green smoke that drives PISO through the
  binding, collects snapshots with SnapshotCollector and checks the
  contract and POD energy concentration end to end.  Its exact-BC
  pattern is inlined from examples/taylor_green.py (tests do not
  import example code) with the copy limited to these few helpers.

Run:  python -m unittest discover -s python/tests -v
(also picked up by the ctest entry `pyfvm_tests` when
FVM_BUILD_PYTHON=ON)
"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np

# Repository python/ directory on sys.path, for the rom package (the
# same bootstrap test_pyfvm.py uses for _path).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from rom import SnapshotCollector, dmd, load_snapshots, pod_decompose  # noqa: E402
from rom.galerkin import GalerkinRom  # noqa: E402
from rom.precomputed import (PrecomputedGalerkinRom,  # noqa: E402
                             ReducedPressureRom,
                             fit_reduced_operators, fit_rhie_chow_fields,
                             make_fit_samples,
                             projected_pressure_snapshots)

# The algebra and contract tests above never touch pyfvm; only the
# Taylor-Green smoke needs the built extension (_path honors
# FVM_PYFVM_DIR, which the ctest entry injects).
try:
    import _path  # noqa: F401,E402
    import pyfvm  # noqa: E402
    HAVE_PYFVM = True
except (ImportError, RuntimeError):
    pyfvm = None
    HAVE_PYFVM = False


class TestPodAlgebra(unittest.TestCase):
    """Pure-algebra POD unit tests (rows = snapshots, modes = rows)."""

    def test_exact_reconstruction_of_low_rank_matrix(self):
        rng = np.random.default_rng(7)
        left = rng.standard_normal((20, 3))
        right = rng.standard_normal((3, 200))
        X = left @ right  # exact rank 3, shape (S=20, N=200)

        result = pod_decompose(X)

        self.assertEqual(result.modes.shape, (20, 200))
        self.assertEqual(result.mean.shape, (200,))
        self.assertEqual(result.singular_values.shape, (20,))
        # rank-3 data: the first 3 modes carry everything (measured
        # 2.3e-16, tolerance 1e-12).
        self.assertLess(result.reconstruction_error(3), 1e-12)
        errors = [result.reconstruction_error(r) for r in range(0, 21)]
        self.assertTrue(all(errors[i] >= errors[i + 1] - 1e-15
                            for i in range(len(errors) - 1)), errors)
        self.assertAlmostEqual(result.energy.sum(), 1.0, delta=1e-12)
        self.assertAlmostEqual(result.cumulative_energy[-1], 1.0,
                               delta=1e-12)

    def test_centering_returns_mean_and_reconstructs(self):
        rng = np.random.default_rng(11)
        fluct = rng.standard_normal((25, 12))
        fluct -= fluct.mean(axis=0)  # zero time-mean fluctuation
        X = 5.0 + fluct

        centered = pod_decompose(X, center=True)

        # the mean field is exactly the constant offset (measured 9e-16)
        np.testing.assert_allclose(centered.mean, 5.0, atol=1e-12)
        np.testing.assert_allclose(
            centered.reconstruct(centered.project(X)), X, atol=1e-10)
        # zero coefficients reconstruct the mean field itself: the
        # centered modes carry none of the constant offset.
        zeros = np.zeros((3, centered.modes.shape[0]))
        np.testing.assert_allclose(centered.reconstruct(zeros),
                                   np.tile(centered.mean, (3, 1)),
                                   atol=1e-12)

        uncentered = pod_decompose(X, center=False)
        self.assertTrue(np.all(uncentered.mean == 0.0))
        np.testing.assert_allclose(
            uncentered.reconstruct(uncentered.project(X)), X, atol=1e-10)


class TestDmdAlgebra(unittest.TestCase):
    """Pure-algebra exact-DMD test on a two-sine signal."""

    def test_two_sine_frequencies_recovered(self):
        dt = 0.01
        times = np.arange(401) * dt
        j = np.arange(40)
        # 0.7 Hz with amplitude 1.0 + 2.3 Hz with amplitude 0.5, each
        # with a per-column phase so the data fills the space.
        X = (1.0 * np.sin(2 * np.pi * 0.7 * times[:, None] + j[None, :])
             + 0.5 * np.sin(2 * np.pi * 2.3 * times[:, None]
                            + 0.3 * j[None, :]))

        result = dmd(X, dt)

        self.assertEqual(result.rank, 4)
        self.assertEqual(result.modes.shape, (40, 4))  # columns = modes
        freqs = np.abs(result.omega.imag) / (2.0 * np.pi)
        oscillating = np.sort(freqs[freqs > 1e-8])
        # each expected frequency appears as a conjugate pair (measured
        # errors ~2e-15, tolerance 1e-6)
        self.assertEqual(oscillating.size, 4)
        np.testing.assert_allclose(oscillating[:2], 0.7, atol=1e-6)
        np.testing.assert_allclose(oscillating[2:], 2.3, atol=1e-6)
        # the larger-amplitude sinusoid dominates (measured 1e-15)
        self.assertAlmostEqual(result.dominant_frequency(), 0.7, delta=1e-6)
        # amplitudes fit the first snapshot: the reconstruction matches
        # the signal at every sampled time (measured 1.7e-13)
        np.testing.assert_allclose(result.reconstruct(times), X, atol=1e-8)


class TestSnapshotContract(unittest.TestCase):
    """npz contract roundtrip and error paths (numpy only)."""

    def test_save_load_roundtrip(self):
        rng = np.random.default_rng(3)
        n = 16
        collector = SnapshotCollector()
        for t in (0.0, 0.5, 1.1):
            base = rng.standard_normal(n)
            collector.append(t, base, base + 1.0, base + 2.0)
        params = {"nx": n, "nu": 0.1, "re": 100.0}
        cx = np.linspace(0.0, math.pi, n)
        cy = np.linspace(0.0, math.pi, n)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snap.npz")
            collector.save(path, case="roundtrip", params=params,
                           cx=cx, cy=cy)
            with np.load(path) as npz:
                for key in ("format_version", "case", "params", "t", "u",
                            "v", "p", "cx", "cy"):
                    self.assertIn(key, npz.files)
            data = load_snapshots(path)

        self.assertEqual(data.format_version, 1)
        self.assertEqual(data.case, "roundtrip")
        self.assertEqual(data.params, params)
        self.assertEqual(data.t.dtype, np.float64)
        np.testing.assert_array_equal(data.t, collector.times)
        for field in ("u", "v", "p"):
            stored = getattr(data, field)
            self.assertEqual(stored.shape, (3, n))
            self.assertEqual(stored.dtype, np.float64)
            np.testing.assert_array_equal(stored,
                                          getattr(collector, field))
        np.testing.assert_array_equal(data.cx, cx)
        np.testing.assert_array_equal(data.cy, cy)

    def test_save_rejects_invalid_stacks(self):
        def reject(collector):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    collector.save(os.path.join(tmp, "bad.npz"))

        # times not strictly increasing
        collector = SnapshotCollector()
        for t in (0.0, 0.1, 0.1):
            collector.append(t, np.zeros(4), np.zeros(4), np.zeros(4))
        reject(collector)
        # fields disagree in size within one snapshot
        collector = SnapshotCollector()
        for t in (0.0, 0.1):
            collector.append(t, np.zeros(4), np.zeros(5), np.zeros(4))
        reject(collector)
        # a field changed size between snapshots
        collector = SnapshotCollector()
        collector.append(0.0, np.zeros(4), np.zeros(4), np.zeros(4))
        collector.append(0.1, np.zeros(5), np.zeros(5), np.zeros(5))
        reject(collector)

    def test_load_rejects_unknown_format_version(self):
        collector = SnapshotCollector()
        for k in range(2):
            collector.append(0.5 * k, np.zeros(6), np.zeros(6), np.zeros(6))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v2.npz")
            collector.save(path)
            with np.load(path) as npz:
                payload = {key: npz[key] for key in npz.files}
            payload["format_version"] = np.int64(2)
            np.savez(path, **payload)
            with self.assertRaises(ValueError):
                load_snapshots(path)


@unittest.skipIf(not HAVE_PYFVM, "pyfvm extension not built")
class TestTaylorGreenSnapshotSmoke(unittest.TestCase):
    """Coarse end-to-end smoke: decaying Taylor-Green vortices on
    [0, pi]^2 (rho = 1, nu = 0.1), PISO with Crank-Nicolson and Central
    convection, exact per-face Dirichlet values refreshed at t^{n+1}
    before every single-step call, snapshots collected per step and
    pushed through the npz contract.  Machine-level assertions only:
    no physical accuracy is claimed at this resolution."""

    NX = 32
    DT = 0.005
    STEPS = 100
    RHO = 1.0
    NU = 0.1
    # Calibration 2026-11-12 (this exact configuration):
    #   reconstruction_error(r_full) = 0.0 exactly (full-rank identity),
    #   cumulative_energy[2] = 0.9999928 -> threshold 0.9999 leaves a
    #   ~14x margin on the observed energy deficiency (1 - e = 7.2e-6).
    ENERGY_RANK = 2
    ENERGY_THRESHOLD = 0.9999

    def _exact_fields(self, mesh, t):
        """Cell-centred exact (u, v, p) at time ``t``."""
        count = mesh.cell_count()
        u = np.empty(count)
        v = np.empty(count)
        p = np.empty(count)
        decay_u = math.exp(-2.0 * self.NU * t)
        decay_p = math.exp(-4.0 * self.NU * t)
        for cell in range(count):
            x, y = mesh.cell_centroid(cell)
            u[cell] = -decay_u * math.cos(x) * math.sin(y)
            v[cell] = decay_u * math.sin(x) * math.cos(y)
            p[cell] = -0.25 * decay_p * (math.cos(2.0 * x)
                                         + math.cos(2.0 * y))
        return u, v, p

    def _refresh_boundary_conditions(self, mesh, bc_u, bc_v, bc_p, t):
        """Impose the exact per-face Dirichlet values at time ``t``."""
        for patch in range(mesh.patch_count()):
            u_faces, v_faces, p_faces = [], [], []
            for face in mesh.patch_faces(patch):
                x, y = mesh.face_centroid(face)
                decay_u = math.exp(-2.0 * self.NU * t)
                decay_p = math.exp(-4.0 * self.NU * t)
                u_faces.append(-decay_u * math.cos(x) * math.sin(y))
                v_faces.append(decay_u * math.sin(x) * math.cos(y))
                p_faces.append(-0.25 * decay_p
                               * (math.cos(2.0 * x) + math.cos(2.0 * y)))
            bc_u.set_values(patch, pyfvm.BCType.Dirichlet,
                            np.array(u_faces))
            bc_v.set_values(patch, pyfvm.BCType.Dirichlet,
                            np.array(v_faces))
            bc_p.set_values(patch, pyfvm.BCType.Dirichlet,
                            np.array(p_faces))

    def test_contract_and_pod_energy(self):
        mesh = pyfvm.CartesianMesh(self.NX, self.NX, 0.0, 0.0,
                                   math.pi, math.pi)
        velocity = pyfvm.VectorField(mesh, "velocity")
        pressure = pyfvm.ScalarField(mesh, "pressure")
        flux = pyfvm.FaceFluxField(mesh, "phi")

        u0, v0, p0 = self._exact_fields(mesh, 0.0)
        velocity.u.to_numpy()[:] = u0
        velocity.v.to_numpy()[:] = v0
        pressure.to_numpy()[:] = p0

        bc_u = pyfvm.BoundaryField(mesh)
        bc_v = pyfvm.BoundaryField(mesh)
        bc_p = pyfvm.BoundaryField(mesh)

        cfg = pyfvm.PisoConfig()
        cfg.dt = self.DT
        cfg.n_steps = 1
        cfg.n_correctors = 2
        cfg.time_scheme = pyfvm.TimeScheme.CrankNicolson
        cfg.scheme = pyfvm.ConvectionScheme.Central
        cfg.solver_config.tolerance = 1e-10
        cfg.solver_config.max_iterations = 20000
        cfg.pressure_solver_config.tolerance = 1e-10
        cfg.pressure_solver_config.max_iterations = 20000

        collector = SnapshotCollector()
        collector.append(0.0, velocity.u.to_numpy(),
                         velocity.v.to_numpy(), pressure.to_numpy())
        for step in range(self.STEPS):
            # exact BCs at the end-of-step level of the implicit operator
            self._refresh_boundary_conditions(
                mesh, bc_u, bc_v, bc_p, (step + 1) * self.DT)
            pyfvm.solve_piso(mesh, self.RHO, self.NU, bc_u, bc_v, bc_p,
                             cfg, velocity, pressure, flux)
            collector.append((step + 1) * self.DT,
                             velocity.u.to_numpy(),
                             velocity.v.to_numpy(),
                             pressure.to_numpy())

        snapshots = self.STEPS + 1
        cells = mesh.cell_count()
        self.assertEqual(len(collector), snapshots)

        params = {"nx": self.NX, "dt": self.DT, "nu": self.NU,
                  "rho": self.RHO}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "taylor_green.npz")
            collector.save(path, case="taylor_green", params=params)
            data = load_snapshots(path)

        # contract: version, params, shapes, strictly increasing times
        self.assertEqual(data.format_version, 1)
        self.assertEqual(data.case, "taylor_green")
        self.assertEqual(data.params, params)
        self.assertEqual(data.t.shape, (snapshots,))
        self.assertTrue(np.all(np.diff(data.t) > 0.0))
        for field in ("u", "v", "p"):
            self.assertEqual(getattr(data, field).shape,
                             (snapshots, cells))
        self.assertTrue(np.isfinite(data.u).all())

        # POD on the u snapshots: full rank reconstructs exactly
        # (measured 0.0) and the energy concentrates in very few modes
        # (measured cumulative_energy[2] = 0.9999928).
        result = pod_decompose(data.u)
        r_full = result.modes.shape[0]
        self.assertEqual(r_full, min(snapshots, cells))
        self.assertEqual(result.modes.shape, (r_full, cells))
        self.assertLess(result.reconstruction_error(r_full), 1e-12)
        self.assertGreaterEqual(
            result.cumulative_energy[self.ENERGY_RANK - 1],
            self.ENERGY_THRESHOLD)


class _TaylorGreenRomFixture:
    """Shared Taylor-Green fixture for the Galerkin ROM test classes:
    16^2, rho = 1, nu = 0.1, exact per-face Dirichlet BCs.  A plain
    mixin (not a TestCase), so unittest never collects it directly."""

    NX = 16
    DT = 0.01
    RHO = 1.0
    NU = 0.1

    def _exact_fields(self, mesh, t):
        """Cell-centred exact (u, v, p) at time ``t``."""
        count = mesh.cell_count()
        u = np.empty(count)
        v = np.empty(count)
        p = np.empty(count)
        decay_u = math.exp(-2.0 * self.NU * t)
        decay_p = math.exp(-4.0 * self.NU * t)
        for cell in range(count):
            x, y = mesh.cell_centroid(cell)
            u[cell] = -decay_u * math.cos(x) * math.sin(y)
            v[cell] = decay_u * math.sin(x) * math.cos(y)
            p[cell] = -0.25 * decay_p * (math.cos(2.0 * x)
                                         + math.cos(2.0 * y))
        return u, v, p

    def _refresh_boundary_conditions(self, mesh, bc_u, bc_v, bc_p, t):
        """Exact per-face Dirichlet values at time ``t``."""
        for patch in range(mesh.patch_count()):
            u_faces, v_faces, p_faces = [], [], []
            for face in mesh.patch_faces(patch):
                x, y = mesh.face_centroid(face)
                decay_u = math.exp(-2.0 * self.NU * t)
                decay_p = math.exp(-4.0 * self.NU * t)
                u_faces.append(-decay_u * math.cos(x) * math.sin(y))
                v_faces.append(decay_u * math.sin(x) * math.cos(y))
                p_faces.append(-0.25 * decay_p
                               * (math.cos(2.0 * x) + math.cos(2.0 * y)))
            bc_u.set_values(patch, pyfvm.BCType.Dirichlet,
                            np.array(u_faces))
            bc_v.set_values(patch, pyfvm.BCType.Dirichlet,
                            np.array(v_faces))
            bc_p.set_values(patch, pyfvm.BCType.Dirichlet,
                            np.array(p_faces))

    def _build_case(self, steps):
        """Integrate Taylor-Green for ``steps`` PISO steps collecting
        one snapshot per step; returns (mesh, collector, bc triplet)."""
        mesh = pyfvm.CartesianMesh(self.NX, self.NX, 0.0, 0.0,
                                   math.pi, math.pi)
        velocity = pyfvm.VectorField(mesh, "velocity")
        pressure = pyfvm.ScalarField(mesh, "pressure")
        flux = pyfvm.FaceFluxField(mesh, "phi")

        u0, v0, p0 = self._exact_fields(mesh, 0.0)
        velocity.u.to_numpy()[:] = u0
        velocity.v.to_numpy()[:] = v0
        pressure.to_numpy()[:] = p0

        bc_u = pyfvm.BoundaryField(mesh)
        bc_v = pyfvm.BoundaryField(mesh)
        bc_p = pyfvm.BoundaryField(mesh)

        cfg = pyfvm.PisoConfig()
        cfg.dt = self.DT
        cfg.n_steps = 1
        cfg.n_correctors = 2
        cfg.time_scheme = pyfvm.TimeScheme.CrankNicolson
        cfg.scheme = pyfvm.ConvectionScheme.Central
        cfg.solver_config.tolerance = 1e-10
        cfg.solver_config.max_iterations = 20000
        cfg.pressure_solver_config.tolerance = 1e-10
        cfg.pressure_solver_config.max_iterations = 20000

        collector = SnapshotCollector()
        collector.append(0.0, velocity.u.to_numpy(),
                         velocity.v.to_numpy(), pressure.to_numpy())
        for step in range(steps):
            self._refresh_boundary_conditions(
                mesh, bc_u, bc_v, bc_p, (step + 1) * self.DT)
            pyfvm.solve_piso(mesh, self.RHO, self.NU, bc_u, bc_v, bc_p,
                             cfg, velocity, pressure, flux)
            collector.append((step + 1) * self.DT,
                             velocity.u.to_numpy(),
                             velocity.v.to_numpy(),
                             pressure.to_numpy())
        return mesh, collector, (bc_u, bc_v, bc_p)

    def _build_rom(self, mesh, collector, bcs, **rom_kwargs):
        from rom.galerkin import GalerkinRom
        pod_u = pod_decompose(collector.u)
        pod_v = pod_decompose(collector.v)
        bc_u, bc_v, _ = bcs
        return GalerkinRom(mesh, self.NU, self.RHO, pod_u, pod_v,
                           bc_u, bc_v, pyfvm.ConvectionScheme.Central,
                           **rom_kwargs)

    def _velocity_error(self, u, v, u_ref, v_ref):
        diff2 = (float(np.mean((u - u_ref) ** 2))
                 + float(np.mean((v - v_ref) ** 2)))
        ref2 = (float(np.mean(u_ref ** 2)) + float(np.mean(v_ref ** 2)))
        return math.sqrt(diff2 / ref2)

    def _set_initial(self, mesh, rom):
        initial = pyfvm.VectorField(mesh, "initial")
        u0, v0, _ = self._exact_fields(mesh, 0.0)
        initial.u.to_numpy()[:] = u0
        initial.v.to_numpy()[:] = v0
        rom.set_state_from_fields(initial)


@unittest.skipIf(not HAVE_PYFVM, "pyfvm extension not built")
class TestGalerkinRomTaylorGreen(_TaylorGreenRomFixture, unittest.TestCase):
    """Velocity-only Galerkin ROM contract tests on Taylor-Green
    (16^2, rho = 1, nu = 0.1, exact per-face Dirichlet BCs refreshed at
    the stage/end-step times).  rom.galerkin imports pyfvm lazily, so
    this class relies on the module-level pyfvm import above.  All
    thresholds are calibrated on 2026-11-13 parent-session
    measurements (plan phase 6.3 change log)."""

    # Calibration 2026-11-13 (16^2, dt = 0.01, full-rank basis):
    #   single-step ROM RK4 vs FOM PISO relative difference 0.006 ->
    #     threshold 0.02 (~3x margin, different integrators);
    #   zero-pressure T = 0.2 error 0.124 -> band [0.05, 0.25] (>= 2x
    #     margin both sides; the band quantifies the structural
    #     pressure leakage, an exact match would mean the pressure
    #     term vanished).
    SINGLE_STEP_MAX = 0.02
    ZERO_PRESSURE_BAND = (0.05, 0.25)
    ZERO_PRESSURE_STEPS = 20  # T = 0.2

    def test_state_roundtrip_full_rank(self):
        mesh, collector, bcs = self._build_case(steps=10)
        rom = self._build_rom(mesh, collector, bcs)

        self._set_initial(mesh, rom)
        u, v = rom.reconstruct_arrays()
        u0, v0, _ = self._exact_fields(mesh, 0.0)
        # full-rank row-orthonormal basis: the least-squares projection
        # is exact (atol 1e-12 is machine-level for 256 cells).
        np.testing.assert_allclose(u, u0, atol=1e-12)
        np.testing.assert_allclose(v, v0, atol=1e-12)

    def test_reduced_mass_matrix_spd(self):
        mesh, collector, bcs = self._build_case(steps=5)
        rom = self._build_rom(mesh, collector, bcs)

        for mass in (rom._mass_u, rom._mass_v):
            np.testing.assert_allclose(mass, mass.T, atol=1e-12)
            self.assertGreater(float(np.linalg.eigvalsh(mass).min()),
                               0.0)

    def test_single_step_consistency(self):
        mesh, collector, bcs = self._build_case(steps=1)
        rom = self._build_rom(mesh, collector, bcs)
        bc_u, bc_v, bc_p = bcs

        # one FOM PISO step from t = 0 (final snapshot of the run).
        u_fom = collector.u[-1].copy()
        v_fom = collector.v[-1].copy()

        # one ROM RK4 step with the exact BCs refreshed at stage times.
        self._set_initial(mesh, rom)

        def bc_refresh(t):
            self._refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p, t)

        rom.advance(0.0, self.DT, 1, bc_refresh=bc_refresh)
        u_rom, v_rom = rom.reconstruct_arrays()

        error = self._velocity_error(u_rom, v_rom, u_fom, v_fom)
        # calibrated 0.006, threshold 0.02 (different integrators).
        self.assertLess(error, self.SINGLE_STEP_MAX)
        self.assertGreater(error, 0.0)

    def test_zero_pressure_leakage_short_horizon(self):
        mesh, collector, bcs = self._build_case(
            steps=self.ZERO_PRESSURE_STEPS)
        rom = self._build_rom(mesh, collector, bcs)
        bc_u, bc_v, bc_p = bcs

        self._set_initial(mesh, rom)

        def bc_refresh(t):
            self._refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p, t)

        rom.advance(0.0, self.DT, self.ZERO_PRESSURE_STEPS,
                    bc_refresh=bc_refresh)
        u_rom, v_rom = rom.reconstruct_arrays()
        u_ex, v_ex, _ = self._exact_fields(
            mesh, self.ZERO_PRESSURE_STEPS * self.DT)

        error = self._velocity_error(u_rom, v_rom, u_ex, v_ex)
        # calibrated 0.124; the band quantifies the structural pressure
        # leakage of the velocity-only ROM (both bounds >= 2x margin).
        self.assertGreaterEqual(error, self.ZERO_PRESSURE_BAND[0])
        self.assertLessEqual(error, self.ZERO_PRESSURE_BAND[1])


@unittest.skipIf(not HAVE_PYFVM, "pyfvm extension not built")
class TestGalerkinRomProjection(_TaylorGreenRomFixture, unittest.TestCase):
    """Pressure-projection mode contract tests (Phase 6.4): every RK4
    stage runs one predictive FOM pressure projection on the frozen
    reconstruction and the residual reads the projected pressure.  Same
    Taylor-Green base as TestGalerkinRomTaylorGreen; thresholds follow
    plan phase 6.4 D4."""

    # Aligned with TestGalerkinRomTaylorGreen.SINGLE_STEP_MAX (one RK4
    # step, different metric reference: exact solution here).
    SINGLE_STEP_MAX = 0.02
    ZERO_PRESSURE_STEPS = 20  # T = 0.2
    # Plan D4: the projection variant must land below half the zero-
    # pressure leakage calibration (0.124); a first measurement above
    # this ceiling is a stop-and-report result.
    PROJECTION_LEAKAGE_MAX = 0.05

    def test_projection_single_step_consistency(self):
        # Mirrors test_single_step_consistency with projection on (no
        # injected pressure): one RK4 step with the exact BCs refreshed
        # at stage times, relative L2 vs the exact solution at T = dt.
        mesh, collector, bcs = self._build_case(steps=1)
        rom = self._build_rom(mesh, collector, bcs,
                              pressure_projection=True)
        bc_u, bc_v, bc_p = bcs

        self._set_initial(mesh, rom)

        def bc_refresh(t):
            self._refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p, t)

        rom.advance(0.0, self.DT, 1, bc_refresh=bc_refresh)
        u_rom, v_rom = rom.reconstruct_arrays()
        u_ex, v_ex, _ = self._exact_fields(mesh, self.DT)

        error = self._velocity_error(u_rom, v_rom, u_ex, v_ex)
        # calibrated 2026-11-14: measured 0.0018 (threshold aligned with
        # the single-step consistency test).
        self.assertLess(error, self.SINGLE_STEP_MAX)
        self.assertGreater(error, 0.0)
        # the per-stage projection ran and published its diagnostics
        self.assertIsNotNone(rom.last_projection)
        for value in rom.last_projection:
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, 0.0)

    def test_projection_leakage_horizon(self):
        # 20 steps (T = 0.2): the per-stage projected pressure must keep
        # the error clearly below the zero-pressure leakage calibration
        # (0.124, structural drift of the velocity-only ROM).
        mesh, collector, bcs = self._build_case(
            steps=self.ZERO_PRESSURE_STEPS)
        rom = self._build_rom(mesh, collector, bcs,
                              pressure_projection=True)
        bc_u, bc_v, bc_p = bcs

        self._set_initial(mesh, rom)

        def bc_refresh(t):
            self._refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p, t)

        rom.advance(0.0, self.DT, self.ZERO_PRESSURE_STEPS,
                    bc_refresh=bc_refresh)
        u_rom, v_rom = rom.reconstruct_arrays()
        u_ex, v_ex, _ = self._exact_fields(
            mesh, self.ZERO_PRESSURE_STEPS * self.DT)

        error = self._velocity_error(u_rom, v_rom, u_ex, v_ex)
        # calibrated 2026-11-14: measured 0.0203 (vs the zero-pressure
        # leakage 0.124 and the oracle floor 0.0091); ceiling = plan D4.
        self.assertLess(error, self.PROJECTION_LEAKAGE_MAX)

    def test_projection_transient_horizon(self):
        # Same 20-step horizon with the transient-diagonal projection
        # (projection_dt = the ROM dt, revised D1): uHat = u_stage +
        # r_s/(a_P + rho V/dt) damps the residual step and lands closer
        # to the oracle floor on this case.
        mesh, collector, bcs = self._build_case(
            steps=self.ZERO_PRESSURE_STEPS)
        rom = self._build_rom(mesh, collector, bcs,
                              pressure_projection=True,
                              projection_dt=self.DT)
        bc_u, bc_v, bc_p = bcs

        self._set_initial(mesh, rom)

        def bc_refresh(t):
            self._refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p, t)

        rom.advance(0.0, self.DT, self.ZERO_PRESSURE_STEPS,
                    bc_refresh=bc_refresh)
        u_rom, v_rom = rom.reconstruct_arrays()
        u_ex, v_ex, _ = self._exact_fields(
            mesh, self.ZERO_PRESSURE_STEPS * self.DT)

        error = self._velocity_error(u_rom, v_rom, u_ex, v_ex)
        # calibrated 2026-11-14: measured 0.0101 (steady-operator
        # projection 0.0203 on the same horizon, oracle floor 0.0091);
        # ceiling = plan D4.
        self.assertLess(error, self.PROJECTION_LEAKAGE_MAX)

    def test_projection_oracle_mutual_exclusion(self):
        # The projection computes the per-stage pressure itself; an
        # injected oracle field would be overwritten, so the constructor
        # rejects the combination.
        mesh, collector, bcs = self._build_case(steps=1)
        pressure = pyfvm.ScalarField(mesh, "oracle")
        with self.assertRaises(ValueError):
            self._build_rom(mesh, collector, bcs, pressure=pressure,
                            pressure_projection=True)

    def test_projection_dedup_trajectory_identity(self):
        # The Phase 6.5a dedup must not change the trajectory: the ROM's
        # projection-mode rhs is bit-identical to the pre-dedup composition
        # of public bindings (mass flux -> plain projection on a scratch
        # copy -> flux restore recompute -> two steady assemble_momentum
        # calls), both at isolated states and over a full RK4 advance.
        mesh, collector, bcs = self._build_case(steps=4)
        rom = self._build_rom(mesh, collector, bcs,
                              pressure_projection=True)
        bc_u, bc_v, bc_p = bcs
        rho, nu = self.RHO, self.NU
        scheme = pyfvm.ConvectionScheme.Central

        velocity = pyfvm.VectorField(mesh, "ref_velocity")
        pressure = pyfvm.ScalarField(mesh, "ref_pressure")
        flux = pyfvm.FaceFluxField(mesh, "ref_phi")
        scratch = pyfvm.VectorField(mesh, "ref_scratch")
        # GalerkinRom's default bc_p is a fresh all-Neumann field (the
        # fixture's Dirichlet bc_p is never passed to the ROM), so the
        # reference must mirror that, not the fixture object.
        ref_bc_p = pyfvm.BoundaryField(mesh)
        volumes = np.array([mesh.cell_volume(cell)
                            for cell in range(mesh.cell_count())])
        mass_u = (rom.modes_u * (rho * volumes)) @ rom.modes_u.T
        mass_v = (rom.modes_v * (rho * volumes)) @ rom.modes_v.T

        def reference_rhs(t, a_u, a_v):
            self._refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p, t)
            velocity.u.to_numpy()[:] = rom.mean_u + a_u @ rom.modes_u
            velocity.v.to_numpy()[:] = rom.mean_v + a_v @ rom.modes_v
            pyfvm.compute_mass_flux(mesh, velocity, rho, bc_u, bc_v, flux)
            scratch.u.to_numpy()[:] = velocity.u.to_numpy()
            scratch.v.to_numpy()[:] = velocity.v.to_numpy()
            pyfvm.pressure_projection(mesh, rho, nu, scheme,
                                      bc_u, bc_v, ref_bc_p, scratch,
                                      pressure, flux)
            # Restore the reconstruction's own mass flux (pre-dedup D1).
            pyfvm.compute_mass_flux(mesh, velocity, rho, bc_u, bc_v, flux)
            kwargs = dict(mu=nu, scheme=scheme, pressure=pressure,
                          bc_p=ref_bc_p, rho=rho)
            sys_u = pyfvm.assemble_momentum(
                mesh, flux, velocity, component=0, bc=bc_u, **kwargs)
            sys_v = pyfvm.assemble_momentum(
                mesh, flux, velocity, component=1, bc=bc_v, **kwargs)
            n = mesh.cell_count()

            def reduced(system, field, modes):
                residual = (system.b
                            - np.bincount(system.rows,
                                          weights=(system.vals
                                                   * field[system.cols]),
                                          minlength=n))
                return modes @ residual

            return (np.linalg.solve(
                        mass_u, reduced(sys_u, velocity.u.to_numpy(),
                                        rom.modes_u)),
                    np.linalg.solve(
                        mass_v, reduced(sys_v, velocity.v.to_numpy(),
                                        rom.modes_v)))

        # Isolated rhs evaluations at random states: the reference call
        # also leaves the shared BCs refreshed at t, exactly what the
        # ROM's bc_refresh would do for its own rhs at the same stage
        # time.
        rng = np.random.default_rng(6)
        for _ in range(3):
            a_u = rng.standard_normal(rom.a_u.size) * 0.1
            a_v = rng.standard_normal(rom.a_v.size) * 0.1
            f_u_ref, f_v_ref = reference_rhs(0.01, a_u, a_v)
            f_u, f_v = rom.rhs(0.01, a_u, a_v)
            np.testing.assert_array_equal(f_u, f_u_ref)
            np.testing.assert_array_equal(f_v, f_v_ref)

        # Full RK4 advance (mirrors GalerkinRom._rk4_step).
        def bc_refresh(t):
            self._refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p, t)

        self._set_initial(mesh, rom)
        rom.advance(0.0, self.DT, 3, bc_refresh=bc_refresh)

        u0, v0, _ = self._exact_fields(mesh, 0.0)
        a_u = (u0 - rom.mean_u) @ rom.modes_u.T
        a_v = (v0 - rom.mean_v) @ rom.modes_v.T
        dt = self.DT
        for step in range(3):
            t = step * dt
            k1u, k1v = reference_rhs(t, a_u, a_v)
            k2u, k2v = reference_rhs(t + 0.5 * dt,
                                     a_u + 0.5 * dt * k1u,
                                     a_v + 0.5 * dt * k1v)
            k3u, k3v = reference_rhs(t + 0.5 * dt,
                                     a_u + 0.5 * dt * k2u,
                                     a_v + 0.5 * dt * k2v)
            k4u, k4v = reference_rhs(t + dt, a_u + dt * k3u,
                                     a_v + dt * k3v)
            a_u = a_u + dt / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
            a_v = a_v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

        np.testing.assert_array_equal(rom.a_u, a_u)
        np.testing.assert_array_equal(rom.a_v, a_v)


class _FrozenBcRomFixture(_TaylorGreenRomFixture):
    """Taylor-Green geometry with the exact BCs FROZEN at t = 0 (Phase
    6.5b fixture): the interior evolves away from the boundary state, so
    the FOM relaxes with real pressure dynamics under FIXED boundary
    conditions - the scope of the precomputed operators.  A plain mixin
    (not a TestCase), like its parent."""

    STEPS = 24
    RANK = 4

    def _build_frozen_case(self):
        """FOM relaxation under frozen BCs; returns (mesh, snapshot
        triplet (u, v, p) arrays, bc triplet)."""
        mesh = pyfvm.CartesianMesh(self.NX, self.NX, 0.0, 0.0,
                                   math.pi, math.pi)
        velocity = pyfvm.VectorField(mesh, "velocity")
        pressure = pyfvm.ScalarField(mesh, "pressure")
        flux = pyfvm.FaceFluxField(mesh, "phi")

        u0, v0, p0 = self._exact_fields(mesh, 0.0)
        velocity.u.to_numpy()[:] = u0
        velocity.v.to_numpy()[:] = v0
        pressure.to_numpy()[:] = p0

        bc_u = pyfvm.BoundaryField(mesh)
        bc_v = pyfvm.BoundaryField(mesh)
        bc_p = pyfvm.BoundaryField(mesh)
        self._refresh_boundary_conditions(mesh, bc_u, bc_v, bc_p, 0.0)

        cfg = pyfvm.PisoConfig()
        cfg.dt = self.DT
        cfg.n_steps = 1
        cfg.n_correctors = 2
        cfg.time_scheme = pyfvm.TimeScheme.CrankNicolson
        cfg.scheme = pyfvm.ConvectionScheme.Central
        cfg.solver_config.tolerance = 1e-10
        cfg.solver_config.max_iterations = 20000
        cfg.pressure_solver_config.tolerance = 1e-10
        cfg.pressure_solver_config.max_iterations = 20000

        count = mesh.cell_count()
        steps = self.STEPS
        snap_u = np.empty((steps + 1, count))
        snap_v = np.empty((steps + 1, count))
        snap_p = np.empty((steps + 1, count))
        snap_u[0], snap_v[0], snap_p[0] = u0, v0, p0
        for step in range(steps):
            pyfvm.solve_piso(mesh, self.RHO, self.NU, bc_u, bc_v, bc_p,
                             cfg, velocity, pressure, flux)
            snap_u[step + 1] = velocity.u.to_numpy()
            snap_v[step + 1] = velocity.v.to_numpy()
            snap_p[step + 1] = pressure.to_numpy()
        return mesh, (snap_u, snap_v, snap_p), (bc_u, bc_v, bc_p)

    def _build_fit(self, case, with_closure, rng):
        """Fit the precomputed operators on the frozen case; returns
        (pod pair, fit)."""
        mesh, (snap_u, snap_v, snap_p), (bc_u, bc_v, _) = case
        pod_u = pod_decompose(snap_u).truncated(self.RANK)
        pod_v = pod_decompose(snap_v).truncated(self.RANK)
        au_samples, av_samples = make_fit_samples(
            pod_u.project(snap_u), pod_v.project(snap_v), rng=rng)
        fit = fit_reduced_operators(
            mesh, self.NU, self.RHO, pod_u, pod_v, bc_u, bc_v,
            pyfvm.ConvectionScheme.Central, au_samples, av_samples,
            velocity_snapshots=(snap_u, snap_v) if with_closure else None,
            pressure_snapshots=snap_p if with_closure else None,
            rng=rng)
        return (pod_u, pod_v), fit

    def _zero_pressure_rom(self, case, pod_pair):
        """GalerkinRom sampler twin with an injectable pressure field
        (the fitter's own construction, exposed for the identity
        checks)."""
        mesh, _, (bc_u, bc_v, _) = case
        field = pyfvm.ScalarField(mesh, "check_pressure")
        rom = GalerkinRom(mesh, self.NU, self.RHO, pod_pair[0],
                          pod_pair[1], bc_u, bc_v,
                          pyfvm.ConvectionScheme.Central, pressure=field)
        return rom, field


@unittest.skipIf(not HAVE_PYFVM, "pyfvm extension not built")
class TestPrecomputedRom(_FrozenBcRomFixture, unittest.TestCase):
    """Precomputed reduced operators (Phase 6.5b) contract tests: the
    quadratic fit is exact (the zero-pressure rhs is a polynomial in the
    joint coefficients for Central convection), the closure reproduces
    the snapshot pressure sources within its linear expressiveness, and
    the fixed-BC scope is enforced."""

    FIT_TOLERANCE = 1e-10
    CLOSURE_TOLERANCE = 0.05

    def test_quadratic_fit_exactness(self):
        # Without pressure snapshots the fit must reproduce the
        # zero-pressure GalerkinRom rhs to machine-level precision at
        # states it never sampled (the quadratic model is exact, not
        # approximate).
        case = self._build_frozen_case()
        pod_pair, fit = self._build_fit(case, with_closure=False,
                                        rng=np.random.default_rng(3))
        rom, _ = self._zero_pressure_rom(case, pod_pair)

        rng = np.random.default_rng(9)
        a_u_snap = pod_pair[0].project(case[1][0])
        scale_u = a_u_snap.std(axis=0).mean()
        for _ in range(6):
            a_u = rng.standard_normal(a_u_snap.shape[1]) * scale_u
            a_v = rng.standard_normal(a_u_snap.shape[1]) * scale_u
            f_ref_u, f_ref_v = rom.rhs(0.0, a_u, a_v)
            f_fit_u, f_fit_v = PrecomputedGalerkinRom(fit).rhs(a_u, a_v)
            np.testing.assert_allclose(f_fit_u, f_ref_u,
                                       rtol=self.FIT_TOLERANCE, atol=0.0)
            np.testing.assert_allclose(f_fit_v, f_ref_v,
                                       rtol=self.FIT_TOLERANCE, atol=0.0)
        self.assertLessEqual(fit.holdout_residual, self.FIT_TOLERANCE)
        self.assertEqual(fit.rank_pressure, 0)

    def test_precomputed_tracks_zero_pressure_rom(self):
        # Same rhs + same RK4 => the closure-free precomputed ROM
        # reproduces the zero-pressure GalerkinRom trajectory (fit
        # tolerance, not bit identity: the polynomial is refitted, not
        # copied).
        case = self._build_frozen_case()
        pod_pair, fit = self._build_fit(case, with_closure=False,
                                        rng=np.random.default_rng(4))
        rom, _ = self._zero_pressure_rom(case, pod_pair)
        mesh, (snap_u, snap_v, _), _ = case

        initial = pyfvm.VectorField(mesh, "initial")
        initial.u.to_numpy()[:] = snap_u[0]
        initial.v.to_numpy()[:] = snap_v[0]
        rom.set_state_from_fields(initial)
        fast = PrecomputedGalerkinRom(fit)
        fast.set_state_from_fields(initial)

        n_steps = 10
        ref = rom.advance(0.0, self.DT, n_steps, record_every=1)
        got = fast.advance(0.0, self.DT, n_steps, record_every=1)
        np.testing.assert_allclose(got[1], ref[1], rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(got[2], ref[2], rtol=1e-8, atol=1e-9)

    def test_pressure_closure_reconstruction(self):
        # (a) b-difference identity: the pressure-source difference of
        # two rhs evaluations is independent of the evaluation state
        # (b^np carries no pressure) - the property the per-mode sources
        # rely on; (b) the affine map reproduces the snapshot pressure
        # coefficients; (c) the folded closure reproduces the snapshot
        # pressure sources at snapshot states.
        case = self._build_frozen_case()
        pod_pair, fit = self._build_fit(case, with_closure=True,
                                        rng=np.random.default_rng(5))
        rom, field = self._zero_pressure_rom(case, pod_pair)
        mesh, (snap_u, snap_v, snap_p), _ = case

        self.assertGreaterEqual(fit.rank_pressure, 1)
        self.assertGreater(fit.pressure_cumulative_energy, 0.99)
        self.assertLessEqual(fit.closure_train_residual,
                             self.CLOSURE_TOLERANCE)

        # (a) state independence of the per-mode source difference.
        a_u_snap = pod_pair[0].project(snap_u)
        a_v_snap = pod_pair[1].project(snap_v)
        for k in (0, fit.rank_pressure - 1):
            sources = []
            for a_u, a_v in ((np.zeros(a_u_snap.shape[1]),
                              np.zeros(a_u_snap.shape[1])),
                             (a_u_snap[7], a_v_snap[7])):
                field.to_numpy()[:] = 0.0
                base_u, base_v = rom.rhs(0.0, a_u, a_v)
                field.to_numpy()[:] = fit.pressure_modes[k]
                f_u, f_v = rom.rhs(0.0, a_u, a_v)
                sources.append((f_u - base_u, f_v - base_v))
            np.testing.assert_allclose(sources[1][0], sources[0][0],
                                       rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(sources[1][1], sources[0][1],
                                       rtol=1e-8, atol=1e-9)

        # (c) folded closure vs the direct snapshot pressure source.
        for row in (3, 12, self.STEPS - 1):
            a_u, a_v = a_u_snap[row], a_v_snap[row]
            field.to_numpy()[:] = 0.0
            base_u, base_v = rom.rhs(0.0, a_u, a_v)
            field.to_numpy()[:] = snap_p[row]
            f_u, f_v = rom.rhs(0.0, a_u, a_v)
            a = np.concatenate((a_u, a_v))
            for direct, closure in (((f_u - base_u), fit.h0_u + fit.H_u @ a),
                                    ((f_v - base_v), fit.h0_v + fit.H_v @ a)):
                rel = (np.linalg.norm(closure - direct)
                       / np.linalg.norm(direct))
                self.assertLess(rel, self.CLOSURE_TOLERANCE)

    def test_precomputed_input_validation(self):
        # Sample-count floor and the paired-snapshot requirement are
        # enforced; the Phase 6 validation-case expansion legalised
        # bc_refresh (the live BC fields feed the M1/M2 pressure
        # solves), and a parameter_fn on a fit built without the
        # parameter column is a dimension-contract violation.
        case = self._build_frozen_case()
        pod_pair, fit = self._build_fit(case, with_closure=False,
                                        rng=np.random.default_rng(6))
        fast = PrecomputedGalerkinRom(fit)
        fast.advance(0.0, self.DT, 1, bc_refresh=lambda t: None)
        mismatched = PrecomputedGalerkinRom(fit,
                                            parameter_fn=lambda t: 1.0)
        with self.assertRaises(ValueError):
            mismatched.advance(0.0, self.DT, 1)

        mesh, (snap_u, snap_v, snap_p), (bc_u, bc_v, _) = case
        pod_u, pod_v = pod_pair
        a_u = pod_u.project(snap_u)[:4]
        a_v = pod_v.project(snap_v)[:4]
        with self.assertRaises(ValueError):
            fit_reduced_operators(mesh, self.NU, self.RHO, pod_u, pod_v,
                                  bc_u, bc_v, pyfvm.ConvectionScheme.Central,
                                  a_u, a_v)
        with self.assertRaises(ValueError):
            fit_reduced_operators(
                mesh, self.NU, self.RHO, pod_u, pod_v, bc_u, bc_v,
                pyfvm.ConvectionScheme.Central,
                *make_fit_samples(a_u, a_v, rng=np.random.default_rng(1)),
                velocity_snapshots=(snap_u, snap_v))


@unittest.skipIf(not HAVE_PYFVM, "pyfvm extension not built")
class TestReducedPressureRom(_FrozenBcRomFixture, unittest.TestCase):
    """Phase 6.5c M1 contract tests: the per-stage full pressure solve
    on offline-fitted Rhie-Chow fields reproduces the GalerkinRom
    projection mode (same equation, pressure applied through the
    projection-solution basis) and stays stable on horizons where the
    6.5b affine closure diverged."""

    # Snapshot-state rhs agreement is ~3e-9; mildly off-manifold states
    # see the projection-basis POD truncation (~4e-5 measured), so the
    # gate follows the plan's model tolerance rather than the in-basis
    # floor.
    RHS_TOLERANCE = 1e-3

    PRESSURE_RANK = 12

    def _build_m1(self, case, rng):
        """(pod pair, fit, field fit) with the projection-solution
        pressure basis (D3 revision)."""
        mesh, (snap_u, snap_v, _), (bc_u, bc_v, _) = case
        pod_u = pod_decompose(snap_u).truncated(self.RANK)
        pod_v = pod_decompose(snap_v).truncated(self.RANK)
        a_u_snap = pod_u.project(snap_u)
        a_v_snap = pod_v.project(snap_v)
        samples = make_fit_samples(a_u_snap, a_v_snap, rng=rng)
        p_proj = projected_pressure_snapshots(
            mesh, self.NU, self.RHO, pod_u, pod_v, bc_u, bc_v,
            pyfvm.BoundaryField(mesh), pyfvm.ConvectionScheme.Central,
            a_u_snap, a_v_snap, self.DT)
        pod_pp = pod_decompose(p_proj)
        fit = fit_reduced_operators(
            mesh, self.NU, self.RHO, pod_u, pod_v, bc_u, bc_v,
            pyfvm.ConvectionScheme.Central, *samples,
            pressure_basis=(pod_pp.modes[:self.PRESSURE_RANK],
                            pod_pp.mean), rng=rng)
        field_fit = fit_rhie_chow_fields(
            mesh, self.NU, self.RHO, pod_u, pod_v, bc_u, bc_v,
            pyfvm.ConvectionScheme.Central, *samples, rng=rng)
        return (pod_u, pod_v), fit, field_fit

    def _projection_rom(self, case, pod_pair):
        mesh, _, (bc_u, bc_v, _) = case
        return GalerkinRom(mesh, self.NU, self.RHO, pod_pair[0],
                           pod_pair[1], bc_u, bc_v,
                           pyfvm.ConvectionScheme.Central,
                           pressure_projection=True, projection_dt=self.DT)

    def test_rhs_matches_projection_rom(self):
        # Snapshot states (in-basis) and perturbed states (mildly
        # off-manifold): the fitted fields + full solve + g0 + G c must
        # track the projection-mode rhs far below the model tolerance.
        case = self._build_frozen_case()
        pod_pair, fit, field_fit = self._build_m1(case,
                                                  np.random.default_rng(2))
        mesh, (snap_u, snap_v, _), (bc_u, bc_v, _) = case
        rom = ReducedPressureRom(fit, field_fit, mesh, self.RHO, bc_u,
                                 bc_v, pyfvm.BoundaryField(mesh), self.DT)
        ref = self._projection_rom(case, pod_pair)
        a_u_snap = pod_pair[0].project(snap_u)
        a_v_snap = pod_pair[1].project(snap_v)

        rng = np.random.default_rng(3)
        states = [(a_u_snap[i], a_v_snap[i])
                  for i in (5, 15, self.STEPS)]
        states.append((a_u_snap[10]
                       + 0.3 * a_u_snap.std(axis=0)
                       * rng.standard_normal(a_u_snap.shape[1]),
                       a_v_snap[10]
                       + 0.3 * a_v_snap.std(axis=0)
                       * rng.standard_normal(a_v_snap.shape[1])))
        for a_u, a_v in states:
            f_u, f_v = rom.rhs(a_u, a_v)
            r_u, r_v = ref.rhs(0.0, a_u, a_v)
            rel = max(np.linalg.norm(f_u - r_u) / np.linalg.norm(r_u),
                      np.linalg.norm(f_v - r_v) / np.linalg.norm(r_v))
            self.assertLess(rel, self.RHS_TOLERANCE)
        self.assertLessEqual(fit.holdout_residual, 1e-10)
        self.assertLessEqual(field_fit.holdout_residual, 1e-10)
        self.assertIsNotNone(rom.last_projection)

    def test_tracks_projection_rom_trajectory(self):
        # Same rhs (to fit tolerance) + same RK4 => the M1 ROM tracks the
        # projection-mode GalerkinRom over a short horizon.
        case = self._build_frozen_case()
        pod_pair, fit, field_fit = self._build_m1(case,
                                                  np.random.default_rng(4))
        mesh, (snap_u, snap_v, _), (bc_u, bc_v, _) = case
        rom = ReducedPressureRom(fit, field_fit, mesh, self.RHO, bc_u,
                                 bc_v, pyfvm.BoundaryField(mesh), self.DT)
        ref = self._projection_rom(case, pod_pair)

        initial = pyfvm.VectorField(mesh, "initial")
        initial.u.to_numpy()[:] = snap_u[0]
        initial.v.to_numpy()[:] = snap_v[0]
        ref.set_state_from_fields(initial)
        rom.set_state_from_fields(initial)
        n_steps = 20
        ref_hist = ref.advance(0.0, self.DT, n_steps, record_every=1)
        got_hist = rom.advance(0.0, self.DT, n_steps, record_every=1)
        # RK4 stages wander slightly off the snapshot manifold, where
        # the rhs difference is the basis truncation (~4e-5/stage); the
        # drift concentrates on the smallest coefficients, hence the
        # generous rtol paired with a tight atol.
        np.testing.assert_allclose(got_hist[1], ref_hist[1],
                                   rtol=0.1, atol=1e-4)
        np.testing.assert_allclose(got_hist[2], ref_hist[2],
                                   rtol=0.1, atol=1e-4)

    def test_long_horizon_stability(self):
        # The 6.5b affine closure diverges on shedding after ~225 steps;
        # the state-dependent pressure solve must stay finite on a long
        # horizon (contract-level stability smoke, no accuracy gate).
        case = self._build_frozen_case()
        _, fit, field_fit = self._build_m1(case, np.random.default_rng(5))
        mesh, (snap_u, snap_v, _), (bc_u, bc_v, _) = case
        rom = ReducedPressureRom(fit, field_fit, mesh, self.RHO, bc_u,
                                 bc_v, pyfvm.BoundaryField(mesh), self.DT)
        initial = pyfvm.VectorField(mesh, "initial")
        initial.u.to_numpy()[:] = snap_u[0]
        initial.v.to_numpy()[:] = snap_v[0]
        rom.set_state_from_fields(initial)
        rom.advance(0.0, self.DT, 6000)
        self.assertTrue(np.isfinite(rom.a_u).all())
        self.assertTrue(np.isfinite(rom.a_v).all())

    def test_dt_p_validation(self):
        # The transient diagonal is mandatory (indefinite steady
        # diagonals otherwise), so dt_p must be positive.
        case = self._build_frozen_case()
        _, fit, field_fit = self._build_m1(case, np.random.default_rng(6))
        mesh, _, (bc_u, bc_v, _) = case
        for bad in (None, 0.0, -0.01):
            with self.assertRaises(ValueError):
                ReducedPressureRom(fit, field_fit, mesh, self.RHO, bc_u,
                                   bc_v, pyfvm.BoundaryField(mesh), bad)


@unittest.skipIf(not HAVE_PYFVM, "pyfvm extension not built")
class TestReducedPressureSolve(_FrozenBcRomFixture, unittest.TestCase):
    """Phase 6.5c M2 contract tests: the reduced basis solve of the
    exported pressure equation tracks the M1 full-solve ROM - same
    assembled equation, solution space shrunk to pbar + span(Psi), so
    the gap is the pressure POD truncation - and stays stable on the
    frozen-BC fixture."""

    # The M2-vs-M1 rhs gap is the projection-solution POD truncation
    # (the same truncation M1's own L2 c-projection sees) plus the
    # Galerkin-vs-L2 coefficient difference: measured 1.1e-6 on snapshot
    # states and 3.7e-4 mildly off-manifold, so the gate keeps the model
    # tolerance with the same margins the M1 class uses.
    RHS_TOLERANCE = 1e-3
    TRAJECTORY_RTOL = 1e-3

    PRESSURE_RANK = 12

    # The M1 build (projection-solution pressure basis, D3 revision) is
    # shared verbatim; only the ROM's solve mode differs.
    _build_m1 = TestReducedPressureRom._build_m1

    def _rom(self, case, field_fit, fit, reduced):
        mesh, _, (bc_u, bc_v, _) = case
        return ReducedPressureRom(fit, field_fit, mesh, self.RHO, bc_u,
                                  bc_v, pyfvm.BoundaryField(mesh), self.DT,
                                  reduced_solve=reduced)

    def test_rhs_matches_full_solve(self):
        # Snapshot and perturbed states: the Galerkin-projected solve
        # must track the M1 full solve far below the model tolerance.
        case = self._build_frozen_case()
        pod_pair, fit, field_fit = self._build_m1(case,
                                                  np.random.default_rng(2))
        _, (snap_u, snap_v, _), _ = case
        full = self._rom(case, field_fit, fit, reduced=False)
        reduced = self._rom(case, field_fit, fit, reduced=True)
        a_u_snap = pod_pair[0].project(snap_u)
        a_v_snap = pod_pair[1].project(snap_v)

        rng = np.random.default_rng(3)
        states = [(a_u_snap[i], a_v_snap[i])
                  for i in (5, 15, self.STEPS)]
        states.append((a_u_snap[10]
                       + 0.3 * a_u_snap.std(axis=0)
                       * rng.standard_normal(a_u_snap.shape[1]),
                       a_v_snap[10]
                       + 0.3 * a_v_snap.std(axis=0)
                       * rng.standard_normal(a_v_snap.shape[1])))
        for a_u, a_v in states:
            f_u, f_v = full.rhs(a_u, a_v)
            g_u, g_v = reduced.rhs(a_u, a_v)
            rel = max(np.linalg.norm(g_u - f_u) / np.linalg.norm(f_u),
                      np.linalg.norm(g_v - f_v) / np.linalg.norm(f_v))
            self.assertLess(rel, self.RHS_TOLERANCE)
        self.assertIsNotNone(reduced.last_projection)

    def test_tracks_full_solve_trajectory(self):
        # Same rhs (to fit tolerance) + same RK4 => the M2 ROM tracks the
        # M1 full-solve ROM over a short horizon: the D9 trajectory gate
        # at CI scale (relative history deviation).
        case = self._build_frozen_case()
        _, fit, field_fit = self._build_m1(case, np.random.default_rng(4))
        mesh, (snap_u, snap_v, _), _ = case
        full = self._rom(case, field_fit, fit, reduced=False)
        reduced = self._rom(case, field_fit, fit, reduced=True)

        initial = pyfvm.VectorField(mesh, "initial")
        initial.u.to_numpy()[:] = snap_u[0]
        initial.v.to_numpy()[:] = snap_v[0]
        full.set_state_from_fields(initial)
        reduced.set_state_from_fields(initial)
        ref = full.advance(0.0, self.DT, 20, record_every=1)
        got = reduced.advance(0.0, self.DT, 20, record_every=1)
        rel_u = (np.linalg.norm(got[1] - ref[1])
                 / np.linalg.norm(ref[1]))
        rel_v = (np.linalg.norm(got[2] - ref[2])
                 / np.linalg.norm(ref[2]))
        self.assertLess(max(rel_u, rel_v), self.TRAJECTORY_RTOL)

    def test_long_horizon_stability(self):
        # Contract-level stability smoke of the reduced solve (the full
        # 15000-step horizon is the shedding evidence run); 2000 steps
        # keep the CI cost bounded.
        case = self._build_frozen_case()
        _, fit, field_fit = self._build_m1(case, np.random.default_rng(5))
        mesh, (snap_u, snap_v, _), _ = case
        rom = self._rom(case, field_fit, fit, reduced=True)
        initial = pyfvm.VectorField(mesh, "initial")
        initial.u.to_numpy()[:] = snap_u[0]
        initial.v.to_numpy()[:] = snap_v[0]
        rom.set_state_from_fields(initial)
        rom.advance(0.0, self.DT, 2000)
        self.assertTrue(np.isfinite(rom.a_u).all())
        self.assertTrue(np.isfinite(rom.a_v).all())

    def test_reduced_solve_rejects_empty_pressure_basis(self):
        # A reduced solve without a pressure basis has nothing to project
        # onto; the constructor must reject it (mirrors the dt_p guard of
        # the M1 class).
        import dataclasses

        case = self._build_frozen_case()
        _, fit, field_fit = self._build_m1(case, np.random.default_rng(6))
        mesh, _, (bc_u, bc_v, _) = case
        empty = dataclasses.replace(
            fit, pressure_modes=np.zeros((0, mesh.cell_count())))
        with self.assertRaises(ValueError):
            ReducedPressureRom(empty, field_fit, mesh, self.RHO, bc_u,
                               bc_v, pyfvm.BoundaryField(mesh), self.DT,
                               reduced_solve=True)


@unittest.skipIf(not HAVE_PYFVM, "pyfvm extension not built")
class TestParameterizedRom(_FrozenBcRomFixture, unittest.TestCase):
    """Phase 6 validation-case machinery: a known boundary parameter
    joins the offline quadratic design (``parameter_bc`` varies the
    live BC per sampled assembly) and the online rhs consumes
    ``parameter_fn(t)`` - a parameterized ReducedPressureRom must then
    track the GalerkinRom projection twin whose BC is refreshed to the
    same value (the wall parameter stands in for Stokes' cos(omega t)).
    """

    RHS_TOLERANCE = 1e-3

    def _parameterized_m1(self, case, rng):
        """(pod pair, fit, field fit, M1 ROM, pbc hook)."""
        mesh, (snap_u, snap_v, _), (bc_u, bc_v, _) = case
        pod_u = pod_decompose(snap_u).truncated(self.RANK)
        pod_v = pod_decompose(snap_v).truncated(self.RANK)
        a_u_snap = pod_u.project(snap_u)
        a_v_snap = pod_v.project(snap_v)
        gammas = np.linspace(-1.0, 1.0, a_u_snap.shape[0])

        def pbc(value):
            bc_u.set(pyfvm.BoundaryField.North, pyfvm.BCType.Dirichlet,
                     float(value))

        au, av, ap = make_fit_samples(a_u_snap, a_v_snap,
                                      parameter=gammas, rng=rng)
        p_proj = projected_pressure_snapshots(
            mesh, self.NU, self.RHO, pod_u, pod_v, bc_u, bc_v,
            pyfvm.BoundaryField(mesh), pyfvm.ConvectionScheme.Central,
            a_u_snap, a_v_snap, self.DT,
            parameter_samples=gammas, parameter_bc=pbc)
        pod_pp = pod_decompose(p_proj)
        # The parameter family fattens the pressure manifold: rank 8
        # leaves a ~3e-2 off-manifold rhs gap, rank 12 saturates it at
        # ~3e-4 (measured 2026-11-19).
        fit = fit_reduced_operators(
            mesh, self.NU, self.RHO, pod_u, pod_v, bc_u, bc_v,
            pyfvm.ConvectionScheme.Central, au, av,
            pressure_basis=(pod_pp.modes[:12], pod_pp.mean),
            parameter_samples=ap, parameter_bc=pbc, rng=rng)
        field_fit = fit_rhie_chow_fields(
            mesh, self.NU, self.RHO, pod_u, pod_v, bc_u, bc_v,
            pyfvm.ConvectionScheme.Central, au, av,
            parameter_samples=ap, parameter_bc=pbc, rng=rng)
        rom = ReducedPressureRom(fit, field_fit, mesh, self.RHO, bc_u,
                                 bc_v, pyfvm.BoundaryField(mesh),
                                 self.DT, parameter_fn=lambda t: 0.5)
        return (pod_u, pod_v), fit, field_fit, rom, pbc

    def test_rhs_matches_projection_with_parameter(self):
        # Snapshot and perturbed states at several parameter values: the
        # parameterized fitted operators + per-stage pressure solve must
        # track the projection twin whose live BC carries the same
        # parameter value (the a-gamma quadratic exactness is asserted
        # by the fit holdouts, both machine precision).
        case = self._build_frozen_case()
        pod_pair, fit, field_fit, rom, pbc = self._parameterized_m1(
            case, np.random.default_rng(2))
        mesh, (snap_u, snap_v, _), (bc_u, bc_v, _) = case
        ref = GalerkinRom(mesh, self.NU, self.RHO, pod_pair[0],
                          pod_pair[1], bc_u, bc_v,
                          pyfvm.ConvectionScheme.Central,
                          pressure_projection=True, projection_dt=self.DT)
        self.assertLessEqual(fit.holdout_residual, 1e-10)
        self.assertLessEqual(field_fit.holdout_residual, 1e-10)

        a_u_snap = pod_pair[0].project(snap_u)
        a_v_snap = pod_pair[1].project(snap_v)
        rng = np.random.default_rng(3)
        states = [(a_u_snap[i], a_v_snap[i], gamma)
                  for i, gamma in ((5, -0.6), (15, 0.3), (24, 0.9))]
        states.append((a_u_snap[10]
                       + 0.3 * a_u_snap.std(axis=0)
                       * rng.standard_normal(a_u_snap.shape[1]),
                       a_v_snap[10]
                       + 0.3 * a_v_snap.std(axis=0)
                       * rng.standard_normal(a_v_snap.shape[1]),
                       0.15))
        for a_u, a_v, gamma in states:
            pbc(gamma)
            f_u, f_v = rom.rhs(a_u, a_v, gamma)
            r_u, r_v = ref.rhs(0.0, a_u, a_v)
            rel = max(np.linalg.norm(f_u - r_u) / np.linalg.norm(r_u),
                      np.linalg.norm(f_v - r_v) / np.linalg.norm(r_v))
            self.assertLess(rel, self.RHS_TOLERANCE)

    def test_parameterized_trajectory_finite(self):
        # End-to-end advance with parameter_fn + bc_refresh tracking a
        # synthetic oscillating parameter stays finite (the machinery
        # integration smoke; accuracy lives in the rhs contract).
        case = self._build_frozen_case()
        pod_pair, fit, field_fit, rom, pbc = self._parameterized_m1(
            case, np.random.default_rng(4))
        mesh, (snap_u, snap_v, _), _ = case
        initial = pyfvm.VectorField(mesh, "initial")
        initial.u.to_numpy()[:] = snap_u[0]
        initial.v.to_numpy()[:] = snap_v[0]
        rom.set_state_from_fields(initial)
        rom.advance(0.0, self.DT, 200, bc_refresh=lambda t: pbc(
            0.8 * np.cos(2.0 * np.pi * t / 0.2)))
        self.assertTrue(np.isfinite(rom.a_u).all())
        self.assertTrue(np.isfinite(rom.a_v).all())


if __name__ == "__main__":
    unittest.main()
