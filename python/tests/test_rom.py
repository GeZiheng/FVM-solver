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


if __name__ == "__main__":
    unittest.main()
