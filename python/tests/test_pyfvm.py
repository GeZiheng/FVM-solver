"""Binding-contract tests for pyfvm.

Scope: these tests guard the *binding layer* (numpy marshaling, index
order, lifetimes, BC passing, IO round-trip) and give each driver a quick
end-to-end smoke through Python. Physical/numerical correctness is owned
by the C++ doctest suite (tests/) and deliberately not duplicated here.

Run:  python -m unittest discover -s python/tests -v
(also registered as the ctest entry `pyfvm_tests` when FVM_BUILD_PYTHON=ON)
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _path  # noqa: F401,E402

import numpy as np
import pyfvm as fvm


def make_cavity_case(n):
    """Small lid-driven cavity (Re = 100) setup shared by the smokes."""
    mesh = fvm.CartesianMesh(n, n, 0.0, 0.0, 1.0, 1.0)
    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")

    bc_u = fvm.BoundaryField(mesh)
    bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 1.0)
    for side in (fvm.BoundaryField.East, fvm.BoundaryField.West,
                 fvm.BoundaryField.South):
        bc_u.set(side, fvm.BCType.Dirichlet, 0.0)
    bc_v = fvm.BoundaryField(mesh)
    for side in (fvm.BoundaryField.North, fvm.BoundaryField.East,
                 fvm.BoundaryField.West, fvm.BoundaryField.South):
        bc_v.set(side, fvm.BCType.Dirichlet, 0.0)
    bc_p = fvm.BoundaryField(mesh)
    return mesh, velocity, pressure, flux, bc_u, bc_v, bc_p


class TestCoreBindings(unittest.TestCase):
    def test_numpy_view_is_zero_copy_and_writable(self):
        mesh = fvm.CartesianMesh(4, 3, 0.0, 0.0, 1.0, 1.0)
        f = fvm.ScalarField(mesh, "T")
        arr = f.to_numpy()
        self.assertEqual(arr.shape, (12,))
        arr[:] = np.arange(12, dtype=float)
        self.assertEqual(f.to_numpy()[mesh.cell_index(2, 1)],
                         6.0)  # cell index = j * nx + i
        f.to_numpy()[mesh.cell_index(0, 0)] = 42.0
        self.assertEqual(arr[0], 42.0)

    def test_field_keeps_mesh_alive(self):
        # The C++ field holds a mesh reference; dropping the Python mesh
        # object must not leave the field dangling.
        f = fvm.ScalarField(fvm.CartesianMesh(4, 4, 0.0, 0.0, 1.0, 1.0), "T")
        f.set_constant(1.0)
        self.assertEqual(f.to_numpy().sum(), 16.0)

    def test_mesh_geometry(self):
        mesh = fvm.CartesianMesh(4, 3, 0.0, 0.0, 1.0, 1.5)
        self.assertEqual(mesh.cell_count(), 12)
        self.assertAlmostEqual(mesh.dx(), 0.25)
        self.assertAlmostEqual(mesh.dy(), 0.5)
        self.assertEqual(mesh.cell_index(2, 1), 6)
        cx, cy = mesh.cell_center(0)
        self.assertAlmostEqual(cx, 0.125)
        self.assertAlmostEqual(cy, 0.25)

    def test_boundary_field_defaults_and_set(self):
        mesh = fvm.CartesianMesh(4, 3, 0.0, 0.0, 1.0, 1.5)
        bc = fvm.BoundaryField(mesh)
        for side in (fvm.BoundaryField.East, fvm.BoundaryField.North,
                     fvm.BoundaryField.West, fvm.BoundaryField.South):
            got = bc.get(side)
            self.assertEqual(got.type, fvm.BCType.Neumann)
            self.assertEqual(got.value, 0.0)
        bc.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 1.0)
        self.assertEqual(bc.get(fvm.BoundaryField.North).type,
                         fvm.BCType.Dirichlet)
        with self.assertRaises(IndexError):  # std::out_of_range
            bc.set(4, fvm.BCType.Dirichlet, 0.0)

    def test_vector_field_size_and_set_constant(self):
        mesh = fvm.CartesianMesh(4, 3, 0.0, 0.0, 1.0, 1.0)
        v = fvm.VectorField(mesh, "velocity")
        self.assertEqual(v.size(), mesh.cell_count())
        v.set_constant(2.5)
        self.assertTrue((v.u.to_numpy() == 2.5).all())
        self.assertTrue((v.v.to_numpy() == 2.5).all())

    def test_out_of_range_indices_raise(self):
        nx, ny = 4, 3
        mesh = fvm.CartesianMesh(nx, ny, 0.0, 0.0, 1.0, 1.0)
        f = fvm.ScalarField(mesh, "T")
        flux = fvm.FaceFluxField(mesh, "phi")
        with self.assertRaises(IndexError):
            f.to_numpy()[mesh.cell_index(nx, 0)]
        # Index is unsigned: negatives are rejected by pybind11's argument
        # conversion (TypeError) before the binding-level range check.
        with self.assertRaises(TypeError):
            f.to_numpy()[mesh.cell_index(-1, 0)]
        with self.assertRaises(IndexError):
            f.to_numpy()[mesh.cell_index(0, ny)] = 1.0
        with self.assertRaises(IndexError):
            flux.phi(mesh.x_face(nx + 1, 0))
        with self.assertRaises(IndexError):
            flux.phi(mesh.y_face(0, ny + 1))
        with self.assertRaises(IndexError):
            mesh.cell_index(nx, 0)
        with self.assertRaises(IndexError):
            mesh.cell_center(mesh.cell_count())
        # The highest valid face indices still work (x: i == nx, y: j == ny).
        flux.phi(mesh.x_face(nx, 0))
        flux.phi(mesh.y_face(0, ny))


class TestTransportBinding(unittest.TestCase):
    def test_pure_diffusion_linear_profile_exact(self):
        # Zero velocity, uniform gamma, T_w = 1, T_e = 0: exact answer is
        # the linear profile T = 1 - x. Exercises the whole solve_transport
        # path (flux construction, BC marshaling, assembly, solve, numpy
        # return order) against a value that is known exactly.
        n = 16
        mesh = fvm.CartesianMesh(n, n, 0.0, 0.0, 1.0, 1.0)
        gamma = fvm.ScalarField(mesh, "gamma")
        gamma.set_constant(1.0)
        velocity = fvm.VectorField(mesh, "velocity")  # stays zero

        bc = fvm.BoundaryField(mesh)
        bc.set(fvm.BoundaryField.West, fvm.BCType.Dirichlet, 1.0)
        bc.set(fvm.BoundaryField.East, fvm.BCType.Dirichlet, 0.0)

        solver_cfg = fvm.SolverConfig()
        solver_cfg.tolerance = 1e-12

        flux = fvm.interpolate_cell_velocity_flux(mesh, velocity, 1.0)
        sol = fvm.solve_transport(mesh, flux, gamma,
                                  fvm.ConvectionScheme.Upwind, bc, solver_cfg)
        x = (np.arange(n) + 0.5) * mesh.dx()
        np.testing.assert_allclose(sol.reshape(n, n),
                                   np.tile(1.0 - x, (n, 1)), atol=1e-8)

    def test_correction_options_plumbing_and_zero_change(self):
        options = fvm.CorrectionOptions()
        self.assertTrue(options.non_orthogonal_diffusion)
        self.assertTrue(options.skew_interpolation)
        self.assertEqual(options.n_non_orthogonal_correctors, 0)
        self.assertEqual(options.non_orth_split,
                         fvm.NonOrthSplit.MinimumCorrection)

        def run_channel_case(config):
            mesh = fvm.CartesianMesh(8, 4, 0.0, 0.0, 1.0, 1.0)
            velocity = fvm.VectorField(mesh, "velocity")
            pressure = fvm.ScalarField(mesh, "pressure")
            flux = fvm.FaceFluxField(mesh, "phi")

            bc_u = fvm.BoundaryField(mesh)
            bc_u.set(fvm.BoundaryField.West, fvm.BCType.Dirichlet, 1.0)
            bc_u.set(fvm.BoundaryField.East, fvm.BCType.Dirichlet, 1.0)
            bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 0.0)
            bc_u.set(fvm.BoundaryField.South, fvm.BCType.Dirichlet, 0.0)
            bc_v = fvm.BoundaryField(mesh)
            for side in (fvm.BoundaryField.East, fvm.BoundaryField.North,
                         fvm.BoundaryField.West, fvm.BoundaryField.South):
                bc_v.set(side, fvm.BCType.Dirichlet, 0.0)
            bc_p = fvm.BoundaryField(mesh)  # natural pressure at both ends

            config.tolerance = 1e-8
            config.max_iterations = 1000
            result = fvm.solve_simple(mesh, 1.0, 0.01, bc_u, bc_v, bc_p,
                                      config, velocity, pressure, flux)
            self.assertTrue(result.converged)
            return (velocity.u.to_numpy(), velocity.v.to_numpy(),
                    pressure.to_numpy())

        enabled_config = fvm.SimpleConfig()
        enabled_config.corrections.n_non_orthogonal_correctors = 2
        over_config = fvm.SimpleConfig()
        over_config.corrections.non_orth_split = fvm.NonOrthSplit.OverRelaxed
        disabled_config = fvm.SimpleConfig()
        disabled_config.corrections.non_orthogonal_diffusion = False
        disabled_config.corrections.skew_interpolation = False

        enabled = run_channel_case(enabled_config)
        over = run_channel_case(over_config)
        disabled = run_channel_case(disabled_config)
        for reference in (enabled, over):
            for on, off in zip(reference, disabled):
                self.assertTrue(np.allclose(on, off, atol=1e-12, rtol=0.0))

    def test_solve_transport_corrections_zero_change_cartesian(self):
        n = 16
        mesh = fvm.CartesianMesh(n, n, 0.0, 0.0, 1.0, 1.0)
        gamma = fvm.ScalarField(mesh, "gamma")
        gamma.set_constant(1.0)
        velocity = fvm.VectorField(mesh, "velocity")  # stays zero
        bc = fvm.BoundaryField(mesh)
        bc.set(fvm.BoundaryField.West, fvm.BCType.Dirichlet, 1.0)
        bc.set(fvm.BoundaryField.East, fvm.BCType.Dirichlet, 0.0)
        solver_cfg = fvm.SolverConfig()
        solver_cfg.tolerance = 1e-12
        flux = fvm.interpolate_cell_velocity_flux(mesh, velocity, 1.0)

        base = fvm.solve_transport(mesh, flux, gamma,
                                   fvm.ConvectionScheme.Upwind, bc,
                                   solver_cfg)
        corrected = fvm.solve_transport(mesh, flux, gamma,
                                        fvm.ConvectionScheme.Upwind, bc,
                                        solver_cfg,
                                        corrections=fvm.CorrectionOptions())
        np.testing.assert_allclose(base, corrected, atol=1e-12)


class TestDriverSmokes(unittest.TestCase):
    """End-to-end smokes: the drivers run to completion through the
    bindings and produce finite, sane fields. Quantitative physics
    assertions live in the C++ suite."""

    def test_simple_cavity_smoke(self):
        mesh, velocity, pressure, flux, bc_u, bc_v, bc_p = make_cavity_case(16)
        cfg = fvm.SimpleConfig()
        cfg.tolerance = 1e-5
        cfg.max_iterations = 2000
        result = fvm.solve_simple(mesh, 1.0, 0.01, bc_u, bc_v, bc_p, cfg,
                                  velocity, pressure, flux)
        self.assertTrue(result.converged)
        self.assertGreater(result.iterations, 0)
        self.assertTrue(np.isfinite(velocity.u.to_numpy()).all())
        self.assertTrue(np.isfinite(pressure.to_numpy()).all())


class TestMeshMismatch(unittest.TestCase):
    """Fields carry the mesh they were created on; passing one built on a
    different mesh to a mesh-taking entry point must raise ValueError."""

    def test_solve_transport_rejects_foreign_gamma(self):
        mesh = fvm.CartesianMesh(4, 4, 0.0, 0.0, 1.0, 1.0)
        other = fvm.CartesianMesh(4, 4, 0.0, 0.0, 1.0, 1.0)
        velocity = fvm.VectorField(mesh, "velocity")
        gamma = fvm.ScalarField(other, "gamma")  # same geometry, wrong mesh
        flux = fvm.interpolate_cell_velocity_flux(mesh, velocity, 1.0)
        bc = fvm.BoundaryField(mesh)
        with self.assertRaises(ValueError):
            fvm.solve_transport(mesh, flux, gamma,
                                fvm.ConvectionScheme.Upwind, bc)

    def test_solve_simple_rejects_foreign_velocity(self):
        mesh, velocity, pressure, flux, bc_u, bc_v, bc_p = make_cavity_case(8)
        other = fvm.CartesianMesh(8, 8, 0.0, 0.0, 1.0, 1.0)
        foreign_velocity = fvm.VectorField(other, "velocity")
        cfg = fvm.SimpleConfig()
        with self.assertRaises(ValueError):
            fvm.solve_simple(mesh, 1.0, 0.01, bc_u, bc_v, bc_p, cfg,
                             foreign_velocity, pressure, flux)

    def test_solve_simple_rejects_foreign_boundary_field(self):
        mesh, velocity, pressure, flux, bc_u, bc_v, bc_p = make_cavity_case(8)
        other = fvm.CartesianMesh(8, 8, 0.0, 0.0, 1.0, 1.0)
        foreign_bc_u = fvm.BoundaryField(other)  # same geometry, wrong mesh
        cfg = fvm.SimpleConfig()
        with self.assertRaises(ValueError):
            fvm.solve_simple(mesh, 1.0, 0.01, foreign_bc_u, bc_v, bc_p, cfg,
                             velocity, pressure, flux)

    def test_write_vti_rejects_foreign_field(self):
        mesh = fvm.CartesianMesh(4, 4, 0.0, 0.0, 1.0, 1.0)
        other = fvm.CartesianMesh(4, 4, 0.0, 0.0, 1.0, 1.0)
        f = fvm.ScalarField(other, "T")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                fvm.write_vti(os.path.join(tmp, "x.vti"), mesh, {"T": f}, {})

    def test_piso_smoke(self):
        mesh, velocity, pressure, flux, bc_u, bc_v, bc_p = make_cavity_case(16)
        cfg = fvm.PisoConfig()
        cfg.dt = 0.005
        cfg.n_steps = 10
        cfg.n_correctors = 2
        cfg.solver_config.tolerance = 1e-9
        result = fvm.solve_piso(mesh, 1.0, 0.01, bc_u, bc_v, bc_p, cfg,
                                velocity, pressure, flux)
        self.assertEqual(result.steps, cfg.n_steps)
        self.assertEqual(len(result.history), cfg.n_steps)
        self.assertTrue(np.isfinite(velocity.u.to_numpy()).all())
        self.assertTrue(np.isfinite(pressure.to_numpy()).all())


class TestIoBinding(unittest.TestCase):
    def test_write_vti_smoke(self):
        mesh = fvm.CartesianMesh(4, 4, 0.0, 0.0, 1.0, 1.0)
        f = fvm.ScalarField(mesh, "T")
        f.set_constant(3.0)
        v = fvm.VectorField(mesh, "U")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "smoke.vti")
            fvm.write_vti(path, mesh, {"T": f}, {"U": v})
            with open(path, "rb") as fh:
                content = fh.read()
            self.assertGreater(len(content), 0)
            self.assertIn(b"ImageData", content)
            self.assertIn(b'"T"', content)
            self.assertIn(b'"U"', content)


class TestSu2Bindings(unittest.TestCase):
    SU2_CONTENT = """NDIME= 2
NPOIN= 6
1 0.0 0.0
2 1.0 0.0
3 0.0 1.0
4 1.0 1.0
5 2.0 0.0
6 2.0 1.0
NELEM= 3
5 1 2 4
5 1 4 3
9 2 4 6 5
NMARK= 4
MARKER_TAG= bottom
MARKER_ELEMS= 2
3 1 2
3 2 5
MARKER_TAG= right
MARKER_ELEMS= 1
3 5 6
MARKER_TAG= top
MARKER_ELEMS= 2
3 6 4
3 4 3
MARKER_TAG= left
MARKER_ELEMS= 1
3 3 1
"""

    def test_point2_geometry_bindings(self):
        cartesian = fvm.CartesianMesh(4, 4, 0.0, 0.0, 1.0, 1.0)
        cx, cy = cartesian.cell_centroid(0)
        self.assertEqual((cx, cy), (0.125, 0.125))
        # Cartesian face 0 is xFace(0, 0): the west boundary face.
        fx, fy = cartesian.face_centroid(0)
        self.assertEqual((fx, fy), (0.0, 0.125))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mixed.su2")
            with open(path, "w", encoding="ascii") as fh:
                fh.write(self.SU2_CONTENT)
            mesh = fvm.read_su2(path)

        cx, cy = mesh.cell_centroid(0)
        self.assertAlmostEqual(cx, 2.0 / 3.0)
        self.assertAlmostEqual(cy, 1.0 / 3.0)
        bottom_face = mesh.patch_faces(0)[0]
        ax, ay = mesh.face_area_vector(bottom_face)
        self.assertAlmostEqual(ax, 0.0)
        self.assertAlmostEqual(ay, -1.0)

    def test_read_su2_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mixed.su2")
            with open(path, "w", encoding="ascii") as fh:
                fh.write(self.SU2_CONTENT)
            mesh = fvm.read_su2(path)

        self.assertEqual(mesh.cell_count(), 3)
        self.assertEqual(mesh.point_count(), 6)
        self.assertEqual(mesh.patch_count(), 4)
        self.assertEqual([mesh.patch_name(i) for i in range(4)],
                         ["bottom", "right", "top", "left"])
        self.assertEqual(mesh.cell_node_count(0), 3)
        self.assertEqual(mesh.cell_node_count(2), 4)
        self.assertEqual(mesh.point(0), (0.0, 0.0))
        self.assertEqual(mesh.point(5), (2.0, 1.0))

    def test_solve_transport_corrections_skew_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mixed.su2")
            with open(path, "w", encoding="ascii") as fh:
                fh.write(self.SU2_CONTENT)
            mesh = fvm.read_su2(path)

        velocity = fvm.VectorField(mesh, "velocity")
        for cell in range(mesh.cell_count()):
            x, y = mesh.cell_centroid(cell)
            velocity.u.to_numpy()[cell] = 1.0 + 2.0 * x - 3.0 * y
            velocity.v.to_numpy()[cell] = 4.0 - x + 2.0 * y
        gamma = fvm.ScalarField(mesh, "gamma")
        gamma.set_constant(1.0)

        bc = fvm.BoundaryField(mesh)
        for patch in range(mesh.patch_count()):
            if mesh.patch_name(patch) == "left":
                bc.set(patch, fvm.BCType.Dirichlet, 1.0)
            elif mesh.patch_name(patch) == "right":
                bc.set(patch, fvm.BCType.Dirichlet, 0.0)

        solver_cfg = fvm.SolverConfig()
        solver_cfg.tolerance = 1e-10
        flux = fvm.interpolate_cell_velocity_flux(mesh, velocity, 2.0)
        phi_explicit = fvm.ScalarField(mesh, "phi_explicit")
        phi_explicit.to_numpy()[:] = np.zeros(mesh.cell_count())
        corrections = fvm.CorrectionOptions()

        solution = fvm.solve_transport(mesh, flux, gamma,
                                       fvm.ConvectionScheme.Central, bc,
                                       solver_cfg,
                                       phi_explicit=phi_explicit,
                                       corrections=corrections)
        self.assertEqual(solution.size, mesh.cell_count())
        self.assertTrue(np.isfinite(solution).all())

    def test_write_vtu_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            su2_path = os.path.join(tmp, "mixed.su2")
            with open(su2_path, "w", encoding="ascii") as fh:
                fh.write(self.SU2_CONTENT)
            mesh = fvm.read_su2(su2_path)

            T = fvm.ScalarField(mesh, "T")
            T.to_numpy()[:] = [1.0, 2.0, 3.0]
            U = fvm.VectorField(mesh, "U")
            U.u.to_numpy()[:] = [1.0, 3.0, 5.0]
            U.v.to_numpy()[:] = [2.0, 4.0, 6.0]

            path = os.path.join(tmp, "smoke.vtu")
            fvm.write_vtu(path, mesh, {"T": T}, {"U": U})
            with open(path, "r", encoding="ascii") as fh:
                content = fh.read()

        self.assertIn("UnstructuredGrid", content)
        self.assertIn('NumberOfPoints="6"', content)
        self.assertIn('NumberOfCells="3"', content)
        self.assertIn("5 5 9", content)
        self.assertIn("3 6 10", content)
        self.assertIn("1 2 0.0", content)

    def test_write_vtu_rejects_foreign_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            su2_path = os.path.join(tmp, "mixed.su2")
            with open(su2_path, "w", encoding="ascii") as fh:
                fh.write(self.SU2_CONTENT)
            mesh = fvm.read_su2(su2_path)
            other = fvm.CartesianMesh(4, 4, 0.0, 0.0, 1.0, 1.0)
            foreign = fvm.ScalarField(other, "T")

            path = os.path.join(tmp, "foreign.vtu")
            with self.assertRaises(ValueError):
                fvm.write_vtu(path, mesh, {"T": foreign}, {})

    def test_read_su2_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing.su2")
            with self.assertRaises(RuntimeError):
                fvm.read_su2(missing)

            invalid = os.path.join(tmp, "invalid.su2")
            with open(invalid, "w", encoding="ascii") as fh:
                fh.write("NDIME= 2\nNPOIN= x\n")
            with self.assertRaises(ValueError):
                fvm.read_su2(invalid)


if __name__ == "__main__":
    unittest.main()
