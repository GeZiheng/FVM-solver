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


if __name__ == "__main__":
    unittest.main()
