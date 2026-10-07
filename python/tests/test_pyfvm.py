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

    bc_u = fvm.BoundaryField()
    bc_u.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 1.0)
    for side in (fvm.BoundaryField.East, fvm.BoundaryField.West,
                 fvm.BoundaryField.South):
        bc_u.set(side, fvm.BCType.Dirichlet, 0.0)
    bc_v = fvm.BoundaryField()
    for side in (fvm.BoundaryField.North, fvm.BoundaryField.East,
                 fvm.BoundaryField.West, fvm.BoundaryField.South):
        bc_v.set(side, fvm.BCType.Dirichlet, 0.0)
    bc_p = fvm.BoundaryField()
    return mesh, velocity, pressure, flux, bc_u, bc_v, bc_p


class TestCoreBindings(unittest.TestCase):
    def test_numpy_view_is_zero_copy_and_writable(self):
        mesh = fvm.CartesianMesh(4, 3, 0.0, 0.0, 1.0, 1.0)
        f = fvm.ScalarField(mesh, "T")
        arr = f.to_numpy()
        self.assertEqual(arr.shape, (12,))
        arr[:] = np.arange(12, dtype=float)
        self.assertEqual(f.value(2, 1), 6.0)  # cell index = j * nx + i
        f.set(0, 0, 42.0)
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
        bc = fvm.BoundaryField()
        for side in (fvm.BoundaryField.East, fvm.BoundaryField.North,
                     fvm.BoundaryField.West, fvm.BoundaryField.South):
            got = bc.get(side)
            self.assertEqual(got.type, fvm.BCType.Neumann)
            self.assertEqual(got.value, 0.0)
        bc.set(fvm.BoundaryField.North, fvm.BCType.Dirichlet, 1.0)
        self.assertEqual(bc.get(fvm.BoundaryField.North).type,
                         fvm.BCType.Dirichlet)
        with self.assertRaises(Exception):
            bc.set(4, fvm.BCType.Dirichlet, 0.0)


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

        bc = fvm.BoundaryField()
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
