#include "BoundaryCondition.h"
#include "Convection.h"
#include "Diffusion.h"
#include "Field.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "TransportEquation.h"
#include <cmath>
#include <doctest/doctest.h>

using namespace fvm::core;
using namespace fvm::math;
using namespace fvm::numerical;

TEST_CASE("Transport: zero velocity reduces to pure diffusion")
{
  CartesianMesh mesh(12, 10, 0.0, 0.0, 1.0, 1.0);
  ScalarField gamma(mesh, "gamma");
  gamma.setConstant(1.5);
  VectorField velocity(mesh, "u"); // zero

  BoundaryField bc;
  bc.set(BoundaryField::West, BCType::Dirichlet, 1.0);
  bc.set(BoundaryField::East, BCType::Dirichlet, 0.0);

  auto sys = assembleTransport(mesh, velocity, gamma, 1.0,
                               ConvectionScheme::Upwind, bc);
  sys.A.finalize();

  SparseMatrix Adiff(mesh.cellCount(), mesh.cellCount());
  Vector bdiff(mesh.cellCount());
  bdiff.setZero();
  assembleDiffusion(mesh, gamma, bc, Adiff, bdiff);
  Adiff.finalize();

  CHECK((sys.A.native() - Adiff.native()).norm() == doctest::Approx(0.0).epsilon(1e-15));
  CHECK((sys.b - bdiff).norm() == doctest::Approx(0.0).epsilon(1e-15));
}

TEST_CASE("Transport: constant source term (Poisson with quadratic exact solution)")
{
  // -phi'' = 2, phi(0) = phi(1) = 0  =>  phi = x(1-x)
  CartesianMesh mesh(40, 20, 0.0, 0.0, 1.0, 1.0);
  ScalarField gamma(mesh, "gamma");
  gamma.setConstant(1.0);
  VectorField velocity(mesh, "u"); // zero

  ScalarField Sc(mesh, "Sc");
  Sc.setConstant(2.0);

  BoundaryField bc;
  bc.set(BoundaryField::West, BCType::Dirichlet, 0.0);
  bc.set(BoundaryField::East, BCType::Dirichlet, 0.0);

  auto sys = assembleTransport(mesh, velocity, gamma, 1.0,
                               ConvectionScheme::Upwind, bc, &Sc);
  sys.A.finalize();

  auto solver = createEigenSparseLU();
  Vector phi = solver->solve(sys.A, sys.b);

  for (Index c = 0; c < mesh.cellCount(); ++c) {
    auto [x, y] = mesh.cellCenter(c);
    CHECK(phi[c] == doctest::Approx(x * (1.0 - x)).epsilon(1e-3));
  }
}

TEST_CASE("Transport: linear source term treated implicitly (Helmholtz)")
{
  // -phi'' + c*phi = f with phi = x(1-x): f = 2 + c*x(1-x), Sp = -c.
  const Scalar cReac = 3.0;
  CartesianMesh mesh(40, 20, 0.0, 0.0, 1.0, 1.0);
  ScalarField gamma(mesh, "gamma");
  gamma.setConstant(1.0);
  VectorField velocity(mesh, "u"); // zero

  ScalarField Sc(mesh, "Sc");
  ScalarField Sp(mesh, "Sp");
  Sp.setConstant(-cReac);
  for (Index c = 0; c < mesh.cellCount(); ++c) {
    auto [x, y] = mesh.cellCenter(c);
    Sc(c) = 2.0 + cReac * x * (1.0 - x);
  }

  BoundaryField bc;
  bc.set(BoundaryField::West, BCType::Dirichlet, 0.0);
  bc.set(BoundaryField::East, BCType::Dirichlet, 0.0);

  auto sys = assembleTransport(mesh, velocity, gamma, 1.0,
                               ConvectionScheme::Upwind, bc, &Sc, &Sp);
  sys.A.finalize();

  auto solver = createEigenSparseLU();
  Vector phi = solver->solve(sys.A, sys.b);

  for (Index c = 0; c < mesh.cellCount(); ++c) {
    auto [x, y] = mesh.cellCenter(c);
    CHECK(phi[c] == doctest::Approx(x * (1.0 - x)).epsilon(1e-3));
  }
}

TEST_CASE("Transport: positive Sp is rejected")
{
  CartesianMesh mesh(8, 8, 0.0, 0.0, 1.0, 1.0);
  ScalarField gamma(mesh, "gamma");
  gamma.setConstant(1.0);
  VectorField velocity(mesh, "u");

  ScalarField Sp(mesh, "Sp");
  Sp.setConstant(1.0); // positive: violates Patankar's rule

  BoundaryField bc;
  CHECK_THROWS_AS(assembleTransport(mesh, velocity, gamma, 1.0,
                                    ConvectionScheme::Upwind, bc, nullptr, &Sp),
                  std::invalid_argument);
}

TEST_CASE("Transport: closed domain with uniform flow annihilates constants")
{
  // All-Neumann walls + uniform inflow/outflow through... walls are
  // zero-flux for diffusion and zero-gradient for convection; the combined
  // operator must still satisfy A * ones = 0 (flux conservation).
  CartesianMesh mesh(16, 12, 0.0, 0.0, 1.0, 1.0);
  ScalarField gamma(mesh, "gamma");
  gamma.setConstant(1.0);
  VectorField velocity(mesh, "u");
  velocity.u().setConstant(1.0);

  BoundaryField bc; // all zero-Neumann

  for (auto scheme : { ConvectionScheme::Upwind, ConvectionScheme::Central }) {
    auto sys = assembleTransport(mesh, velocity, gamma, 1.0, scheme, bc);
    sys.A.finalize();

    Vector ones = Vector::Ones(mesh.cellCount());
    Vector Aones = sys.A.native() * ones;
    CHECK(Aones.norm() == doctest::Approx(0.0).epsilon(1e-12));
  }
}
