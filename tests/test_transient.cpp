#include "BoundaryCondition.h"
#include "Convection.h"
#include "Field.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "TransportEquation.h"
#include <cmath>
#include <doctest/doctest.h>
#include <unsupported/Eigen/MatrixFunctions>

using namespace fvm::core;
using namespace fvm::math;
using namespace fvm::numerical;

namespace
{

constexpr Scalar kPi = 3.14159265358979323846;

// Recirculating divergence-free velocity (streamfunction
// psi = sin(pi x) sin(pi y)).
VectorField makeRecirculatingVelocity(const CartesianMesh& mesh)
{
    VectorField velocity(mesh, "u");
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        velocity.u()(c) = kPi * std::sin(kPi * x) * std::cos(kPi * y);
        velocity.v()(c) = -kPi * std::cos(kPi * x) * std::sin(kPi * y);
    }
    return velocity;
}

// Exact solution of the semi-discrete system V d(phi)/dt = -A_sp phi with
// homogeneous Dirichlet BCs and zero velocity: phi(t) = exp(-A_sp t / V) phi0.
Vector exactDiscreteSolution(const SparseMatrix& A,
    const Vector& phi0,
    Scalar t,
    Scalar vol)
{
    const Eigen::MatrixXd dense = A.native().toDense();
    return (-dense * (t / vol)).exp() * phi0;
}

// Time-march phi0 for `n` steps of size t/n with the given scheme.
Vector marchTransient(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    const ScalarField& gamma,
    const BoundaryField& bc,
    const Vector& phi0,
    Scalar t,
    int n,
    TimeScheme scheme)
{
    ScalarField phi(mesh, "phi");
    phi.data() = phi0;

    auto solver = createEigenSparseLU();
    for (int step = 0; step < n; ++step)
    {
        auto sys = assembleTransientTransport(mesh,
            flux,
            gamma,
            ConvectionScheme::Upwind,
            bc,
            phi,
            t / n,
            scheme);
        sys.A.finalize();
        phi.data() = solver->solve(sys.A, sys.b);
    }
    return phi.data();
}

} // namespace

TEST_CASE("Transient: assembly matches the theta-scheme identity (Euler)")
{
    CartesianMesh mesh(10, 8, 0.0, 0.0, 1.0, 1.0);
    const Scalar rho = 1.1;
    ScalarField gamma(mesh, "gamma");
    gamma.setConstant(0.7);

    VectorField velocity = makeRecirculatingVelocity(mesh);
    const FaceFluxField flux = interpolateCellVelocityFlux(mesh, velocity, rho);

    BoundaryField bc;
    bc.set(BoundaryField::West, BCType::Dirichlet, 1.0);
    bc.set(BoundaryField::East, BCType::Dirichlet, 0.0);

    ScalarField phiOld(mesh, "phi");
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        phiOld(c) = std::sin(kPi * x) * std::cos(0.5 * kPi * y);
    }

    const Scalar dt = 0.02;
    const Scalar vol = mesh.cellVolume(0);

    auto full = assembleTransport(
        mesh, flux, gamma, ConvectionScheme::Upwind, bc);
    full.A.finalize();

    auto trans = assembleTransientTransport(mesh,
        flux,
        gamma,
        ConvectionScheme::Upwind,
        bc,
        phiOld,
        dt,
        TimeScheme::Euler);
    trans.A.finalize();

    // theta = 1: A = A_sp + (V/dt) I, b = (V/dt) phi_old + b_sp.
    SparseMatrix expectedA(mesh.cellCount(), mesh.cellCount());
    {
        auto rebuild = assembleTransport(
            mesh, flux, gamma, ConvectionScheme::Upwind, bc);
        expectedA = rebuild.A;
    }
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        expectedA.insert(c, c, vol / dt);
    }
    expectedA.finalize();
    CHECK((trans.A.native() - expectedA.native()).norm() < 1e-12);

    const Vector expectedB = (vol / dt) * phiOld.data() + full.b;
    CHECK((trans.b - expectedB).norm() < 1e-12);
}

TEST_CASE("Transient: assembly matches the theta-scheme identity (CN)")
{
    CartesianMesh mesh(10, 8, 0.0, 0.0, 1.0, 1.0);
    ScalarField gamma(mesh, "gamma");
    gamma.setConstant(0.7);

    VectorField velocity = makeRecirculatingVelocity(mesh);
    const FaceFluxField flux = interpolateCellVelocityFlux(mesh, velocity, 1.0);

    BoundaryField bc;
    bc.set(BoundaryField::West, BCType::Dirichlet, 1.0);
    bc.set(BoundaryField::East, BCType::Dirichlet, 0.0);

    ScalarField phiOld(mesh, "phi");
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        phiOld(c) = std::cos(kPi * x) * std::sin(kPi * y);
    }

    const Scalar dt = 0.05;
    const Scalar vol = mesh.cellVolume(0);

    // Rebuild an unfinalized spatial operator so its triplets can be scaled.
    auto spatial = assembleTransport(
        mesh, flux, gamma, ConvectionScheme::Upwind, bc);
    SparseMatrix probe = spatial.A;
    probe.finalize();
    const Vector AphiOld = probe.native() * phiOld.data();

    auto trans = assembleTransientTransport(mesh,
        flux,
        gamma,
        ConvectionScheme::Upwind,
        bc,
        phiOld,
        dt,
        TimeScheme::CrankNicolson);
    trans.A.finalize();

    // theta = 1/2: A = 1/2 A_sp + (V/dt) I,
    //   b = (V/dt) phi_old - 1/2 A_sp phi_old + b_sp.
    spatial.A.scale(0.5);
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        spatial.A.insert(c, c, vol / dt);
    }
    spatial.A.finalize();
    CHECK((trans.A.native() - spatial.A.native()).norm() < 1e-12);

    const Vector expectedB
        = (vol / dt) * phiOld.data() - 0.5 * AphiOld + spatial.b;
    CHECK((trans.b - expectedB).norm() < 1e-12);
}

TEST_CASE("Transient: backward Euler is first order in time")
{
    // Isolate the temporal error by comparing against the exact solution
    // of the same semi-discrete operator, exp(-A_sp t) phi0.
    CartesianMesh mesh(8, 8, 0.0, 0.0, 1.0, 1.0);
    ScalarField gamma(mesh, "gamma");
    gamma.setConstant(0.1);
    VectorField velocity(mesh, "u"); // zero
    const FaceFluxField flux = interpolateCellVelocityFlux(mesh, velocity, 1.0);

    BoundaryField bc; // all Dirichlet 0
    for (int side = 0; side < 4; ++side)
        bc.set(side, BCType::Dirichlet, 0.0);

    ScalarField phi(mesh, "phi");
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        phi(c) = std::sin(kPi * x) * std::sin(kPi * y);
    }

    auto spatial = assembleTransport(
        mesh, flux, gamma, ConvectionScheme::Upwind, bc);
    spatial.A.finalize();

    const Scalar tEnd = 0.1;
    const Scalar vol = mesh.cellVolume(0);
    const Vector exact
        = exactDiscreteSolution(spatial.A, phi.data(), tEnd, vol);

    auto errorAt = [&](int n)
    {
        const Vector num = marchTransient(
            mesh, flux, gamma, bc, phi.data(), tEnd, n, TimeScheme::Euler);
        return (num - exact).norm() / exact.norm();
    };

    const Scalar e1 = errorAt(8);
    const Scalar e2 = errorAt(16);
    const Scalar ratio = e1 / e2;

    CHECK(e1 > 0.0);
    // Backward Euler: halving dt halves the error (ratio ~ 2).
    CHECK(ratio > 1.7);
    CHECK(ratio < 2.3);
}

TEST_CASE("Transient: Crank-Nicolson is second order in time")
{
    CartesianMesh mesh(8, 8, 0.0, 0.0, 1.0, 1.0);
    ScalarField gamma(mesh, "gamma");
    gamma.setConstant(0.1);
    VectorField velocity(mesh, "u"); // zero
    const FaceFluxField flux = interpolateCellVelocityFlux(mesh, velocity, 1.0);

    BoundaryField bc; // all Dirichlet 0
    for (int side = 0; side < 4; ++side)
        bc.set(side, BCType::Dirichlet, 0.0);

    ScalarField phi(mesh, "phi");
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        phi(c) = std::sin(kPi * x) * std::sin(kPi * y);
    }

    auto spatial = assembleTransport(
        mesh, flux, gamma, ConvectionScheme::Upwind, bc);
    spatial.A.finalize();

    const Scalar tEnd = 0.1;
    const Scalar vol = mesh.cellVolume(0);
    const Vector exact
        = exactDiscreteSolution(spatial.A, phi.data(), tEnd, vol);

    auto errorAt = [&](int n)
    {
        const Vector num = marchTransient(mesh,
            flux,
            gamma,
            bc,
            phi.data(),
            tEnd,
            n,
            TimeScheme::CrankNicolson);
        return (num - exact).norm() / exact.norm();
    };

    const Scalar e1 = errorAt(8);
    const Scalar e2 = errorAt(16);
    const Scalar ratio = e1 / e2;

    CHECK(e1 > 0.0);
    // Crank-Nicolson: halving dt quarters the error (ratio ~ 4).
    CHECK(ratio > 3.3);
    CHECK(ratio < 4.7);
}

TEST_CASE("Transient: constant field is preserved in a closed domain")
{
    // Closed domain (all zero-flux walls) with uniform flow: the steady
    // operator annihilates constants, so the transient system maps a
    // constant phi_old to the same constant.
    CartesianMesh mesh(12, 10, 0.0, 0.0, 1.0, 1.0);
    ScalarField gamma(mesh, "gamma");
    gamma.setConstant(1.0);
    VectorField velocity(mesh, "u");
    velocity.u().setConstant(1.0);
    const FaceFluxField flux = interpolateCellVelocityFlux(mesh, velocity, 1.0);

    BoundaryField bc; // all zero-Neumann

    ScalarField phiOld(mesh, "phi");
    phiOld.setConstant(3.5);

    const Scalar dt = 0.01;
    const Scalar vol = mesh.cellVolume(0);

    auto trans = assembleTransientTransport(mesh,
        flux,
        gamma,
        ConvectionScheme::Upwind,
        bc,
        phiOld,
        dt,
        TimeScheme::CrankNicolson);
    trans.A.finalize();

    const Vector residual = trans.A.native() * phiOld.data() - trans.b;
    CHECK(residual.norm() < 1e-12);

    // phi = const is thus the exact solution of the step.
    const Scalar expected = (vol / dt) * 3.5;
    CHECK((trans.b.array() - expected).abs().maxCoeff() < 1e-12);
}
