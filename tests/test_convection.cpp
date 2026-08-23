#include "BoundaryCondition.h"
#include "Convection.h"
#include "Diffusion.h"
#include "Field.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "SparseMatrix.h"
#include <cmath>
#include <doctest/doctest.h>

using namespace fvm::core;
using namespace fvm::math;
using namespace fvm::numerical;

namespace
{

// Exact solution of rho*u*phi' - gamma*phi'' = 0 on [0,1],
// phi(0) = 0, phi(1) = 1, with Pe = rho*u/gamma.
Scalar exactConvDiff(Scalar x, Scalar Pe)
{
    return (std::exp(Pe * x) - 1.0) / (std::exp(Pe) - 1.0);
}

struct ConvDiffSetup
{
    CartesianMesh mesh;
    ScalarField gamma;
    VectorField velocity;

    ConvDiffSetup(Index n, Scalar gammaVal)
        : mesh(n, n, 0.0, 0.0, 1.0, 1.0)
        , gamma(mesh, "gamma")
        , velocity(mesh, "u")
    {
        gamma.setConstant(gammaVal);
        velocity.u().setConstant(1.0);
        velocity.v().setConstant(0.0);
    }
};

Scalar solveAndL2Error(Index n, ConvectionScheme scheme, Scalar gammaVal)
{
    ConvDiffSetup s(n, gammaVal);
    const Scalar rho = 1.0;
    const Scalar Pe = rho * 1.0 / gammaVal; // L = 1, u = 1

    BoundaryField bc; // north/south: zero-flux (solution is 1D)
    bc.set(BoundaryField::West, BCType::Dirichlet, 0.0);
    bc.set(BoundaryField::East, BCType::Dirichlet, 1.0);

    SparseMatrix A(s.mesh.cellCount(), s.mesh.cellCount());
    Vector b(s.mesh.cellCount());
    b.setZero();
    assembleDiffusion(s.mesh, s.gamma, bc, A, b);
    assembleConvection(s.mesh, s.velocity, rho, scheme, bc, A, b);
    A.finalize();

    auto solver = createEigenSparseLU();
    Vector phi = solver->solve(A, b);

    Scalar err2 = 0.0;
    for (Index c = 0; c < s.mesh.cellCount(); ++c)
    {
        auto [x, y] = s.mesh.cellCenter(c);
        Scalar d = phi[c] - exactConvDiff(x, Pe);
        err2 += d * d;
    }
    return std::sqrt(err2 / static_cast<Scalar>(s.mesh.cellCount()));
}

} // namespace

TEST_CASE("Convection: upwind is first-order accurate (1D conv-diff, Pe = 10)")
{
    const Scalar gammaVal = 0.1;
    Scalar prevErr = -1.0, prevH = 0.0;

    for (Index n : { 40, 80, 160 })
    {
        Scalar l2 = solveAndL2Error(n, ConvectionScheme::Upwind, gammaVal);
        Scalar h = 1.0 / static_cast<Scalar>(n);
        if (prevErr > 0.0)
        {
            Scalar order = std::log(prevErr / l2) / std::log(prevH / h);
            CHECK(order == doctest::Approx(1.0).epsilon(0.2));
        }
        prevErr = l2;
        prevH = h;
    }
}

TEST_CASE(
    "Convection: central differencing is second-order accurate (1D conv-diff, "
    "Pe = 10)")
{
    const Scalar gammaVal = 0.1;
    Scalar prevErr = -1.0, prevH = 0.0;

    for (Index n : { 40, 80, 160 })
    {
        Scalar l2 = solveAndL2Error(n, ConvectionScheme::Central, gammaVal);
        Scalar h = 1.0 / static_cast<Scalar>(n);
        if (prevErr > 0.0)
        {
            Scalar order = std::log(prevErr / l2) / std::log(prevH / h);
            CHECK(order == doctest::Approx(2.0).epsilon(0.2));
        }
        prevErr = l2;
        prevH = h;
    }
}

TEST_CASE(
    "Convection: upwind stays bounded where central oscillates (high cell Pe)")
{
    // Pe = 100 with only 10 cells: cell Pe = 10 >> 2 -> CD oscillates,
    // UD must remain within [phi_west, phi_east] = [0, 1].
    const Scalar gammaVal = 0.01;
    const Index n = 10;

    for (auto scheme : { ConvectionScheme::Upwind, ConvectionScheme::Central })
    {
        ConvDiffSetup s(n, gammaVal);
        BoundaryField bc;
        bc.set(BoundaryField::West, BCType::Dirichlet, 0.0);
        bc.set(BoundaryField::East, BCType::Dirichlet, 1.0);

        SparseMatrix A(s.mesh.cellCount(), s.mesh.cellCount());
        Vector b(s.mesh.cellCount());
        b.setZero();
        assembleDiffusion(s.mesh, s.gamma, bc, A, b);
        assembleConvection(s.mesh, s.velocity, 1.0, scheme, bc, A, b);
        A.finalize();

        auto solver = createEigenSparseLU();
        Vector phi = solver->solve(A, b);

        Scalar lo = phi.minCoeff(), hi = phi.maxCoeff();
        if (scheme == ConvectionScheme::Upwind)
        {
            CHECK(lo >= -1e-12);
            CHECK(hi <= 1.0 + 1e-12);
        }
        else
        {
            // Sanity check that the test case is actually in the oscillatory
            // regime.
            CHECK((lo < -1e-6 || hi > 1.0 + 1e-6));
        }
    }
}

TEST_CASE(
    "Convection: constant field is preserved (consistency with uniform flow)")
{
    // Uniform flow u = (U, 0) in a closed-boundary box: the convective
    // operator must annihilate constant fields (row sums vanish).
    CartesianMesh mesh(16, 12, 0.0, 0.0, 1.0, 1.0);
    VectorField velocity(mesh, "u");
    velocity.u().setConstant(2.0);
    velocity.v().setConstant(0.0);

    BoundaryField bc; // all zero-Neumann (zero-gradient inflow at west)

    for (auto scheme : { ConvectionScheme::Upwind, ConvectionScheme::Central })
    {
        SparseMatrix A(mesh.cellCount(), mesh.cellCount());
        Vector b(mesh.cellCount());
        b.setZero();
        assembleConvection(mesh, velocity, 1.0, scheme, bc, A, b);
        A.finalize();

        Vector ones = Vector::Ones(mesh.cellCount());
        Vector Aones = A.native() * ones;
        CHECK(Aones.norm() == doctest::Approx(0.0).epsilon(1e-12));
        CHECK(b.norm() == doctest::Approx(0.0).epsilon(1e-12));
    }
}
