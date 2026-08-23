#include "BoundaryCondition.h"
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

// Solve the assembled diffusion system with SparseLU.
Vector solveDiffusion(const CartesianMesh& mesh,
    const ScalarField& gamma,
    const BoundaryField& bc,
    const ScalarField* source = nullptr)
{
    SparseMatrix A(mesh.cellCount(), mesh.cellCount());
    Vector b(mesh.cellCount());
    b.setZero();
    if (source)
    {
        for (Index c = 0; c < mesh.cellCount(); ++c)
        {
            b(c) += (*source)(c)*mesh.cellVolume(c);
        }
    }
    assembleDiffusion(mesh, gamma, bc, A, b);
    A.finalize();

    auto solver = createEigenSparseLU();
    return solver->solve(A, b);
}

ScalarField constantGamma(const CartesianMesh& mesh, Scalar value)
{
    ScalarField gamma(mesh, "gamma");
    gamma.setConstant(value);
    return gamma;
}

} // namespace

TEST_CASE("Diffusion: 1D linear exact solution with Dirichlet BCs")
{
    // -d2phi/dx2 = 0, phi(0) = 1, phi(1) = 0  =>  phi(x) = 1 - x
    CartesianMesh mesh(20, 10, 0.0, 0.0, 1.0, 1.0);
    auto gamma = constantGamma(mesh, 1.0);

    BoundaryField bc; // north/south default to zero-flux
    bc.set(BoundaryField::West, BCType::Dirichlet, 1.0);
    bc.set(BoundaryField::East, BCType::Dirichlet, 0.0);

    Vector phi = solveDiffusion(mesh, gamma, bc);

    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        CHECK(phi[c] == doctest::Approx(1.0 - x).epsilon(1e-10));
    }
}

TEST_CASE(
    "Diffusion: matrix is SPD with Dirichlet BC (symmetry + CG convergence)")
{
    CartesianMesh mesh(16, 16, 0.0, 0.0, 1.0, 1.0);
    auto gamma = constantGamma(mesh, 1.0);

    BoundaryField bc;
    bc.set(BoundaryField::West, BCType::Dirichlet, 1.0);
    bc.set(BoundaryField::East, BCType::Dirichlet, 0.0);

    SparseMatrix A(mesh.cellCount(), mesh.cellCount());
    Vector b(mesh.cellCount());
    b.setZero();
    assembleDiffusion(mesh, gamma, bc, A, b);
    A.finalize();

    // Symmetry
    const auto& An = A.native();
    Scalar asymNorm = (An - Eigen::SparseMatrix<Scalar>(An.transpose())).norm();
    CHECK(asymNorm == doctest::Approx(0.0).epsilon(1e-12));

    // CG converges (only valid for SPD systems)
    SolverConfig cfg;
    cfg.tolerance = 1e-10;
    auto cg = createEigenCG(cfg);
    Vector phi = cg->solve(A, b);
    CHECK(cg->lastResidual() < 1e-10);

    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        CHECK(phi[c] == doctest::Approx(1.0 - x).epsilon(1e-6));
    }
}

TEST_CASE("Diffusion: mixed Dirichlet/Neumann BCs give exact linear profile")
{
    // phi(0) = 0, phi'(1) = 1  =>  phi(x) = x
    CartesianMesh mesh(20, 10, 0.0, 0.0, 1.0, 1.0);
    auto gamma = constantGamma(mesh, 1.0);

    BoundaryField bc;
    bc.set(BoundaryField::West, BCType::Dirichlet, 0.0);
    bc.set(BoundaryField::East, BCType::Neumann, 1.0); // outward dphi/dn = 1

    Vector phi = solveDiffusion(mesh, gamma, bc);

    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        CHECK(phi[c] == doctest::Approx(x).epsilon(1e-10));
    }
}

TEST_CASE("Diffusion: manufactured solution converges at second order")
{
    // phi = x^2,  -phi'' = -2.
    // West: Dirichlet phi(0) = 0;  East: Neumann phi'(1) = 2 (exact flux).
    // The only discretization error comes from the cell-centered Dirichlet
    // flux at the west face -> global L2 error should be O(h^2).
    Scalar prevErr = -1.0;
    Scalar prevH = 0.0;

    for (Index n : { 10, 20, 40 })
    {
        CartesianMesh mesh(n, n, 0.0, 0.0, 1.0, 1.0);
        auto gamma = constantGamma(mesh, 1.0);

        ScalarField source(mesh, "source");
        source.setConstant(-2.0);

        BoundaryField
            bc; // north/south default zero-flux (consistent: dphi/dy = 0)
        bc.set(BoundaryField::West, BCType::Dirichlet, 0.0);
        bc.set(BoundaryField::East, BCType::Neumann, 2.0);

        Vector phi = solveDiffusion(mesh, gamma, bc, &source);

        Scalar err2 = 0.0;
        for (Index c = 0; c < mesh.cellCount(); ++c)
        {
            auto [x, y] = mesh.cellCenter(c);
            Scalar d = phi[c] - x * x;
            err2 += d * d;
        }
        Scalar l2 = std::sqrt(err2 / static_cast<Scalar>(mesh.cellCount()));
        Scalar h = 1.0 / static_cast<Scalar>(n);

        if (prevErr > 0.0)
        {
            Scalar order = std::log(prevErr / l2) / std::log(prevH / h);
            CHECK(order == doctest::Approx(2.0).epsilon(0.15));
        }
        prevErr = l2;
        prevH = h;
    }
}

TEST_CASE(
    "Diffusion: fully closed domain (all zero Neumann) conserves the mean")
{
    // With all-Neumann BCs, A * ones = 0 (row sums vanish): the operator
    // annihilates constants, i.e. fluxes are conservative.
    CartesianMesh mesh(12, 8, 0.0, 0.0, 1.0, 1.0);
    auto gamma = constantGamma(mesh, 2.5);

    BoundaryField bc; // default: zero-flux everywhere

    SparseMatrix A(mesh.cellCount(), mesh.cellCount());
    Vector b(mesh.cellCount());
    b.setZero();
    assembleDiffusion(mesh, gamma, bc, A, b);
    A.finalize();

    Vector ones = Vector::Ones(mesh.cellCount());
    Vector Aones = A.native() * ones;
    CHECK(Aones.norm() == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(b.norm() == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("Diffusion: default BoundaryField equals explicit zero-Neumann")
{
    CartesianMesh mesh(8, 8, 0.0, 0.0, 1.0, 1.0);
    auto gamma = constantGamma(mesh, 1.0);

    BoundaryField bcDefault;
    BoundaryField bcExplicit;
    for (int side = 0; side < 4; ++side)
    {
        bcExplicit.set(side, BCType::Neumann, 0.0);
    }

    auto assemble = [&](const BoundaryField& bc)
    {
        SparseMatrix A(mesh.cellCount(), mesh.cellCount());
        Vector b(mesh.cellCount());
        b.setZero();
        assembleDiffusion(mesh, gamma, bc, A, b);
        A.finalize();
        return A;
    };

    SparseMatrix A1 = assemble(bcDefault);
    SparseMatrix A2 = assemble(bcExplicit);
    Scalar diff = (A1.native() - A2.native()).norm();
    CHECK(diff == doctest::Approx(0.0).epsilon(1e-15));
}
