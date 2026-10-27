#include "BoundaryCondition.h"
#include "CorrectionOptions.h"
#include "Diffusion.h"
#include "Field.h"
#include "GridOperators.h"
#include "LinearSolver.h"
#include "CartesianMesh.h"
#include "MeshGeomCache.h"
#include "ObliqueMeshFixtures.h"
#include "SparseMatrix.h"
#include "UnstructuredMesh.h"
#include <cmath>
#include <doctest/doctest.h>
#include <vector>

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

namespace
{
Scalar obliqueLinear(const Point2& point)
{
    return 1.0 + 2.0 * point.x - 3.0 * point.y;
}

Vector solveWithCG(SparseMatrix A, const Vector& b)
{
    A.finalize();
    SolverConfig config;
    config.tolerance = 1.0e-12;
    auto cg = createEigenCG(config);
    return cg->solve(A, b);
}
} // namespace

TEST_CASE("Diffusion: 1D linear exact solution with Dirichlet BCs")
{
    // -d2phi/dx2 = 0, phi(0) = 1, phi(1) = 0  =>  phi(x) = 1 - x
    CartesianMesh mesh(20, 10, 0.0, 0.0, 1.0, 1.0);
    auto gamma = constantGamma(mesh, 1.0);

    BoundaryField bc(mesh); // north/south default to zero-flux
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

    BoundaryField bc(mesh);
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

    BoundaryField bc(mesh);
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
            bc(mesh); // north/south default zero-flux (consistent: dphi/dy = 0)
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

    BoundaryField bc(mesh); // default: zero-flux everywhere

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

TEST_CASE("Diffusion: linear field exact on oblique pair")
{
    for (const NonOrthSplit split : { NonOrthSplit::MinimumCorrection,
             NonOrthSplit::OverRelaxed })
    {
        const UnstructuredMesh mesh(makeObliquePair());
        const Mesh& base = mesh;
        ScalarField gamma(base, "gamma");
        gamma.setConstant(1.0);

        ScalarField phi(base, "phi");
        for (Index cell = 0; cell < base.cellCount(); ++cell)
        {
            phi(cell) = obliqueLinear(base.cellCentroid(cell));
        }

        BoundaryField bc(base);
        for (Index patch = 0; patch < bc.patchCount(); ++patch)
        {
            const Index face = base.patchFaces(patch)[0];
            const Point2 area = base.faceAreaVector(face);
            const Scalar areaNorm = std::hypot(area.x, area.y);
            const Scalar normalDerivative
                = (2.0 * area.x - 3.0 * area.y) / areaNorm;
            bc.set(patch, BCType::Neumann, normalDerivative);
        }

        SparseMatrix aOn(base.cellCount(), base.cellCount());
        SparseMatrix aOff(base.cellCount(), base.cellCount());
        Vector bOn(base.cellCount());
        Vector bOff(base.cellCount());
        bOn.setZero();
        bOff.setZero();
        CorrectionOptions corrections;
        corrections.nonOrthSplit = split;
        assembleDiffusion(base, gamma, bc, aOn, bOn, &phi, corrections);
        assembleDiffusion(base, gamma, bc, aOff, bOff, nullptr, corrections);
        aOn.finalize();
        aOff.finalize();
        SparseMatrix aOnDense = aOn;
        SparseMatrix aOffDense = aOff;
        aOnDense.finalize();
        aOffDense.finalize();

        CHECK((aOnDense.native().toDense()
                   - aOffDense.native().toDense()).norm()
            == doctest::Approx(0.0).epsilon(1e-15));

        const std::vector<Point2> gradients
            = leastSquaresGradients(base, phi, bc);
        const MeshGeomCache geometry(base);
        Vector expectedCorrection = Vector::Zero(base.cellCount());
        for (Index face = 0; face < base.faceCount(); ++face)
        {
            if (base.isBoundaryFace(face))
            {
                continue;
            }
            const Index owner = base.faceOwner(face);
            const Index neighbour = base.faceNeighbour(face);
            const Scalar g = geometry.interpWeight(face);
            const Point2 gradF{ g * gradients[owner].x
                    + (1.0 - g) * gradients[neighbour].x,
                g * gradients[owner].y
                    + (1.0 - g) * gradients[neighbour].y };
            const Point2 correction
                = geometry.nonOrthCorrection(face, split);
            const Scalar flux
                = correction.x * gradF.x + correction.y * gradF.y;
            expectedCorrection(owner) += flux;
            expectedCorrection(neighbour) -= flux;
        }
        CHECK((bOn - bOff - expectedCorrection).lpNorm<Eigen::Infinity>()
            == doctest::Approx(0.0).epsilon(1e-12));

        const Vector residualOn = aOn.native() * phi.data() - bOn;
        const Vector residualOff = aOff.native() * phi.data() - bOff;
        CHECK(residualOn.lpNorm<Eigen::Infinity>() <= 1.0e-10);
        CHECK(residualOff.lpNorm<Eigen::Infinity>() > 1.0e-6);
    }
}

TEST_CASE("Diffusion: constant field stays conservative on oblique pair")
{
    for (const NonOrthSplit split : { NonOrthSplit::MinimumCorrection,
             NonOrthSplit::OverRelaxed })
    {
        const UnstructuredMesh mesh(makeObliquePair());
        const Mesh& base = mesh;
        ScalarField gamma(base, "gamma");
        gamma.setConstant(1.0);
        ScalarField phi(base, "phi");
        phi.setConstant(2.7);

        BoundaryField bc(base); // exact zero normal derivative for a constant.

        SparseMatrix a(base.cellCount(), base.cellCount());
        Vector b(base.cellCount());
        b.setZero();
        CorrectionOptions corrections;
        corrections.nonOrthSplit = split;
        assembleDiffusion(base, gamma, bc, a, b, &phi, corrections);
        a.finalize();

        const Vector ones = Vector::Ones(base.cellCount());
        const Vector aOnes = a.native() * ones;
        CHECK(aOnes.norm() == doctest::Approx(0.0).epsilon(1e-12));
        CHECK(b.norm() == doctest::Approx(0.0).epsilon(1e-12));
    }
}

TEST_CASE("Diffusion: second order on sheared triangle meshes")
{
    auto exact = [](const Point2& point)
    {
        return 1.0 + point.x * point.x + 0.5 * point.y * point.y
            - 0.3 * point.x * point.y;
    };

    for (const NonOrthSplit split : { NonOrthSplit::MinimumCorrection,
             NonOrthSplit::OverRelaxed })
    {
        for (const Index n : { Index{ 4 }, Index{ 8 }, Index{ 16 } })
        {
            const UnstructuredMesh mesh(makeShearedTriMesh(n));
            const Mesh& base = mesh;
            ScalarField gamma(base, "gamma");
            gamma.setConstant(1.0);
            ScalarField source(base, "source");
            source.setConstant(3.0);

            BoundaryField bc(base);
            for (Index patch = 0; patch < bc.patchCount(); ++patch)
            {
                const Index face = base.patchFaces(patch)[0];
                bc.set(patch, BCType::Dirichlet, exact(base.faceCentroid(face)));
            }

            Scalar previousOnError = 0.0;
            Scalar previousOffError = 0.0;
            for (const bool correctionOn : { true, false })
            {
                CorrectionOptions corrections;
                corrections.nonOrthogonalDiffusion = correctionOn;
                corrections.nonOrthSplit = split;
                ScalarField phi(base, "phi");
                for (Index cell = 0; cell < base.cellCount(); ++cell)
                {
                    phi(cell) = exact(base.cellCentroid(cell));
                }

                for (int sweep = 0; sweep < 3; ++sweep)
                {
                    SparseMatrix a(base.cellCount(), base.cellCount());
                    Vector b(base.cellCount());
                    b.setZero();
                    assembleDiffusion(base, gamma, bc, a, b, &phi, corrections);
                    for (Index cell = 0; cell < base.cellCount(); ++cell)
                    {
                        b(cell) += source(cell) * base.cellVolume(cell);
                    }
                    phi.data() = solveWithCG(a, b);
                }

                Scalar error = 0.0;
                for (Index cell = 0; cell < base.cellCount(); ++cell)
                {
                    error = std::max(error,
                        std::abs(phi(cell) - exact(base.cellCentroid(cell))));
                }

                if (correctionOn && previousOnError > 0.0)
                {
                    const Scalar rate = previousOnError / error;
                    CHECK(rate > 1.7);
                    CHECK(rate < 2.3);
                }
                if (!correctionOn && previousOffError > 0.0)
                {
                    CHECK(previousOffError / error < 1.5);
                }
                if (correctionOn)
                {
                    previousOnError = error;
                }
                else
                {
                    previousOffError = error;
                }
            }
        }
    }
}

TEST_CASE(
    "Diffusion: split fixed points agree exactly for linear fields only")
{
    auto quadratic = [](const Point2& point)
    {
        return 1.0 + point.x * point.x + 0.5 * point.y * point.y
            - 0.3 * point.x * point.y;
    };

    const UnstructuredMesh mesh(makeShearedTriMesh(8));
    const Mesh& base = mesh;
    ScalarField gamma(base, "gamma");
    gamma.setConstant(1.0);

    ScalarField sourceLinear(base, "sourceLinear");
    sourceLinear.setConstant(0.0);
    ScalarField sourceQuadratic(base, "sourceQuadratic");
    sourceQuadratic.setConstant(3.0);

    // Every boundary face of the sheared fixture is non-orthogonal (the
    // slanted sides ~26.5 deg, the horizontal edges ~63.4 deg), so the
    // uncorrected Dirichlet beta-form would carry an O(h) tangential error
    // and the linear field would not be the exact discrete solution.  Pure
    // Neumann data g_n = grad(phi).n_hat is exact for any face orientation,
    // the singular system is consistent (constant gradient closes on
    // itself), and the CG solve from a zero start is compared after
    // removing the free constant.
    BoundaryField bcLinear(base);
    for (Index patch = 0; patch < bcLinear.patchCount(); ++patch)
    {
        const Index face = base.patchFaces(patch)[0];
        const Point2 area = base.faceAreaVector(face);
        const Scalar areaNorm = std::hypot(area.x, area.y);
        bcLinear.set(patch,
            BCType::Neumann,
            (2.0 * area.x - 3.0 * area.y) / areaNorm);
    }

    BoundaryField bcQuadratic(base);
    for (Index patch = 0; patch < bcQuadratic.patchCount(); ++patch)
    {
        const Index face = base.patchFaces(patch)[0];
        bcQuadratic.set(patch, BCType::Dirichlet,
            quadratic(base.faceCentroid(face)));
    }

    constexpr int maxSweeps = 200;
    constexpr Scalar sweepTolerance = 1.0e-12;
    auto solveDeferred = [&](const ScalarField& sourceField,
        const BoundaryField& bc,
        NonOrthSplit split,
        int& sweepCount)
    {
        CorrectionOptions corrections;
        corrections.nonOrthSplit = split;
        ScalarField phi(base, "phi");
        phi.setZero();
        bool converged = false;
        Scalar lastChange = 0.0;
        for (int sweep = 0; sweep < maxSweeps; ++sweep)
        {
            SparseMatrix a(base.cellCount(), base.cellCount());
            Vector b(base.cellCount());
            b.setZero();
            assembleDiffusion(base, gamma, bc, a, b, &phi, corrections);
            for (Index cell = 0; cell < base.cellCount(); ++cell)
            {
                b(cell) += sourceField(cell) * base.cellVolume(cell);
            }
            const Vector previous = phi.data();
            phi.data() = solveWithCG(a, b);
            lastChange
                = (phi.data() - previous).lpNorm<Eigen::Infinity>();
            if (lastChange < sweepTolerance)
            {
                converged = true;
                sweepCount = sweep + 1;
                break;
            }
        }
        if (!converged)
        {
            FAIL("Deferred-correction sweeps did not converge; last max "
                    "change="
                << lastChange);
        }
        return phi.data();
    };

    // Linear LSQ gradients satisfy G.d = dPhi exactly, so both splits have
    // the same fixed point and that point is the analytic linear field.
    int minimumSweeps = 0;
    int overRelaxedSweeps = 0;
    const Vector linearMinimum = solveDeferred(sourceLinear,
        bcLinear,
        NonOrthSplit::MinimumCorrection,
        minimumSweeps);
    const Vector linearOverRelaxed = solveDeferred(sourceLinear,
        bcLinear,
        NonOrthSplit::OverRelaxed,
        overRelaxedSweeps);
    // Pure Neumann fixes the solution only up to a constant: remove the
    // mean offset of each mode before comparing with the analytic field.
    Scalar minOffset = 0.0;
    Scalar overOffset = 0.0;
    for (Index cell = 0; cell < base.cellCount(); ++cell)
    {
        const Point2 center = base.cellCentroid(cell);
        minOffset += linearMinimum(cell) - obliqueLinear(center);
        overOffset += linearOverRelaxed(cell) - obliqueLinear(center);
    }
    minOffset /= static_cast<Scalar>(base.cellCount());
    overOffset /= static_cast<Scalar>(base.cellCount());
    CHECK(std::abs(minOffset - overOffset) <= 1.0e-8);
    for (Index cell = 0; cell < base.cellCount(); ++cell)
    {
        const Point2 center = base.cellCentroid(cell);
        CHECK(std::abs(linearMinimum(cell) - minOffset - obliqueLinear(center))
            <= 1.0e-8);
        CHECK(std::abs(linearOverRelaxed(cell) - overOffset
                - obliqueLinear(center))
            <= 1.0e-8);
    }

    // For a quadratic field LSQ gives G.d != dPhi at O(h^2).  The two
    // fixed-point equations therefore differ by
    // sum gamma (alphaOr - alphaMin) (G.d - dPhi); this is the same O(h^2)
    // split-dependence seen in the OpenFOAM NC=8 cross-check residual.
    minimumSweeps = 0;
    overRelaxedSweeps = 0;
    const Vector quadraticMinimum = solveDeferred(sourceQuadratic,
        bcQuadratic,
        NonOrthSplit::MinimumCorrection,
        minimumSweeps);
    const Vector quadraticOverRelaxed = solveDeferred(sourceQuadratic,
        bcQuadratic,
        NonOrthSplit::OverRelaxed,
        overRelaxedSweeps);
    CHECK((quadraticMinimum - quadraticOverRelaxed)
        .lpNorm<Eigen::Infinity>() <= 2.0e-2);
}

TEST_CASE("Diffusion: default BoundaryField equals explicit zero-Neumann")
{
    CartesianMesh mesh(8, 8, 0.0, 0.0, 1.0, 1.0);
    auto gamma = constantGamma(mesh, 1.0);

    BoundaryField bcDefault(mesh);
    BoundaryField bcExplicit(mesh);
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
