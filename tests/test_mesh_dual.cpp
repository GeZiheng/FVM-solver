#include "CartesianMesh.h"
#include "Convection.h"
#include "Diffusion.h"
#include "MeshGeomCache.h"
#include "GridOperators.h"
#include "Momentum.h"
#include "Simple.h"
#include "TransportEquation.h"
#include "UnstructuredMesh.h"

#include <cmath>
#include <Eigen/Dense>
#include <doctest/doctest.h>
#include <string>
#include <vector>

using namespace fvm::core;
using namespace fvm::math;
using namespace fvm::numerical;

namespace
{
bool samePoint(const Point2& first, const Point2& second)
{
    return std::abs(first.x - second.x) <= 1.0e-12
        && std::abs(first.y - second.y) <= 1.0e-12;
}

bool sameCellPair(const Mesh& mesh,
    Index first,
    Index second,
    Index face)
{
    const Index owner = mesh.faceOwner(face);
    const Index neighbour = mesh.faceNeighbour(face);
    return (owner == first && neighbour == second)
        || (owner == second && neighbour == first);
}

/// Map every face of `left` to the semantically identical face of `right`
/// (same owner/neighbour pair, or same owner and patch name on the boundary,
/// and the same centroid).  The two representations use different face ids.
std::vector<Index> matchFaces(const Mesh& left, const Mesh& right)
{
    std::vector<Index> matchingFace(left.faceCount(), right.faceCount());
    for (Index face = 0; face < left.faceCount(); ++face)
    {
        const Index owner = left.faceOwner(face);
        const Index neighbour = left.faceNeighbour(face);
        const bool boundary = left.isBoundaryFace(face);
        const std::string patchName = boundary
            ? left.patchName(left.facePatch(face))
            : std::string();
        Index found = 0;
        for (Index candidate = 0; candidate < right.faceCount(); ++candidate)
        {
            bool semanticMatch = false;
            if (boundary)
            {
                semanticMatch = right.isBoundaryFace(candidate)
                    && right.faceOwner(candidate) == owner
                    && right.patchName(right.facePatch(candidate)) == patchName;
            }
            else
            {
                semanticMatch
                    = sameCellPair(right, owner, neighbour, candidate);
            }

            if (semanticMatch && samePoint(left.faceCentroid(face),
                    right.faceCentroid(candidate)))
            {
                matchingFace[face] = candidate;
                ++found;
            }
        }
        REQUIRE(found == 1);
    }
    return matchingFace;
}

void compareMomentumAssemblies(const MomentumAssembly& left,
    const MomentumAssembly& right)
{
    SparseMatrix aLeft = left.system.A;
    SparseMatrix aRight = right.system.A;
    aLeft.finalize();
    aRight.finalize();

    const Eigen::MatrixXd denseLeft = aLeft.native();
    const Eigen::MatrixXd denseRight = aRight.native();
    CHECK((denseLeft - denseRight).lpNorm<Eigen::Infinity>() <= 1.0e-12);
    CHECK((left.system.b - right.system.b).lpNorm<Eigen::Infinity>()
        <= 1.0e-12);
    CHECK((left.diag - right.diag).lpNorm<Eigen::Infinity>() <= 1.0e-12);
    CHECK((left.rhsNoPressure - right.rhsNoPressure).lpNorm<Eigen::Infinity>()
        <= 1.0e-12);
}
} // namespace

TEST_CASE("Operators agree on dual structured mesh representations")
{
    const Index nx = 4;
    const Index ny = 3;
    const CartesianMesh cartesian(nx, ny, 0.0, 0.0, 2.0, 1.5);
    UnstructuredMesh unstructured = makeStructuredMesh(nx, ny, 0.0, 0.0, 2.0, 1.5);

    REQUIRE(cartesian.cellCount() == unstructured.cellCount());
    REQUIRE(cartesian.faceCount() == unstructured.faceCount());

    // Match global faces geometrically and by owner/neighbour (or patch on
    // the boundary); the two representations deliberately use different ids.
    std::vector<Index> matchingFace(cartesian.faceCount());
    for (Index face = 0; face < cartesian.faceCount(); ++face)
    {
        const Index owner = cartesian.faceOwner(face);
        const Index neighbour = cartesian.faceNeighbour(face);
        const bool boundary = cartesian.isBoundaryFace(face);
        const std::string patchName = boundary
            ? cartesian.patchName(cartesian.facePatch(face))
            : std::string();
        Index match = unstructured.faceCount();
        Index found = 0;

        for (Index candidate = 0; candidate < unstructured.faceCount();
            ++candidate)
        {
            bool semanticMatch = false;
            if (boundary)
            {
                semanticMatch
                    = unstructured.isBoundaryFace(candidate)
                    && unstructured.faceOwner(candidate) == owner
                    && unstructured.patchName(
                        unstructured.facePatch(candidate)) == patchName;
            }
            else
            {
                semanticMatch = sameCellPair(unstructured,
                    owner,
                    neighbour,
                    candidate);
            }

            if (semanticMatch && samePoint(cartesian.faceCentroid(face),
                    unstructured.faceCentroid(candidate)))
            {
                match = candidate;
                ++found;
            }
        }

        REQUIRE(found == 1);
        matchingFace[face] = match;
    }

    ScalarField gammaC(cartesian, "gamma");
    ScalarField gammaU(unstructured, "gamma");
    VectorField velocityC(cartesian, "velocity");
    VectorField velocityU(unstructured, "velocity");
    for (Index cell = 0; cell < cartesian.cellCount(); ++cell)
    {
        gammaC(cell) = 0.3 + 0.01 * static_cast<Scalar>(cell % 3);
        gammaU(cell) = gammaC(cell);
        velocityC.u()(cell) = 0.2 + 0.03 * static_cast<Scalar>(cell);
        velocityC.v()(cell) = -0.1 + 0.02 * static_cast<Scalar>(cell);
        velocityU.u()(cell) = velocityC.u()(cell);
        velocityU.v()(cell) = velocityC.v()(cell);
    }

    const FaceFluxField fluxC
        = interpolateCellVelocityFlux(cartesian, velocityC, 1.2);
    const FaceFluxField fluxU
        = interpolateCellVelocityFlux(unstructured, velocityU, 1.2);
    for (Index face = 0; face < cartesian.faceCount(); ++face)
    {
        CHECK(std::abs(fluxC[face] - fluxU[matchingFace[face]]) <= 1.0e-12);
    }

    BoundaryField bcC(cartesian);
    BoundaryField bcU(unstructured);
    bcC.set(BoundaryField::East, BCType::Dirichlet, 1.0);
    bcC.set(BoundaryField::West, BCType::Dirichlet, 0.0);
    bcU.set(BoundaryField::East, BCType::Dirichlet, 1.0);
    bcU.set(BoundaryField::West, BCType::Dirichlet, 0.0);

    {
        SparseMatrix AC(cartesian.cellCount(), cartesian.cellCount());
        SparseMatrix AU(unstructured.cellCount(), unstructured.cellCount());
        Vector bC(cartesian.cellCount());
        Vector bU(unstructured.cellCount());
        bC.setZero();
        bU.setZero();
        assembleDiffusion(cartesian, gammaC, bcC, AC, bC);
        assembleDiffusion(unstructured, gammaU, bcU, AU, bU);
        AC.finalize();
        AU.finalize();

        const Eigen::MatrixXd left = AC.native();
        const Eigen::MatrixXd right = AU.native();
        CHECK((left - right).lpNorm<Eigen::Infinity>() <= 1.0e-12);
        CHECK((bC - bU).lpNorm<Eigen::Infinity>() <= 1.0e-12);
    }

    for (const ConvectionScheme scheme :
        { ConvectionScheme::Upwind, ConvectionScheme::Central })
    {
        SparseMatrix AC(cartesian.cellCount(), cartesian.cellCount());
        SparseMatrix AU(unstructured.cellCount(), unstructured.cellCount());
        Vector bC(cartesian.cellCount());
        Vector bU(unstructured.cellCount());
        bC.setZero();
        bU.setZero();
        assembleConvection(cartesian, fluxC, scheme, bcC, AC, bC);
        assembleConvection(unstructured, fluxU, scheme, bcU, AU, bU);
        AC.finalize();
        AU.finalize();

        const Eigen::MatrixXd left = AC.native();
        const Eigen::MatrixXd right = AU.native();
        CHECK((left - right).lpNorm<Eigen::Infinity>() <= 1.0e-12);
        CHECK((bC - bU).lpNorm<Eigen::Infinity>() <= 1.0e-12);
    }

    ScalarField sourceC(cartesian, "source");
    ScalarField sourceU(unstructured, "source");
    for (Index cell = 0; cell < cartesian.cellCount(); ++cell)
    {
        sourceC(cell) = 1.0 + 0.1 * static_cast<Scalar>(cell);
        sourceU(cell) = sourceC(cell);
    }

    {
        EquationSystem steadyC = assembleTransport(cartesian,
            fluxC,
            gammaC,
            ConvectionScheme::Central,
            bcC,
            &sourceC);
        steadyC.A.finalize();
        EquationSystem steadyU = assembleTransport(unstructured,
            fluxU,
            gammaU,
            ConvectionScheme::Central,
            bcU,
            &sourceU);
        steadyU.A.finalize();
        Eigen::MatrixXd left = steadyC.A.native();
        Eigen::MatrixXd right = steadyU.A.native();
        CHECK((left - right).lpNorm<Eigen::Infinity>() <= 1.0e-12);
        CHECK((steadyC.b - steadyU.b).lpNorm<Eigen::Infinity>() <= 1.0e-12);

        ScalarField oldC(cartesian, "old");
        ScalarField oldU(unstructured, "old");
        for (Index cell = 0; cell < cartesian.cellCount(); ++cell)
        {
            oldC(cell) = 0.5 * static_cast<Scalar>(cell);
            oldU(cell) = oldC(cell);
        }
        EquationSystem transientC = assembleTransientTransport(cartesian,
            fluxC,
            gammaC,
            ConvectionScheme::Central,
            bcC,
            oldC,
            0.1,
            TimeScheme::Euler,
            &sourceC);
        EquationSystem transientU = assembleTransientTransport(
            unstructured,
            fluxU,
            gammaU,
            ConvectionScheme::Central,
            bcU,
            oldU,
            0.1,
            TimeScheme::Euler,
            &sourceU);
        transientC.A.finalize();
        left = transientC.A.native();
        transientU.A.finalize();
        right = transientU.A.native();
        CHECK((left - right).lpNorm<Eigen::Infinity>() <= 1.0e-12);
        CHECK((transientC.b - transientU.b).lpNorm<Eigen::Infinity>()
            <= 1.0e-12);
    }

    for (Index cell = 0; cell < cartesian.cellCount(); ++cell)
    {
        ScalarField phiC(cartesian, "phi");
        ScalarField phiU(unstructured, "phi");
        phiC(cell) = 0.7 + 0.05 * static_cast<Scalar>(cell);
        phiU(cell) = phiC(cell);
        CHECK(std::abs(cellGradient(cartesian, phiC, bcC, cell, 0)
                  - cellGradient(unstructured, phiU, bcU, cell, 0))
            <= 1.0e-12);
        CHECK(std::abs(cellGradient(cartesian, phiC, bcC, cell, 1)
                  - cellGradient(unstructured, phiU, bcU, cell, 1))
            <= 1.0e-12);
    }
}

TEST_CASE("Cartesian and unstructured mesh representations agree")
{
    const std::vector<std::pair<Index, Index>> dimensions = {
        { 3, 2 },
        { 5, 4 },
        { 1, 3 }
    };

    for (const auto& [nx, ny] : dimensions)
    {
        CAPTURE(nx);
        CAPTURE(ny);

        const CartesianMesh cartesian(nx, ny, 0.0, 0.0, 2.0, 1.0);
        const UnstructuredMesh unstructured = makeStructuredMesh(nx, ny, 0.0, 0.0, 2.0, 1.0);
        const Mesh& left = cartesian;
        const Mesh& right = unstructured;

        REQUIRE(left.cellCount() == right.cellCount());
        REQUIRE(left.faceCount() == right.faceCount());

        for (Index cell = 0; cell < left.cellCount(); ++cell)
        {
            REQUIRE(left.cellVolume(cell) == doctest::Approx(right.cellVolume(cell)));
            CHECK(samePoint(left.cellCentroid(cell), right.cellCentroid(cell)));

            Point2 leftSum;
            Point2 rightSum;
            for (const Index face : left.cellFaces(cell))
            {
                const Point2 area = left.faceAreaVector(face);
                const Scalar orientation = left.faceOwner(face) == cell ? 1.0 : -1.0;
                leftSum.x += orientation * area.x;
                leftSum.y += orientation * area.y;
            }
            for (const Index face : right.cellFaces(cell))
            {
                const Point2 area = right.faceAreaVector(face);
                const Scalar orientation = right.faceOwner(face) == cell ? 1.0 : -1.0;
                rightSum.x += orientation * area.x;
                rightSum.y += orientation * area.y;
            }
            CHECK(std::abs(leftSum.x) < 1.0e-12);
            CHECK(std::abs(leftSum.y) < 1.0e-12);
            CHECK(std::abs(rightSum.x) < 1.0e-12);
            CHECK(std::abs(rightSum.y) < 1.0e-12);
        }

        const MeshGeomCache leftGeometry(left);
        const MeshGeomCache rightGeometry(right);
        std::vector<Index> matchCount(right.faceCount(), 0);

        for (Index leftFace = 0; leftFace < left.faceCount(); ++leftFace)
        {
            const Index owner = left.faceOwner(leftFace);
            const Index neighbour = left.faceNeighbour(leftFace);
            const bool boundary = left.isBoundaryFace(leftFace);
            const std::string& patchName = boundary
                ? left.patchName(left.facePatch(leftFace))
                : left.patchName(0);

            Index match = right.faceCount();
            Index found = 0;
            for (Index rightFace = 0; rightFace < right.faceCount(); ++rightFace)
            {
                bool semanticMatch = false;
                if (boundary)
                {
                    semanticMatch = right.isBoundaryFace(rightFace)
                        && right.faceOwner(rightFace) == owner
                        && right.patchName(right.facePatch(rightFace)) == patchName;
                }
                else
                {
                    semanticMatch = sameCellPair(right, owner, neighbour, rightFace);
                }

                if (semanticMatch && samePoint(left.faceCentroid(leftFace),
                        right.faceCentroid(rightFace)))
                {
                    match = rightFace;
                    ++found;
                }
            }

            REQUIRE(found == 1);
            ++matchCount[match];

            const Point2 expectedArea = left.faceAreaVector(leftFace);
            const Point2 actualArea = right.faceAreaVector(match);
            CHECK(std::abs(expectedArea.x - actualArea.x) <= 1.0e-12);
            CHECK(std::abs(expectedArea.y - actualArea.y) <= 1.0e-12);
            CHECK(samePoint(left.faceCentroid(leftFace), right.faceCentroid(match)));
            CHECK(std::abs(leftGeometry.interpWeight(leftFace)
                - rightGeometry.interpWeight(match)) <= 1.0e-12);

            if (!boundary)
            {
                CHECK(std::abs(leftGeometry.innerCoef(leftFace)
                    - rightGeometry.innerCoef(match)) <= 1.0e-12);
            }
        }

        for (const Index count : matchCount)
        {
            CHECK(count == 1);
        }

        for (Index face = 0; face < left.faceCount(); ++face)
        {
            if (!left.isBoundaryFace(face))
            {
                CHECK(leftGeometry.interpWeight(face) == doctest::Approx(0.5));
                const Point2 correction = leftGeometry.nonOrthCorrection(face);
                CHECK(std::abs(correction.x) < 1.0e-12);
                CHECK(std::abs(correction.y) < 1.0e-12);
            }
        }
        CHECK(leftGeometry.maxNonOrthogonality() < 1.0e-9);

        for (Index face = 0; face < left.faceCount(); ++face)
        {
            const Point2 area = left.faceAreaVector(face);
            const Scalar expectedInner = std::abs(area.x) > 0.0
                ? std::abs(area.x) / cartesian.dx()
                : std::abs(area.y) / cartesian.dy();

            if (left.isBoundaryFace(face))
            {
                const Scalar expectedBoundary = std::abs(area.x) > 0.0
                    ? std::abs(area.x) / (cartesian.dx() / 2.0)
                    : std::abs(area.y) / (cartesian.dy() / 2.0);
                CHECK(leftGeometry.boundaryCoef(face) == doctest::Approx(expectedBoundary));
            }
            else
            {
                CHECK(leftGeometry.innerCoef(face) == doctest::Approx(expectedInner));
            }
        }
    }
}

TEST_CASE("Coupled momentum and SIMPLE agree on dual representations")
{
    const Index nx = 5;
    const Index ny = 4;
    const CartesianMesh cartesian(nx, ny, 0.0, 0.0, 1.0, 1.0);
    UnstructuredMesh unstructured = makeStructuredMesh(
        nx, ny, 0.0, 0.0, 1.0, 1.0);

    const std::vector<Index> matchingFace = matchFaces(cartesian, unstructured);

    // Identical initial fields by cell id (the factory creates cells in the
    // same j-major order as CartesianMesh::cellIndex).
    VectorField velocityC(cartesian, "velocity");
    VectorField velocityU(unstructured, "velocity");
    ScalarField pressureC(cartesian, "pressure");
    ScalarField pressureU(unstructured, "pressure");
    for (Index cell = 0; cell < cartesian.cellCount(); ++cell)
    {
        velocityC.u()(cell) = 0.1 + 0.02 * static_cast<Scalar>(cell);
        velocityC.v()(cell) = -0.05 + 0.01 * static_cast<Scalar>(cell);
        pressureC(cell) = 0.02 * static_cast<Scalar>(cell);
        velocityU.u()(cell) = velocityC.u()(cell);
        velocityU.v()(cell) = velocityC.v()(cell);
        pressureU(cell) = pressureC(cell);
    }

    // Channel-style BCs: west inlet, east outlet (anchored p), walls.
    const auto setChannelBCs = [](BoundaryField& bcU,
                                   BoundaryField& bcV,
                                   BoundaryField& bcP)
    {
        bcU.set(BoundaryField::East, BCType::Neumann, 0.0);
        bcU.set(BoundaryField::North, BCType::Dirichlet, 0.0);
        bcU.set(BoundaryField::West, BCType::Dirichlet, 1.0);
        bcU.set(BoundaryField::South, BCType::Dirichlet, 0.0);
        bcV.set(BoundaryField::East, BCType::Neumann, 0.0);
        bcV.set(BoundaryField::North, BCType::Dirichlet, 0.0);
        bcV.set(BoundaryField::West, BCType::Dirichlet, 0.0);
        bcV.set(BoundaryField::South, BCType::Dirichlet, 0.0);
        bcP.set(BoundaryField::East, BCType::Dirichlet, 0.0);
        bcP.set(BoundaryField::North, BCType::Neumann, 0.0);
        bcP.set(BoundaryField::West, BCType::Neumann, 0.0);
        bcP.set(BoundaryField::South, BCType::Neumann, 0.0);
    };
    BoundaryField bcUC(cartesian), bcVC(cartesian), bcPC(cartesian);
    BoundaryField bcUU(unstructured), bcVU(unstructured), bcPU(unstructured);
    setChannelBCs(bcUC, bcVC, bcPC);
    setChannelBCs(bcUU, bcVU, bcPU);

    FaceFluxField fluxC0
        = interpolateCellVelocityFlux(cartesian, velocityC, 1.0);
    FaceFluxField fluxU0
        = interpolateCellVelocityFlux(unstructured, velocityU, 1.0);
    for (Index face = 0; face < cartesian.faceCount(); ++face)
    {
        CHECK(std::abs(fluxC0[face] - fluxU0[matchingFace[face]]) <= 1.0e-12);
    }

    // 1. Momentum assembly, both components, under-relaxed and transient.
    for (const Scalar relaxation : {0.7, 1.0})
    {
        for (const int component : {0, 1})
        {
            const BoundaryField& bcC = (component == 0) ? bcUC : bcVC;
            const BoundaryField& bcU = (component == 0) ? bcUU : bcVU;
            const MomentumAssembly mC = assembleMomentum(cartesian,
                fluxC0,
                velocityC,
                0.02,
                ConvectionScheme::Upwind,
                bcC,
                component,
                pressureC,
                bcPC,
                relaxation,
                TimeTerm{});
            const MomentumAssembly mU = assembleMomentum(unstructured,
                fluxU0,
                velocityU,
                0.02,
                ConvectionScheme::Upwind,
                bcU,
                component,
                pressureU,
                bcPU,
                relaxation,
                TimeTerm{});
            compareMomentumAssemblies(mC, mU);
        }
    }

    {
        TimeTerm time;
        time.rho = 1.0;
        time.dt = 0.05;
        time.scheme = TimeScheme::Euler;

        for (const int component : {0, 1})
        {
            const BoundaryField& bcC = (component == 0) ? bcUC : bcVC;
            const BoundaryField& bcU = (component == 0) ? bcUU : bcVU;
            const MomentumAssembly mC = assembleMomentum(cartesian,
                fluxC0,
                velocityC,
                0.02,
                ConvectionScheme::Upwind,
                bcC,
                component,
                pressureC,
                bcPC,
                1.0,
                time);
            const MomentumAssembly mU = assembleMomentum(unstructured,
                fluxU0,
                velocityU,
                0.02,
                ConvectionScheme::Upwind,
                bcU,
                component,
                pressureU,
                bcPU,
                1.0,
                time);
            compareMomentumAssemblies(mC, mU);
        }
    }

    // 2. End-to-end SIMPLE: identical trajectories (every assembled system
    // agrees to 1e-12), so the converged fields must agree to round-off
    // amplification.  1e-8 leaves margin over that while still locking the
    // coupled solver's mesh independence.
    SimpleConfig config;
    config.tolerance = 1.0e-8;
    config.maxIterations = 1000;

    const SimpleResult resultC = solveSimple(cartesian,
        1.0,
        0.02,
        bcUC,
        bcVC,
        bcPC,
        config,
        velocityC,
        pressureC,
        fluxC0);
    const SimpleResult resultU = solveSimple(unstructured,
        1.0,
        0.02,
        bcUU,
        bcVU,
        bcPU,
        config,
        velocityU,
        pressureU,
        fluxU0);
    REQUIRE(resultC.converged);
    REQUIRE(resultU.converged);

    for (Index cell = 0; cell < cartesian.cellCount(); ++cell)
    {
        CHECK(std::abs(velocityC.u()(cell) - velocityU.u()(cell)) <= 1.0e-8);
        CHECK(std::abs(velocityC.v()(cell) - velocityU.v()(cell)) <= 1.0e-8);
        CHECK(std::abs(pressureC(cell) - pressureU(cell)) <= 1.0e-8);
    }
    for (Index face = 0; face < cartesian.faceCount(); ++face)
    {
        CHECK(std::abs(fluxC0[face] - fluxU0[matchingFace[face]]) <= 1.0e-8);
    }
}
