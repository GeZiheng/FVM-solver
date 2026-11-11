#include "BoundaryCondition.h"
#include "CartesianMesh.h"
#include "Convection.h"
#include "Diffusion.h"
#include "Field.h"
#include "FluxField.h"
#include "GridOperators.h"
#include "LinearSolver.h"
#include "Momentum.h"
#include "ObliqueMeshFixtures.h"
#include "Pressure.h"
#include "Simple.h"
#include "SparseMatrix.h"
#include "TimeScheme.h"
#include "UnstructuredMesh.h"
#include <algorithm>
#include <cmath>
#include <doctest/doctest.h>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

using namespace fvm::core;
using namespace fvm::math;
using namespace fvm::numerical;

namespace
{

Scalar linear(const Point2& point)
{
    return 1.0 + 2.0 * point.x + 0.25 * point.y;
}

Vector solveCG(SparseMatrix& matrix, const Vector& rhs)
{
    matrix.finalize();
    SolverConfig config;
    config.tolerance = 1.0e-12;
    auto solver = createEigenCG(config);
    return solver->solve(matrix, rhs);
}

Scalar shearedLinearError(const Mesh& mesh)
{
    ScalarField gamma(mesh, "gamma");
    gamma.setConstant(1.0);

    BoundaryField bc(mesh);
    for (Index patch = 0; patch < bc.patchCount(); ++patch)
    {
        const auto faces = mesh.patchFaces(patch);
        std::vector<Scalar> values;
        values.reserve(faces.size());
        for (const Index face : faces)
        {
            values.push_back(linear(mesh.faceCentroid(face)));
        }
        bc.setValues(patch, BCType::Dirichlet, std::move(values));
    }

    const CorrectionOptions corrections;
    ScalarField phi(mesh, "phi");
    for (int sweep = 0; sweep < 8; ++sweep)
    {
        SparseMatrix a(mesh.cellCount(), mesh.cellCount());
        Vector b(mesh.cellCount());
        b.setZero();
        assembleDiffusion(mesh, gamma, bc, a, b, &phi, corrections);
        const Vector previous = phi.data();
        phi.data() = solveCG(a, b);
        const Scalar change
            = (phi.data() - previous).lpNorm<Eigen::Infinity>();
        if (change < 1.0e-12)
        {
            break;
        }
    }
    Scalar maxError = 0.0;
    for (Index cell = 0; cell < mesh.cellCount(); ++cell)
    {
        maxError = std::max(maxError,
            std::abs(phi.data()[cell] - linear(mesh.cellCentroid(cell))));
    }
    return maxError;
}

Vector solveCartesianOneDDiffusion(const CartesianMesh& mesh)
{
    ScalarField gamma(mesh, "gamma");
    gamma.setConstant(1.0);

    // phi = 1 + 2x; the north/south face arrays prescribe the exact
    // zero normal derivative while the end faces carry phi_b arrays.
    BoundaryField bc(mesh);
    for (const Index patch : { BoundaryField::East,
             BoundaryField::North,
             BoundaryField::West,
             BoundaryField::South })
    {
        const auto faces = mesh.patchFaces(patch);
        std::vector<Scalar> values;
        values.reserve(faces.size());
        for (const Index face : faces)
        {
            if (patch == BoundaryField::East)
            {
                values.push_back(1.0 + 2.0 * mesh.faceCentroid(face).x);
            }
            else if (patch == BoundaryField::West)
            {
                values.push_back(1.0);
            }
            else
            {
                values.push_back(0.0);
            }
        }
        const bool isEndPatch = patch == BoundaryField::East
            || patch == BoundaryField::West;
        bc.setValues(patch,
            isEndPatch ? BCType::Dirichlet : BCType::Neumann,
            std::move(values));
    }

    SparseMatrix a(mesh.cellCount(), mesh.cellCount());
    Vector b(mesh.cellCount());
    b.setZero();
    assembleDiffusion(mesh, gamma, bc, a, b);
    return solveCG(a, b);
}

std::vector<Scalar> correctedFluxWithOutletPressure(bool perFaceOutletP)
{
    CartesianMesh mesh(8, 4, 0.0, 0.0, 1.0, 1.0);
    const Scalar rho = 1.0;
    const Scalar mu = 0.01;
    const Scalar outletP = 0.37;

    BoundaryField bcU(mesh);
    bcU.set(BoundaryField::West, BCType::Dirichlet, 1.0);
    bcU.set(BoundaryField::East, BCType::Neumann, 0.0);
    bcU.set(BoundaryField::North, BCType::Dirichlet, 0.0);
    bcU.set(BoundaryField::South, BCType::Dirichlet, 0.0);

    BoundaryField bcV(mesh);
    for (const Index patch : { BoundaryField::East,
             BoundaryField::North,
             BoundaryField::West,
             BoundaryField::South })
    {
        bcV.set(patch, BCType::Dirichlet, 0.0);
    }

    BoundaryField bcP(mesh);
    bcP.set(BoundaryField::West, BCType::Neumann, 0.0);
    if (perFaceOutletP)
    {
        const auto faces = mesh.patchFaces(BoundaryField::East);
        bcP.setValues(BoundaryField::East,
            BCType::Dirichlet,
            std::vector<Scalar>(faces.size(), outletP));
    }
    else
    {
        bcP.set(BoundaryField::East, BCType::Dirichlet, outletP);
    }

    VectorField velocity(mesh, "velocity");
    ScalarField pressure(mesh, "pressure");
    FaceFluxField flux(mesh, "flux");
    initializeMassFlux(mesh, velocity, rho, bcU, bcV, bcP, flux);

    SolverConfig solverConfig;
    solverConfig.tolerance = 1.0e-11;
    solverConfig.maxIterations = 10000;
    auto momSolver = createEigenBiCGSTAB(solverConfig);
    auto pSolver = createEigenCG(solverConfig);

    const MomentumPrediction pred = predictMomentum(mesh,
        flux,
        velocity,
        mu,
        ConvectionScheme::Upwind,
        bcU,
        bcV,
        pressure,
        bcP,
        1.0,
        TimeTerm{},
        *momSolver);
    correctPressure(mesh,
        rho,
        pred,
        bcU,
        bcV,
        bcP,
        *pSolver,
        1.0,
        velocity,
        pressure,
        flux);

    std::vector<Scalar> result(static_cast<std::size_t>(mesh.faceCount()));
    for (Index f = 0; f < mesh.faceCount(); ++f)
    {
        result[static_cast<std::size_t>(f)] = flux[f];
    }
    return result;
}

} // namespace

TEST_CASE("Boundary face values reject invalid patches, lengths and faces")
{
    CartesianMesh mesh(4, 3, 0.0, 0.0, 1.0, 1.0);
    BoundaryField bc(mesh);

    CHECK_THROWS_AS(bc.setValues(BoundaryField::West,
                        BCType::Dirichlet,
                        { 1.0, 2.0 }),
        std::invalid_argument);
    CHECK_THROWS_AS(bc.setValues(4, BCType::Dirichlet, { 1.0 }),
        std::out_of_range);
    CHECK_FALSE(bc.hasFaceValues(BoundaryField::West));

    const Index interiorFace = mesh.xFace(1, 0);
    REQUIRE_FALSE(mesh.isBoundaryFace(interiorFace));
    CHECK_THROWS_AS(bc.valueAt(interiorFace), std::invalid_argument);
}

TEST_CASE("Boundary face values fall back to patch constants on both meshes")
{
    const auto checkFallback = [](const Mesh& mesh)
    {
        BoundaryField bc(mesh);
        for (Index patch = 0; patch < bc.patchCount(); ++patch)
        {
            bc.set(patch, BCType::Dirichlet, 2.0 + static_cast<Scalar>(patch));
        }

        for (Index patch = 0; patch < bc.patchCount(); ++patch)
        {
            CHECK_FALSE(bc.hasFaceValues(patch));
            const Scalar expected = bc.get(patch).value;
            for (const Index face : mesh.patchFaces(patch))
            {
                CHECK(bc.valueAt(face) == doctest::Approx(expected));
            }
        }
    };

    CartesianMesh cartesian(4, 3, 0.0, 0.0, 1.0, 1.0);
    checkFallback(cartesian);

    UnstructuredMesh structured = makeStructuredMesh(4, 3, 0.0, 0.0, 1.0, 1.0);
    checkFallback(structured);
}

TEST_CASE("setValues scatters values in patchFaces traversal order")
{
    CartesianMesh mesh(4, 3, 0.0, 0.0, 1.0, 1.0);
    BoundaryField bc(mesh);

    for (const Index patch : { BoundaryField::East,
             BoundaryField::North,
             BoundaryField::West,
             BoundaryField::South })
    {
        const auto faces = mesh.patchFaces(patch);
        std::vector<Scalar> values;
        values.reserve(faces.size());
        for (std::size_t k = 0; k < faces.size(); ++k)
        {
            values.push_back(1.0 + 2.0 * static_cast<Scalar>(k)
                + static_cast<Scalar>(patch));
        }

        bc.setValues(patch, BCType::Dirichlet, values);
        CHECK(bc.hasFaceValues(patch));
        for (std::size_t k = 0; k < faces.size(); ++k)
        {
            CHECK(bc.valueAt(faces[k]) == doctest::Approx(values[k]));
        }
    }
}

TEST_CASE("Per-face Dirichlet values are exact on Cartesian and O(h) on "
          "sheared meshes")
{
    CartesianMesh cartesian(20, 3, 0.0, 0.0, 1.0, 1.0);
    const Vector cartesianSolution = solveCartesianOneDDiffusion(cartesian);
    for (Index cell = 0; cell < cartesian.cellCount(); ++cell)
    {
        const Point2 center = cartesian.cellCentroid(cell);
        CHECK(cartesianSolution[cell]
            == doctest::Approx(1.0 + 2.0 * center.x).epsilon(1.0e-10));
    }

    // Dirichlet boundary faces carry no explicit non-orthogonal completion
    // term, so the boundary truncation error is O(h) (docs/numerical.md,
    // split fixed-point test).  Refining the sheared mesh must reduce the
    // error: that verifies the per-face values are consumed correctly.
    const Scalar coarseError
        = shearedLinearError(UnstructuredMesh(makeShearedTriMesh(4)));
    const Scalar fineError
        = shearedLinearError(UnstructuredMesh(makeShearedTriMesh(8)));
    CHECK(coarseError <= 0.15);
    CHECK(fineError <= 0.08);
    CHECK(fineError < coarseError);
}

TEST_CASE("Upwind convection consumes per-face inflow values in the RHS")
{
    CartesianMesh mesh(2, 2, 0.0, 0.0, 1.0, 1.0);
    FaceFluxField flux(mesh, "phi");
    for (Index j = 0; j < mesh.ny(); ++j)
    {
        flux[mesh.xFace(0, j)] = -2.0; // uniform inflow
        flux[mesh.xFace(1, j)] = 2.0;
        flux[mesh.xFace(2, j)] = 2.0;
    }

    BoundaryField constant(mesh);
    constant.set(BoundaryField::West, BCType::Dirichlet, 2.0);

    BoundaryField perFace(mesh);
    perFace.setValues(BoundaryField::West,
        BCType::Dirichlet,
        { 2.0, 2.0 });

    auto assemble = [&](const BoundaryField& bc)
    {
        SparseMatrix a(mesh.cellCount(), mesh.cellCount());
        Vector b(mesh.cellCount());
        b.setZero();
        assembleConvection(mesh,
            flux,
            ConvectionScheme::Upwind,
            bc,
            a,
            b);
        a.finalize();
        return std::make_pair(a, b);
    };

    const auto constantSystem = assemble(constant);
    const auto perFaceSystem = assemble(perFace);
    CHECK((constantSystem.first.native() - perFaceSystem.first.native()).norm()
        == doctest::Approx(0.0).epsilon(1.0e-15));
    CHECK((constantSystem.second - perFaceSystem.second)
            .lpNorm<Eigen::Infinity>()
        == doctest::Approx(0.0).epsilon(1.0e-15));

    perFace.setValues(BoundaryField::West,
        BCType::Dirichlet,
        { 1.0, 3.0 });
    const auto distinctSystem = assemble(perFace);

    // The west face flux is -2, so the upwind inflow source is -F*phi_b.
    Vector expected = Vector::Zero(mesh.cellCount());
    expected(mesh.cellIndex(0, 0)) = 2.0 * 1.0;
    expected(mesh.cellIndex(0, 1)) = 2.0 * 3.0;
    CHECK((distinctSystem.second - expected).lpNorm<Eigen::Infinity>()
        == doctest::Approx(0.0).epsilon(1.0e-14));
}

TEST_CASE("correctPressure preserves the prescribed parabolic inlet flux")
{
    CartesianMesh mesh(8, 4, 0.0, 0.0, 1.0, 1.0);
    const Scalar rho = 1.0;
    const Scalar mu = 0.01;

    BoundaryField bcU(mesh);
    const auto inletFaces = mesh.patchFaces(BoundaryField::West);
    std::vector<Scalar> inletU;
    inletU.reserve(inletFaces.size());
    Scalar analyticInflow = 0.0;
    for (const Index face : inletFaces)
    {
        const Scalar y = mesh.faceCentroid(face).y;
        const Scalar value = 4.0 * y * (1.0 - y);
        inletU.push_back(value);
        analyticInflow += value * mesh.dy();
    }
    bcU.setValues(BoundaryField::West, BCType::Dirichlet, inletU);
    bcU.set(BoundaryField::East, BCType::Neumann, 0.0);
    bcU.set(BoundaryField::North, BCType::Dirichlet, 0.0);
    bcU.set(BoundaryField::South, BCType::Dirichlet, 0.0);

    BoundaryField bcV(mesh);
    bcV.set(BoundaryField::West, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::East, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::North, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::South, BCType::Dirichlet, 0.0);

    BoundaryField bcP(mesh);
    bcP.set(BoundaryField::West, BCType::Neumann, 0.0);
    bcP.set(BoundaryField::East, BCType::Dirichlet, 0.0);

    VectorField velocity(mesh, "velocity");
    ScalarField pressure(mesh, "pressure");
    FaceFluxField flux(mesh, "flux");
    initializeMassFlux(mesh,
        velocity,
        rho,
        bcU,
        bcV,
        bcP,
        flux);

    SolverConfig solverConfig;
    solverConfig.tolerance = 1.0e-11;
    solverConfig.maxIterations = 10000;
    auto momSolver = createEigenBiCGSTAB(solverConfig);
    auto pSolver = createEigenCG(solverConfig);

    const MomentumPrediction pred = predictMomentum(mesh,
        flux,
        velocity,
        mu,
        ConvectionScheme::Upwind,
        bcU,
        bcV,
        pressure,
        bcP,
        1.0,
        TimeTerm{},
        *momSolver);
    correctPressure(mesh,
        rho,
        pred,
        bcU,
        bcV,
        bcP,
        *pSolver,
        1.0,
        velocity,
        pressure,
        flux);

    Scalar inletFlux = 0.0;
    for (const Index face : mesh.patchFaces(BoundaryField::West))
    {
        inletFlux += flux[face];
    }
    CHECK(inletFlux
        == doctest::Approx(-analyticInflow).epsilon(1.0e-11));
}

TEST_CASE("Per-face constant outlet pressure matches the patch constant "
          "corrected flux")
{
    const auto patchConstant = correctedFluxWithOutletPressure(false);
    const auto perFace = correctedFluxWithOutletPressure(true);
    REQUIRE(patchConstant.size() == perFace.size());
    for (std::size_t f = 0; f < patchConstant.size(); ++f)
    {
        CHECK(perFace[f]
            == doctest::Approx(patchConstant[f]).epsilon(1.0e-12));
    }
}

TEST_CASE("SIMPLE converges with a per-face parabolic inlet profile")
{
    auto makeInletBCs = [](bool parabolic)
    {
        CartesianMesh mesh(16, 8, 0.0, 0.0, 1.0, 1.0);
        BoundaryField bcU(mesh);
        if (parabolic)
        {
            std::vector<Scalar> inletU;
            for (const Index face : mesh.patchFaces(BoundaryField::West))
            {
                const Scalar y = mesh.faceCentroid(face).y;
                inletU.push_back(1.5 * y * (1.0 - y));
            }
            bcU.setValues(BoundaryField::West,
                BCType::Dirichlet,
                std::move(inletU));
        }
        else
        {
            bcU.set(BoundaryField::West, BCType::Dirichlet, 0.25);
        }
        bcU.set(BoundaryField::East, BCType::Neumann, 0.0);
        bcU.set(BoundaryField::North, BCType::Dirichlet, 0.0);
        bcU.set(BoundaryField::South, BCType::Dirichlet, 0.0);

        BoundaryField bcV(mesh);
        bcV.set(BoundaryField::West, BCType::Dirichlet, 0.0);
        bcV.set(BoundaryField::East, BCType::Dirichlet, 0.0);
        bcV.set(BoundaryField::North, BCType::Dirichlet, 0.0);
        bcV.set(BoundaryField::South, BCType::Dirichlet, 0.0);

        BoundaryField bcP(mesh);
        bcP.set(BoundaryField::East, BCType::Dirichlet, 0.0);

        VectorField velocity(mesh, "velocity");
        ScalarField pressure(mesh, "pressure");
        FaceFluxField flux(mesh, "phi");

        SimpleConfig config;
        config.tolerance = 1.0e-6;
        config.maxIterations = 3000;
        config.relaxationU = 0.7;
        config.relaxationP = 0.3;
        config.solverConfig.tolerance = 1.0e-9;
        config.solverConfig.maxIterations = 2000;

        const SimpleResult result = solveSimple(mesh,
            1.0,
            0.01,
            bcU,
            bcV,
            bcP,
            config,
            velocity,
            pressure,
            flux);
        CHECK(result.converged);
        Vector outletU = velocity.u().data();
        return std::make_tuple(
            result.converged, std::move(outletU), std::move(mesh));
    };

    const auto parabolic = makeInletBCs(true);
    const auto uniform = makeInletBCs(false);

    REQUIRE(std::get<0>(parabolic));
    REQUIRE(std::get<0>(uniform));
    CHECK((std::get<1>(parabolic) - std::get<1>(uniform))
            .lpNorm<Eigen::Infinity>()
        > 1.0e-3);

    const auto outletCenter
        = std::get<2>(parabolic).cellIndex(15, 4);
    const auto outletNearWall
        = std::get<2>(parabolic).cellIndex(15, 1);
    CHECK(std::get<1>(parabolic)(outletCenter)
        > std::get<1>(parabolic)(outletNearWall));
}
