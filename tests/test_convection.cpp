#include "BoundaryCondition.h"
#include "Convection.h"
#include "Diffusion.h"
#include "Field.h"
#include "LinearSolver.h"
#include "CartesianMesh.h"
#include "ObliqueMeshFixtures.h"
#include "SparseMatrix.h"
#include "UnstructuredMesh.h"
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

    BoundaryField bc(s.mesh); // north/south: zero-flux (solution is 1D)
    bc.set(BoundaryField::West, BCType::Dirichlet, 0.0);
    bc.set(BoundaryField::East, BCType::Dirichlet, 1.0);

    SparseMatrix A(s.mesh.cellCount(), s.mesh.cellCount());
    Vector b(s.mesh.cellCount());
    b.setZero();
    assembleDiffusion(s.mesh, s.gamma, bc, A, b);
    assembleConvection(s.mesh,
        interpolateCellVelocityFlux(s.mesh, s.velocity, rho),
        scheme,
        bc,
        A,
        b);
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
        BoundaryField bc(s.mesh);
        bc.set(BoundaryField::West, BCType::Dirichlet, 0.0);
        bc.set(BoundaryField::East, BCType::Dirichlet, 1.0);

        SparseMatrix A(s.mesh.cellCount(), s.mesh.cellCount());
        Vector b(s.mesh.cellCount());
        b.setZero();
        assembleDiffusion(s.mesh, s.gamma, bc, A, b);
        assembleConvection(s.mesh,
            interpolateCellVelocityFlux(s.mesh, s.velocity, 1.0),
            scheme,
            bc,
            A,
            b);
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

    BoundaryField bc(mesh); // all zero-Neumann (zero-gradient inflow at west)

    for (auto scheme : { ConvectionScheme::Upwind, ConvectionScheme::Central })
    {
        SparseMatrix A(mesh.cellCount(), mesh.cellCount());
        Vector b(mesh.cellCount());
        b.setZero();
        assembleConvection(mesh,
            interpolateCellVelocityFlux(mesh, velocity, 1.0),
            scheme,
            bc,
            A,
            b);
        A.finalize();

        Vector ones = Vector::Ones(mesh.cellCount());
        Vector Aones = A.native() * ones;
        CHECK(Aones.norm() == doctest::Approx(0.0).epsilon(1e-12));
        CHECK(b.norm() == doctest::Approx(0.0).epsilon(1e-12));
    }
}

TEST_CASE("Cartesian skew correction is a no-op")
{
    CartesianMesh mesh(4, 3, 0.0, 0.0, 1.0, 1.0);
    VectorField velocity(mesh, "u");
    ScalarField explicitPhi(mesh, "phi");
    for (Index cell = 0; cell < mesh.cellCount(); ++cell)
    {
        const auto [x, y] = mesh.cellCenter(cell);
        velocity.u()(cell) = 1.0 + 2.0 * x - 3.0 * y;
        velocity.v()(cell) = 4.0 - x + 2.0 * y;
        explicitPhi(cell) = 0.5 + x - 2.0 * y;
    }

    BoundaryField bc(mesh); // all zero-Neumann
    CorrectionOptions corrections;

    for (auto scheme : { ConvectionScheme::Central, ConvectionScheme::Upwind })
    {
        SparseMatrix aOn(mesh.cellCount(), mesh.cellCount());
        SparseMatrix aOff(mesh.cellCount(), mesh.cellCount());
        Vector bOn = Vector::Zero(mesh.cellCount());
        Vector bOff = Vector::Zero(mesh.cellCount());

        assembleConvection(mesh,
            interpolateCellVelocityFlux(mesh, velocity, 2.0),
            scheme,
            bc,
            aOn,
            bOn,
            &explicitPhi,
            corrections);
        assembleConvection(mesh,
            interpolateCellVelocityFlux(mesh, velocity, 2.0),
            scheme,
            bc,
            aOff,
            bOff);
        aOn.finalize();
        aOff.finalize();

        const Scalar matrixDiff = (aOn.native() - aOff.native()).norm();
        const Scalar rhsDiff = (bOn - bOff).norm();
        CHECK(matrixDiff <= 1.0e-15);
        CHECK(rhsDiff <= 1.0e-15);
    }
}

TEST_CASE(
    "computeMassFlux is exact for a linear velocity on the oblique pair")
{
    UnstructuredMesh mesh(makeObliquePair());
    VectorField velocity(mesh, "u");
    for (Index cell = 0; cell < mesh.cellCount(); ++cell)
    {
        const auto [x, y] = mesh.cellCentroid(cell);
        velocity.u()(cell) = 1.0 + 2.0 * x - 3.0 * y;
        velocity.v()(cell) = 4.0 - x + 2.0 * y;
    }

    // Each patch has one straight edge, so its outward-normal derivative is
    // constant for a linear field.
    BoundaryField bcU(mesh);
    BoundaryField bcV(mesh);
    const Point2 uGradient{ 2.0, -3.0 };
    const Point2 vGradient{ -1.0, 2.0 };
    for (Index patch = 0; patch < mesh.patchCount(); ++patch)
    {
        const Index face = mesh.patchFaces(patch)[0];
        const Point2 area = mesh.faceAreaVector(face);
        const Scalar areaNorm = std::hypot(area.x, area.y);
        bcU.set(patch,
            BCType::Neumann,
            (uGradient.x * area.x + uGradient.y * area.y) / areaNorm);
        bcV.set(patch,
            BCType::Neumann,
            (vGradient.x * area.x + vGradient.y * area.y) / areaNorm);
    }

    CorrectionOptions off;
    off.skewInterpolation = false;
    FaceFluxField fluxOn(mesh, "phi");
    FaceFluxField fluxOff(mesh, "phi");
    computeMassFlux(mesh,
        velocity,
        2.0,
        bcU,
        bcV,
        fluxOn,
        CorrectionOptions{});
    computeMassFlux(mesh, velocity, 2.0, bcU, bcV, fluxOff, off);

    Index interiorFace = mesh.faceCount();
    for (Index face = 0; face < mesh.faceCount(); ++face)
    {
        if (!mesh.isBoundaryFace(face))
        {
            interiorFace = face;
            break;
        }
    }
    REQUIRE(interiorFace < mesh.faceCount());

    const Point2 faceCenter = mesh.faceCentroid(interiorFace);
    const Point2 area = mesh.faceAreaVector(interiorFace);
    const Scalar uFace = 1.0 + 2.0 * faceCenter.x - 3.0 * faceCenter.y;
    const Scalar vFace = 4.0 - faceCenter.x + 2.0 * faceCenter.y;
    const Scalar expected = 2.0 * (uFace * area.x + vFace * area.y);
    CHECK(std::abs(fluxOn[interiorFace] - expected) <= 1.0e-10);
    CHECK(std::abs(fluxOff[interiorFace] - expected) > 1.0e-6);
}

TEST_CASE("interpolateCellVelocityFlux leaves boundary faces unchanged")
{
    CartesianMesh cartesian(4, 3, 0.0, 0.0, 1.0, 1.0);
    VectorField cartesianVelocity(cartesian, "u");
    for (Index cell = 0; cell < cartesian.cellCount(); ++cell)
    {
        const auto [x, y] = cartesian.cellCenter(cell);
        cartesianVelocity.u()(cell) = 1.0 + 2.0 * x - 3.0 * y;
        cartesianVelocity.v()(cell) = 4.0 - x + 2.0 * y;
    }

    CorrectionOptions on;
    CorrectionOptions off;
    off.skewInterpolation = false;
    const FaceFluxField fluxOn = interpolateCellVelocityFlux(
        cartesian,
        cartesianVelocity,
        2.0,
        on);
    const FaceFluxField fluxOff = interpolateCellVelocityFlux(
        cartesian,
        cartesianVelocity,
        2.0,
        off);
    for (Index face = 0; face < cartesian.faceCount(); ++face)
    {
        CHECK(std::abs(fluxOn[face] - fluxOff[face]) <= 1.0e-15);
    }

    UnstructuredMesh mesh(makeObliquePair());
    VectorField velocity(mesh, "u");
    for (Index cell = 0; cell < mesh.cellCount(); ++cell)
    {
        const auto [x, y] = mesh.cellCentroid(cell);
        velocity.u()(cell) = 1.0 + 2.0 * x - 3.0 * y;
        velocity.v()(cell) = 4.0 - x + 2.0 * y;
    }
    const FaceFluxField flux = interpolateCellVelocityFlux(mesh, velocity, 2.0, on);
    for (Index face = 0; face < mesh.faceCount(); ++face)
    {
        if (!mesh.isBoundaryFace(face))
            continue;

        const Point2 area = mesh.faceAreaVector(face);
        const auto [uOwner, vOwner] = velocity(mesh.faceOwner(face));
        const Scalar expected = 2.0 * (uOwner * area.x + vOwner * area.y);
        CHECK(std::abs(flux[face] - expected) <= 1.0e-15);
    }
}
