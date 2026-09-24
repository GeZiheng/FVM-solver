#include "Convection.h"
#include "Field.h"
#include "FluxField.h"
#include "Mesh.h"
#include "Simple.h"
#include "SparseMatrix.h"
#include <cmath>
#include <doctest/doctest.h>

using namespace fvm::core;
using namespace fvm::math;
using namespace fvm::numerical;

TEST_CASE("FluxField: face counts and indexing on a 3x2 mesh")
{
    CartesianMesh mesh(3, 2, 0.0, 0.0, 3.0, 2.0);
    FaceFluxField flux(mesh, "phi");

    CHECK(flux.xFaceCount() == 4 * 2);
    CHECK(flux.yFaceCount() == 3 * 3);

    // Distinct (i, j) faces map to distinct storage entries.
    flux.x(0, 0) = 1.0;
    flux.x(3, 1) = 2.0;
    flux.y(0, 0) = 3.0;
    flux.y(2, 2) = 4.0;
    CHECK(flux.x(0, 0) == 1.0);
    CHECK(flux.x(3, 1) == 2.0);
    CHECK(flux.y(0, 0) == 3.0);
    CHECK(flux.y(2, 2) == 4.0);
}

TEST_CASE(
    "FluxField: outwardFlux sign convention matches the mesh face "
    "convention")
{
    CartesianMesh mesh(3, 2, 0.0, 0.0, 3.0, 2.0);
    FaceFluxField flux(mesh, "phi");

    // Interior cell (1, 0): east +x(2,0), west -x(1,0),
    //                       north +y(1,1), south -y(1,0).
    const Index cell = mesh.cellIndex(1, 0);
    flux.x(2, 0) = 10.0;
    flux.x(1, 0) = 20.0;
    flux.y(1, 1) = 30.0;
    flux.y(1, 0) = 40.0;

    CHECK(flux.outwardFlux(cell, 0) == doctest::Approx(10.0));  // east
    CHECK(flux.outwardFlux(cell, 1) == doctest::Approx(30.0));  // north
    CHECK(flux.outwardFlux(cell, 2) == doctest::Approx(-20.0)); // west
    CHECK(flux.outwardFlux(cell, 3) == doctest::Approx(-40.0)); // south
}

TEST_CASE("FluxField: cellImbalance sums the signed outward fluxes")
{
    CartesianMesh mesh(3, 3, 0.0, 0.0, 3.0, 3.0);
    FaceFluxField flux(mesh, "phi");

    // Interior cell (1, 1): inflow 2 from west, 3 from south,
    // outflow 4 east, 1 north -> imbalance = 4 - (-2) ... computed as
    // x(2,1) - x(1,1) + y(1,2) - y(1,1) = 4 - 2 + 1 - 3 = 0.
    flux.x(1, 1) = 2.0; // west face: +x flux INTO the cell
    flux.x(2, 1) = 4.0; // east face: outflow
    flux.y(1, 1) = 3.0; // south face: +y flux INTO the cell
    flux.y(1, 2) = 1.0; // north face: outflow

    const Index cell = mesh.cellIndex(1, 1);
    CHECK(flux.cellImbalance(cell) == doctest::Approx(0.0));

    flux.y(1, 2) = 2.0; // one extra unit out -> imbalance 1
    CHECK(flux.cellImbalance(cell) == doctest::Approx(1.0));
}

TEST_CASE(
    "FluxField: computeMassFlux interpolates interior faces and "
    "honors Dirichlet velocity BCs")
{
    CartesianMesh mesh(4, 3, 0.0, 0.0, 4.0, 3.0);
    const Scalar rho = 1.5;

    VectorField velocity(mesh, "u");
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        velocity.u()(c) = 1.0 + 0.1 * c;
        velocity.v()(c) = -2.0 + 0.2 * c;
    }

    BoundaryField bcU; // west Dirichlet, rest zero-gradient
    bcU.set(BoundaryField::West, BCType::Dirichlet, 7.0);
    BoundaryField bcV; // north Dirichlet
    bcV.set(BoundaryField::North, BCType::Dirichlet, -5.0);

    FaceFluxField flux(mesh, "phi");
    computeMassFlux(mesh, velocity, rho, bcU, bcV, flux);
    const Scalar dy = mesh.dy();
    const Scalar dx = mesh.dx();

    // Interior x-face between cells (1,1) and (2,1): arithmetic mean.
    const Index c11 = mesh.cellIndex(1, 1);
    const Index c21 = mesh.cellIndex(2, 1);
    CHECK(flux.x(2, 1)
          == doctest::Approx(rho * 0.5 * (velocity.u()(c11) + velocity.u()(c21))
                             * dy));

    // West boundary: Dirichlet value everywhere.
    for (Index j = 0; j < mesh.ny(); ++j)
    {
        CHECK(flux.x(0, j) == doctest::Approx(rho * 7.0 * dy));
    }
    // East boundary: zero-gradient -> adjacent cell velocity.
    for (Index j = 0; j < mesh.ny(); ++j)
    {
        CHECK(flux.x(mesh.nx(), j)
              == doctest::Approx(rho * velocity.u()(mesh.nx() - 1, j) * dy));
    }
    // North boundary: Dirichlet v.
    for (Index i = 0; i < mesh.nx(); ++i)
    {
        CHECK(flux.y(i, mesh.ny()) == doctest::Approx(rho * -5.0 * dx));
    }
    // South boundary: zero-gradient -> adjacent cell velocity.
    for (Index i = 0; i < mesh.nx(); ++i)
    {
        CHECK(flux.y(i, 0) == doctest::Approx(rho * velocity.v()(i, 0) * dx));
    }
}

TEST_CASE(
    "FluxField: flux-based convection assembly uses the stored "
    "face fluxes directly")
{
    // Two cells side by side, unit spacing -> face area 1. A prescribed
    // flux F = 2 flows from cell 0 to cell 1 across the interior x-face;
    // all boundary fluxes are zero (no boundary contributions).
    CartesianMesh mesh(2, 1, 0.0, 0.0, 2.0, 1.0);
    FaceFluxField flux(mesh, "phi");
    flux.x(1, 0) = 2.0;

    BoundaryField bc;
    const Index n = mesh.cellCount();

    {
        SparseMatrix A(n, n);
        Vector b(n);
        b.setZero();
        assembleConvection(mesh, flux, ConvectionScheme::Upwind, bc, A, b);
        A.finalize();

        // Upwind, F > 0 (0 -> 1): A(0,0) += F, A(1,0) -= F.
        CHECK(A.native().coeff(0, 0) == doctest::Approx(2.0));
        CHECK(A.native().coeff(1, 0) == doctest::Approx(-2.0));
        CHECK(A.native().coeff(0, 1) == doctest::Approx(0.0));
        CHECK(A.native().coeff(1, 1) == doctest::Approx(0.0));
        CHECK(b.norm() == doctest::Approx(0.0).epsilon(1e-15));
    }
    {
        SparseMatrix A(n, n);
        Vector b(n);
        b.setZero();
        assembleConvection(mesh, flux, ConvectionScheme::Central, bc, A, b);
        A.finalize();

        // Central: face value = 0.5 (phi_0 + phi_1), h = F / 2 = 1.
        CHECK(A.native().coeff(0, 0) == doctest::Approx(1.0));
        CHECK(A.native().coeff(0, 1) == doctest::Approx(1.0));
        CHECK(A.native().coeff(1, 0) == doctest::Approx(-1.0));
        CHECK(A.native().coeff(1, 1) == doctest::Approx(-1.0));
    }
}

TEST_CASE(
    "FluxField: SIMPLE-corrected flux is conservative on a small "
    "lid-driven cavity")
{
    const Index n = 8;
    CartesianMesh mesh(n, n, 0.0, 0.0, 1.0, 1.0);

    BoundaryField bcU;
    bcU.set(BoundaryField::North, BCType::Dirichlet, 1.0);
    bcU.set(BoundaryField::East, BCType::Dirichlet, 0.0);
    bcU.set(BoundaryField::West, BCType::Dirichlet, 0.0);
    bcU.set(BoundaryField::South, BCType::Dirichlet, 0.0);
    BoundaryField bcV;
    bcV.set(BoundaryField::North, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::East, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::West, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::South, BCType::Dirichlet, 0.0);
    BoundaryField bcP; // zero-gradient everywhere (pure Neumann)

    VectorField velocity(mesh, "u");
    ScalarField pressure(mesh, "p");
    FaceFluxField flux(mesh, "phi");

    SimpleConfig config;
    config.tolerance = 1e-6;
    config.maxIterations = 3000;
    config.solverConfig.tolerance = 1e-9;
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
    REQUIRE(result.converged);

    // Every cell's net outflow vanishes up to the pressure solver's
    // relative accuracy (1e-9); the wall fluxes are exactly zero.
    Scalar maxImbalance = 0.0;
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        maxImbalance = std::max(maxImbalance, std::abs(flux.cellImbalance(c)));
    }
    CHECK(maxImbalance < 1e-8);

    for (Index j = 0; j < mesh.ny(); ++j)
    {
        CHECK(std::abs(flux.x(0, j)) == doctest::Approx(0.0).epsilon(1e-15));
        CHECK(std::abs(flux.x(n, j)) == doctest::Approx(0.0).epsilon(1e-15));
    }
    for (Index i = 0; i < mesh.nx(); ++i)
    {
        CHECK(std::abs(flux.y(i, 0)) == doctest::Approx(0.0).epsilon(1e-15));
        CHECK(std::abs(flux.y(i, n)) == doctest::Approx(0.0).epsilon(1e-15));
    }
}
