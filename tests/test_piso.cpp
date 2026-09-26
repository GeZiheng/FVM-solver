#include "Field.h"
#include "FluxField.h"
#include "Mesh.h"
#include "Piso.h"
#include <cmath>
#include <doctest/doctest.h>

using namespace fvm::core;
using namespace fvm::numerical;

namespace
{

constexpr Scalar kPi = 3.14159265358979323846;

// Exact solution of the impulsive start of Couette flow between y = 0
// (fixed) and y = 1 (moving with u = 1), no-slip walls, u(y, 0) = 0:
//   u(y, t) = y + (2/pi) sum_{n>=1} ((-1)^n / n) sin(n pi y)
//                                      exp(-n^2 pi^2 nu t).
Scalar exactCouette(Scalar y, Scalar t, Scalar nu)
{
    Scalar sum = 0.0;
    for (int n = 1; n <= 200; ++n)
    {
        const Scalar sign = (n % 2 == 0) ? 1.0 : -1.0;
        sum += sign / n * std::sin(n * kPi * y) * std::exp(-n * n * kPi * kPi * nu * t);
    }
    return y + (2.0 / kPi) * sum;
}

} // namespace

TEST_CASE("PISO: impulsive Couette start matches the exact series")
{
    // The flow is x-independent (zero-gradient side walls), so the exact
    // 1D series applies. Square cells keep the pressure operator isotropic,
    // which the diagonal-preconditioned CG needs.
    const Index n = 32;
    CartesianMesh mesh(n, n, 0.0, 0.0, 1.0, 1.0);
    const Scalar rho = 1.0;
    const Scalar mu = 0.1; // nu = mu / rho = 0.1
    const Scalar nu = mu / rho;

    BoundaryField bcU; // top lid drives; west/east zero-gradient (default)
    bcU.set(BoundaryField::North, BCType::Dirichlet, 1.0);
    bcU.set(BoundaryField::South, BCType::Dirichlet, 0.0);

    BoundaryField bcV; // no penetration on all sides
    bcV.set(BoundaryField::North, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::South, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::West, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::East, BCType::Dirichlet, 0.0);

    BoundaryField bcP; // zero-gradient everywhere (closed domain)

    VectorField velocity(mesh, "u");
    ScalarField pressure(mesh, "p");
    FaceFluxField flux(mesh, "phi");

    PisoConfig config;
    config.dt = 0.005;
    config.nSteps = 100; // t = 0.5
    config.nCorrectors = 1;
    config.timeScheme = TimeScheme::Euler;
    config.solverConfig.tolerance = 1e-9;
    config.solverConfig.maxIterations = 20000;
    config.verbose = false;

    const PisoResult result = solvePiso(mesh,
        rho,
        mu,
        bcU,
        bcV,
        bcP,
        config,
        velocity,
        pressure,
        flux);

    CHECK(result.steps == config.nSteps);

    const Scalar tEnd = config.dt * config.nSteps;
    Scalar maxErr = 0.0;
    Scalar maxFluxImbalance = 0.0;
    Scalar maxAbsV = 0.0;
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        maxErr = std::max(
            maxErr, std::abs(velocity.u()(c) - exactCouette(y, tEnd, nu)));
        maxFluxImbalance
            = std::max(maxFluxImbalance, std::abs(flux.cellImbalance(c)));
        maxAbsV = std::max(maxAbsV, std::abs(velocity.v()(c)));
    }

    CHECK(maxErr < 0.02);
    CHECK(maxFluxImbalance < 1e-7);
    CHECK(maxAbsV < 1e-4);

    // The profile is bracketed by the wall speeds.
    for (Index j = 0; j < n; ++j)
    {
        const Scalar u = velocity.u()(0, j);
        CHECK(u >= -1e-6);
        CHECK(u <= 1.0 + 1e-6);
    }
}

TEST_CASE("PISO: transient pressure-driven channel reaches Poiseuille flow")
{
    // Unit channel driven by p(west) = 1, p(east) = 0, no-slip walls.
    // Exact fully-developed profile u(y) = y (1 - y) / 2, Q = 1/12.
    const Index n = 16;
    CartesianMesh mesh(n, n, 0.0, 0.0, 1.0, 1.0);
    const Scalar rho = 1.0;
    const Scalar mu = 1.0;

    BoundaryField bcU; // west/east zero-gradient (default)
    bcU.set(BoundaryField::North, BCType::Dirichlet, 0.0);
    bcU.set(BoundaryField::South, BCType::Dirichlet, 0.0);

    BoundaryField bcV;
    bcV.set(BoundaryField::North, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::South, BCType::Dirichlet, 0.0);

    BoundaryField bcP;
    bcP.set(BoundaryField::West, BCType::Dirichlet, 1.0);
    bcP.set(BoundaryField::East, BCType::Dirichlet, 0.0);

    VectorField velocity(mesh, "u");
    ScalarField pressure(mesh, "p");
    FaceFluxField flux(mesh, "phi");

    PisoConfig config;
    config.dt = 0.02;
    config.nSteps = 200; // t = 4 (several viscous times L^2 / nu = 1)
    config.nCorrectors = 2;
    config.timeScheme = TimeScheme::Euler;
    config.solverConfig.tolerance = 1e-9;
    config.solverConfig.maxIterations = 20000;

    const PisoResult result = solvePiso(mesh,
        rho,
        mu,
        bcU,
        bcV,
        bcP,
        config,
        velocity,
        pressure,
        flux);

    CHECK(result.steps == config.nSteps);
    CHECK(result.history.back().continuity < 1e-5);

    Scalar err2 = 0.0;
    Scalar ref2 = 0.0;
    Scalar flowRate = 0.0;
    Scalar maxAbsV = 0.0;
    Scalar maxFluxImbalance = 0.0;
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        const Scalar uExact = 0.5 * y * (1.0 - y);
        err2 += (velocity.u()(c) - uExact) * (velocity.u()(c) - uExact);
        ref2 += uExact * uExact;
        flowRate += velocity.u()(c) * mesh.dy();
        maxAbsV = std::max(maxAbsV, std::abs(velocity.v()(c)));
        maxFluxImbalance
            = std::max(maxFluxImbalance, std::abs(flux.cellImbalance(c)));
    }
    flowRate /= n;

    CHECK(std::sqrt(err2 / ref2) < 0.08);
    CHECK(flowRate == doctest::Approx(1.0 / 12.0).epsilon(0.08));
    CHECK(maxAbsV < 0.01);
    CHECK(maxFluxImbalance < 1e-7);
}
