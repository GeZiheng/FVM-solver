#include "Field.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "Simple.h"
#include "TransportEquation.h"
#include <cmath>
#include <doctest/doctest.h>

using namespace fvm::core;
using namespace fvm::math;
using namespace fvm::numerical;

namespace
{

// Recirculating divergence-free velocity (streamfunction
// psi = sin(pi x) sin(pi y)) used by the transport tests.
VectorField makeRecirculatingVelocity(const CartesianMesh& mesh)
{
    VectorField velocity(mesh, "u");
    const Scalar pi = 3.14159265358979323846;
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        velocity.u()(c) = pi * std::sin(pi * x) * std::cos(pi * y);
        velocity.v()(c) = -pi * std::cos(pi * x) * std::sin(pi * y);
    }
    return velocity;
}

} // namespace

TEST_CASE("Simple: momentum assembly matches transport without pressure "
          "or relaxation")
{
    CartesianMesh mesh(12, 10, 0.0, 0.0, 1.0, 1.0);
    const Scalar rho = 1.2;
    const Scalar mu = 0.05;

    VectorField velocity = makeRecirculatingVelocity(mesh);
    ScalarField pressure(mesh, "p"); // zero

    ScalarField muField(mesh, "mu");
    muField.setConstant(mu);

    BoundaryField bc;
    bc.set(BoundaryField::West, BCType::Dirichlet, 1.0);
    bc.set(BoundaryField::East, BCType::Dirichlet, 0.0);

    // Reference: generic scalar transport of the u-component.
    auto ref = assembleTransport(mesh,
        velocity,
        muField,
        rho,
        ConvectionScheme::Upwind,
        bc);
    ref.A.finalize();

    auto mom = assembleMomentum(mesh,
        velocity,
        rho,
        mu,
        ConvectionScheme::Upwind,
        bc,
        0,
        pressure,
        BoundaryField{},
        1.0);
    mom.system.A.finalize();

    CHECK((mom.system.A.native() - ref.A.native()).norm()
          == doctest::Approx(0.0).epsilon(1e-12));
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        CHECK(mom.system.b(c) == doctest::Approx(ref.b(c)).epsilon(1e-12));
        CHECK(mom.rhsNoPressure(c)
              == doctest::Approx(ref.b(c)).epsilon(1e-12));
    }
}

TEST_CASE("Simple: under-relaxation preserves the residual at u_old")
{
    CartesianMesh mesh(12, 10, 0.0, 0.0, 1.0, 1.0);
    const Scalar rho = 1.0;
    const Scalar mu = 0.1;

    VectorField velocity = makeRecirculatingVelocity(mesh);
    ScalarField pressure(mesh, "p"); // zero

    ScalarField muField(mesh, "mu");
    muField.setConstant(mu);

    BoundaryField bc;
    bc.set(BoundaryField::West, BCType::Dirichlet, 1.0);
    bc.set(BoundaryField::East, BCType::Dirichlet, 0.0);

    // Unrelaxed reference system for the u-component.
    auto ref = assembleTransport(mesh,
        velocity,
        muField,
        rho,
        ConvectionScheme::Upwind,
        bc);
    ref.A.finalize();

    const Scalar alpha = 0.5;
    auto mom = assembleMomentum(mesh,
        velocity,
        rho,
        mu,
        ConvectionScheme::Upwind,
        bc,
        0,
        pressure,
        BoundaryField{},
        alpha);
    mom.system.A.finalize();

    // Algebraic identity of Patankar under-relaxation: at phi = phi_old the
    // residual of the relaxed system equals the unrelaxed residual,
    //   (A + (1/a-1) diag(A)) u_old - (b + (1/a-1) diag(A) u_old)
    //       = A u_old - b,
    // so the unrelaxed solution is a fixed point of the relaxed system.
    const Vector& uOld = velocity.u().data();
    const Vector resRelaxed = mom.system.A.native() * uOld - mom.system.b;
    const Vector resRef = ref.A.native() * uOld - ref.b;
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        CHECK(resRelaxed(c) == doctest::Approx(resRef(c)).epsilon(1e-10));
    }

    // Relaxed diagonal = unrelaxed diagonal / alpha.
    auto momNoRelax = assembleMomentum(mesh,
        velocity,
        rho,
        mu,
        ConvectionScheme::Upwind,
        bc,
        0,
        pressure,
        BoundaryField{},
        1.0);
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        CHECK(mom.diag(c)
              == doctest::Approx(momNoRelax.diag(c) / alpha).epsilon(1e-12));
    }

    // The unrelaxed solution solves the relaxed system too (fixed point),
    // verified through the actual solver path. Since assembleMomentum uses
    // `velocity` both as convecting velocity and as u_old, first iterate to
    // a self-consistent state where u_old solves its own unrelaxed system.
    VectorField fpVel = makeRecirculatingVelocity(mesh);
    auto directSolver = createEigenSparseLU();
    Vector cur = fpVel.u().data();
    for (int it = 0; it < 50; ++it)
    {
        fpVel.u().data() = cur;
        auto sys = assembleTransport(mesh,
            fpVel,
            muField,
            rho,
            ConvectionScheme::Upwind,
            bc);
        sys.A.finalize();
        const Vector next = directSolver->solve(sys.A, sys.b);
        if ((next - cur).lpNorm<Eigen::Infinity>() < 1e-10)
        {
            cur = next;
            break;
        }
        cur = next;
    }
    fpVel.u().data() = cur; // self-consistent: cur solves A(cur) x = b(cur)

    auto momFp = assembleMomentum(mesh,
        fpVel,
        rho,
        mu,
        ConvectionScheme::Upwind,
        bc,
        0,
        pressure,
        BoundaryField{},
        alpha);
    momFp.system.A.finalize();
    const Vector solFp = directSolver->solve(momFp.system.A, momFp.system.b);
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        // Tolerance accounts for the Picard loop's finite convergence
        // tolerance and SparseLU roundoff; the fixed-point property itself
        // is exact (see the residual identity checked above).
        CHECK(solFp(c) == doctest::Approx(cur(c)).epsilon(1e-5));
    }
}

TEST_CASE("Simple: pressure gradient enters only the full right-hand side")
{
    CartesianMesh mesh(8, 6, 0.0, 0.0, 1.0, 1.0);
    VectorField velocity(mesh, "u"); // zero convection
    ScalarField pressure(mesh, "p");

    // Linear pressure p = x -> dp/dx = 1, dp/dy = 0 everywhere. The
    // Dirichlet pressure BCs at west/east are consistent with p = x, so
    // the Gauss gradient is exact also in the boundary-adjacent cells.
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        pressure(c) = x;
    }

    BoundaryField bc; // zero-flux walls (keeps system simple)
    BoundaryField bcP;
    bcP.set(BoundaryField::West, BCType::Dirichlet, 0.0);
    bcP.set(BoundaryField::East, BCType::Dirichlet, 1.0);
    const Scalar vol = mesh.cellVolume(0);

    auto mom = assembleMomentum(mesh,
        velocity,
        1.0,
        0.1,
        ConvectionScheme::Upwind,
        bc,
        0,
        pressure,
        bcP,
        1.0);

    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        // b = b0 - dp/dx * vol, with b0 = 0 here (no convection, and a
        // pure-Neumann diffusion operator contributes nothing to b).
        CHECK(mom.rhsNoPressure(c)
              == doctest::Approx(0.0).epsilon(1e-12));
        CHECK(mom.system.b(c) == doctest::Approx(-vol).epsilon(1e-10));
    }

    // v-component sees dp/dy = 0.
    auto momV = assembleMomentum(mesh,
        velocity,
        1.0,
        0.1,
        ConvectionScheme::Upwind,
        bc,
        1,
        pressure,
        bcP,
        1.0);
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        CHECK(momV.system.b(c) == doctest::Approx(0.0).epsilon(1e-12));
    }
}

TEST_CASE("Simple: pressure-driven channel flow reproduces Poiseuille "
          "profile")
{
    // Unit square channel: walls at north/south (no-slip), pressure
    // difference between west (p=1) and east (p=0) drives the flow.
    // Exact fully-developed solution: u(y) = y (1 - y) / 2, v = 0.
    const Index nx = 32, ny = 32;
    CartesianMesh mesh(nx, ny, 0.0, 0.0, 1.0, 1.0);
    const Scalar rho = 1.0;
    const Scalar mu = 1.0;

    BoundaryField bcU; // west/east: zero-gradient (default Neumann 0)
    bcU.set(BoundaryField::North, BCType::Dirichlet, 0.0);
    bcU.set(BoundaryField::South, BCType::Dirichlet, 0.0);

    BoundaryField bcV;
    bcV.set(BoundaryField::North, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::South, BCType::Dirichlet, 0.0);

    BoundaryField bcP; // north/south: zero-gradient (default)
    bcP.set(BoundaryField::West, BCType::Dirichlet, 1.0);
    bcP.set(BoundaryField::East, BCType::Dirichlet, 0.0);

    VectorField velocity(mesh, "u");
    ScalarField pressure(mesh, "p");

    SimpleConfig config;
    config.tolerance = 1e-7;
    config.maxIterations = 2000;
    config.relaxationU = 0.7;
    config.relaxationP = 0.3;
    config.solverConfig.tolerance = 1e-9;
    config.solverConfig.maxIterations = 2000;

    const SimpleResult result
        = solveSimple(mesh, rho, mu, bcU, bcV, bcP, config, velocity,
            pressure);

    CHECK(result.converged);
    CHECK(result.history.back().continuity < config.tolerance);

    // Compare against the parabolic profile u(y) = y (1 - y) / 2.
    Scalar err2 = 0.0, ref2 = 0.0, flowRate = 0.0, maxAbsV = 0.0;
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        const Scalar uExact = 0.5 * y * (1.0 - y);
        err2 += (velocity.u()(c) - uExact) * (velocity.u()(c) - uExact);
        ref2 += uExact * uExact;
        flowRate += velocity.u()(c) * mesh.dy();
        maxAbsV = std::max(maxAbsV, std::abs(velocity.v()(c)));
    }
    flowRate /= nx; // average over columns

    CHECK(std::sqrt(err2 / ref2) < 0.05);
    // Exact flow rate Q = u_max * 2/3 = 1/12.
    CHECK(flowRate == doctest::Approx(1.0 / 12.0).epsilon(0.05));
    CHECK(maxAbsV < 0.01 * 0.125);
}

TEST_CASE("Simple: lid-driven cavity at Re=100 converges with recirculation")
{
    const Index n = 32;
    CartesianMesh mesh(n, n, 0.0, 0.0, 1.0, 1.0);
    const Scalar rho = 1.0;
    const Scalar uLid = 1.0;
    const Scalar mu = 0.01; // Re = rho U L / mu = 100

    BoundaryField bcU; // walls default Neumann -> set all to no-slip
    bcU.set(BoundaryField::North, BCType::Dirichlet, uLid);
    bcU.set(BoundaryField::East, BCType::Dirichlet, 0.0);
    bcU.set(BoundaryField::West, BCType::Dirichlet, 0.0);
    bcU.set(BoundaryField::South, BCType::Dirichlet, 0.0);

    BoundaryField bcV;
    bcV.set(BoundaryField::North, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::East, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::West, BCType::Dirichlet, 0.0);
    bcV.set(BoundaryField::South, BCType::Dirichlet, 0.0);

    BoundaryField bcP; // zero-gradient everywhere (default)

    VectorField velocity(mesh, "u");
    ScalarField pressure(mesh, "p");

    SimpleConfig config;
    config.tolerance = 1e-6;
    config.maxIterations = 3000;
    config.relaxationU = 0.7;
    config.relaxationP = 0.3;
    config.solverConfig.tolerance = 1e-9;
    config.solverConfig.maxIterations = 2000;

    const SimpleResult result
        = solveSimple(mesh, rho, mu, bcU, bcV, bcP, config, velocity,
            pressure);

    CHECK(result.converged);
    CHECK(result.history.back().continuity < config.tolerance);

    // Boundedness: u is bracketed by its boundary values [0, U_lid] up to
    // a small correction-step overshoot.
    const Scalar uMin = velocity.u().data().minCoeff();
    const Scalar uMax = velocity.u().data().maxCoeff();
    CHECK(uMax <= uLid + 0.05);
    CHECK(uMin >= -0.6);

    // Recirculation pattern of the primary vortex:
    // u > 0 near the lid, u < 0 near the bottom (vertical centerline);
    // v > 0 near the left wall, v < 0 near the right wall (horizontal
    // centerline).
    CHECK(velocity.u()(n / 2, n - 2) > 0.0);
    CHECK(velocity.u()(n / 2, 1) < 0.0);
    CHECK(velocity.v()(2, n / 2) > 0.0);
    CHECK(velocity.v()(n - 3, n / 2) < 0.0);
}
