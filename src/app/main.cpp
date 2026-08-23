#include <cmath>
#include <iostream>

#include "Convection.h"
#include "Diffusion.h"
#include "Field.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "TransportEquation.h"
#include "VtkWriter.h"

using namespace fvm::core;
using namespace fvm::math;
using namespace fvm::numerical;
using namespace fvm::io;

int main()
{
    // Steady convection-diffusion in a unit square with a recirculating
    // divergence-free velocity field (streamfunction psi = sin(pi x) sin(pi y)).
    // Left wall hot (T = 1), right wall cold (T = 0), top/bottom adiabatic.
    const Index nx = 64, ny = 64;
    CartesianMesh mesh(nx, ny, 0.0, 0.0, 1.0, 1.0);
    std::cout << "Mesh: " << nx << " x " << ny << " cells\n";

    const Scalar rho = 1.0;
    const Scalar gammaVal = 0.01; // Pe ~ 100

    ScalarField gamma(mesh, "gamma");
    gamma.setConstant(gammaVal);

    VectorField velocity(mesh, "velocity");
    const Scalar pi = 3.14159265358979323846;
    for (Index c = 0; c < mesh.cellCount(); ++c)
    {
        auto [x, y] = mesh.cellCenter(c);
        velocity.u()(c) = pi * std::sin(pi * x) * std::cos(pi * y);
        velocity.v()(c) = -pi * std::cos(pi * x) * std::sin(pi * y);
    }

    BoundaryField bc; // north/south default to zero-flux (adiabatic)
    bc.set(BoundaryField::West, BCType::Dirichlet, 1.0);
    bc.set(BoundaryField::East, BCType::Dirichlet, 0.0);

    auto sys = assembleTransport(mesh,
        velocity,
        gamma,
        rho,
        ConvectionScheme::Upwind,
        bc);
    sys.A.finalize();

    SolverConfig cfg;
    cfg.tolerance = 1e-10;
    cfg.maxIterations = 2000;
    auto solver = createEigenBiCGSTAB(cfg);
    Vector sol = solver->solve(sys.A, sys.b);
    std::cout << "BiCGSTAB: " << solver->lastIterations()
              << " iterations, residual " << solver->lastResidual() << "\n";

    ScalarField temperature(mesh, "temperature");
    temperature.data() = sol;

    const std::string filename = "convection_diffusion.vti";
    VtkWriter::write(filename,
        mesh,
        { { "temperature", &temperature } },
        { { "velocity", &velocity } });

    std::cout << "Wrote " << filename << "\n";
    std::cout << "Open with ParaView to visualize.\n";

    return 0;
}
