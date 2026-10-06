#include "Simple.h"
#include "Convection.h"
#include "Momentum.h"
#include "Pressure.h"
#include "TimeScheme.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace fvm::numerical
{

SimpleResult solveSimple(const CartesianMesh& mesh,
    Scalar rho,
    Scalar mu,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const BoundaryField& bcP,
    const SimpleConfig& config,
    VectorField& velocity,
    ScalarField& pressure,
    FaceFluxField& flux)
{
    if (config.relaxationP <= 0.0 || config.relaxationP > 1.0)
    {
        throw std::
            invalid_argument("solveSimple: relaxationP must be in (0, 1]");
    }

    const Index nCells = mesh.cellCount();

    // Initialize the persistent flux field from the velocity field and
    // its BCs (OpenFOAM createPhi).
    computeMassFlux(mesh, velocity, rho, bcU, bcV, flux);

    // With pure Neumann pressure BCs the pressure-correction equation is
    // only compatible when the boundary fluxes sum to zero (OpenFOAM
    // adjustPhi); catch unbalanced velocity BCs here instead of solving
    // an inconsistent system.
    bool hasDirichletP = false;
    for (int side = 0; side < 4; ++side)
    {
        if (bcP.get(side).type == BCType::Dirichlet)
            hasDirichletP = true;
    }
    if (!hasDirichletP)
        checkFluxCompatibility(flux);

    auto momSolver = createEigenBiCGSTAB(config.solverConfig);
    auto pSolver = createEigenCG(config.solverConfig);

    SimpleResult result;

    for (int iter = 1; iter <= config.maxIterations; ++iter)
    {
        const MomentumPrediction pred = predictMomentum(mesh,
            flux,
            velocity,
            mu,
            config.scheme,
            bcU,
            bcV,
            pressure,
            bcP,
            config.relaxationU,
            TimeTerm{},
            *momSolver);

        CorrectorResult corr;
        try
        {
            corr = correctPressure(mesh,
                rho,
                pred,
                bcU,
                bcV,
                bcP,
                *pSolver,
                config.relaxationP,
                velocity,
                pressure,
                flux);
        }
        catch (const std::exception& e)
        {
            std::cerr << "solveSimple iter " << iter
                      << ": pressure-correction solve failed: " << e.what()
                      << " (lastIterations=" << pSolver->lastIterations()
                      << ", lastResidual=" << pSolver->lastResidual() << ")"
                      << std::endl;
            throw;
        }

        // ---- Residuals and convergence --------------------------------
        Scalar maxSpeed = 0.0;
        for (Index P = 0; P < nCells; ++P)
        {
            maxSpeed = std::max(maxSpeed,
                std::hypot(velocity.u()(P), velocity.v()(P)));
        }
        const Scalar fRef
            = std::max(rho * maxSpeed * 0.5 * (mesh.dx() + mesh.dy()), 1e-30);
        const Scalar uRef = std::max(maxSpeed, 1e-30);

        SimpleResiduals res;
        res.continuity = corr.maxImbalance / fRef;
        res.u = corr.duMax / uRef;
        res.v = corr.dvMax / uRef;
        res.pressure = corr.dpMax;
        result.history.push_back(res);

        if (config.verbose)
        {
            std::cout << "SIMPLE iter " << iter
                      << ": continuity=" << res.continuity << " du=" << res.u
                      << " dv=" << res.v << " dp=" << res.pressure << std::endl;
        }

        if (res.continuity < config.tolerance && res.u < config.tolerance
            && res.v < config.tolerance)
        {
            result.converged = true;
            result.iterations = iter;
            return result;
        }
    }

    result.iterations = config.maxIterations;
    return result;
}

} // namespace fvm::numerical
