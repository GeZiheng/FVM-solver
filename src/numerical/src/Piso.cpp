#include "Piso.h"
#include "Convection.h"
#include "Momentum.h"
#include "Pressure.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace fvm::numerical
{

PisoResult solvePiso(const CartesianMesh& mesh,
    Scalar rho,
    Scalar mu,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const BoundaryField& bcP,
    const PisoConfig& config,
    VectorField& velocity,
    ScalarField& pressure,
    FaceFluxField& flux)
{
    if (config.dt <= 0.0)
    {
        throw std::invalid_argument("solvePiso: dt must be positive");
    }
    if (config.nSteps <= 0)
    {
        throw std::invalid_argument("solvePiso: nSteps must be positive");
    }
    if (config.nCorrectors <= 0)
    {
        throw std::invalid_argument("solvePiso: nCorrectors must be positive");
    }

    const Index nCells = mesh.cellCount();

    // Initialize the persistent flux from the velocity field and its BCs
    // (OpenFOAM createPhi). Pure-Neumann pressure is only compatible when
    // the boundary fluxes balance (OpenFOAM adjustPhi).
    computeMassFlux(mesh, velocity, rho, bcU, bcV, flux);

    bool hasDirichletP = false;
    for (int side = 0; side < 4; ++side)
    {
        if (bcP.get(side).type == BCType::Dirichlet)
            hasDirichletP = true;
    }
    if (!hasDirichletP)
        checkFluxCompatibility(flux);

    auto momSolver = fvm::math::createEigenBiCGSTAB(config.solverConfig);
    auto pSolver = fvm::math::createEigenCG(config.solverConfig);

    TimeTerm time;
    time.rho = rho;
    time.dt = config.dt;
    time.scheme = config.timeScheme;

    const Scalar hRef = 0.5 * (mesh.dx() + mesh.dy());

    PisoResult result;
    Scalar t = 0.0;

    for (int step = 1; step <= config.nSteps; ++step)
    {
        // One momentum predictor (no under-relaxation in PISO). The ddt
        // term references the old-time velocity currently in `velocity`.
        MomentumPrediction pred = predictMomentum(mesh,
            flux,
            velocity,
            mu,
            config.scheme,
            bcU,
            bcV,
            pressure,
            bcP,
            1.0,
            time,
            *momSolver);

        // Several pressure correctors; the cumulative reconstruction keeps
        // the velocity consistent with the full corrected pressure.
        for (int corrector = 0; corrector < config.nCorrectors; ++corrector)
        {
            // OpenFOAM re-evaluates HbyA = rAU*UEqn.H() at the top of every
            // pressure corrector, so the second and later correctors solve
            // an equation built from the velocity corrected by the previous
            // one (fvMatrix::H() reads the matrix's current psi_). With a
            // frozen uHat instead, the loop reaches its own fixed point after
            // one sweep, i.e. algebraically nCorrectors = 1, which is unstable
            // for this class of problems (OpenFOAM diverges there too).
            if (corrector > 0)
            {
                refreshUHat(pred, velocity);
            }

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
                flux,
                /*cumulativeVelocityCorrection=*/true);
        }

        t += config.dt;

        // True post-correction continuity error and max speed.
        Scalar maxImbalance = 0.0;
        Scalar maxSpeed = 0.0;
        for (Index P = 0; P < nCells; ++P)
        {
            maxImbalance
                = std::max(maxImbalance, std::abs(flux.cellImbalance(P)));
            maxSpeed = std::max(
                maxSpeed, std::hypot(velocity.u()(P), velocity.v()(P)));
        }
        const Scalar fRef
            = std::max(rho * maxSpeed * hRef, 1e-30);

        PisoStepInfo info;
        info.time = t;
        info.continuity = maxImbalance / fRef;
        info.maxSpeed = maxSpeed;
        result.history.push_back(info);
        result.steps = step;

        if (config.verbose)
        {
            std::cout << "PISO step " << step << " (t=" << t
                      << "): continuity=" << info.continuity
                      << " maxSpeed=" << info.maxSpeed << std::endl;
        }
    }

    return result;
}

} // namespace fvm::numerical
