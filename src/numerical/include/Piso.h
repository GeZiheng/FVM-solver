#pragma once

#include "BoundaryCondition.h"
#include "Convection.h"
#include "Field.h"
#include "FluxField.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "TimeScheme.h"
#include "Types.h"

#include <vector>

using fvm::core::CartesianMesh;
using fvm::core::FaceFluxField;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::VectorField;

namespace fvm::numerical
{

/**
 * @brief Configuration for the PISO transient solver.
 */
struct PisoConfig
{
    Scalar dt = 0.01; ///< Time-step size (> 0).
    int nSteps = 100; ///< Number of time steps to advance.
    int nCorrectors = 2; ///< Pressure correctors per time step (>= 1).
    TimeScheme timeScheme = TimeScheme::Euler; ///< ddt scheme.
    ConvectionScheme scheme = ConvectionScheme::Upwind;
    fvm::math::SolverConfig solverConfig = {};
    bool verbose = false;
};

/**
 * @brief Per-time-step diagnostics of the PISO loop.
 */
struct PisoStepInfo
{
    Scalar time = 0.0;       ///< Time at the end of the step.
    Scalar continuity = 0.0; ///< Scaled max |cell mass imbalance| after
                             ///< the correctors (true continuity error).
    Scalar maxSpeed = 0.0;   ///< Max |u| after the step.
};

/**
 * @brief Outcome of a PISO run.
 */
struct PisoResult
{
    int steps = 0;
    std::vector<PisoStepInfo> history;
};

/**
 * @brief Advance the transient incompressible Navier-Stokes equations with
 * PISO on a collocated grid (Rhie-Chow interpolation).
 *
 * Each time step performs one momentum predictor
 * (predictMomentum, no under-relaxation) followed by `nCorrectors`
 * pressure-correction steps (correctPressure). The momentum ddt term is
 * the theta scheme selected by PisoConfig::timeScheme. The persistent
 * face flux is carried between steps and is conservative to the pressure
 * solver's accuracy.
 *
 * A pure-Neumann pressure (closed domain) is tolerated: the reference
 * cell is eliminated in the correction equation, and the boundary fluxes
 * are checked for compatibility at entry (checkFluxCompatibility).
 *
 * @param mesh     Computational mesh.
 * @param rho      Density (constant).
 * @param mu       Dynamic viscosity (constant).
 * @param bcU      Boundary conditions for the u-velocity component.
 * @param bcV      Boundary conditions for the v-velocity component.
 * @param bcP      Pressure boundary conditions.
 * @param config   Algorithm configuration.
 * @param velocity In/out: initial field -> field after the last step.
 * @param pressure In/out: initial guess -> pressure after the last step.
 * @param flux     Out: persistent conservative face mass-flux field.
 */
PisoResult solvePiso(const CartesianMesh& mesh,
    Scalar rho,
    Scalar mu,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const BoundaryField& bcP,
    const PisoConfig& config,
    VectorField& velocity,
    ScalarField& pressure,
    FaceFluxField& flux);

} // namespace fvm::numerical
