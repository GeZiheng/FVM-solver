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

/// Configuration for the PISO transient solver.
struct PisoConfig
{
    Scalar dt = 0.01;    ///< Time-step size (> 0).
    int nSteps = 100;    ///< Number of time steps to advance.
    int nCorrectors = 2; ///< Pressure correctors per step (>= 2 for stability).
    TimeScheme timeScheme = TimeScheme::Euler; ///< ddt scheme.
    ConvectionScheme scheme = ConvectionScheme::Upwind;
    fvm::math::SolverConfig solverConfig = {};
    bool verbose = false;
};

/// Per-time-step diagnostics of the PISO loop.
struct PisoStepInfo
{
    Scalar time = 0.0;       ///< Time at the end of the step.
    Scalar continuity = 0.0; ///< Scaled max |cell mass imbalance| after
                             ///< the correctors (true continuity error).
    Scalar maxSpeed = 0.0;   ///< Max |u| after the step.
};

/// Outcome of a PISO run.
struct PisoResult
{
    int steps = 0;
    std::vector<PisoStepInfo> history;
};

/// Advance the transient incompressible Navier-Stokes equations with PISO on a
/// collocated grid (Rhie-Chow interpolation).
///
/// Each step is one momentum predictor (no under-relaxation) plus
/// `nCorrectors` pressure corrections.  Before every corrector after the first
/// the Rhie-Chow data is re-evaluated from the velocity corrected so far
/// (OpenFOAM `HbyA = rAU*UEqn.H()`); without that refresh the loop stalls after
/// one sweep (see `refreshUHat`).  The ddt term is the theta scheme in
/// `PisoConfig::timeScheme`; the persistent flux is carried between steps and
/// is conservative to the pressure solver's accuracy.  Closed domains
/// (pure-Neumann pressure) are supported via reference-cell elimination and the
/// entry flux-compatibility check.
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
