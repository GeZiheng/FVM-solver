#pragma once

#include "BoundaryCondition.h"
#include "Convection.h"
#include "Field.h"
#include "FluxField.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "Types.h"

#include <vector>

namespace fvm::numerical
{
using fvm::core::CartesianMesh;
using fvm::core::FaceFluxField;
using fvm::core::Index;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::VectorField;

/// Configuration for the SIMPLE algorithm.
struct SimpleConfig
{
    int maxIterations = 500;
    Scalar tolerance = 1e-6;  ///< Tolerance on the scaled residuals.
    Scalar relaxationU = 0.7; ///< Momentum under-relaxation in (0, 1].
    Scalar relaxationP = 0.3; ///< Pressure under-relaxation in (0, 1].
    ConvectionScheme scheme = ConvectionScheme::Upwind;
    fvm::math::SolverConfig solverConfig = {};
    bool verbose = false;
};

/// Per-iteration residual snapshot (scaling is documented in docs/numerical.md).
struct SimpleResiduals
{
    Scalar continuity = 0.0; ///< Scaled max cell mass imbalance.
    Scalar u = 0.0;          ///< Max |u - u_old| after correction.
    Scalar v = 0.0;          ///< Max |v - v_old| after correction.
    Scalar pressure = 0.0;   ///< Max |alpha_p * p'|.
};

/// Outcome of a SIMPLE run.
struct SimpleResult
{
    bool converged = false;
    int iterations = 0;
    std::vector<SimpleResiduals> history;
};

/// Solve the steady incompressible Navier-Stokes equations with SIMPLE on a
/// collocated grid (Rhie-Chow interpolation).
///
/// `velocity` and `pressure` are in/out (initial guess -> converged fields).
/// `flux` is re-initialized on entry from `velocity` and the velocity BCs
/// (computeMassFlux); on return it holds the corrected conservative flux
/// (net outflow zero to the pressure solver's accuracy).  The iteration
/// itself lives in the shared predictor/corrector; this file owns the loop.
SimpleResult solveSimple(const CartesianMesh& mesh,
    Scalar rho,
    Scalar mu,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const BoundaryField& bcP,
    const SimpleConfig& config,
    VectorField& velocity,
    ScalarField& pressure,
    FaceFluxField& flux);

} // namespace fvm::numerical
