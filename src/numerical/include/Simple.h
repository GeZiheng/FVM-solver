#pragma once

#include "BoundaryCondition.h"
#include "Convection.h"
#include "Field.h"
#include "FluxField.h"
#include "LinearSolver.h"
#include "Mesh.h"
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
 * @brief Configuration for the SIMPLE algorithm.
 */
struct SimpleConfig
{
    int maxIterations = 500;
    Scalar tolerance = 1e-6;  ///< Convergence tolerance on the scaled
                              ///< continuity residual (max cell mass
                              ///< imbalance / characteristic flux).
    Scalar relaxationU = 0.7; ///< Under-relaxation for momentum (0, 1].
    Scalar relaxationP = 0.3; ///< Under-relaxation for pressure (0, 1].
    ConvectionScheme scheme = ConvectionScheme::Upwind;
    fvm::math::SolverConfig solverConfig = {};
    bool verbose = false;
};

/**
 * @brief Per-iteration residual snapshot of the SIMPLE loop.
 */
struct SimpleResiduals
{
    Scalar continuity = 0.0; ///< Scaled max cell mass imbalance.
    Scalar u = 0.0;          ///< Max |u - u_old| after correction.
    Scalar v = 0.0;          ///< Max |v - v_old| after correction.
    Scalar pressure = 0.0;   ///< Max |alpha_p * p'|.
};

/**
 * @brief Outcome of a SIMPLE run.
 */
struct SimpleResult
{
    bool converged = false;
    int iterations = 0;
    std::vector<SimpleResiduals> history;
};

/**
 * @brief Solve the steady incompressible Navier-Stokes equations with the
 * SIMPLE algorithm on a collocated grid (Rhie-Chow interpolation).
 *
 *   div(rho u u) = -grad(p) + div(mu grad u),
 *   div(u) = 0.
 *
 * Each iteration calls the shared momentum predictor (Momentum.h) and
 * pressure corrector (Pressure.h); this file only owns the SIMPLE loop.
 *
 * @param mesh     Computational mesh.
 * @param rho      Density (constant).
 * @param mu       Dynamic viscosity (constant).
 * @param bcU      Boundary conditions for the u-velocity component.
 * @param bcV      Boundary conditions for the v-velocity component.
 * @param bcP      Boundary conditions for pressure (walls: zero-gradient
 *                 Neumann; fixed-pressure boundaries: Dirichlet).
 * @param config   Algorithm configuration.
 * @param velocity In/out: initial guess -> converged velocity.
 * @param pressure In/out: initial guess -> converged pressure.
 * @param flux     Out: persistent face mass-flux field. Re-initialized
 *                 on entry from `velocity` and the velocity BCs
 *                 (computeMassFlux); on return it holds the corrected
 *                 conservative flux (each cell's net outflow is zero up
 *                 to the pressure-solver accuracy).
 */
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
