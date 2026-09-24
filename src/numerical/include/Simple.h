#pragma once

#include "BoundaryCondition.h"
#include "Convection.h"
#include "Field.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "TransportEquation.h"
#include "Types.h"

#include <vector>

using fvm::core::CartesianMesh;
using fvm::core::FaceFluxField;
using fvm::core::Index;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::Vector;
using fvm::core::VectorField;

namespace fvm::numerical
{

/**
 * @brief Configuration for the SIMPLE algorithm.
 */
struct SimpleConfig
{
    int maxIterations = 500;
    Scalar tolerance = 1e-6; ///< Convergence tolerance on the scaled
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
 * @brief Assembled momentum equation for one velocity component.
 *
 * The system solves
 *
 *   A u* = b,   b = b0 - grad(p) * vol,
 *
 * with Patankar under-relaxation applied:
 *
 *   A(P,P) -> A(P,P) / alpha,
 *   b      -> b + (1 - alpha) / alpha * A0(P,P) * u_old(P).
 *
 * `diag` holds the relaxed diagonal A(P,P) and `rhsNoPressure` the
 * right-hand side without the pressure-gradient source; together they
 * provide the H/a_P data needed by the Rhie-Chow interpolation.
 */
struct MomentumAssembly
{
    EquationSystem system;
    Vector diag;
    Vector rhsNoPressure;
};

/**
 * @brief Assemble the momentum equation for one velocity component from
 * a precomputed face mass-flux field.
 *
 * Reuses the diffusion (gamma = mu) and flux-based convection
 * operators, then adds the pressure-gradient source (central
 * differences, one-sided at boundary cells) and under-relaxation.
 *
 * @param mesh       Computational mesh.
 * @param flux       Convecting face mass-flux field (e.g. the persistent
 *                   flux maintained by solveSimple).
 * @param velocity   Current velocity field (u_old for the relaxation
 *                   source).
 * @param mu         Dynamic viscosity (constant).
 * @param scheme     Convection scheme.
 * @param bc         Boundary conditions for this velocity component.
 * @param component  0 for the u-equation, 1 for the v-equation.
 * @param pressure   Current pressure field.
 * @param bcP        Pressure boundary conditions (the pressure-gradient
 *                   source uses the Dirichlet boundary values).
 * @param relaxation Under-relaxation factor in (0, 1].
 */
MomentumAssembly assembleMomentum(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    const VectorField& velocity,
    Scalar mu,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    int component,
    const ScalarField& pressure,
    const BoundaryField& bcP,
    Scalar relaxation);

/**
 * @brief Result of the momentum predictor: the predicted velocity plus
 * the Rhie-Chow data (uHat = H/a_P, d = vol/a_P) needed by the pressure
 * corrector.
 */
struct MomentumPrediction
{
    MomentumAssembly momU;
    MomentumAssembly momV;
    Vector uStar; ///< Predicted u (solution of the relaxed momentum eq).
    Vector vStar; ///< Predicted v.
    Vector uHatU; ///< u velocity without the pressure-gradient part.
    Vector uHatV; ///< v velocity without the pressure-gradient part.
    Vector dU;    ///< vol / a_P of the u equation.
    Vector dV;    ///< vol / a_P of the v equation.
};

/**
 * @brief Run the momentum predictor: assemble and solve the u/v momentum
 * equations and compute the Rhie-Chow data (OpenFOAM UEqn.H).
 *
 * @param mesh       Computational mesh.
 * @param flux       Convecting face mass-flux field.
 * @param velocity   Current velocity field (u_old for under-relaxation).
 * @param mu         Dynamic viscosity (constant).
 * @param scheme     Convection scheme.
 * @param bcU        Boundary conditions for the u component.
 * @param bcV        Boundary conditions for the v component.
 * @param pressure   Current pressure field.
 * @param bcP        Pressure boundary conditions.
 * @param relaxationU Momentum under-relaxation factor in (0, 1].
 * @param momSolver  Linear solver used for both momentum systems.
 */
MomentumPrediction predictMomentum(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    const VectorField& velocity,
    Scalar mu,
    ConvectionScheme scheme,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const ScalarField& pressure,
    const BoundaryField& bcP,
    Scalar relaxationU,
    fvm::math::LinearSolver& momSolver);

/**
 * @brief Outcome of one pressure-correction step.
 */
struct CorrectorResult
{
    Scalar maxImbalance = 0.0; ///< Pre-correction max |mass imbalance|.
    Scalar duMax = 0.0;        ///< Max |u_new - u_old| after correction.
    Scalar dvMax = 0.0;        ///< Max |v_new - v_old| after correction.
    Scalar dpMax = 0.0;        ///< Max |relaxationP * p'|.
};

/**
 * @brief Run one pressure-correction step (OpenFOAM pEqn.H): write the
 * Rhie-Chow predicted flux into the persistent flux field, assemble and
 * solve the pressure-correction equation, then correct the face fluxes
 * (conservative), the cell-centered velocity and the pressure in place.
 *
 * With pure Neumann pressure BCs the correction equation is singular;
 * the reference cell 0 is eliminated (p'_0 = 0).
 *
 * @param mesh        Computational mesh.
 * @param rho         Density (constant).
 * @param pred        Momentum predictor result (Rhie-Chow data).
 * @param bcU         Boundary conditions for the u component.
 * @param bcV         Boundary conditions for the v component.
 * @param bcP         Pressure boundary conditions.
 * @param pSolver     Linear solver for the (SPD) correction system.
 * @param relaxationP Pressure under-relaxation factor in (0, 1].
 * @param velocity    In/out: corrected cell-centered velocity.
 * @param pressure    In/out: pressure += relaxationP * p'.
 * @param flux        In/out: overwritten with the predicted flux, then
 *                    corrected in place (conservative up to the pressure
 *                    solver accuracy).
 */
CorrectorResult correctPressure(const CartesianMesh& mesh,
    Scalar rho,
    const MomentumPrediction& pred,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const BoundaryField& bcP,
    fvm::math::LinearSolver& pSolver,
    Scalar relaxationP,
    VectorField& velocity,
    ScalarField& pressure,
    FaceFluxField& flux);

/**
 * @brief Solve the steady incompressible Navier-Stokes equations with the
 * SIMPLE algorithm on a collocated grid (Rhie-Chow interpolation).
 *
 *   div(rho u u) = -grad(p) + div(mu grad u),
 *   div(u) = 0.
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
