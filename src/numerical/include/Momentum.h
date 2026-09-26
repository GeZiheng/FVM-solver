#pragma once

#include "BoundaryCondition.h"
#include "Convection.h"
#include "Field.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "TimeScheme.h"
#include "TransportEquation.h"
#include "Types.h"

using fvm::core::CartesianMesh;
using fvm::core::FaceFluxField;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::Vector;
using fvm::core::VectorField;

namespace fvm::numerical
{

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
 * @param time       Transient ddt term (rho V/dt). If inactive the
 *                   assembly is steady and `relaxation` applies
 *                   (Patankar). If active, `relaxation` must be 1 and
 *                   the theta-scheme ddt term is added instead.
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
    Scalar relaxation,
    const TimeTerm& time = {});

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
 * @param time        Transient ddt term (see assembleMomentum); inactive
 *                    for steady SIMPLE.
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
    const TimeTerm& time,
    fvm::math::LinearSolver& momSolver);

} // namespace fvm::numerical
