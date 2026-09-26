#pragma once

#include "BoundaryCondition.h"
#include "Field.h"
#include "FluxField.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "Momentum.h"
#include "Types.h"

using fvm::core::CartesianMesh;
using fvm::core::FaceFluxField;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::VectorField;

namespace fvm::numerical
{

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
 * @param cumulativeVelocityCorrection
 *                    When true, the cell velocity is reconstructed from
 *                    uHat and the full (relaxed) pressure, making repeated
 *                    calls with the same predictor correct. Requires
 *                    relaxationP == 1 (PISO). When false (steady SIMPLE)
 *                    velocity = uStar - d grad(p') as before.
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
    FaceFluxField& flux,
    bool cumulativeVelocityCorrection = false);

} // namespace fvm::numerical
