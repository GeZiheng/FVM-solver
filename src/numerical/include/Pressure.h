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

/// Outcome of one pressure-correction step.
struct CorrectorResult
{
    Scalar maxImbalance = 0.0; ///< Max |mass imbalance| entering the correction
                               ///< (after the closed-domain flux adjustment).
    Scalar duMax = 0.0;        ///< Max |u_new - u_old| after correction.
    Scalar dvMax = 0.0;        ///< Max |v_new - v_old| after correction.
    Scalar dpMax = 0.0;        ///< Max |relaxationP * p'|.
};

/// Run one pressure-correction step (OpenFOAM pEqn.H): write the Rhie-Chow
/// predicted flux into the persistent flux field, assemble and solve the p'
/// equation, then correct the face flux, the cell velocity and the pressure
/// in place (`pressure += relaxationP * p'`; the flux is conservative up to the
/// pressure solver's accuracy).
///
/// Invariants (see docs/numerical.md):
///  - pure-Neumann p is singular: reference cell 0 is eliminated (p'_0 = 0);
///  - the predicted boundary flux is built exactly like the interior Rhie-Chow
///    flux (pressure-free uHat + face-normal pressure gradient, none on
///    zero-gradient-p sides), and a closed domain is rebalanced first
///    (adjustPhi style) so the system stays solvable;
///  - pressure is updated for *all* cells before the velocity reconstruction
///    (two passes), because the cumulative reconstruction reads grad(p) of the
///    full field.
///
/// `cumulativeVelocityCorrection` selects the reconstruction: true (PISO)
/// rebuilds `u = uHat - d grad(p)` from the full pressure, so repeated calls
/// with the same predictor accumulate -- it requires `relaxationP == 1`; false
/// (steady SIMPLE) keeps `u = uStar - d grad(p')`.
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
