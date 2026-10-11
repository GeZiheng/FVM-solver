#pragma once

#include "BoundaryCondition.h"
#include "Field.h"
#include "FluxField.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "Momentum.h"
#include "Types.h"

namespace fvm::numerical
{
using fvm::core::CartesianMesh;
using fvm::core::FaceFluxField;
using fvm::core::Index;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::Vector;
using fvm::core::VectorField;

/// Outcome of one pressure-correction step.
struct CorrectorResult
{
    Scalar maxImbalance = 0.0; ///< Max |mass imbalance| this correction had to
                               ///< remove: initial residual of the absolute-p
                               ///< equation at the previous pressure (after the
                               ///< closed-domain flux adjustment).
    Scalar duMax = 0.0;        ///< Max |u_new - u_old| after correction.
    Scalar dvMax = 0.0;        ///< Max |v_new - v_old| after correction.
    Scalar dpMax = 0.0;        ///< Max |relaxationP * (p_solved - p_old)|.
};

/// Run one pressure-correction step (OpenFOAM pEqn): write the pressure-free
/// Rhie-Chow flux phiHbyA = rho S uHat_f into the persistent flux field,
/// assemble and solve the absolute-pressure equation
///   sum_f F_f = 0 with  F_f = phiHbyA_f - C_f (p_N - p_P),
/// then rebuild the face flux (`phi = phiHbyA - pEqn.flux()`, conservative to
/// the pressure solver's accuracy), the cell velocity `u = uHat - d grad(p)`
/// and the stored pressure (`pressure += relaxationP * (p_solved - p_old)`).
///
/// Right-hand side: b = -sum_f outward(phiHbyA_f) + sum_Dirichlet C_b p_b.
/// Solving for the absolute pressure instead of p' is algebraically identical
/// (p' = p_solved - p_old solves the same system) but avoids forming the
/// cancelling difference between the old pressure gradient and the matrix.
///
/// Invariants (see docs/numerical.md):
///  - pure-Neumann p is singular: reference cell 0 is eliminated and kept at its
///    previous value (its equation is redundant), and the dropped column is
///    moved to the right-hand side of its neighbours;
///  - the boundary flux is as pressure-free as the interior one (uHat or the
///    velocity BC), and a closed domain is rebalanced first (adjustPhi style)
///    so the system stays solvable;
///  - the flux and the velocity use the *unrelaxed* solved pressure; only the
///    stored field is relaxed, so steady SIMPLE applies the full p' correction
///    to the velocity and relaxationP of it to the pressure.
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

} // namespace fvm::numerical
