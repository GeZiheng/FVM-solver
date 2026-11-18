#pragma once

#include "BoundaryCondition.h"
#include "CorrectionOptions.h"
#include "Field.h"
#include "FluxField.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "Momentum.h"
#include "SparseMatrix.h"
#include "Types.h"

namespace fvm::numerical
{
using fvm::core::Mesh;
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

/// Assembled pressure system exported without solving (Phase 6.5c M2).
/// `A` is the *uneliminated* N x N operator of the absolute-pressure
/// equation: in the pure-Neumann case its constants null space is left in
/// instead of reference-cell eliminating, because a mean-centred POD test
/// space is orthogonal to the constants and removes the null space through
/// the projection on its own.  `b` is the first-pass right-hand side (the
/// explicit cross-pressure term is evaluated on the incoming pressure, and
/// the closed-domain adjustPhi rebalance has run), so with
/// nNonOrthogonalCorrectors = 0 it is the same equation the solving path
/// assembles - before its reference-cell elimination, whose solution also
/// satisfies this system up to the redundant-row residual.  The exporting
/// call's maxImbalance is the full-space residual max|b - A p_old|;
/// du/dv/dpMax stay 0 and `pressure` is left untouched.
struct PressureSystemExport
{
    fvm::math::SparseMatrix A{0, 0}; ///< Finalized operator, nCells x nCells.
    Vector b;                        ///< First-pass RHS, nCells.
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
///
/// With `exportSystem` set (Phase 6.5c M2) the corrector assembles the
/// pressure equation, writes the uneliminated system into it and returns
/// without solving or correcting anything (see PressureSystemExport); the
/// null (default) keeps the solving behaviour bit-identical.
CorrectorResult correctPressure(const Mesh& mesh,
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
    const CorrectionOptions& corrections = {},
    PressureSystemExport* exportSystem = nullptr);

} // namespace fvm::numerical
