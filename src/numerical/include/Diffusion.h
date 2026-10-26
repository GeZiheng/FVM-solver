#pragma once

#include "BoundaryCondition.h"
#include "CorrectionOptions.h"
#include "Field.h"
#include "Mesh.h"
#include "SparseMatrix.h"
#include "Types.h"

namespace fvm::numerical
{
using fvm::core::Index;
using fvm::core::Mesh;
using fvm::core::ScalarField;
using fvm::core::Vector;
using fvm::math::SparseMatrix;

/// Assemble -div(gamma grad(phi)) into A*phi = b, accumulating into A and b
/// (callers zero them first; A must not be finalized).
///
/// Interior faces: D_f = gamma_f * alpha_f, where alpha_f follows
/// `corrections.nonOrthSplit`: (S.d)/(d.d) for MinimumCorrection (default;
/// equals S_f/d_PN on orthogonal meshes) or (S.S)/max(S.d, 0.05|S||d|) for
/// OverRelaxed (gamma_f = arithmetic mean).  Dirichlet (phi = phi_b):
/// D_b = gamma_P * beta_f with
/// beta_f = (S.S)/(S.delta); A(P,P) += D_b, b(P) += D_b * phi_b.
/// Neumann (d(phi)/dn = g, outward): b(P) += gamma_P * g * |S_f|.
///
/// When `explicitPhi` is supplied and `corrections.nonOrthogonalDiffusion` is
/// true, the interior cross-diffusion term gamma_f*(k_f . G_f) is added to
/// the RHS only, with k_f = S - alpha_f*d matching the selected split.  It
/// is a deferred correction: callers refresh it by re-assembling with the
/// latest available iterate.  Boundary faces receive no correction, so A
/// remains symmetric.
///
/// The matrix is symmetric positive semi-definite (definite with any Dirichlet).
void assembleDiffusion(const Mesh& mesh,
    const ScalarField& gamma,
    const BoundaryField& bc,
    SparseMatrix& A,
    Vector& b,
    const ScalarField* explicitPhi = nullptr,
    const CorrectionOptions& corrections = {});

} // namespace fvm::numerical
