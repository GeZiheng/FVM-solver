#pragma once

#include "BoundaryCondition.h"
#include "Field.h"
#include "Mesh.h"
#include "SparseMatrix.h"
#include "Types.h"

using fvm::core::CartesianMesh;
using fvm::core::ScalarField;
using fvm::core::Vector;
using fvm::math::SparseMatrix;

namespace fvm::numerical
{

/// Assemble -div(gamma grad(phi)) into A*phi = b, accumulating into A and b
/// (callers zero them first; A must not be finalized).
///
/// Interior faces: D_f = gamma_f * S_f / d_PN (gamma_f = arithmetic mean).
/// Dirichlet (phi = phi_b): A(P,P) += D_b, b(P) += D_b * phi_b.
/// Neumann (d(phi)/dn = g, outward): b(P) += gamma_P * g * S_f.
///
/// The matrix is symmetric positive semi-definite (definite with any Dirichlet).
void assembleDiffusion(const CartesianMesh& mesh,
    const ScalarField& gamma,
    const BoundaryField& bc,
    SparseMatrix& A,
    Vector& b);

} // namespace fvm::numerical
