#pragma once

#include "BoundaryCondition.h"
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
/// Interior faces: D_f = gamma_f * alpha_f with alpha_f = (S.d)/(d.d) from
/// MeshGeomCache (gamma_f = arithmetic mean; equals S_f/d_PN on orthogonal
/// meshes).  Dirichlet (phi = phi_b): D_b = gamma_P * beta_f with
/// beta_f = (S.S)/(S.delta); A(P,P) += D_b, b(P) += D_b * phi_b.
/// Neumann (d(phi)/dn = g, outward): b(P) += gamma_P * g * |S_f|.
///
/// The matrix is symmetric positive semi-definite (definite with any Dirichlet).
void assembleDiffusion(const Mesh& mesh,
    const ScalarField& gamma,
    const BoundaryField& bc,
    SparseMatrix& A,
    Vector& b);

} // namespace fvm::numerical
