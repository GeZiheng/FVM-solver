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

/**
 * @brief Assemble the diffusion operator -div(gamma * grad(phi)) into A * phi =
 * b.
 *
 * Interior faces use central differencing:
 *   D_f = gamma_f * S_f / d_PN   (gamma_f: arithmetic mean of cell values)
 *
 * Boundary faces:
 *   - Dirichlet (phi = phi_b): D_b = gamma_P * S_f / d_Pb
 *       A(P,P) += D_b,  b(P) += D_b * phi_b
 *   - Neumann (d(phi)/dn = g, outward normal): flux is known
 *       b(P) += gamma_P * g * S_f
 *
 * With Dirichlet/Neumann BCs only, the resulting matrix is symmetric
 * positive semi-definite (SPD if any Dirichlet BC is present).
 *
 * @note A and b are accumulated into (not reset); callers must zero them
 *       beforehand if assembling a fresh system. A must not be finalized.
 */
void assembleDiffusion(const CartesianMesh& mesh,
    const ScalarField& gamma,
    const BoundaryField& bc,
    SparseMatrix& A,
    Vector& b);

} // namespace fvm::numerical
