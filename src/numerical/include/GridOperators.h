#pragma once

#include "BoundaryCondition.h"
#include "Field.h"
#include "Mesh.h"
#include "Types.h"

using fvm::core::CartesianMesh;
using fvm::core::Index;
using fvm::core::Scalar;
using fvm::core::ScalarField;

namespace fvm::numerical
{

/**
 * @brief Cell-center gradient of a scalar field, component `dir`, via the
 * Gauss divergence theorem:
 *
 *   grad(phi)_P = (1 / vol_P) * sum_f phi_f * n_f,dir * S_f.
 *
 * Interior face values are arithmetic means; boundary faces take the
 * Dirichlet value or the cell value (zero normal gradient) for Neumann.
 * On a uniform grid this reduces to central differences in the interior.
 *
 * This is a shared discretization operator: it is used by the momentum
 * pressure-gradient source and by the pressure-correction velocity
 * reconstruction. Future operators (divergence, ...) should live here too
 * so that other solvers can reuse them.
 *
 * @param mesh Computational mesh.
 * @param phi  Cell-centered scalar field.
 * @param bc   Boundary conditions (used for boundary face values).
 * @param P    Cell index.
 * @param dir  Gradient direction: 0 = x, 1 = y.
 */
Scalar cellGradient(const CartesianMesh& mesh,
    const ScalarField& phi,
    const BoundaryField& bc,
    Index P,
    int dir);

} // namespace fvm::numerical
