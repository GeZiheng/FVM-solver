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

/// Cell-center gradient of `phi`, component `dir`, by the Gauss theorem:
///   grad(phi)_P = (1 / vol_P) * sum_f phi_f * n_f,dir * S_f
/// (arithmetic face means in the interior; Dirichlet value or cell value on
/// boundaries; central differences on a uniform grid).
///
/// Shared operator: used by the momentum pressure-gradient source and by the
/// pressure-correction velocity reconstruction.  Put future operators
/// (divergence, ...) here too so other solvers can reuse them.
Scalar cellGradient(const CartesianMesh& mesh,
    const ScalarField& phi,
    const BoundaryField& bc,
    Index P,
    int dir);

} // namespace fvm::numerical
