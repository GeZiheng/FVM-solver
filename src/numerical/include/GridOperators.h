#pragma once

#include "BoundaryCondition.h"
#include "Field.h"
#include "Mesh.h"
#include "Types.h"

#include <vector>

namespace fvm::numerical
{
using fvm::core::Index;
using fvm::core::Mesh;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::Point2;

/// Cell-center gradient of `phi`, component `dir`, by the Gauss theorem:
///   grad(phi)_P = (1 / vol_P) * sum_f phi_f * n_f,dir * S_f
/// (arithmetic face means in the interior; Dirichlet value or cell value on
/// boundaries; central differences on a uniform grid).
///
/// Shared operator: used by the momentum pressure-gradient source and by the
/// pressure-correction velocity reconstruction.  Put future operators
/// (divergence, ...) here too so other solvers can reuse them.
Scalar cellGradient(const Mesh& mesh,
    const ScalarField& phi,
    const BoundaryField& bc,
    Index P,
    int dir);

/// Cell-centred least-squares gradient of `phi`, exact for linear fields on
/// arbitrary meshes.  Per cell P, solve the 2x2 normal equations built from
/// one row per incident face: interior (x_N-x_P, phi_N-phi_P, w=1/|d|);
/// Dirichlet boundary (x_f-x_P, phi_b-phi_P, w=1/|x_f-x_P|); Neumann
/// boundary (n_hat, g_n, w=1) with n_hat the owner-outward unit normal and
/// g_n the prescribed outward normal derivative.  Falls back to the
/// Green-Gauss cellGradient for a cell whose normal matrix is singular.
std::vector<Point2> leastSquaresGradients(const Mesh& mesh,
    const ScalarField& phi,
    const BoundaryField& bc);

} // namespace fvm::numerical
