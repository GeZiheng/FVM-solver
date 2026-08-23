#pragma once

#include "BoundaryCondition.h"
#include "Field.h"
#include "Mesh.h"
#include "SparseMatrix.h"
#include "Types.h"

using fvm::core::CartesianMesh;
using fvm::core::Scalar;
using fvm::core::Vector;
using fvm::core::VectorField;
using fvm::math::SparseMatrix;

namespace fvm::numerical
{

/**
 * @brief Convection interpolation scheme for face values.
 */
enum class ConvectionScheme
{
    Upwind, ///< First-order upwind (bounded, diagonally dominant)
    Central ///< Second-order central differencing (may be unbounded)
};

/**
 * @brief Assemble the convection operator div(rho * u * phi) into A * phi = b.
 *
 * Interior face mass flux: F_f = rho * (u_f . n) * S_f, with u_f the
 * arithmetic mean of the adjacent cell velocities.
 *
 * Boundary faces (F_b = rho * (u_P . n) * S_f):
 *   - Outflow (F_b > 0): face value extrapolated from the cell (upwind),
 *       A(P,P) += F_b.
 *   - Inflow (F_b < 0) with Dirichlet BC: face value = phi_b (known),
 *       b(P) -= F_b * phi_b.
 *   - Inflow (F_b < 0) with Neumann BC: zero normal gradient is assumed,
 *       face value = phi_P, A(P,P) += F_b.
 *
 * @note A and b are accumulated into (not reset); A must not be finalized.
 */
void assembleConvection(const CartesianMesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    SparseMatrix& A,
    Vector& b);

} // namespace fvm::numerical
