#pragma once

#include "BoundaryCondition.h"
#include "Field.h"
#include "FluxField.h"
#include "Mesh.h"
#include "SparseMatrix.h"
#include "Types.h"

using fvm::core::CartesianMesh;
using fvm::core::FaceFluxField;
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
 * @brief Build the face mass-flux field from a cell-centered velocity,
 * using the cell-center velocity on boundary faces.
 *
 * Interior faces: F_f = rho * (u_f . n) * S_f with u_f the arithmetic
 * mean of the adjacent cell velocities. Boundary faces: F_b uses the
 * adjacent cell velocity (suitable for transport problems with a given
 * velocity field, where no velocity BCs are at hand).
 */
FaceFluxField interpolateCellVelocityFlux(const CartesianMesh& mesh,
    const VectorField& velocity,
    Scalar rho);

/**
 * @brief Fill the face mass-flux field from a cell-centered velocity,
 * honoring the velocity boundary conditions on boundary faces
 * (OpenFOAM-style createPhi).
 *
 * Interior faces: F_f = rho * (u_f . n) * S_f with u_f the arithmetic
 * mean of the adjacent cell velocities. Boundary faces use the
 * Dirichlet value of the matching velocity component (u on east/west,
 * v on north/south) when prescribed, otherwise the adjacent cell
 * velocity (zero normal gradient).
 *
 * The field is filled in place (FaceFluxField is not assignable because
 * it references its mesh).
 */
void computeMassFlux(const CartesianMesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    FaceFluxField& flux);

/**
 * @brief Assemble the convection operator div(F * phi) into A * phi = b
 * from a precomputed face mass-flux field.
 *
 * The flux field provides F_f on every face (positive along the
 * coordinate direction); boundary-face fluxes are taken from the field
 * as outward fluxes.
 *
 * Boundary faces (F_b > 0 outflow / F_b < 0 inflow):
 *   - Outflow: face value extrapolated from the cell (upwind),
 *       A(P,P) += F_b.
 *   - Inflow with Dirichlet BC: face value = phi_b (known),
 *       b(P) -= F_b * phi_b.
 *   - Inflow with Neumann BC: zero normal gradient is assumed,
 *       face value = phi_P, A(P,P) += F_b.
 *
 * @note A and b are accumulated into (not reset); A must not be finalized.
 */
void assembleConvection(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    SparseMatrix& A,
    Vector& b);

} // namespace fvm::numerical
