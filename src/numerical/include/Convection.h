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

/// Convection interpolation scheme for face values.
enum class ConvectionScheme
{
    Upwind, ///< First-order upwind (bounded, diagonally dominant)
    Central ///< Second-order central differencing (may be unbounded)
};

/// Build the face mass flux F_f = rho * (u_f . n) * S_f from a cell-centered
/// velocity: arithmetic mean in the interior, the adjacent cell value on
/// boundary faces (for transport with a prescribed velocity, no BCs at hand).
FaceFluxField interpolateCellVelocityFlux(const CartesianMesh& mesh,
    const VectorField& velocity,
    Scalar rho);

/// Fill the face mass flux from a cell-centered velocity, honouring the velocity
/// BCs on boundary faces (OpenFOAM-style createPhi): Dirichlet value of the
/// matching component (u on east/west, v on north/south), else the adjacent cell
/// velocity (zero normal gradient).  Filled in place -- FaceFluxField references
/// its mesh and is not assignable.
void computeMassFlux(const CartesianMesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    FaceFluxField& flux);

/// Throw std::runtime_error if the net boundary outflow is not zero (relative
/// tolerance).  With pure-Neumann pressure the correction equation is only
/// compatible when the boundary fluxes sum to zero (OpenFOAM adjustPhi checks
/// the same invariant); call it after computeMassFlux to fail fast on
/// unbalanced velocity BCs.  Scaled by sum_b |F_b|.
void checkFluxCompatibility(const FaceFluxField& flux, Scalar relTol = 1e-10);

/// Assemble div(F phi) into A*phi = b from a face mass-flux field (positive
/// along the coordinate direction; boundary entries are outward fluxes).
///
/// Boundary faces, with F_b > 0 outflow / F_b < 0 inflow:
///   outflow            -> upwind face value,      A(P,P) += F_b
///   inflow, Dirichlet  -> phi_b (known),          b(P)   -= F_b * phi_b
///   inflow, Neumann    -> zero normal gradient,   A(P,P) += F_b
///
/// Accumulates into A and b (A must not be finalized).
void assembleConvection(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    SparseMatrix& A,
    Vector& b);

} // namespace fvm::numerical
