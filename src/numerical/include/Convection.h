#pragma once

#include "BoundaryCondition.h"
#include "CorrectionOptions.h"
#include "Field.h"
#include "FluxField.h"
#include "Mesh.h"
#include "SparseMatrix.h"
#include "Types.h"

namespace fvm::numerical
{
using fvm::core::FaceFluxField;
using fvm::core::Index;
using fvm::core::Mesh;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::Vector;
using fvm::core::VectorField;
using fvm::math::SparseMatrix;

/// Convection interpolation scheme for face values.
enum class ConvectionScheme
{
    Upwind, ///< First-order upwind (bounded, diagonally dominant)
    Central ///< Second-order central differencing (may be unbounded)
};

/// Build the face mass flux F_f = rho * (u_f . n) * S_f from a cell-centered
/// velocity: linear interpolation in the interior and the adjacent owner
/// value on boundary faces.  With skew interpolation enabled, interior linear
/// interpolation is corrected by the owner/neighbour LSQ gradients at the
/// skew vector.  This overload has no velocity BCs and therefore uses zero-
/// Neumann gradient rows; prefer `computeMassFlux` when exact Dirichlet rows
/// are available.  Boundary faces are not skew-corrected.
FaceFluxField interpolateCellVelocityFlux(const Mesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    const CorrectionOptions& corrections = {});

/// Fill the face mass flux from a cell-centered velocity, honouring the velocity
/// BCs on boundary faces (OpenFOAM-style createPhi): Dirichlet value of the
/// patch's u and v components (where prescribed; otherwise the adjacent owner
/// value, i.e. zero normal gradient), projected onto the owner-outward face
/// area vector.  With skew interpolation enabled, interior linear
/// interpolation is corrected by the component LSQ gradients; Dirichlet BCs
/// supply the exact boundary rows.  Boundary face logic is unchanged.
/// Filled in place -- FaceFluxField references its mesh and is not assignable.
void computeMassFlux(const Mesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    FaceFluxField& flux,
    const CorrectionOptions& corrections = {});

/// Throw std::runtime_error if the net boundary outflow is not zero (relative
/// tolerance).  With pure-Neumann pressure the correction equation is only
/// compatible when the boundary fluxes sum to zero (OpenFOAM adjustPhi checks
/// the same invariant); call it after computeMassFlux to fail fast on
/// unbalanced velocity BCs.  Scaled by sum_b |F_b|.
void checkFluxCompatibility(const FaceFluxField& flux, Scalar relTol = 1e-10);

/// True if any patch of `bcP` prescribes a Dirichlet pressure.  A
/// Dirichlet-p patch makes the pressure equation definite.
bool hasDirichletPressure(const BoundaryField& bcP);

/// Shared entry step of solveSimple and solvePiso: build the persistent flux
/// from the velocity field and its BCs (OpenFOAM createPhi), then fail fast
/// when a closed domain (pure-Neumann p) has an unbalanced boundary flux
/// (OpenFOAM adjustPhi), which would make the correction equation
/// incompatible.
void initializeMassFlux(const Mesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const BoundaryField& bcP,
    FaceFluxField& flux,
    const CorrectionOptions& corrections = {});

/// Assemble div(F phi) into A*phi = b from a face mass-flux field (interior
/// values owner-to-neighbour; boundary values owner-outward).
///
/// Boundary faces, with F_b > 0 outflow / F_b < 0 inflow:
///   outflow            -> upwind face value,      A(P,P) += F_b
///   inflow, Dirichlet  -> phi_b (known),          b(P)   -= F_b * phi_b
///   inflow, Neumann    -> zero normal gradient,   A(P,P) += F_b
///
/// Accumulates into A and b (A must not be finalized).
///
/// With `explicitPhi` and Central skew interpolation enabled, interior face
/// values are also corrected by `G_f . s_f`; that explicit increment enters
/// only the RHS (`b(O) -= F_f * corr`, `b(N) += F_f * corr`).  Upwind is not
/// corrected because its face value is an owner-centred value, and boundary
/// faces receive no skew correction.
void assembleConvection(const Mesh& mesh,
    const FaceFluxField& flux,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    SparseMatrix& A,
    Vector& b,
    const ScalarField* explicitPhi = nullptr,
    const CorrectionOptions& corrections = {});

} // namespace fvm::numerical
