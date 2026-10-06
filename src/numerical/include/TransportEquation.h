#pragma once

#include "BoundaryCondition.h"
#include "Convection.h"
#include "Diffusion.h"
#include "Field.h"
#include "Mesh.h"
#include "SparseMatrix.h"
#include "TimeScheme.h"
#include "Types.h"

using fvm::core::CartesianMesh;
using fvm::core::Index;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::Vector;
using fvm::core::VectorField;
using fvm::math::SparseMatrix;

namespace fvm::numerical
{

/// Assembled linear system A*phi = b (unfinalized matrix).
struct EquationSystem
{
    SparseMatrix A;
    Vector b;

    explicit EquationSystem(Index n)
        : A(n, n)
        , b(n)
    {
        b.setZero();
    }
};

/// Assemble the steady scalar transport equation
///   div(F phi) = div(gamma grad(phi)) + S(phi),  S = Sc + Sp*phi (per volume)
/// from a pre-computed (typically conservative) face mass-flux field, into an
/// unfinalized EquationSystem (call `A.finalize()` before solving).
///
/// Patankar source linearization: Sp must be <= 0 and goes to the diagonal
/// implicitly; a positive Sp is rejected with an exception.
EquationSystem assembleTransport(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    const ScalarField& gamma,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    const ScalarField* Sc = nullptr,
    const ScalarField* Sp = nullptr);

/// Assemble one implicit theta-scheme time step of the scalar transport
/// equation, reusing the steady assembly (A_sp, b_sp) and adding (V/dt) I:
///   (theta A_sp + (V/dt) I) phi^{n+1}
///       = (V/dt) phi^n - (1 - theta) A_sp phi^n + b_sp
/// theta = 1 is backward Euler, theta = 1/2 Crank-Nicolson.  The convecting
/// flux is frozen over the step, as are the source terms.
EquationSystem assembleTransientTransport(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    const ScalarField& gamma,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    const ScalarField& phiOld,
    Scalar dt,
    TimeScheme timeScheme,
    const ScalarField* Sc = nullptr,
    const ScalarField* Sp = nullptr);

} // namespace fvm::numerical
