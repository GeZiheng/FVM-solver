#pragma once

#include "BoundaryCondition.h"
#include "Convection.h"
#include "CorrectionOptions.h"
#include "Diffusion.h"
#include "Field.h"
#include "Mesh.h"
#include "SparseMatrix.h"
#include "TimeScheme.h"
#include "Types.h"

namespace fvm::numerical
{
using fvm::core::Index;
using fvm::core::Mesh;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::Vector;
using fvm::core::VectorField;
using fvm::math::SparseMatrix;

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
///
/// `explicitPhi` is the field frozen for deferred non-orthogonal diffusion;
/// callers pass the latest iteration for Newton/Picard refreshes.
EquationSystem assembleTransport(const Mesh& mesh,
    const FaceFluxField& flux,
    const ScalarField& gamma,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    const ScalarField* Sc = nullptr,
    const ScalarField* Sp = nullptr,
    const ScalarField* explicitPhi = nullptr,
    const CorrectionOptions& corrections = {});

/// Assemble one implicit theta-scheme time step of the scalar transport
/// equation, reusing the steady assembly (A_sp, b_sp) and adding (V/dt) I:
///   (theta A_sp + (V/dt) I) phi^{n+1}
///       = (V/dt) phi^n - (1 - theta) A_sp phi^n + b_sp
/// theta = 1 is backward Euler, theta = 1/2 Crank-Nicolson.  The convecting
/// flux is frozen over the step, as are the source terms.
/// `explicitPhi` is likewise frozen: use the last iteration or previous time
/// layer as appropriate to the caller.
EquationSystem assembleTransientTransport(const Mesh& mesh,
    const FaceFluxField& flux,
    const ScalarField& gamma,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    const ScalarField& phiOld,
    Scalar dt,
    TimeScheme timeScheme,
    const ScalarField* Sc = nullptr,
    const ScalarField* Sp = nullptr,
    const ScalarField* explicitPhi = nullptr,
    const CorrectionOptions& corrections = {});

} // namespace fvm::numerical
