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

/**
 * @brief Assembled linear system A * phi = b (unfinalized matrix).
 */
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

/**
 * @brief Assemble the steady scalar transport equation
 *
 *   div(F * phi) = div(gamma * grad(phi)) + S(phi),
 *   S(phi) = Sc + Sp * phi   (per unit volume),
 *
 * from a precomputed (typically conservative) face mass-flux field,
 * e.g. the flux produced by solveSimple, or one built with
 * interpolateCellVelocityFlux for a given velocity field.
 *
 * into an unfinalized EquationSystem (call system.A.finalize() before solving).
 *
 * Source-term linearization follows Patankar's rule: Sp must be non-positive
 * and is treated implicitly (added to the diagonal). A positive Sp would
 * destroy diagonal dominance, so it is rejected with an exception.
 *
 * @param mesh   Computational mesh.
 * @param flux   Face mass-flux field (see Convection.h).
 * @param gamma  Diffusion coefficient field.
 * @param scheme Convection scheme (Upwind or Central).
 * @param bc     Boundary conditions on the four domain sides.
 * @param Sc     Constant part of the volumetric source (optional).
 * @param Sp     Linear part of the volumetric source (optional, must be <= 0).
 */
EquationSystem assembleTransport(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    const ScalarField& gamma,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    const ScalarField* Sc = nullptr,
    const ScalarField* Sp = nullptr);

/**
 * @brief Assemble the unsteady scalar transport equation over one time
 * step with the implicit theta scheme.
 *
 *     (V/dt)(phi^{n+1} - phi^n) + theta L(phi^{n+1})
 *         + (1 - theta) L(phi^n) = b,
 *
 * where L(phi) = div(F phi) - div(gamma grad phi) is the spatial
 * (convection + diffusion) operator and b collects the source and
 * Dirichlet boundary terms. The returned system solves for phi^{n+1}:
 *
 *     (theta A_sp + (V/dt) I) phi^{n+1}
 *         = (V/dt) phi^n - (1 - theta) A_sp phi^n + b_sp,
 *
 * with A_sp / b_sp the steady assembly of assembleTransport. For
 * theta = 1 this is backward Euler; theta = 1/2 gives Crank-Nicolson.
 *
 * The convecting flux is frozen over the step (standard for scalar
 * transport with a prescribed velocity). Source terms (Sc, Sp) are
 * assumed constant over the step.
 *
 * @param mesh      Computational mesh.
 * @param flux      Face mass-flux field (frozen over the step).
 * @param gamma     Diffusion coefficient field.
 * @param scheme    Convection scheme (Upwind or Central).
 * @param bc        Boundary conditions on the four domain sides.
 * @param phiOld    Field at the old time level phi^n.
 * @param dt        Time-step size (must be > 0).
 * @param timeScheme Time-integration scheme (Euler or CrankNicolson).
 * @param Sc        Constant part of the volumetric source (optional).
 * @param Sp        Linear part of the volumetric source (optional, <= 0).
 */
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
