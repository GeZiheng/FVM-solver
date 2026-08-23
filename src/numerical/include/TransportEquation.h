#pragma once

#include "BoundaryCondition.h"
#include "Convection.h"
#include "Diffusion.h"
#include "Field.h"
#include "Mesh.h"
#include "SparseMatrix.h"
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
 *   div(rho * u * phi) = div(gamma * grad(phi)) + S(phi),
 *   S(phi) = Sc + Sp * phi   (per unit volume),
 *
 * into an unfinalized EquationSystem (call system.A.finalize() before solving).
 *
 * Source-term linearization follows Patankar's rule: Sp must be non-positive
 * and is treated implicitly (added to the diagonal). A positive Sp would
 * destroy diagonal dominance, so it is rejected with an exception.
 *
 * @param mesh     Computational mesh.
 * @param velocity Convecting velocity field (cell-centered).
 * @param gamma    Diffusion coefficient field.
 * @param rho      Density (constant).
 * @param scheme   Convection scheme (Upwind or Central).
 * @param bc       Boundary conditions on the four domain sides.
 * @param Sc       Constant part of the volumetric source (optional).
 * @param Sp       Linear part of the volumetric source (optional, must be <= 0).
 */
EquationSystem assembleTransport(const CartesianMesh& mesh,
    const VectorField& velocity,
    const ScalarField& gamma,
    Scalar rho,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    const ScalarField* Sc = nullptr,
    const ScalarField* Sp = nullptr);

} // namespace fvm::numerical
