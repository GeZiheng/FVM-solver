#pragma once

#include "BoundaryCondition.h"
#include "Convection.h"
#include "Field.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "TimeScheme.h"
#include "TransportEquation.h"
#include "Types.h"

using fvm::core::CartesianMesh;
using fvm::core::FaceFluxField;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::Vector;
using fvm::core::VectorField;

namespace fvm::numerical
{

/// Assembled momentum equation for one velocity component.  `diag` holds the
/// (relaxed) diagonal a_P and `rhsNoPressure` the right-hand side without the
/// pressure-gradient source; together they give the Rhie-Chow data H/a_P.
struct MomentumAssembly
{
    EquationSystem system;
    Vector diag;
    Vector rhsNoPressure;
};

/// Assemble the momentum equation for one velocity component from a
/// pre-computed face mass-flux field: diffusion + convection + pressure-gradient
/// source + under-relaxation (or the transient ddt term).
/// @note `time.active()` requires `relaxation == 1`; the two are alternatives.
MomentumAssembly assembleMomentum(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    const VectorField& velocity,
    Scalar mu,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    int component,
    const ScalarField& pressure,
    const BoundaryField& bcP,
    Scalar relaxation,
    const TimeTerm& time = {});

/// Momentum predictor result: the predicted velocities plus the Rhie-Chow data
/// (`uHat = H/a_P`, `d = V/a_P`) consumed by the pressure corrector.
struct MomentumPrediction
{
    MomentumAssembly momU;
    MomentumAssembly momV;
    Vector uStar; ///< Predicted u (solution of the relaxed momentum equation).
    Vector vStar; ///< Predicted v.
    Vector uHatU; ///< u without the pressure-gradient contribution.
    Vector uHatV; ///< v without the pressure-gradient contribution.
    Vector dU;    ///< V / a_P of the u equation.
    Vector dV;    ///< V / a_P of the v equation.
};

/// Assemble and solve the u/v momentum equations and compute the Rhie-Chow data
/// (OpenFOAM UEqn.H).
MomentumPrediction predictMomentum(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    const VectorField& velocity,
    Scalar mu,
    ConvectionScheme scheme,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const ScalarField& pressure,
    const BoundaryField& bcP,
    Scalar relaxationU,
    const TimeTerm& time,
    fvm::math::LinearSolver& momSolver);

/// Re-evaluate the Rhie-Chow data `uHat = H(u)/a_P` from the current velocity
/// (OpenFOAM `HbyA = rAU*UEqn.H()`).  OpenFOAM re-evaluates it at *every*
/// corrector because `fvMatrix::H()` reads the matrix's current `psi_`.
///
/// `predictMomentum` evaluates uHat once, at the predicted velocity; a PISO
/// driver must call this before each *additional* corrector.  With a frozen
/// uHat the loop reaches its own fixed point after one sweep, i.e.
/// `nCorrectors = 1`, which diverges for these problems (OpenFOAM too).
///
/// `pred.momU`/`momV` must be the finalized assemblies from `predictMomentum`;
/// `uHatU`/`uHatV` are overwritten from `velocity`.
void refreshUHat(MomentumPrediction& pred, const VectorField& velocity);

} // namespace fvm::numerical
