#include "Convection.h"
#include "GridOperators.h"
#include "Pressure.h"
#include "Mesh.h"
#include "MeshGeomCache.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace fvm::numerical
{
using fvm::core::MeshGeomCache;
using fvm::core::Point2;

CorrectorResult correctPressure(const Mesh& mesh,
    Scalar rho,
    const MomentumPrediction& pred,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const BoundaryField& bcP,
    fvm::math::LinearSolver& pSolver,
    Scalar relaxationP,
    VectorField& velocity,
    ScalarField& pressure,
    FaceFluxField& flux,
    const CorrectionOptions& corrections)
{
    if (relaxationP <= 0.0 || relaxationP > 1.0)
    {
        throw std::
            invalid_argument("correctPressure: relaxationP must be in (0, 1]");
    }

    const Index nCells = mesh.cellCount();
    const Index nFaces = mesh.faceCount();

    // Pure-Neumann p makes the equation singular (constants null space):
    // eliminate reference cell 0 and keep its value fixed (its continuity
    // equation is redundant). Any Dirichlet p patch makes the system definite
    // -> keep all cells.
    const bool pinReference = !hasDirichletPressure(bcP);
    const Index nP = pinReference ? nCells - 1 : nCells;
    const auto rowOf = [pinReference](Index cell) -> Index
    {
        return pinReference ? cell - 1 : cell;
    };
    const auto isReference = [pinReference](Index cell)
    {
        return pinReference && cell == 0;
    };

    const Vector& uHatU = pred.uHatU;
    const Vector& uHatV = pred.uHatV;
    const Vector& dU = pred.dU;
    const Vector& dV = pred.dV;

    if (corrections.nNonOrthogonalCorrectors < 0)
    {
        throw std::invalid_argument(
            "correctPressure: nNonOrthogonalCorrectors must be non-negative");
    }

    // ---- Pressure equation for the absolute pressure (OpenFOAM pEqn) ----
    // Cell continuity sum_f F_f = 0 with
    //   F_f = phiHbyA_f - C_f (p_N - p_P)   (interior)
    //   F_b = phiHbyA_b - C_b (p_b - p_P)   (Dirichlet p)
    //   F_b = phiHbyA_b                      (Neumann p)
    // -> A p = b,  b = -sum_f outward(phiHbyA_f) + sum_Dirichlet C_b p_b.
    // The pressure-free flux phiHbyA = rho (uHat . S)_f is written into the
    // persistent flux and corrected in place below (phi = phiHbyA -
    // pEqn.flux()), so the corrected flux stays conservative to the solver
    // accuracy.
    EquationSystem pSys(nP);
    Vector massImbalance(nCells); // net outward phiHbyA (adjusted below)
    Vector rhsExtra(nCells);      // known boundary / reference pressure terms
    massImbalance.setZero();
    rhsExtra.setZero();

    // Pressure coefficient of every face, derived once during the assembly
    // and reused by the flux correction.  Keeping the single derivation is
    // what makes the equation and its correction provably consistent; a face
    // with no pressure term (Neumann p) keeps 0.
    Vector faceC(nFaces);
    faceC.setZero();
    Vector faceCrossP(nFaces);
    faceCrossP.setZero();

    // Derived face geometry, computed once per corrector (never per cell).
    const MeshGeomCache geom(mesh);

    // Value of the eliminated reference cell (kept at its previous level).
    const Scalar pRefValue = pressure(0);

    // Explicit cross-pressure flux.  It does not overwrite the persistent
    // phiHbyA/massImbalance; the effective pressure-free flux enters only
    // through the assembled RHS and the matching final flux correction.
    auto crossPressureOut = [&](const ScalarField& pField)
    {
        Vector crossOut = Vector::Zero(nCells);
        if (!corrections.nonOrthogonalDiffusion)
        {
            return crossOut;
        }

        const std::vector<Point2> gradients
            = leastSquaresGradients(mesh, pField, bcP);
        for (Index f = 0; f < nFaces; ++f)
        {
            if (mesh.isBoundaryFace(f))
            {
                continue;
            }

            const Index o = mesh.faceOwner(f);
            const Index n = mesh.faceNeighbour(f);
            const Scalar g = geom.interpWeight(f);
            const Point2 d = geom.innerDelta(f);
            const Point2 gradF{ g * gradients[o].x
                    + (1.0 - g) * gradients[n].x,
                g * gradients[o].y + (1.0 - g) * gradients[n].y };
            const Scalar dd = d.x * d.x + d.y * d.y;
            const Scalar projection = (gradF.x * d.x + gradF.y * d.y) / dd;
            const Point2 residualGradient{ gradF.x - projection * d.x,
                gradF.y - projection * d.y };
            const Scalar dUf = g * dU(o) + (1.0 - g) * dU(n);
            const Scalar dVf = g * dV(o) + (1.0 - g) * dV(n);
            const Point2 area = mesh.faceAreaVector(f);
            const Scalar crossP = rho
                * (dUf * residualGradient.x * area.x
                    + dVf * residualGradient.y * area.y);

            crossOut(o) += crossP;
            crossOut(n) -= crossP;
            faceCrossP(f) = crossP;
        }
        return crossOut;
    };

    Vector crossOut = crossPressureOut(pressure);

    for (Index f = 0; f < nFaces; ++f)
    {
        const Index o = mesh.faceOwner(f);
        const Point2 S = mesh.faceAreaVector(f);
        const Index n = mesh.faceNeighbour(f);

        if (n != Mesh::kInvalidCell)
        {
            // Interior face, visited exactly once.  Rhie-Chow face velocity
            // with geometric interpolation weights:
            //   phiHbyA_f = rho (interp(uHat) . S)_f,
            //   C_f = rho (dU_f dx Sx + dV_f dy Sy) / (d.d),
            // with d = centroid_N - centroid_O.  On a Cartesian mesh this
            // reduces to the classic rho d_bar S / delta keyed on the face
            // component (u for x-faces, v for y-faces).
            const Scalar g = geom.interpWeight(f);
            const Point2 cO = mesh.cellCentroid(o);
            const Point2 cN = mesh.cellCentroid(n);
            const Scalar ddx = cN.x - cO.x;
            const Scalar ddy = cN.y - cO.y;
            const Scalar dd = ddx * ddx + ddy * ddy;

            const Scalar uF = g * uHatU(o) + (1.0 - g) * uHatU(n);
            const Scalar vF = g * uHatV(o) + (1.0 - g) * uHatV(n);
            const Scalar F = rho * (uF * S.x + vF * S.y);

            const Scalar dUf = g * dU(o) + (1.0 - g) * dU(n);
            const Scalar dVf = g * dV(o) + (1.0 - g) * dV(n);
            const Scalar C = rho * (dUf * ddx * S.x + dVf * ddy * S.y) / dd;

            // Store phiHbyA (owner -> neighbour) and the pressure coefficient.
            flux[f] = F;
            faceC(f) = C;

            massImbalance(o) += F;
            massImbalance(n) -= F;

            if (!isReference(o))
            {
                pSys.A.insert(rowOf(o), rowOf(o), C);
                if (!isReference(n))
                    pSys.A.insert(rowOf(o), rowOf(n), -C);
            }
            if (!isReference(n))
            {
                pSys.A.insert(rowOf(n), rowOf(n), C);
                if (!isReference(o))
                    pSys.A.insert(rowOf(n), rowOf(o), -C);
            }

            // Reference-cell elimination: the surviving row lost its -C p_0
            // entry, so move the known value to the right-hand side.
            if (isReference(o))
                rhsExtra(n) += C * pRefValue;
            if (isReference(n))
                rhsExtra(o) += C * pRefValue;
        }
        else
        {
            // Boundary face: the same pressure-free uHat flux as the
            // interior (Dirichlet velocity value where prescribed), with no
            // pressure term beyond the Dirichlet-p correction below.  u_b
            // must come from uHat (or the velocity BC), never from uStar.
            const Index patch = mesh.facePatch(f);
            const BoundaryCondition& pCond = bcP.get(patch);
            const BoundaryCondition& uCond = bcU.get(patch);
            const BoundaryCondition& vCond = bcV.get(patch);

            const Scalar ub
                = (uCond.type == BCType::Dirichlet) ? uCond.value : uHatU(o);
            const Scalar vb
                = (vCond.type == BCType::Dirichlet) ? vCond.value : uHatV(o);

            // S is owner-outward, so Fb is the owner-outward boundary flux.
            const Scalar Fb = rho * (ub * S.x + vb * S.y);
            massImbalance(o) += Fb;
            flux[f] = Fb;

            if (pCond.type == BCType::Dirichlet)
            {
                // Dirichlet-p: C_b scales (p_b - p_P) in the corrected flux.
                // Derived once here; the correction after the solve reuses
                // it.  Over-relaxed form: the implicit boundary gradient is
                // (p_b - p_P) * S / (S.delta), so each component's mobility
                // weighs its own S component; with dU = dV this reduces to
                // rAU * |S|^2 / (S.delta), the same form as the diffusion
                // boundary coefficient beta_f and OpenFOAM's nonOrthDeltaCoeffs
                // boundary limit.  The denominator floor guards degenerate
                // faces (S.delta <= 0.05 |S| |delta|).
                const Point2 cO = mesh.cellCentroid(o);
                const Point2 cF = mesh.faceCentroid(f);
                const Scalar del = cF.x - cO.x;
                const Scalar dey = cF.y - cO.y;
                const Scalar sDotDelta = S.x * del + S.y * dey;
                const Scalar sDeltaFloor = 0.05 * std::sqrt(
                    (S.x * S.x + S.y * S.y) * (del * del + dey * dey));
                const Scalar denominator
                    = std::max(sDotDelta, sDeltaFloor);
                const Scalar Cb = rho
                    * (dU(o) * S.x * S.x + dV(o) * S.y * S.y) / denominator;
                faceC(f) = Cb;

                if (!isReference(o))
                    pSys.A.insert(rowOf(o), rowOf(o), Cb);
                rhsExtra(o) += Cb * pCond.value;
            }
        }
    }

    // ---- Closed-domain compatibility (OpenFOAM adjustPhi) -------------
    // Pure-Neumann p: the p terms cannot change the net boundary outflow
    // (interior terms cancel, Neumann faces take no pressure term), so an
    // unbalanced phiHbyA makes the system inconsistent and silently violates
    // the eliminated cell's continuity.  Spread the residual uniformly per
    // unit area over the faces whose normal velocity is not fixed; the same
    // adjustment must reach the right-hand side, which is why massImbalance
    // is updated here too.
    if (pinReference)
    {
        Scalar net = 0.0;
        Scalar scale = 0.0;
        Scalar adjustableArea = 0.0;
        std::vector<Index> adjustableFaces;

        for (Index p = 0; p < mesh.patchCount(); ++p)
        {
            const bool uFixed = bcU.get(p).type == BCType::Dirichlet;
            const bool vFixed = bcV.get(p).type == BCType::Dirichlet;
            for (const Index f : mesh.patchFaces(p))
            {
                const Point2 S = mesh.faceAreaVector(f);
                net += flux[f];
                scale += std::abs(flux[f]);

                // A face is not adjustable when the velocity component
                // dominating its normal is prescribed.  On a Cartesian mesh
                // this is exactly the old rule: x-faces keyed on u,
                // y-faces on v.
                const bool normalFixed
                    = (std::abs(S.x) >= std::abs(S.y)) ? uFixed : vFixed;
                if (!normalFixed)
                {
                    adjustableFaces.push_back(f);
                    adjustableArea += std::hypot(S.x, S.y);
                }
            }
        }

        if (adjustableArea > 0.0 && std::abs(net) > 1e-14 * scale)
        {
            // Uniform outward flux per unit area over the adjustable faces.
            const Scalar deltaPerArea = -net / adjustableArea;
            for (const Index f : adjustableFaces)
            {
                const Point2 S = mesh.faceAreaVector(f);
                const Scalar add = deltaPerArea * std::hypot(S.x, S.y);
                flux[f] += add;
                massImbalance(mesh.faceOwner(f)) += add;
            }
        }
    }

    // ---- Assemble the right-hand side and solve for the absolute p ----
    // b = -sum_f outward(phiHbyA_f) + sum_Dirichlet C_b p_b + reference term.
    Vector pOld(nP);
    if (pinReference)
        pOld = pressure.data().tail(nCells - 1);
    else
        pOld = pressure.data();

    const auto assembleRhs = [&](const Vector& currentCrossOut)
    {
        Vector rhs(nP);
        for (Index P = 0; P < nCells; ++P)
        {
            if (!isReference(P))
            {
                rhs(rowOf(P)) = -(massImbalance(P) - currentCrossOut(P))
                    + rhsExtra(P);
            }
        }
        return rhs;
    };

    Vector b = assembleRhs(crossOut);

    pSys.A.finalize();

    // Initial residual at the previous pressure: the mass imbalance this
    // correction has to remove (identical to the p' form's right-hand side).
    CorrectorResult result;
    const Vector residual = b - pSys.A.native() * pOld;
    result.maxImbalance = residual.cwiseAbs().maxCoeff();
    if (pinReference)
    {
        // The eliminated row is redundant: its imbalance is minus the sum of
        // the solved rows (the adjusted boundary fluxes balance).
        result.maxImbalance
            = std::max(result.maxImbalance, std::abs(residual.sum()));
    }

    const int subSolves = corrections.nNonOrthogonalCorrectors + 1;
    Vector pSol;
    for (int subIteration = 0; subIteration < subSolves; ++subIteration)
    {
        if (subIteration > 0)
        {
            ScalarField pIter(mesh, "pIter");
            if (pinReference)
            {
                pIter.data()(0) = pRefValue;
                pIter.data().tail(nCells - 1) = pSol;
            }
            else
            {
                pIter.data() = pSol;
            }
            crossOut = crossPressureOut(pIter);
            b = assembleRhs(crossOut);
        }
        pSol = pSolver.solve(pSys.A, b);
    }

    // Absolute pressure used to rebuild the flux and the velocity; the
    // eliminated reference cell keeps its previous value.
    ScalarField pNew(mesh, "p");
    if (pinReference)
    {
        pNew.data()(0) = pRefValue;
        pNew.data().tail(nCells - 1) = pSol;
    }
    else
    {
        pNew.data() = pSol;
    }

    // ---- Corrections -------------------------------------------------
    // 1. Conservative flux correction (OpenFOAM pEqn.flux), one pass over
    // every face carrying a pressure coefficient:
    //   F_f = phiHbyA_f - C_f (p_N - p_P) on interior faces, and
    //   F_b = phiHbyA_b - C_b (p_b - p_P) on Dirichlet-p faces.  Each cell's
    //   net outflow then equals the linear solver's residual.  The
    //   coefficients are the ones stored by the assembly above, never
    //   recomputed here.
    for (Index f = 0; f < nFaces; ++f)
    {
        flux[f] -= faceCrossP(f);
    }

    for (Index f = 0; f < nFaces; ++f)
    {
        if (faceC(f) == 0.0)
            continue;

        const Index o = mesh.faceOwner(f);
        const Index n = mesh.faceNeighbour(f);
        if (n != Mesh::kInvalidCell)
        {
            flux[f] -= faceC(f) * (pNew(n) - pNew(o));
        }
        else if (bcP.get(mesh.facePatch(f)).type == BCType::Dirichlet)
        {
            flux[f] -= faceC(f) * (bcP.get(mesh.facePatch(f)).value - pNew(o));
        }
    }

    // 2a. Relaxed pressure update (OpenFOAM p.relax): p += alpha_p (p_solved
    // - p_old).  PISO uses alpha_p = 1, so the stored field is the solution.
    for (Index P = 0; P < nCells; ++P)
    {
        const Scalar dp = relaxationP * (pNew(P) - pressure(P));
        pressure(P) += dp;
        result.dpMax = std::max(result.dpMax, std::abs(dp));
    }

    // 2b. Cell-centered velocity reconstruction from the *unrelaxed* solved
    // pressure: u = uHat - d grad(p) (OpenFOAM U = HbyA - rAU*grad(p), but
    // with the unrelaxed p, unlike OpenFOAM's steady p.relax() ordering -
    // see docs/numerical.md).  For SIMPLE this equals u* - d grad(p')
    // because uHat = u* + d grad(p_old); for PISO it is the cumulative
    // reconstruction across correctors.
    for (Index P = 0; P < nCells; ++P)
    {
        const Scalar uNew
            = uHatU(P) - dU(P) * cellGradient(mesh, pNew, bcP, P, 0);
        const Scalar vNew
            = uHatV(P) - dV(P) * cellGradient(mesh, pNew, bcP, P, 1);
        result.duMax = std::max(result.duMax, std::abs(uNew - velocity.u()(P)));
        result.dvMax = std::max(result.dvMax, std::abs(vNew - velocity.v()(P)));
        velocity.u()(P) = uNew;
        velocity.v()(P) = vNew;
    }

    return result;
}

} // namespace fvm::numerical
