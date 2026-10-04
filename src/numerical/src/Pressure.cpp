#include "Pressure.h"
#include "GridOperators.h"

#include <cmath>
#include <stdexcept>

namespace fvm::numerical
{

namespace
{

/**
 * @brief Outward-normal boundary velocity for the momentum component
 * associated with the given face (u on east/west, v on north/south).
 * Dirichlet BCs prescribe the boundary value; otherwise the cell value
 * (zero normal gradient) is used.
 */
Scalar boundaryNormalVelocity(int face,
    const BoundaryField& bcComp,
    const Vector& starComp,
    Index P)
{
    const BoundaryCondition& cond = bcComp.get(face);
    const Scalar val
        = (cond.type == BCType::Dirichlet) ? cond.value : starComp(P);
    return (face == BoundaryField::West || face == BoundaryField::South) ? -val
                                                                         : val;
}

} // namespace

CorrectorResult correctPressure(const CartesianMesh& mesh,
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
    bool cumulativeVelocityCorrection)
{
    if (relaxationP <= 0.0 || relaxationP > 1.0)
    {
        throw std::
            invalid_argument("correctPressure: relaxationP must be in (0, 1]");
    }
    if (cumulativeVelocityCorrection && relaxationP != 1.0)
    {
        throw std::invalid_argument(
            "correctPressure: cumulativeVelocityCorrection requires "
            "relaxationP == 1");
    }

    const Index nCells = mesh.cellCount();

    // With pure Neumann pressure BCs the correction equation is singular
    // (null space: constants). Eliminate the reference cell 0 (p'_0 = 0);
    // its continuity equation is redundant because the mass imbalances
    // sum to zero. With at least one Dirichlet side the system is already
    // definite and all cells are kept.
    bool hasDirichletP = false;
    for (int side = 0; side < 4; ++side)
    {
        if (bcP.get(side).type == BCType::Dirichlet)
            hasDirichletP = true;
    }
    const bool pinReference = !hasDirichletP;
    const Index nP = pinReference ? nCells - 1 : nCells;
    const auto rowOf = [pinReference](Index cell) -> Index
    {
        return pinReference ? cell - 1 : cell;
    };
    const auto isReference = [pinReference](Index cell)
    {
        return pinReference && cell == 0;
    };

    // Pressure-correction BCs: p' = 0 wherever p is fixed (Dirichlet),
    // zero gradient elsewhere.
    BoundaryField bcPrime = bcP;
    for (int side = 0; side < 4; ++side)
    {
        if (bcPrime.get(side).type == BCType::Dirichlet)
            bcPrime.set(side, BCType::Dirichlet, 0.0);
    }

    const Vector& uStar = pred.uStar;
    const Vector& vStar = pred.vStar;
    const Vector& uHatU = pred.uHatU;
    const Vector& uHatV = pred.uHatV;
    const Vector& dU = pred.dU;
    const Vector& dV = pred.dV;

    // ---- Pressure-correction equation -------------------------------
    // Continuity per cell: sum_f F_f = 0 with
    //   F_f = F*_f - rho d_f S_f (p'_N - p'_P) / delta  (interior),
    //   F_b = F*_b + rho d_P S_f p'_P / dist            (Dirichlet p),
    //   F_b = F*_b                                      (Neumann p).
    // Collecting the p' terms on the left gives the diffusion-like
    // system A p' = -massImbalance, where A has coefficient rho * d
    // and massImbalance(P) is the predicted net outflow of cell P.
    // The predicted flux F* is written into the persistent flux
    // field; the correction step below applies the p' correction to
    // it in place, which makes the stored flux conservative up to the
    // solver accuracy.
    EquationSystem pSys(nP);
    Vector massImbalance(nCells);
    massImbalance.setZero();

    for (Index P = 0; P < nCells; ++P)
    {
        const auto [iP, jP] = mesh.cellIJ(P);
        for (int face = 0; face < 4; ++face)
        {
            const Index N = mesh.neighbor(P, face);
            const Scalar Sf = mesh.faceArea(face);
            const bool xFace
                = (face == BoundaryField::East || face == BoundaryField::West);
            const Vector& dC = xFace ? dU : dV;
            const Vector& uHat = xFace ? uHatU : uHatV;

            if (N != nCells)
            {
                if (face != BoundaryField::East && face != BoundaryField::North)
                    continue; // interior faces once

                const Scalar delta = mesh.cellToCellDistance(P, face);
                const Scalar dF = 0.5 * (dC(P) + dC(N));
                const Scalar uHatF = 0.5 * (uHat(P) + uHat(N));
                const Scalar F
                    = rho * Sf
                      * (uHatF - dF * (pressure(N) - pressure(P)) / delta);
                const Scalar C = rho * dF * Sf / delta;

                // Store the predicted flux (positive P -> N, the
                // stored convention).
                if (face == BoundaryField::East)
                    flux.x(iP + 1, jP) = F;
                else
                    flux.y(iP, jP + 1) = F;

                massImbalance(P) += F;
                massImbalance(N) -= F;

                if (!isReference(P))
                {
                    pSys.A.insert(rowOf(P), rowOf(P), C);
                    if (!isReference(N))
                        pSys.A.insert(rowOf(P), rowOf(N), -C);
                }
                if (!isReference(N))
                {
                    pSys.A.insert(rowOf(N), rowOf(N), C);
                    if (!isReference(P))
                        pSys.A.insert(rowOf(N), rowOf(P), -C);
                }
            }
            else
            {
                // Boundary face. The predicted flux must be the Rhie-Chow
                // flux at the OLD pressure, exactly like the interior
                // faces: the pressure-free velocity (uHat) plus the
                // face-normal pressure gradient. On a zero-gradient p
                // boundary the normal gradient is zero, so no pressure
                // term is added there. Using the cell-centred gradient
                // (via uStar) instead double-counts the fixed boundary
                // pressure once the correction adds its increment, which
                // makes the corrector iteration diverge.
                const BoundaryCondition& pCond = bcP.get(face);
                const Scalar ub = boundaryNormalVelocity(face,
                    xFace ? bcU : bcV,
                    xFace ? uHatU : uHatV,
                    P);
                Scalar Fb = rho * ub * Sf;
                if (pCond.type == BCType::Dirichlet)
                {
                    Fb -= rho * dC(P) * Sf
                          * (pCond.value - pressure(P))
                          / mesh.cellToFaceDistance(face);
                }
                massImbalance(P) += Fb;

                // Store the outward flux (west/south faces are
                // stored positive along +x/+y, hence the sign).
                switch (face)
                {
                    case BoundaryField::East:
                        flux.x(iP + 1, jP) = Fb;
                        break;
                    case BoundaryField::West:
                        flux.x(iP, jP) = -Fb;
                        break;
                    case BoundaryField::North:
                        flux.y(iP, jP + 1) = Fb;
                        break;
                    default: // South
                        flux.y(iP, jP) = -Fb;
                        break;
                }

                if (pCond.type == BCType::Dirichlet && !isReference(P))
                {
                    // p' = 0 at the boundary: correction Cb * p'_P.
                    const Scalar Cb
                        = rho * dC(P) * Sf / mesh.cellToFaceDistance(face);
                    pSys.A.insert(rowOf(P), rowOf(P), Cb);
                }
            }
        }
    }

    CorrectorResult result;
    result.maxImbalance = massImbalance.cwiseAbs().maxCoeff();

    // ---- Closed-domain compatibility (OpenFOAM adjustPhi) -------------
    // With a pure-Neumann pressure the p' correction cannot change the net
    // boundary outflow: interior corrections cancel in pairs and Neumann
    // boundaries take no p' term. An unbalanced predicted boundary flux
    // therefore makes the correction equation inconsistent, and the
    // reference-cell elimination silently violates the eliminated cell's
    // continuity (its equation is the redundant one that gets dropped).
    // Redistribute the residual over the boundary faces whose normal
    // velocity is not fixed, so the p' system stays solvable and the
    // stored flux stays conservative.
    if (pinReference)
    {
        const Index nx = mesh.nx();
        const Index ny = mesh.ny();
        const Scalar Sx = mesh.faceArea(BoundaryField::East);
        const Scalar Sy = mesh.faceArea(BoundaryField::North);

        Scalar net = 0.0;
        Scalar scale = 0.0;
        Scalar adjustableArea = 0.0;
        for (Index j = 0; j < ny; ++j)
        {
            const Scalar westOut = -flux.x(0, j);
            const Scalar eastOut = flux.x(nx, j);
            net += westOut + eastOut;
            scale += std::abs(westOut) + std::abs(eastOut);
            if (bcU.get(BoundaryField::West).type != BCType::Dirichlet)
                adjustableArea += Sx;
            if (bcU.get(BoundaryField::East).type != BCType::Dirichlet)
                adjustableArea += Sx;
        }
        for (Index i = 0; i < nx; ++i)
        {
            const Scalar southOut = -flux.y(i, 0);
            const Scalar northOut = flux.y(i, ny);
            net += southOut + northOut;
            scale += std::abs(southOut) + std::abs(northOut);
            if (bcV.get(BoundaryField::South).type != BCType::Dirichlet)
                adjustableArea += Sy;
            if (bcV.get(BoundaryField::North).type != BCType::Dirichlet)
                adjustableArea += Sy;
        }

        if (adjustableArea > 0.0 && std::abs(net) > 1e-14 * scale)
        {
            // Uniform outward flux per unit area over the adjustable faces.
            const Scalar delta = -net / adjustableArea;

            for (Index j = 0; j < ny; ++j)
            {
                if (bcU.get(BoundaryField::West).type != BCType::Dirichlet)
                {
                    flux.x(0, j) -= delta * Sx; // outward = -x(0, j)
                    massImbalance(mesh.cellIndex(0, j)) += delta * Sx;
                }
                if (bcU.get(BoundaryField::East).type != BCType::Dirichlet)
                {
                    flux.x(nx, j) += delta * Sx; // outward = +x(nx, j)
                    massImbalance(mesh.cellIndex(nx - 1, j)) += delta * Sx;
                }
            }
            for (Index i = 0; i < nx; ++i)
            {
                if (bcV.get(BoundaryField::South).type != BCType::Dirichlet)
                {
                    flux.y(i, 0) -= delta * Sy; // outward = -y(i, 0)
                    massImbalance(mesh.cellIndex(i, 0)) += delta * Sy;
                }
                if (bcV.get(BoundaryField::North).type != BCType::Dirichlet)
                {
                    flux.y(i, ny) += delta * Sy; // outward = +y(i, ny)
                    massImbalance(mesh.cellIndex(i, ny - 1)) += delta * Sy;
                }
            }
            result.maxImbalance = massImbalance.cwiseAbs().maxCoeff();
        }
    }

    for (Index P = 0; P < nCells; ++P)
    {
        if (!isReference(P))
            pSys.b(rowOf(P)) = -massImbalance(P);
    }
    pSys.A.finalize();
    const Vector pCorrSol = pSolver.solve(pSys.A, pSys.b);

    ScalarField pCorr(mesh, "pCorr");
    if (pinReference)
    {
        pCorr.data()(0) = 0.0;
        pCorr.data().tail(nCells - 1) = pCorrSol;
    }
    else
    {
        pCorr.data() = pCorrSol;
    }

    // ---- Corrections -------------------------------------------------
    // 1. Conservative flux correction (OpenFOAM pEqn.flux):
    //   interior faces:         F_f -= C (p'_N - p'_P),
    //   Dirichlet-p boundaries: F_b += Cb p'_P (outward; west/south
    //   are stored with a sign, so they subtract instead).
    // After this pass each cell's net outflow equals the linear
    // solver's residual: the stored flux is conservative up to the
    // solver accuracy.
    const Index nx = mesh.nx();
    const Index ny = mesh.ny();
    const Scalar Sx = mesh.faceArea(BoundaryField::East);
    const Scalar Sy = mesh.faceArea(BoundaryField::North);
    const Scalar hx = mesh.cellToFaceDistance(BoundaryField::East);
    const Scalar hy = mesh.cellToFaceDistance(BoundaryField::North);

    for (Index j = 0; j < ny; ++j)
    {
        for (Index i = 1; i < nx; ++i)
        {
            const Index P = mesh.cellIndex(i - 1, j);
            const Index N = mesh.cellIndex(i, j);
            const Scalar dF = 0.5 * (dU(P) + dU(N));
            const Scalar C = rho * dF * Sx / mesh.dx();
            flux.x(i, j) -= C * (pCorr(N) - pCorr(P));
        }
    }
    for (Index j = 1; j < ny; ++j)
    {
        for (Index i = 0; i < nx; ++i)
        {
            const Index P = mesh.cellIndex(i, j - 1);
            const Index N = mesh.cellIndex(i, j);
            const Scalar dF = 0.5 * (dV(P) + dV(N));
            const Scalar C = rho * dF * Sy / mesh.dy();
            flux.y(i, j) -= C * (pCorr(N) - pCorr(P));
        }
    }
    if (bcP.get(BoundaryField::West).type == BCType::Dirichlet)
    {
        for (Index j = 0; j < ny; ++j)
        {
            const Index P = mesh.cellIndex(0, j);
            const Scalar Cb = rho * dU(P) * Sx / hx;
            flux.x(0, j) -= Cb * pCorr(P); // outward = -x(0, j)
        }
    }
    if (bcP.get(BoundaryField::East).type == BCType::Dirichlet)
    {
        for (Index j = 0; j < ny; ++j)
        {
            const Index P = mesh.cellIndex(nx - 1, j);
            const Scalar Cb = rho * dU(P) * Sx / hx;
            flux.x(nx, j) += Cb * pCorr(P); // outward = +x(nx, j)
        }
    }
    if (bcP.get(BoundaryField::South).type == BCType::Dirichlet)
    {
        for (Index i = 0; i < nx; ++i)
        {
            const Index P = mesh.cellIndex(i, 0);
            const Scalar Cb = rho * dV(P) * Sy / hy;
            flux.y(i, 0) -= Cb * pCorr(P); // outward = -y(i, 0)
        }
    }
    if (bcP.get(BoundaryField::North).type == BCType::Dirichlet)
    {
        for (Index i = 0; i < nx; ++i)
        {
            const Index P = mesh.cellIndex(i, ny - 1);
            const Scalar Cb = rho * dV(P) * Sy / hy;
            flux.y(i, ny) += Cb * pCorr(P); // outward = +y(i, ny)
        }
    }

    // 2a. Update the pressure (all cells first, Jacobi-style).
    // The cumulative reconstruction below evaluates grad(p) of the FULL
    // corrected pressure. Updating pressure in the same loop that
    // reconstructs the velocity makes the gradient read a partially
    // updated field, which is order dependent and wrong by orders of
    // magnitude (verified against OpenFOAM: cell-by-cell agreement to
    // ~1e-11 is only recovered with this two-pass split).
    for (Index P = 0; P < nCells; ++P)
    {
        const Scalar dp = relaxationP * pCorr(P);
        pressure(P) += dp;
        result.dpMax = std::max(result.dpMax, std::abs(dp));
    }

    // 2b. Cell-centered velocity reconstruction.
    for (Index P = 0; P < nCells; ++P)
    {
        Scalar uNew;
        Scalar vNew;
        if (cumulativeVelocityCorrection)
        {
            // Reconstitute U = uHat - d grad(p) from the full pressure.
            // relaxationP == 1 guarantees the full p' correction is stored
            // in `pressure`, so repeated calls (PISO correctors) accumulate
            // correctly instead of dropping earlier corrections.
            uNew = uHatU(P) - dU(P) * cellGradient(mesh, pressure, bcP, P, 0);
            vNew = uHatV(P) - dV(P) * cellGradient(mesh, pressure, bcP, P, 1);
        }
        else
        {
            // Steady SIMPLE: apply the unrelaxed p' correction directly.
            uNew = uStar(P) - dU(P) * cellGradient(mesh, pCorr, bcPrime, P, 0);
            vNew = vStar(P) - dV(P) * cellGradient(mesh, pCorr, bcPrime, P, 1);
        }
        result.duMax = std::max(result.duMax, std::abs(uNew - velocity.u()(P)));
        result.dvMax = std::max(result.dvMax, std::abs(vNew - velocity.v()(P)));
        velocity.u()(P) = uNew;
        velocity.v()(P) = vNew;
    }

    return result;
}

} // namespace fvm::numerical
