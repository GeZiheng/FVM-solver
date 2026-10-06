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
    FaceFluxField& flux)
{
    if (relaxationP <= 0.0 || relaxationP > 1.0)
    {
        throw std::
            invalid_argument("correctPressure: relaxationP must be in (0, 1]");
    }

    const Index nCells = mesh.cellCount();

    // Pure-Neumann p makes the equation singular (constants null space):
    // eliminate reference cell 0 and keep its value fixed (its continuity
    // equation is redundant). Any Dirichlet p side makes the system definite
    // -> keep all cells.
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

    const Vector& uHatU = pred.uHatU;
    const Vector& uHatV = pred.uHatV;
    const Vector& dU = pred.dU;
    const Vector& dV = pred.dV;

    // ---- Pressure equation for the absolute pressure (OpenFOAM pEqn) ----
    // Cell continuity sum_f F_f = 0 with
    //   F_f = rho S_f uHat_f - C_f (p_N - p_P)   (interior, C_f = rho d_f S_f/delta)
    //   F_b = rho S_f u_b    - C_b (p_b - p_P)   (Dirichlet p)
    //   F_b = rho S_f u_b                        (Neumann p)
    // -> A p = b,  b = -sum_f outward(phiHbyA_f) + sum_Dirichlet C_b p_b.
    // The pressure-free flux phiHbyA = rho S_f uHat_f is written into the
    // persistent flux and corrected in place below (phi = phiHbyA - pEqn.flux()),
    // so the corrected flux stays conservative to the solver accuracy.
    EquationSystem pSys(nP);
    Vector massImbalance(nCells); // net outward phiHbyA (adjusted below)
    Vector rhsExtra(nCells);      // known boundary / reference pressure terms
    massImbalance.setZero();
    rhsExtra.setZero();

    // Value of the eliminated reference cell (kept at its previous level).
    const Scalar pRefValue = pressure(0);

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
                const Scalar F = rho * Sf * uHatF; // phiHbyA: no pressure
                const Scalar C = rho * dF * Sf / delta;

                // Store phiHbyA (positive P -> N, stored convention).
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

                // Reference-cell elimination: the neighbour row lost the
                // -C p_0 entry, so move the known value to the right-hand side.
                if (isReference(P))
                    rhsExtra(N) += C * pRefValue;
            }
            else
            {
                // Boundary face: the same pressure-free uHat flux as the
                // interior; a Dirichlet-p face adds C_b p_b to the right-hand
                // side (the unknown C_b p_P stays on the diagonal).  u_b must
                // come from uHat (or the velocity BC), never from uStar.
                const BoundaryCondition& pCond = bcP.get(face);
                const Scalar ub = boundaryNormalVelocity(face,
                    xFace ? bcU : bcV,
                    xFace ? uHatU : uHatV,
                    P);
                const Scalar Fb = rho * ub * Sf;
                massImbalance(P) += Fb;

                // Outward flux (west/south are stored along +x/+y -> sign).
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

                if (pCond.type == BCType::Dirichlet)
                {
                    const Scalar Cb
                        = rho * dC(P) * Sf / mesh.cellToFaceDistance(face);
                    if (!isReference(P))
                        pSys.A.insert(rowOf(P), rowOf(P), Cb);
                    rhsExtra(P) += Cb * pCond.value;
                }
            }
        }
    }

    // ---- Closed-domain compatibility (OpenFOAM adjustPhi) -------------
    // Pure-Neumann p: the p terms cannot change the net boundary outflow
    // (interior terms cancel, Neumann faces take no pressure term), so an
    // unbalanced phiHbyA makes the system inconsistent and silently violates
    // the eliminated cell's continuity.  Spread the residual over the faces
    // whose normal velocity is not fixed; the same adjustment must reach the
    // right-hand side, which is why massImbalance is updated here too.
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
        }
    }

    // ---- Assemble the right-hand side and solve for the absolute p ----
    // b = -sum_f outward(phiHbyA_f) + sum_Dirichlet C_b p_b + reference term.
    Vector pOld(nP);
    if (pinReference)
        pOld = pressure.data().tail(nCells - 1);
    else
        pOld = pressure.data();

    Vector b(nP);
    for (Index P = 0; P < nCells; ++P)
    {
        if (!isReference(P))
            b(rowOf(P)) = -massImbalance(P) + rhsExtra(P);
    }

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

    const Vector pSol = pSolver.solve(pSys.A, b);

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
    // 1. Conservative flux correction (OpenFOAM pEqn.flux):
    //   F_f = phiHbyA_f - C (p_N - p_P) on interior faces, and
    //   F_b = phiHbyA_b - Cb (p_b - p_P) on Dirichlet-p faces (stored sign on
    //   west/south is flipped).  Each cell's net outflow then equals the linear
    //   solver's residual.
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
            flux.x(i, j) -= C * (pNew(N) - pNew(P));
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
            flux.y(i, j) -= C * (pNew(N) - pNew(P));
        }
    }
    if (bcP.get(BoundaryField::West).type == BCType::Dirichlet)
    {
        const Scalar pb = bcP.get(BoundaryField::West).value;
        for (Index j = 0; j < ny; ++j)
        {
            const Index P = mesh.cellIndex(0, j);
            const Scalar Cb = rho * dU(P) * Sx / hx;
            flux.x(0, j) += Cb * (pb - pNew(P)); // outward = -x(0, j)
        }
    }
    if (bcP.get(BoundaryField::East).type == BCType::Dirichlet)
    {
        const Scalar pb = bcP.get(BoundaryField::East).value;
        for (Index j = 0; j < ny; ++j)
        {
            const Index P = mesh.cellIndex(nx - 1, j);
            const Scalar Cb = rho * dU(P) * Sx / hx;
            flux.x(nx, j) -= Cb * (pb - pNew(P)); // outward = +x(nx, j)
        }
    }
    if (bcP.get(BoundaryField::South).type == BCType::Dirichlet)
    {
        const Scalar pb = bcP.get(BoundaryField::South).value;
        for (Index i = 0; i < nx; ++i)
        {
            const Index P = mesh.cellIndex(i, 0);
            const Scalar Cb = rho * dV(P) * Sy / hy;
            flux.y(i, 0) += Cb * (pb - pNew(P)); // outward = -y(i, 0)
        }
    }
    if (bcP.get(BoundaryField::North).type == BCType::Dirichlet)
    {
        const Scalar pb = bcP.get(BoundaryField::North).value;
        for (Index i = 0; i < nx; ++i)
        {
            const Index P = mesh.cellIndex(i, ny - 1);
            const Scalar Cb = rho * dV(P) * Sy / hy;
            flux.y(i, ny) -= Cb * (pb - pNew(P)); // outward = +y(i, ny)
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
    // pressure: u = uHat - d grad(p) (OpenFOAM U = HbyA - rAU*grad(p), but with
    // the unrelaxed p, unlike OpenFOAM's steady p.relax() ordering - see
    // docs/numerical.md).  For SIMPLE this equals u* - d grad(p') because
    // uHat = u* + d grad(p_old); for PISO it is the cumulative reconstruction
    // across correctors.
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
