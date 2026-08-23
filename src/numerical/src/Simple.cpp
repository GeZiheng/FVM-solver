#include "Simple.h"
#include "Diffusion.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace fvm::numerical
{

namespace
{

/**
 * @brief Cell-center gradient of a scalar field, component `dir`, via the
 * Gauss divergence theorem:
 *
 *   grad(phi)_P = (1 / vol_P) * sum_f phi_f * n_f,dir * S_f.
 *
 * Interior face values are arithmetic means; boundary faces take the
 * Dirichlet value or the cell value (zero normal gradient) for Neumann.
 * On a uniform grid this reduces to central differences in the interior.
 */
Scalar cellGradient(const CartesianMesh& mesh,
    const ScalarField& phi,
    const BoundaryField& bc,
    Index P,
    int dir)
{
    const Index nCells = mesh.cellCount();
    Scalar sum = 0.0;

    for (int face = 0; face < 4; ++face)
    {
        const auto normal = mesh.faceNormal(face);
        const Scalar nd = (dir == 0) ? normal.first : normal.second;
        if (nd == 0.0)
            continue;

        const Index N = mesh.neighbor(P, face);
        Scalar phiF;
        if (N != nCells)
        {
            phiF = 0.5 * (phi(P) + phi(N));
        }
        else
        {
            const BoundaryCondition& cond = bc.get(face);
            phiF = (cond.type == BCType::Dirichlet) ? cond.value : phi(P);
        }
        sum += phiF * nd * mesh.faceArea(face);
    }

    return sum / mesh.cellVolume(P);
}

/**
 * @brief Velocity without the pressure-gradient contribution:
 * uHat = (b_noPressure - sum_{N != P} A(P,N) u*_N) / diag(P).
 */
Vector computeUHat(const MomentumAssembly& mom, const Vector& uStar)
{
    const auto& A = mom.system.A.native();
    const Vector Au = A * uStar;
    // off-diagonal part: A u* - diag * u*
    const Vector offDiag = Au - mom.diag.cwiseProduct(uStar);
    return (mom.rhsNoPressure - offDiag).cwiseQuotient(mom.diag);
}

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
    return (face == BoundaryField::West || face == BoundaryField::South)
               ? -val
               : val;
}

} // namespace

MomentumAssembly assembleMomentum(const CartesianMesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    Scalar mu,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    int component,
    const ScalarField& pressure,
    const BoundaryField& bcP,
    Scalar relaxation)
{
    if (component != 0 && component != 1)
    {
        throw std::invalid_argument(
            "assembleMomentum: component must be 0 (u) or 1 (v)");
    }
    if (relaxation <= 0.0 || relaxation > 1.0)
    {
        throw std::invalid_argument(
            "assembleMomentum: relaxation must be in (0, 1]");
    }

    const Index nCells = mesh.cellCount();
    const Scalar vol = mesh.cellVolume(0);

    ScalarField muField(mesh, "mu");
    muField.setConstant(mu);

    MomentumAssembly result{ EquationSystem(nCells), Vector(nCells),
        Vector(nCells) };
    result.rhsNoPressure.setZero();

    // Convection-diffusion part (no pressure source yet).
    assembleDiffusion(mesh, muField, bc, result.system.A,
        result.rhsNoPressure);
    assembleConvection(mesh, velocity, rho, scheme, bc, result.system.A,
        result.rhsNoPressure);

    // Read the unrelaxed diagonal from a finalized copy.
    SparseMatrix probe = result.system.A;
    probe.finalize();
    const Vector aP0 = probe.native().diagonal();

    const Scalar invAlpha = 1.0 / relaxation;
    const ScalarField& phiOld
        = (component == 0) ? velocity.u() : velocity.v();

    result.diag = aP0 * invAlpha;

    for (Index P = 0; P < nCells; ++P)
    {
        // Patankar under-relaxation: A(P,P) /= alpha,
        // b += (1 - alpha) / alpha * aP0 * phi_old.
        result.system.A.insert(P, P, aP0(P) * (invAlpha - 1.0));
        result.rhsNoPressure(P)
            += (1.0 - relaxation) * invAlpha * aP0(P) * phiOld(P);
    }

    // Full right-hand side: add the pressure-gradient source.
    result.system.b = result.rhsNoPressure;
    for (Index P = 0; P < nCells; ++P)
    {
        result.system.b(P)
            -= cellGradient(mesh, pressure, bcP, P, component) * vol;
    }

    return result;
}

SimpleResult solveSimple(const CartesianMesh& mesh,
    Scalar rho,
    Scalar mu,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const BoundaryField& bcP,
    const SimpleConfig& config,
    VectorField& velocity,
    ScalarField& pressure)
{
    if (config.relaxationP <= 0.0 || config.relaxationP > 1.0)
    {
        throw std::invalid_argument(
            "solveSimple: relaxationP must be in (0, 1]");
    }

    const Index nCells = mesh.cellCount();
    const Scalar vol = mesh.cellVolume(0);

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
    const auto rowOf = [pinReference](Index cell) -> Index {
        return pinReference ? cell - 1 : cell;
    };
    const auto isReference
        = [pinReference](Index cell) { return pinReference && cell == 0; };

    // Pressure-correction BCs: p' = 0 wherever p is fixed (Dirichlet),
    // zero gradient elsewhere.
    BoundaryField bcPrime = bcP;
    for (int side = 0; side < 4; ++side)
    {
        if (bcPrime.get(side).type == BCType::Dirichlet)
            bcPrime.set(side, BCType::Dirichlet, 0.0);
    }

    auto momSolver = createEigenBiCGSTAB(config.solverConfig);
    auto pSolver = createEigenCG(config.solverConfig);

    ScalarField pCorr(mesh, "pCorr");
    SimpleResult result;

    for (int iter = 1; iter <= config.maxIterations; ++iter)
    {
        // ---- 1. Momentum predictor ------------------------------------
        auto momU = assembleMomentum(mesh,
            velocity,
            rho,
            mu,
            config.scheme,
            bcU,
            0,
            pressure,
            bcP,
            config.relaxationU);
        momU.system.A.finalize();
        const Vector uStar = momSolver->solve(momU.system.A, momU.system.b);

        auto momV = assembleMomentum(mesh,
            velocity,
            rho,
            mu,
            config.scheme,
            bcV,
            1,
            pressure,
            bcP,
            config.relaxationU);
        momV.system.A.finalize();
        const Vector vStar = momSolver->solve(momV.system.A, momV.system.b);

        // ---- 2. Rhie-Chow data: uHat and d = vol / a_P ----------------
        const Vector uHatU = computeUHat(momU, uStar);
        const Vector uHatV = computeUHat(momV, vStar);
        const Vector dU = momU.diag.cwiseInverse() * vol;
        const Vector dV = momV.diag.cwiseInverse() * vol;

        // ---- 3. Pressure-correction equation --------------------------
        // Continuity per cell: sum_f F_f = 0 with
        //   F_f = F*_f - rho d_f S_f (p'_N - p'_P) / delta  (interior),
        //   F_b = F*_b + rho d_P S_f p'_P / dist            (Dirichlet p),
        //   F_b = F*_b                                      (Neumann p).
        // Collecting the p' terms on the left gives the diffusion-like
        // system A p' = -massImbalance, where A has coefficient rho * d
        // and massImbalance(P) is the predicted net outflow of cell P.
        EquationSystem pSys(nP);
        Vector massImbalance(nCells);
        massImbalance.setZero();

        for (Index P = 0; P < nCells; ++P)
        {
            for (int face = 0; face < 4; ++face)
            {
                const Index N = mesh.neighbor(P, face);
                const Scalar Sf = mesh.faceArea(face);
                const bool xFace = (face == BoundaryField::East
                                    || face == BoundaryField::West);
                const Vector& dC = xFace ? dU : dV;
                const Vector& uHat = xFace ? uHatU : uHatV;

                if (N != nCells)
                {
                    if (face != BoundaryField::East
                        && face != BoundaryField::North)
                        continue; // interior faces once

                    const Scalar delta = mesh.cellToCellDistance(P, face);
                    const Scalar dF = 0.5 * (dC(P) + dC(N));
                    const Scalar uHatF = 0.5 * (uHat(P) + uHat(N));
                    const Scalar F = rho * Sf
                                     * (uHatF
                                        - dF * (pressure(N) - pressure(P))
                                              / delta);
                    const Scalar C = rho * dF * Sf / delta;

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
                    // Boundary face: predicted flux from the velocity BC.
                    const Scalar ub = boundaryNormalVelocity(face,
                        xFace ? bcU : bcV,
                        xFace ? uStar : vStar,
                        P);
                    massImbalance(P) += rho * ub * Sf;

                    if (bcP.get(face).type == BCType::Dirichlet
                        && !isReference(P))
                    {
                        // p' = 0 at the boundary: correction Cb * p'_P.
                        const Scalar Cb = rho * dC(P) * Sf
                                          / mesh.cellToFaceDistance(face);
                        pSys.A.insert(rowOf(P), rowOf(P), Cb);
                    }
                }
            }
        }

        // Scaled continuity residual (pre-correction mass imbalance).
        const Scalar maxImbalance = massImbalance.cwiseAbs().maxCoeff();

        for (Index P = 0; P < nCells; ++P)
        {
            if (!isReference(P))
                pSys.b(rowOf(P)) = -massImbalance(P);
        }
        pSys.A.finalize();
        Vector pCorrSol;
        try
        {
            pCorrSol = pSolver->solve(pSys.A, pSys.b);
        }
        catch (const std::exception& e)
        {
            std::cerr << "solveSimple iter " << iter
                      << ": pressure-correction solve failed: " << e.what()
                      << " (lastIterations=" << pSolver->lastIterations()
                      << ", lastResidual=" << pSolver->lastResidual() << ")"
                      << std::endl;
            throw;
        }
        if (pinReference)
        {
            pCorr.data()(0) = 0.0;
            pCorr.data().tail(nCells - 1) = pCorrSol;
        }
        else
        {
            pCorr.data() = pCorrSol;
        }

        // ---- 4. Corrections -------------------------------------------
        Scalar duMax = 0.0, dvMax = 0.0, dpMax = 0.0;
        for (Index P = 0; P < nCells; ++P)
        {
            const Scalar uNew
                = uStar(P) - dU(P) * cellGradient(mesh, pCorr, bcPrime, P, 0);
            const Scalar vNew
                = vStar(P) - dV(P) * cellGradient(mesh, pCorr, bcPrime, P, 1);
            duMax = std::max(duMax, std::abs(uNew - velocity.u()(P)));
            dvMax = std::max(dvMax, std::abs(vNew - velocity.v()(P)));
            velocity.u()(P) = uNew;
            velocity.v()(P) = vNew;

            const Scalar dp = config.relaxationP * pCorr(P);
            pressure(P) += dp;
            dpMax = std::max(dpMax, std::abs(dp));
        }

        // ---- 5. Residuals and convergence -----------------------------
        Scalar maxSpeed = 0.0;
        for (Index P = 0; P < nCells; ++P)
        {
            maxSpeed = std::max(maxSpeed,
                std::hypot(velocity.u()(P), velocity.v()(P)));
        }
        const Scalar fRef
            = std::max(rho * maxSpeed * 0.5 * (mesh.dx() + mesh.dy()),
                1e-30);
        const Scalar uRef = std::max(maxSpeed, 1e-30);

        SimpleResiduals res;
        res.continuity = maxImbalance / fRef;
        res.u = duMax / uRef;
        res.v = dvMax / uRef;
        res.pressure = dpMax;
        result.history.push_back(res);

        if (config.verbose)
        {
            std::cout << "SIMPLE iter " << iter << ": continuity="
                      << res.continuity << " du=" << res.u
                      << " dv=" << res.v << " dp=" << res.pressure
                      << std::endl;
        }

        if (res.continuity < config.tolerance && res.u < config.tolerance
            && res.v < config.tolerance)
        {
            result.converged = true;
            result.iterations = iter;
            return result;
        }
    }

    result.iterations = config.maxIterations;
    return result;
}

} // namespace fvm::numerical
