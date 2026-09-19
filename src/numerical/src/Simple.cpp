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
    const FaceFluxField& flux,
    const VectorField& velocity,
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
    assembleConvection(mesh, flux, scheme, bc, result.system.A,
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
    fvm::math::LinearSolver& momSolver)
{
    const Scalar vol = mesh.cellVolume(0);

    MomentumPrediction pred{ assembleMomentum(mesh,
                                 flux,
                                 velocity,
                                 mu,
                                 scheme,
                                 bcU,
                                 0,
                                 pressure,
                                 bcP,
                                 relaxationU),
        assembleMomentum(mesh,
            flux,
            velocity,
            mu,
            scheme,
            bcV,
            1,
            pressure,
            bcP,
            relaxationU),
        Vector(),
        Vector(),
        Vector(),
        Vector(),
        Vector(),
        Vector() };

    pred.momU.system.A.finalize();
    pred.uStar = momSolver.solve(pred.momU.system.A, pred.momU.system.b);
    pred.momV.system.A.finalize();
    pred.vStar = momSolver.solve(pred.momV.system.A, pred.momV.system.b);

    pred.uHatU = computeUHat(pred.momU, pred.uStar);
    pred.uHatV = computeUHat(pred.momV, pred.vStar);
    pred.dU = pred.momU.diag.cwiseInverse() * vol;
    pred.dV = pred.momV.diag.cwiseInverse() * vol;

    return pred;
}

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
        throw std::invalid_argument(
            "correctPressure: relaxationP must be in (0, 1]");
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
                const Scalar F = rho * Sf
                                 * (uHatF
                                    - dF * (pressure(N) - pressure(P)) / delta);
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
                // Boundary face: predicted flux from the velocity BC.
                const Scalar ub = boundaryNormalVelocity(face,
                    xFace ? bcU : bcV,
                    xFace ? uStar : vStar,
                    P);
                const Scalar Fb = rho * ub * Sf;
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

                if (bcP.get(face).type == BCType::Dirichlet && !isReference(P))
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

    // 2. Cell-centered velocity and pressure corrections.
    for (Index P = 0; P < nCells; ++P)
    {
        const Scalar uNew
            = uStar(P) - dU(P) * cellGradient(mesh, pCorr, bcPrime, P, 0);
        const Scalar vNew
            = vStar(P) - dV(P) * cellGradient(mesh, pCorr, bcPrime, P, 1);
        result.duMax = std::max(result.duMax, std::abs(uNew - velocity.u()(P)));
        result.dvMax = std::max(result.dvMax, std::abs(vNew - velocity.v()(P)));
        velocity.u()(P) = uNew;
        velocity.v()(P) = vNew;

        const Scalar dp = relaxationP * pCorr(P);
        pressure(P) += dp;
        result.dpMax = std::max(result.dpMax, std::abs(dp));
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
    ScalarField& pressure,
    FaceFluxField& flux)
{
    if (config.relaxationP <= 0.0 || config.relaxationP > 1.0)
    {
        throw std::invalid_argument(
            "solveSimple: relaxationP must be in (0, 1]");
    }

    const Index nCells = mesh.cellCount();

    // Initialize the persistent flux field from the velocity field and
    // its BCs (OpenFOAM createPhi).
    computeMassFlux(mesh, velocity, rho, bcU, bcV, flux);

    // With pure Neumann pressure BCs the pressure-correction equation is
    // only compatible when the boundary fluxes sum to zero (OpenFOAM
    // adjustPhi); catch unbalanced velocity BCs here instead of solving
    // an inconsistent system.
    bool hasDirichletP = false;
    for (int side = 0; side < 4; ++side)
    {
        if (bcP.get(side).type == BCType::Dirichlet)
            hasDirichletP = true;
    }
    if (!hasDirichletP)
        checkFluxCompatibility(flux);

    auto momSolver = createEigenBiCGSTAB(config.solverConfig);
    auto pSolver = createEigenCG(config.solverConfig);

    SimpleResult result;

    for (int iter = 1; iter <= config.maxIterations; ++iter)
    {
        const MomentumPrediction pred = predictMomentum(mesh,
            flux,
            velocity,
            mu,
            config.scheme,
            bcU,
            bcV,
            pressure,
            bcP,
            config.relaxationU,
            *momSolver);

        CorrectorResult corr;
        try
        {
            corr = correctPressure(mesh,
                rho,
                pred,
                bcU,
                bcV,
                bcP,
                *pSolver,
                config.relaxationP,
                velocity,
                pressure,
                flux);
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

        // ---- Residuals and convergence --------------------------------
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
        res.continuity = corr.maxImbalance / fRef;
        res.u = corr.duMax / uRef;
        res.v = corr.dvMax / uRef;
        res.pressure = corr.dpMax;
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
