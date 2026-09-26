#include "Momentum.h"
#include "Diffusion.h"
#include "GridOperators.h"

#include <stdexcept>

namespace fvm::numerical
{

namespace
{

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
    Scalar relaxation,
    const TimeTerm& time)
{
    if (component != 0 && component != 1)
    {
        throw std::invalid_argument(
            "assembleMomentum: component must be 0 (u) or 1 (v)");
    }
    if (relaxation <= 0.0 || relaxation > 1.0)
    {
        throw std::
            invalid_argument("assembleMomentum: relaxation must be in (0, 1]");
    }
    if (time.active() && relaxation != 1.0)
    {
        throw std::invalid_argument(
            "assembleMomentum: transient momentum requires relaxation == 1");
    }

    const Index nCells = mesh.cellCount();
    const Scalar vol = mesh.cellVolume(0);

    ScalarField muField(mesh, "mu");
    muField.setConstant(mu);

    MomentumAssembly result{
        EquationSystem(nCells), Vector(nCells), Vector(nCells)
    };
    result.rhsNoPressure.setZero();

    // Convection-diffusion part (no pressure source yet).
    assembleDiffusion(mesh, muField, bc, result.system.A, result.rhsNoPressure);
    assembleConvection(mesh,
        flux,
        scheme,
        bc,
        result.system.A,
        result.rhsNoPressure);

    // Read the unrelaxed diagonal from a finalized copy.
    SparseMatrix probe = result.system.A;
    probe.finalize();
    const Vector aP0 = probe.native().diagonal();

    const ScalarField& phiOld = (component == 0) ? velocity.u() : velocity.v();

    if (time.active())
    {
        // theta-scheme ddt term: diagonal rho V/dt, right-hand side
        // rho V/dt phi_old, and the spatial operator scaled by theta.
        // The explicit old-time contribution - (1 - theta) A_spatial
        // phi_old is moved to the right-hand side (zero for Euler).
        const Scalar theta = time.theta();
        const Scalar ddtDiag = time.rho * vol / time.dt;

        Vector AphiOld;
        if (theta < 1.0)
        {
            AphiOld = probe.native() * phiOld.data();
        }

        result.system.A.scale(theta);
        for (Index P = 0; P < nCells; ++P)
        {
            result.system.A.insert(P, P, ddtDiag);
            result.rhsNoPressure(P) += ddtDiag * phiOld(P);
            if (theta < 1.0)
            {
                result.rhsNoPressure(P) -= (1.0 - theta) * AphiOld(P);
            }
        }
        result.diag = theta * aP0;
        result.diag.array() += ddtDiag;
    }
    else
    {
        const Scalar invAlpha = 1.0 / relaxation;

        result.diag = aP0 * invAlpha;

        for (Index P = 0; P < nCells; ++P)
        {
            // Patankar under-relaxation: A(P,P) /= alpha,
            // b += (1 - alpha) / alpha * aP0 * phi_old.
            result.system.A.insert(P, P, aP0(P) * (invAlpha - 1.0));
            result.rhsNoPressure(P)
                += (1.0 - relaxation) * invAlpha * aP0(P) * phiOld(P);
        }
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
    const TimeTerm& time,
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
                                 relaxationU,
                                 time),
        assembleMomentum(mesh,
            flux,
            velocity,
            mu,
            scheme,
            bcV,
            1,
            pressure,
            bcP,
            relaxationU,
            time),
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

} // namespace fvm::numerical
