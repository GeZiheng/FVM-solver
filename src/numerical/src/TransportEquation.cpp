#include "TransportEquation.h"
#include <stdexcept>

namespace fvm::numerical
{

EquationSystem assembleTransport(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    const ScalarField& gamma,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    const ScalarField* Sc,
    const ScalarField* Sp)
{
    EquationSystem sys(mesh.cellCount());

    assembleDiffusion(mesh, gamma, bc, sys.A, sys.b);
    assembleConvection(mesh, flux, scheme, bc, sys.A, sys.b);

    const Index nCells = mesh.cellCount();
    const Scalar vol = mesh.cellVolume(0);

    if (Sc)
    {
        for (Index c = 0; c < nCells; ++c)
        {
            sys.b(c) += (*Sc)(c)*vol;
        }
    }

    if (Sp)
    {
        for (Index c = 0; c < nCells; ++c)
        {
            if ((*Sp)(c) > 0.0)
            {
                throw std::invalid_argument(
                    "assembleTransport: Sp must be non-positive (Patankar "
                    "linearization)");
            }
            sys.A.insert(c, c, -(*Sp)(c)*vol);
        }
    }

    return sys;
}

EquationSystem assembleTransientTransport(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    const ScalarField& gamma,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    const ScalarField& phiOld,
    Scalar dt,
    TimeScheme timeScheme,
    const ScalarField* Sc,
    const ScalarField* Sp)
{
    if (dt <= 0.0)
    {
        throw std::invalid_argument(
            "assembleTransientTransport: dt must be positive");
    }

    const Index nCells = mesh.cellCount();
    const Scalar vol = mesh.cellVolume(0);
    const Scalar theta = thetaOf(timeScheme);
    const Scalar invDt = 1.0 / dt;

    // Full spatial operator (unscaled), reused for the explicit part.
    EquationSystem sys
        = assembleTransport(mesh, flux, gamma, scheme, bc, Sc, Sp);

    SparseMatrix spatial = sys.A; // copy of the accumulated triplets
    const Vector bSpatial = sys.b; // full spatial right-hand side
    spatial.finalize();
    const Vector AphiOld = spatial.native() * phiOld.data();

    // Left-hand side: theta * A_spatial + (V/dt) I.
    sys.A.scale(theta);
    for (Index c = 0; c < nCells; ++c)
    {
        sys.A.insert(c, c, vol * invDt);
    }

    // Right-hand side:
    //   (V/dt) phi^n - (1 - theta) A_spatial phi^n + b_spatial.
    sys.b = (vol * invDt) * phiOld.data() - (1.0 - theta) * AphiOld + bSpatial;

    return sys;
}

} // namespace fvm::numerical
