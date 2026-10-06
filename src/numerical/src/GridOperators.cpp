#include "GridOperators.h"

namespace fvm::numerical
{

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

} // namespace fvm::numerical
