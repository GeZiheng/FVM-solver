#include "GridOperators.h"
#include "Mesh.h"

#include <cmath>

namespace fvm::numerical
{
using fvm::core::Point2;

Scalar cellGradient(const Mesh& mesh,
    const ScalarField& phi,
    const BoundaryField& bc,
    Index P,
    int dir)
{
    Scalar sum = 0.0;

    for (const Index face : mesh.cellFaces(P))
    {
        const Point2 area = mesh.faceAreaVector(face);
        const Scalar orientation = (mesh.faceOwner(face) == P) ? 1.0 : -1.0;
        const Scalar nd = orientation * ((dir == 0) ? area.x : area.y);

        Scalar phiF;
        if (mesh.isBoundaryFace(face))
        {
            const BoundaryCondition& cond = bc.get(mesh.facePatch(face));
            phiF = (cond.type == BCType::Dirichlet) ? cond.value : phi(P);
        }
        else
        {
            const Index owner = mesh.faceOwner(face);
            const Index neighbour = mesh.faceNeighbour(face);
            // Distance-weighted linear interpolation of the owner value,
            // identical to MeshGeomCache::interpWeight but computed inline:
            // this is a per-cell API, so constructing an O(N) cache here
            // would make a full gradient sweep O(N^2).
            const Point2 centroidO = mesh.cellCentroid(owner);
            const Point2 centroidN = mesh.cellCentroid(neighbour);
            const Point2 centroidF = mesh.faceCentroid(face);
            const Scalar toNeighbour = std::hypot(centroidN.x - centroidO.x,
                centroidN.y - centroidO.y);
            const Scalar faceToNeighbour = std::hypot(centroidN.x - centroidF.x,
                centroidN.y - centroidF.y);
            const Scalar g = faceToNeighbour / toNeighbour;
            phiF = g * phi(owner) + (1.0 - g) * phi(neighbour);
        }

        sum += phiF * nd;
    }

    return sum / mesh.cellVolume(P);
}

} // namespace fvm::numerical
