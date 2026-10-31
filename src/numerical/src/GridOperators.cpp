#include "GridOperators.h"
#include "Mesh.h"

#include <cmath>
#include <vector>

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

std::vector<Point2> leastSquaresGradients(const Mesh& mesh,
    const ScalarField& phi,
    const BoundaryField& bc)
{
    std::vector<Point2> gradients(mesh.cellCount());

    for (Index P = 0; P < mesh.cellCount(); ++P)
    {
        const Point2 cellCentroid = mesh.cellCentroid(P);
        Scalar a00 = 0.0;
        Scalar a01 = 0.0;
        Scalar a11 = 0.0;
        Scalar b0 = 0.0;
        Scalar b1 = 0.0;

        for (const Index face : mesh.cellFaces(P))
        {
            Point2 direction;
            Scalar residual = 0.0;
            Scalar weight = 0.0;

            if (!mesh.isBoundaryFace(face))
            {
                const Index owner = mesh.faceOwner(face);
                const Index neighbour = mesh.faceNeighbour(face);
                const Index Q = (owner == P) ? neighbour : owner;
                const Point2 neighbourCentroid = mesh.cellCentroid(Q);
                direction = { neighbourCentroid.x - cellCentroid.x,
                    neighbourCentroid.y - cellCentroid.y };
                residual = phi(Q) - phi(P);
                weight = 1.0
                    / std::hypot(direction.x, direction.y);
            }
            else
            {
                const BoundaryCondition& condition = bc.get(mesh.facePatch(face));
                const Point2 faceCentroid = mesh.faceCentroid(face);
                direction = { faceCentroid.x - cellCentroid.x,
                    faceCentroid.y - cellCentroid.y };
                if (condition.type == BCType::Dirichlet)
                {
                    residual = condition.value - phi(P);
                    weight = 1.0
                        / std::hypot(direction.x, direction.y);
                }
                else
                {
                    const Point2 area = mesh.faceAreaVector(face);
                    const Scalar areaNorm = std::hypot(area.x, area.y);
                    direction = { area.x / areaNorm, area.y / areaNorm };
                    residual = condition.value;
                    weight = 1.0;
                }
            }

            const Scalar scaledWeight = weight * weight;
            const Scalar dx = scaledWeight * direction.x;
            const Scalar dy = scaledWeight * direction.y;
            a00 += dx * direction.x;
            a01 += dx * direction.y;
            a11 += dy * direction.y;
            b0 += dx * residual;
            b1 += dy * residual;
        }

        const Scalar determinant = a00 * a11 - a01 * a01;
        const Scalar trace = a00 + a11;
        const bool degenerate = !std::isfinite(determinant)
            || determinant <= 1.0e-24 * trace * trace;

        if (degenerate)
        {
            gradients[P] = { cellGradient(mesh, phi, bc, P, 0),
                cellGradient(mesh, phi, bc, P, 1) };
            continue;
        }

        gradients[P] = { (a11 * b0 - a01 * b1) / determinant,
            (a00 * b1 - a01 * b0) / determinant };
        if (!std::isfinite(gradients[P].x)
            || !std::isfinite(gradients[P].y))
        {
            gradients[P] = { cellGradient(mesh, phi, bc, P, 0),
                cellGradient(mesh, phi, bc, P, 1) };
        }
    }

    return gradients;
}

} // namespace fvm::numerical
