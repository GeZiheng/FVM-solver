#include "MeshGeomCache.h"
#include "Mesh.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fvm::core
{
namespace
{
Scalar squaredDistance(const Point2& first, const Point2& second)
{
    const Scalar dx = second.x - first.x;
    const Scalar dy = second.y - first.y;
    return dx * dx + dy * dy;
}

Scalar dotProduct(const Point2& first, const Point2& second)
{
    return first.x * second.x + first.y * second.y;
}
} // namespace

MeshGeomCache::MeshGeomCache(const Mesh& mesh)
    : faceCount_(mesh.faceCount())
    , faceOwners_(faceCount_)
    , faceNeighbours_(faceCount_)
    , faceCentroids_(faceCount_)
    , faceAreaVectors_(faceCount_)
    , interpWeights_(faceCount_, 1.0)
    , innerCoefficients_(faceCount_)
    , boundaryCoefficients_(faceCount_)
    , nonOrthCorrections_(faceCount_)
{
    maxNonOrthogonality_ = 0.0;
    for (Index face = 0; face < faceCount_; ++face)
    {
        faceOwners_[face] = mesh.faceOwner(face);
        faceNeighbours_[face] = mesh.faceNeighbour(face);
        faceCentroids_[face] = mesh.faceCentroid(face);
        faceAreaVectors_[face] = mesh.faceAreaVector(face);
    }

    constexpr Scalar degreesPerRadian = 180.0 / 3.14159265358979323846;
    for (Index face = 0; face < faceCount_; ++face)
    {
        const Point2& areaVector = faceAreaVectors_[face];
        const Point2 ownerCentroid = mesh.cellCentroid(faceOwners_[face]);
        if (faceNeighbours_[face] == Mesh::kInvalidCell)
        {
            const Point2 delta{ faceCentroids_[face].x - ownerCentroid.x,
                faceCentroids_[face].y - ownerCentroid.y };
            boundaryCoefficients_[face] = dotProduct(areaVector, areaVector)
                                          / dotProduct(areaVector, delta);
            continue;
        }

        const Point2 neighbourCentroid
            = mesh.cellCentroid(faceNeighbours_[face]);
        const Point2 delta{ neighbourCentroid.x - ownerCentroid.x,
            neighbourCentroid.y - ownerCentroid.y };

        const Scalar neighbourDistance
            = std::sqrt(squaredDistance(faceCentroids_[face],
                neighbourCentroid));
        const Scalar cellDistance
            = std::sqrt(squaredDistance(ownerCentroid, neighbourCentroid));
        interpWeights_[face] = neighbourDistance / cellDistance;

        innerCoefficients_[face]
            = dotProduct(areaVector, delta) / dotProduct(delta, delta);
        nonOrthCorrections_[face]
            = { areaVector.x - innerCoefficients_[face] * delta.x,
                  areaVector.y - innerCoefficients_[face] * delta.y };

        const Scalar cosine
            = std::clamp(dotProduct(areaVector, delta)
                             / (std::sqrt(dotProduct(areaVector, areaVector))
                                 * std::sqrt(dotProduct(delta, delta))),
                -1.0,
                1.0);
        maxNonOrthogonality_ = std::max(maxNonOrthogonality_,
            std::acos(cosine) * degreesPerRadian);
    }
}

Scalar MeshGeomCache::interpWeight(Index face) const
{
    if (face >= faceCount_)
    {
        throw std::out_of_range("Invalid face index");
    }
    return interpWeights_[face];
}

Scalar MeshGeomCache::innerCoef(Index face) const
{
    if (face >= faceCount_)
    {
        throw std::out_of_range("Invalid face index");
    }
    if (faceNeighbours_[face] == Mesh::kInvalidCell)
    {
        throw std::
            invalid_argument("innerCoef is valid for interior faces only");
    }
    return innerCoefficients_[face];
}

Scalar MeshGeomCache::boundaryCoef(Index face) const
{
    if (face >= faceCount_)
    {
        throw std::out_of_range("Invalid face index");
    }
    if (faceNeighbours_[face] != Mesh::kInvalidCell)
    {
        throw std::
            invalid_argument("boundaryCoef is valid for boundary faces only");
    }
    return boundaryCoefficients_[face];
}

Point2 MeshGeomCache::nonOrthCorrection(Index face) const
{
    if (face >= faceCount_)
    {
        throw std::out_of_range("Invalid face index");
    }
    if (faceNeighbours_[face] == Mesh::kInvalidCell)
    {
        throw std::invalid_argument(
            "nonOrthCorrection is valid for interior faces only");
    }
    return nonOrthCorrections_[face];
}

} // namespace fvm::core
