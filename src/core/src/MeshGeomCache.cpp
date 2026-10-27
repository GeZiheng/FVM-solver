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
    , innerDeltas_(faceCount_)
    , boundaryCoefficients_(faceCount_)
    , nonOrthAngles_(faceCount_)
    , skewVectors_(faceCount_)
{
    maxNonOrthogonality_ = 0.0;
    maxSkewness_ = 0.0;
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
        innerDeltas_[face] = delta;

        const Scalar neighbourDistance
            = std::sqrt(squaredDistance(faceCentroids_[face],
                neighbourCentroid));
        const Scalar cellDistance
            = std::sqrt(squaredDistance(ownerCentroid, neighbourCentroid));
        interpWeights_[face] = neighbourDistance / cellDistance;

        const Scalar cosine
            = std::clamp(dotProduct(areaVector, delta)
                             / (std::sqrt(dotProduct(areaVector, areaVector))
                                 * std::sqrt(dotProduct(delta, delta))),
                -1.0,
                1.0);
        nonOrthAngles_[face] = std::acos(cosine) * degreesPerRadian;
        maxNonOrthogonality_ = std::max(maxNonOrthogonality_,
            nonOrthAngles_[face]);

        skewVectors_[face]
            = { faceCentroids_[face].x
                  - (interpWeights_[face] * ownerCentroid.x
                      + (1.0 - interpWeights_[face]) * neighbourCentroid.x),
                faceCentroids_[face].y
                  - (interpWeights_[face] * ownerCentroid.y
                      + (1.0 - interpWeights_[face]) * neighbourCentroid.y) };
        const Scalar skewnessRatio = std::sqrt(
            dotProduct(skewVectors_[face], skewVectors_[face]))
            / std::sqrt(dotProduct(delta, delta));
        maxSkewness_ = std::max(maxSkewness_, skewnessRatio);
    }
}

Point2 MeshGeomCache::innerDelta(Index face) const
{
    if (face >= faceCount_)
    {
        throw std::out_of_range("Invalid face index");
    }
    if (faceNeighbours_[face] == Mesh::kInvalidCell)
    {
        throw std::invalid_argument(
            "innerDelta is valid for interior faces only");
    }
    return innerDeltas_[face];
}

Scalar MeshGeomCache::interpWeight(Index face) const
{
    if (face >= faceCount_)
    {
        throw std::out_of_range("Invalid face index");
    }
    return interpWeights_[face];
}

Scalar MeshGeomCache::innerCoef(Index face, NonOrthSplit split) const
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

    const Point2& area = faceAreaVectors_[face];
    const Point2& delta = innerDeltas_[face];
    const Scalar areaDotArea = dotProduct(area, area);
    const Scalar deltaDotDelta = dotProduct(delta, delta);
    if (split == NonOrthSplit::OverRelaxed)
    {
        const Scalar areaDistanceFloor = 0.05 * std::sqrt(areaDotArea)
            * std::sqrt(deltaDotDelta);
        return areaDotArea
            / std::max(dotProduct(area, delta), areaDistanceFloor);
    }
    return dotProduct(area, delta) / deltaDotDelta;
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

Point2 MeshGeomCache::nonOrthCorrection(Index face, NonOrthSplit split) const
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

    const Point2& area = faceAreaVectors_[face];
    const Point2& delta = innerDeltas_[face];
    const Scalar alpha = innerCoef(face, split);
    return { area.x - alpha * delta.x, area.y - alpha * delta.y };
}

Scalar MeshGeomCache::nonOrthogonality(Index face) const
{
    if (face >= faceCount_)
    {
        throw std::out_of_range("Invalid face index");
    }
    if (faceNeighbours_[face] == Mesh::kInvalidCell)
    {
        throw std::invalid_argument(
            "nonOrthogonality is valid for interior faces only");
    }
    return nonOrthAngles_[face];
}

Point2 MeshGeomCache::skewVector(Index face) const
{
    if (face >= faceCount_)
    {
        throw std::out_of_range("Invalid face index");
    }
    if (faceNeighbours_[face] == Mesh::kInvalidCell)
    {
        throw std::invalid_argument(
            "skewVector is valid for interior faces only");
    }
    return skewVectors_[face];
}

} // namespace fvm::core
