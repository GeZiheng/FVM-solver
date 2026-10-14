#pragma once

#include "Types.h"

#include <cstddef>
#include <vector>

namespace fvm::core
{

class Mesh;

/// Precomputed face-based geometric coefficients for an arbitrary 2D mesh.
class MeshGeomCache
{
public:
    explicit MeshGeomCache(const Mesh& mesh);

    /// Linear interpolation weight from the owner to the neighbour.  Boundary
    /// faces have no neighbour and therefore use an owner-only weight of 1.
    Scalar interpWeight(Index face) const;

    /// Interior-face diffusion coefficient (S.d)/(d.d), where d points from
    /// the owner centroid to the neighbour centroid.
    Scalar innerCoef(Index face) const;

    /// Boundary-face diffusion coefficient (S.S)/(S.delta), where delta points
    /// from the owner centroid to the face centroid.
    Scalar boundaryCoef(Index face) const;

    /// Interior-face non-orthogonal correction S - alpha*d.  It is zero when
    /// the face area vector is parallel to the owner-neighbour distance.
    Point2 nonOrthCorrection(Index face) const;

    /// Largest interior-face non-orthogonality angle in degrees.
    Scalar maxNonOrthogonality() const
    {
        return maxNonOrthogonality_;
    }

private:
    std::size_t faceCount_ = 0;
    std::vector<Index> faceOwners_;
    std::vector<Index> faceNeighbours_;
    std::vector<Point2> faceCentroids_;
    std::vector<Point2> faceAreaVectors_;
    std::vector<Scalar> interpWeights_;
    std::vector<Scalar> innerCoefficients_;
    std::vector<Scalar> boundaryCoefficients_;
    std::vector<Point2> nonOrthCorrections_;
    Scalar maxNonOrthogonality_ = 0.0;
};

} // namespace fvm::core
