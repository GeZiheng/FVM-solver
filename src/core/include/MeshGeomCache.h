#pragma once

#include "Types.h"

#include <cstddef>
#include <vector>

namespace fvm::core
{

class Mesh;

/// Interior-face non-orthogonal flux split family for the (alpha, k) pairs.
/// Core owns the geometric decomposition; numerical code selects which
/// family to use.  Both reduce to |S|/|d| on orthogonal faces.
enum class NonOrthSplit
{
    MinimumCorrection, ///< alpha = (S.d)/(d.d), the canonical d-projection.
    OverRelaxed        ///< alpha = (S.S)/max(S.d, 0.05|S||d|).
};

/// Precomputed face-based geometric coefficients for an arbitrary 2D mesh.
class MeshGeomCache
{
public:
    explicit MeshGeomCache(const Mesh& mesh);

    /// Linear interpolation weight from the owner to the neighbour.  Boundary
    /// faces have no neighbour and therefore use an owner-only weight of 1.
    Scalar interpWeight(Index face) const;

    /// Interior-face diffusion coefficient for the selected split:
    /// MinimumCorrection (default): (S.d)/(d.d); OverRelaxed:
    /// (S.S)/max(S.d, 0.05*|S||d|) (the floor mirrors OpenFOAM's mesh-level
    /// nonOrthDeltaCoeffs guard).  Computed on demand from the cached area
    /// vectors and centroid deltas -- roughly ten flops, far below the cost
    /// of the surrounding assembly work.
    Scalar innerCoef(Index face,
        NonOrthSplit split = NonOrthSplit::MinimumCorrection) const;

    /// Owner-to-neighbour centroid vector x_N - x_O. Interior faces only.
    Point2 innerDelta(Index face) const;

    /// Boundary-face diffusion coefficient (S.S)/(S.delta), where delta points
    /// from the owner centroid to the face centroid.
    Scalar boundaryCoef(Index face) const;

    /// Interior-face non-orthogonal correction S - alpha*d for the selected
    /// split (zero when the area vector is parallel to the centroid delta).
    /// Always pass the same split used for innerCoef so the (alpha, k) pair
    /// completes to the full S.G face flux.
    Point2 nonOrthCorrection(Index face,
        NonOrthSplit split = NonOrthSplit::MinimumCorrection) const;

    /// Interior-face non-orthogonality angle in degrees:
    /// acos(clamp((S.d)/(|S||d|))). Interior faces only.
    Scalar nonOrthogonality(Index face) const;

    /// Skewness vector s_f = x_f - (g*x_O + (1-g)*x_N), g = interpWeight(f).
    /// Interior faces only.
    Point2 skewVector(Index face) const;

    /// Largest interior-face non-orthogonality angle in degrees.
    Scalar maxNonOrthogonality() const
    {
        return maxNonOrthogonality_;
    }

    /// Max over interior faces of |s_f|/|d| (0.0 when there are none).
    Scalar maxSkewness() const
    {
        return maxSkewness_;
    }

private:
    std::size_t faceCount_ = 0;
    std::vector<Index> faceOwners_;
    std::vector<Index> faceNeighbours_;
    std::vector<Point2> faceCentroids_;
    std::vector<Point2> faceAreaVectors_;
    std::vector<Scalar> interpWeights_;
    std::vector<Point2> innerDeltas_;
    std::vector<Scalar> boundaryCoefficients_;
    std::vector<Scalar> nonOrthAngles_;
    std::vector<Point2> skewVectors_;
    Scalar maxNonOrthogonality_ = 0.0;
    Scalar maxSkewness_ = 0.0;
};

} // namespace fvm::core
