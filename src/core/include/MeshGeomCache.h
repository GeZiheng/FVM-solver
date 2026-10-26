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

    /// Interior-face over-relaxed diffusion coefficient
    /// (S.S)/max(S.d, 0.05*|S||d|). Interior faces only.  The floor mirrors
    /// OpenFOAM's mesh-level nonOrthDeltaCoeffs guard against degenerate faces.
    Scalar innerCoefOverRelaxed(Index face) const;

    /// Owner-to-neighbour centroid vector x_N - x_O. Interior faces only.
    Point2 innerDelta(Index face) const;

    /// Boundary-face diffusion coefficient (S.S)/(S.delta), where delta points
    /// from the owner centroid to the face centroid.
    Scalar boundaryCoef(Index face) const;

    /// Interior-face non-orthogonal correction S - alpha*d.  It is zero when
    /// the face area vector is parallel to the owner-neighbour distance.
    Point2 nonOrthCorrection(Index face) const;

    /// Interior-face over-relaxed correction vector S - alphaOr*d.
    Point2 nonOrthCorrectionOverRelaxed(Index face) const;

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
    std::vector<Scalar> innerCoefficients_;
    std::vector<Scalar> innerCoefficientsOverRelaxed_;
    std::vector<Point2> innerDeltas_;
    std::vector<Scalar> boundaryCoefficients_;
    std::vector<Point2> nonOrthCorrections_;
    std::vector<Point2> nonOrthCorrectionsOverRelaxed_;
    std::vector<Scalar> nonOrthAngles_;
    std::vector<Point2> skewVectors_;
    Scalar maxNonOrthogonality_ = 0.0;
    Scalar maxSkewness_ = 0.0;
};

} // namespace fvm::core
