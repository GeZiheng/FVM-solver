#pragma once

#include "Types.h"
#include <array>
#include <utility>
#include <vector>

namespace fvm::core
{

/// 2D uniform Cartesian mesh.  Cells are (i, j) with i along x, j along y and
/// linear index cellIndex = j * nx + i.
///
/// Face indices (used by every boundary array in the project):
///   0 = east (+x), 1 = north (+y), 2 = west (-x), 3 = south (-y)
class CartesianMesh
{
public:
    CartesianMesh(Index nx,
        Index ny,
        Scalar xMin,
        Scalar yMin,
        Scalar xMax,
        Scalar yMax);

    // Grid dimensions
    Index nx() const
    {
        return nx_;
    }
    Index ny() const
    {
        return ny_;
    }
    Index cellCount() const
    {
        return nx_ * ny_;
    }

    // Geometry
    Scalar dx() const
    {
        return dx_;
    }
    Scalar dy() const
    {
        return dy_;
    }
    Scalar cellVolume(Index cell) const;

    /// Cell-center coordinates (x, y).
    std::pair<Scalar, Scalar> cellCenter(Index cell) const;
    std::pair<Scalar, Scalar> cellCenter(Index i, Index j) const;

    // Index mapping
    Index cellIndex(Index i, Index j) const
    {
        return j * nx_ + i;
    }
    std::pair<Index, Index> cellIJ(Index cell) const
    {
        return { cell % nx_, cell / nx_ };
    }

    /// Neighbour cell across `face`; returns cellCount() on a boundary.
    Index neighbor(Index cell, int face) const;

    /// Face area (2D: edge length).
    Scalar faceArea(int face) const;

    /// Outward face normal.
    std::pair<Scalar, Scalar> faceNormal(int face) const;

    /// Cell-center to neighbour-center distance; 0 on a boundary.
    Scalar cellToCellDistance(Index cell, int face) const;

    /// Cell-center to face-center distance.
    Scalar cellToFaceDistance(int face) const;

    /// True if the given cell face lies on the domain boundary.
    bool isBoundaryFace(Index cell, int face) const;

    // Domain bounds
    Scalar xMin() const
    {
        return xMin_;
    }
    Scalar yMin() const
    {
        return yMin_;
    }
    Scalar xMax() const
    {
        return xMax_;
    }
    Scalar yMax() const
    {
        return yMax_;
    }

private:
    Index nx_, ny_;
    Scalar xMin_, yMin_, xMax_, yMax_;
    Scalar dx_, dy_;
    Scalar volume_;
};

} // namespace fvm::core
