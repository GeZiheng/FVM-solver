#pragma once

#include "Types.h"
#include <array>
#include <utility>
#include <vector>

namespace fvm::core
{

/**
 * @brief 2D uniform Cartesian mesh.
 *
 * Layout: cells are indexed as (i, j) where i runs along x and j along y.
 * Linear index: cellIndex = j * nx + i.
 *
 * Face indexing convention:
 *   - Face 0 (east)  : normal = (+1, 0)
 *   - Face 1 (north) : normal = (0, +1)
 *   - Face 2 (west)  : normal = (-1, 0)
 *   - Face 3 (south) : normal = (0, -1)
 */
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

    /**
     * @brief Cell-center coordinates.
     * @return (x, y) of cell center.
     */
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

    /**
     * @brief Get neighbor cell across a face.
     * @param cell Cell index.
     * @param face Face index (0=east, 1=north, 2=west, 3=south).
     * @return Neighbor cell index, or cellCount() if boundary.
     */
    Index neighbor(Index cell, int face) const;

    /**
     * @brief Face area (2D: length of edge).
     */
    Scalar faceArea(int face) const;

    /**
     * @brief Outward face normal vector.
     */
    std::pair<Scalar, Scalar> faceNormal(int face) const;

    /**
     * @brief Distance from cell center to neighbor cell center across face.
     * Returns 0 if neighbor is boundary.
     */
    Scalar cellToCellDistance(Index cell, int face) const;

    /**
     * @brief Distance from cell center to face center.
     */
    Scalar cellToFaceDistance(int face) const;

    /**
     * @brief Check if cell is on a boundary.
     * @param cell Cell index.
     * @param face Face index.
     * @return true if the face is on domain boundary.
     */
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
