#pragma once

#include "Types.h"
#include "Mesh.h"
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace fvm::core
{

/// 2D uniform Cartesian mesh.  Cells are (i, j) with i along x, j along y and
/// linear index cellIndex = j * nx + i.
///
/// cellFaces(cell) lists the faces in the order [east, north, west, south];
/// xFace(i, j) / yFace(i, j) expose the global face ids.  Patches are created
/// in the order 0 = east, 1 = north, 2 = west, 3 = south.
class CartesianMesh : public Mesh
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
    Index cellCount() const override
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
    Scalar cellVolume(Index cell) const override;

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

    // Generic mesh interface: global face and patch topology.
    Index faceCount() const override;
    Index faceOwner(Index face) const override;
    Index faceNeighbour(Index face) const override;
    bool isBoundaryFace(Index face) const override;
    IndexSpan cellFaces(Index cell) const override;
    Point2 cellCentroid(Index cell) const override;
    Point2 faceCentroid(Index face) const override;
    Point2 faceAreaVector(Index face) const override;
    Index patchCount() const override;
    const std::string& patchName(Index patch) const override;
    IndexSpan patchFaces(Index patch) const override;
    Index facePatch(Index face) const override;

    /// Global id of the vertical face at x = xMin + i*dx, y-cell j.
    Index xFace(Index i, Index j) const;

    /// Global id of the horizontal face at y = yMin + j*dy, x-cell i.
    Index yFace(Index i, Index j) const;

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
    std::vector<Index> cellFaceIds_;
    std::vector<Index> cellFaceOffsets_;
    std::vector<Index> patchFaceIds_;
    std::vector<Index> patchFaceOffsets_;
    std::array<std::string, 4> patchNames_;
};

} // namespace fvm::core
