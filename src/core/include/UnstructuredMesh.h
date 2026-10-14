#pragma once

#include "Mesh.h"
#include "Types.h"

#include <string>
#include <utility>
#include <vector>

namespace fvm::core
{

/// One boundary patch described by unordered vertex-pair edges.
struct PatchInput
{
    std::string name;
    std::vector<std::pair<Index, Index>> edges;
};

/// Polygonal input description for an UnstructuredMesh.
struct MeshInput
{
    std::vector<Point2> points;
    std::vector<std::vector<Index>> cells;
    std::vector<PatchInput> patches;
};

/// A validated two-dimensional polygonal mesh with explicit face topology.
///
/// Cell loops are normalized to counter-clockwise order.  The area vector of a
/// face is derived from its owner's traversal, so it points from the owner
/// towards the neighbour on interior faces and outwards on boundary faces.
class UnstructuredMesh final : public Mesh
{
public:
    explicit UnstructuredMesh(const MeshInput& input);

    Index cellCount() const override
    {
        return cellVolumes_.size();
    }

    Index faceCount() const override
    {
        return faces_.size();
    }

    Index faceOwner(Index face) const override;
    Index faceNeighbour(Index face) const override;
    bool isBoundaryFace(Index face) const override;
    IndexSpan cellFaces(Index cell) const override;
    Scalar cellVolume(Index cell) const override;
    Point2 cellCentroid(Index cell) const override;
    Point2 faceCentroid(Index face) const override;
    Point2 faceAreaVector(Index face) const override;

    Index patchCount() const override
    {
        return patchNames_.size();
    }

    const std::string& patchName(Index patch) const override;
    IndexSpan patchFaces(Index patch) const override;
    Index facePatch(Index face) const override;

private:
    struct Face
    {
        Index owner = 0;
        Index neighbour = kInvalidCell;
        Index v0 = 0;
        Index v1 = 0;
    };

    std::vector<Face> faces_;
    std::vector<std::vector<Index>> cellLoops_;
    std::vector<Scalar> cellVolumes_;
    std::vector<Point2> cellCentroids_;
    std::vector<Point2> faceCentroids_;
    std::vector<Point2> faceAreaVectors_;
    std::vector<Index> facePatches_;

    std::vector<Index> cellFaceIds_;
    std::vector<Index> cellFaceOffsets_;
    std::vector<Index> patchFaceIds_;
    std::vector<Index> patchFaceOffsets_;
    std::vector<std::string> patchNames_;
};

/// Build a structured quadrilateral mesh from grid points and CCW cells.
/// The patches follow the CartesianMesh compatibility order:
/// 0 = east, 1 = north, 2 = west, 3 = south.
UnstructuredMesh makeStructuredMesh(Index nx,
    Index ny,
    Scalar xMin,
    Scalar yMin,
    Scalar xMax,
    Scalar yMax);

} // namespace fvm::core
