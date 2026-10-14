#include "UnstructuredMesh.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace fvm::core
{
namespace
{
std::pair<Index, Index> faceKey(Index first, Index second)
{
    return std::minmax(first, second);
}

Scalar polygonTwiceArea(const std::vector<Index>& loop, const std::vector<Point2>& points)
{
    Scalar twiceArea = 0.0;
    for (Index local = 0; local < loop.size(); ++local)
    {
        const Point2& current = points[loop[local]];
        const Point2& next = points[loop[(local + 1) % loop.size()]];
        twiceArea += current.x * next.y - next.x * current.y;
    }
    return twiceArea;
}
} // namespace

UnstructuredMesh::UnstructuredMesh(const MeshInput& input)
{
    for (const std::vector<Index>& loop : input.cells)
    {
        if (loop.size() < 3)
        {
            throw std::invalid_argument("Mesh cells must have at least three vertices");
        }
        for (const Index vertex : loop)
        {
            if (vertex >= input.points.size())
            {
                throw std::out_of_range("Cell vertex index is out of range");
            }
        }
    }
    for (const PatchInput& patch : input.patches)
    {
        for (const auto& [first, second] : patch.edges)
        {
            if (first >= input.points.size() || second >= input.points.size())
            {
                throw std::out_of_range("Patch vertex index is out of range");
            }
        }
    }

    // A face area vector follows the owner's edge orientation.  Normalizing
    // every cell to CCW order therefore makes each vector owner-outward.
    cellLoops_ = input.cells;
    for (std::vector<Index>& loop : cellLoops_)
    {
        if (polygonTwiceArea(loop, input.points) < 0.0)
        {
            std::reverse(loop.begin(), loop.end());
        }
    }

    std::map<std::pair<Index, Index>, Index> faceIds;
    cellFaceOffsets_.push_back(0);
    for (Index cell = 0; cell < cellLoops_.size(); ++cell)
    {
        const std::vector<Index>& loop = cellLoops_[cell];
        for (Index local = 0; local < loop.size(); ++local)
        {
            const Index first = loop[local];
            const Index second = loop[(local + 1) % loop.size()];
            const auto key = faceKey(first, second);
            const auto existing = faceIds.find(key);
            if (existing == faceIds.end())
            {
                const Index face = faces_.size();
                faces_.push_back({ cell, kInvalidCell, first, second });
                faceIds.emplace(key, face);
                cellFaceIds_.push_back(face);
                continue;
            }

            Face& face = faces_[existing->second];
            if (face.neighbour != kInvalidCell || face.owner == cell
                || (face.v0 == first && face.v1 == second))
            {
                throw std::invalid_argument("Non-manifold or inconsistent mesh orientation");
            }
            face.neighbour = cell;
            cellFaceIds_.push_back(existing->second);
        }
        cellFaceOffsets_.push_back(cellFaceIds_.size());
    }

    cellVolumes_.reserve(cellLoops_.size());
    cellCentroids_.reserve(cellLoops_.size());
    for (const std::vector<Index>& loop : cellLoops_)
    {
        const Scalar twiceArea = polygonTwiceArea(loop, input.points);
        if (std::abs(twiceArea) <= 0.0)
        {
            throw std::invalid_argument("Mesh cells must have non-zero area");
        }

        Scalar centroidX = 0.0;
        Scalar centroidY = 0.0;
        for (Index local = 0; local < loop.size(); ++local)
        {
            const Point2& current = input.points[loop[local]];
            const Point2& next = input.points[loop[(local + 1) % loop.size()]];
            const Scalar cross = current.x * next.y - next.x * current.y;
            centroidX += (current.x + next.x) * cross;
            centroidY += (current.y + next.y) * cross;
        }

        const Scalar area = 0.5 * twiceArea;
        cellVolumes_.push_back(area);
        cellCentroids_.push_back({ centroidX / (6.0 * area), centroidY / (6.0 * area) });
    }

    faceCentroids_.reserve(faces_.size());
    faceAreaVectors_.reserve(faces_.size());
    for (const Face& face : faces_)
    {
        const Point2& first = input.points[face.v0];
        const Point2& second = input.points[face.v1];
        faceCentroids_.push_back({
            0.5 * (first.x + second.x),
            0.5 * (first.y + second.y)
        });
        faceAreaVectors_.push_back({ second.y - first.y, first.x - second.x });
    }

    for (Index cell = 0; cell < cellLoops_.size(); ++cell)
    {
        Point2 sum;
        Scalar maxArea = 0.0;
        const IndexSpan localFaces = cellFaces(cell);
        for (const Index face : localFaces)
        {
            const Scalar orientation = faces_[face].owner == cell ? 1.0 : -1.0;
            sum.x += orientation * faceAreaVectors_[face].x;
            sum.y += orientation * faceAreaVectors_[face].y;
            maxArea = std::max(maxArea,
                std::abs(faceAreaVectors_[face].x) + std::abs(faceAreaVectors_[face].y));
        }
        const Scalar residual = std::abs(sum.x) + std::abs(sum.y);
        if (residual > 1.0e-9 * maxArea)
        {
            throw std::invalid_argument("Cell face area vectors do not close");
        }
    }

    facePatches_.assign(faces_.size(), kInvalidCell);
    std::set<std::pair<Index, Index>> patchEdges;
    std::set<Index> patchFaces;
    for (const PatchInput& patch : input.patches)
    {
        for (const auto& [first, second] : patch.edges)
        {
            const auto key = faceKey(first, second);
            if (!patchEdges.insert(key).second)
            {
                throw std::invalid_argument("Patch edge is duplicated");
            }
            const auto found = faceIds.find(key);
            if (found == faceIds.end())
            {
                throw std::invalid_argument("Patch edge is not a mesh edge");
            }
            const Index face = found->second;
            if (!isBoundaryFace(face))
            {
                throw std::invalid_argument("Patch edge must describe a boundary face");
            }
            if (!patchFaces.insert(face).second)
            {
                throw std::invalid_argument("Boundary face belongs to more than one patch edge");
            }
        }
    }

    patchNames_.reserve(input.patches.size());
    patchFaceOffsets_.push_back(0);
    for (const PatchInput& patch : input.patches)
    {
        patchNames_.push_back(patch.name);
        for (const auto& [first, second] : patch.edges)
        {
            const Index face = faceIds.at(faceKey(first, second));
            patchFaceIds_.push_back(face);
            facePatches_[face] = patchNames_.size() - 1;
        }
        patchFaceOffsets_.push_back(patchFaceIds_.size());
    }

    Index boundaryFaceCount = 0;
    for (const Face& face : faces_)
    {
        if (face.neighbour == kInvalidCell)
        {
            ++boundaryFaceCount;
        }
    }
    if (patchFaceIds_.size() != boundaryFaceCount)
    {
        throw std::invalid_argument("Every boundary face must belong to exactly one patch");
    }
}

Index UnstructuredMesh::faceOwner(Index face) const
{
    if (face >= faceCount())
    {
        throw std::out_of_range("Invalid face index");
    }
    return faces_[face].owner;
}

Index UnstructuredMesh::faceNeighbour(Index face) const
{
    if (face >= faceCount())
    {
        throw std::out_of_range("Invalid face index");
    }
    return faces_[face].neighbour;
}

bool UnstructuredMesh::isBoundaryFace(Index face) const
{
    return faceNeighbour(face) == kInvalidCell;
}

IndexSpan UnstructuredMesh::cellFaces(Index cell) const
{
    if (cell >= cellCount())
    {
        throw std::out_of_range("Invalid cell index");
    }
    const Index start = cellFaceOffsets_[cell];
    return IndexSpan(cellFaceIds_.data() + start, cellFaceOffsets_[cell + 1] - start);
}

Scalar UnstructuredMesh::cellVolume(Index cell) const
{
    if (cell >= cellCount())
    {
        throw std::out_of_range("Invalid cell index");
    }
    return cellVolumes_[cell];
}

Point2 UnstructuredMesh::cellCentroid(Index cell) const
{
    if (cell >= cellCount())
    {
        throw std::out_of_range("Invalid cell index");
    }
    return cellCentroids_[cell];
}

Point2 UnstructuredMesh::faceCentroid(Index face) const
{
    if (face >= faceCount())
    {
        throw std::out_of_range("Invalid face index");
    }
    return faceCentroids_[face];
}

Point2 UnstructuredMesh::faceAreaVector(Index face) const
{
    if (face >= faceCount())
    {
        throw std::out_of_range("Invalid face index");
    }
    return faceAreaVectors_[face];
}

const std::string& UnstructuredMesh::patchName(Index patch) const
{
    if (patch >= patchCount())
    {
        throw std::out_of_range("Invalid patch index");
    }
    return patchNames_[patch];
}

IndexSpan UnstructuredMesh::patchFaces(Index patch) const
{
    if (patch >= patchCount())
    {
        throw std::out_of_range("Invalid patch index");
    }
    const Index start = patchFaceOffsets_[patch];
    return IndexSpan(patchFaceIds_.data() + start, patchFaceOffsets_[patch + 1] - start);
}

Index UnstructuredMesh::facePatch(Index face) const
{
    if (face >= faceCount())
    {
        throw std::out_of_range("Invalid face index");
    }
    if (facePatches_[face] == kInvalidCell)
    {
        throw std::out_of_range("Face is not on a patch");
    }
    return facePatches_[face];
}

UnstructuredMesh makeStructuredMesh(Index nx,
    Index ny,
    Scalar xMin,
    Scalar yMin,
    Scalar xMax,
    Scalar yMax)
{
    if (nx == 0 || ny == 0)
    {
        throw std::invalid_argument("Mesh dimensions must be positive");
    }

    MeshInput input;
    input.points.resize((nx + 1) * (ny + 1));
    for (Index j = 0; j <= ny; ++j)
    {
        for (Index i = 0; i <= nx; ++i)
        {
            const Index vertex = j * (nx + 1) + i;
            input.points[vertex] = {
                xMin + static_cast<Scalar>(i) * (xMax - xMin) / static_cast<Scalar>(nx),
                yMin + static_cast<Scalar>(j) * (yMax - yMin) / static_cast<Scalar>(ny)
            };
        }
    }

    input.cells.reserve(nx * ny);
    for (Index j = 0; j < ny; ++j)
    {
        for (Index i = 0; i < nx; ++i)
        {
            const Index lowerLeft = j * (nx + 1) + i;
            input.cells.push_back({
                lowerLeft,
                lowerLeft + 1,
                lowerLeft + nx + 2,
                lowerLeft + nx + 1
            });
        }
    }

    input.patches.reserve(4);
    input.patches.push_back({ "east", {} });
    input.patches.push_back({ "north", {} });
    input.patches.push_back({ "west", {} });
    input.patches.push_back({ "south", {} });
    for (Index j = 0; j < ny; ++j)
    {
        input.patches[0].edges.emplace_back((j + 1) * (nx + 1) - 1,
            (j + 2) * (nx + 1) - 1);
    }
    for (Index i = 0; i < nx; ++i)
    {
        input.patches[1].edges.emplace_back(ny * (nx + 1) + i,
            ny * (nx + 1) + i + 1);
    }
    for (Index j = 0; j < ny; ++j)
    {
        input.patches[2].edges.emplace_back(j * (nx + 1),
            (j + 1) * (nx + 1));
    }
    for (Index i = 0; i < nx; ++i)
    {
        input.patches[3].edges.emplace_back(i, i + 1);
    }

    return UnstructuredMesh(input);
}

} // namespace fvm::core
