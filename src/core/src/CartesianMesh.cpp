#include "CartesianMesh.h"
#include <stdexcept>

namespace fvm::core
{

CartesianMesh::CartesianMesh(Index nx,
    Index ny,
    Scalar xMin,
    Scalar yMin,
    Scalar xMax,
    Scalar yMax)
    : nx_(nx)
    , ny_(ny)
    , xMin_(xMin)
    , yMin_(yMin)
    , xMax_(xMax)
    , yMax_(yMax)
{
    if (nx == 0 || ny == 0)
    {
        throw std::invalid_argument("Mesh dimensions must be positive");
    }
    dx_ = (xMax - xMin) / static_cast<Scalar>(nx);
    dy_ = (yMax - yMin) / static_cast<Scalar>(ny);
    volume_ = dx_ * dy_;

    patchNames_ = { "east", "north", "west", "south" };

    cellFaceOffsets_.push_back(0);
    for (Index j = 0; j < ny_; ++j)
    {
        for (Index i = 0; i < nx_; ++i)
        {
            cellFaceIds_.push_back(xFace(i + 1, j)); // east
            cellFaceIds_.push_back(yFace(i, j + 1)); // north
            cellFaceIds_.push_back(xFace(i, j));     // west
            cellFaceIds_.push_back(yFace(i, j));     // south
            cellFaceOffsets_.push_back(cellFaceIds_.size());
        }
    }

    patchFaceOffsets_.push_back(0);
    for (Index j = 0; j < ny_; ++j)
    {
        patchFaceIds_.push_back(xFace(nx_, j));
    }
    patchFaceOffsets_.push_back(patchFaceIds_.size());
    for (Index i = 0; i < nx_; ++i)
    {
        patchFaceIds_.push_back(yFace(i, ny_));
    }
    patchFaceOffsets_.push_back(patchFaceIds_.size());
    for (Index j = 0; j < ny_; ++j)
    {
        patchFaceIds_.push_back(xFace(0, j));
    }
    patchFaceOffsets_.push_back(patchFaceIds_.size());
    for (Index i = 0; i < nx_; ++i)
    {
        patchFaceIds_.push_back(yFace(i, 0));
    }
    patchFaceOffsets_.push_back(patchFaceIds_.size());
}

Index CartesianMesh::faceCount() const
{
    return (nx_ + 1) * ny_ + nx_ * (ny_ + 1);
}

Index CartesianMesh::xFace(Index i, Index j) const
{
    if (i > nx_ || j >= ny_)
    {
        throw std::out_of_range("Invalid x-face indices");
    }
    return j * (nx_ + 1) + i;
}

Index CartesianMesh::yFace(Index i, Index j) const
{
    if (i >= nx_ || j > ny_)
    {
        throw std::out_of_range("Invalid y-face indices");
    }
    return (nx_ + 1) * ny_ + j * nx_ + i;
}

Index CartesianMesh::faceOwner(Index face) const
{
    if (face >= faceCount())
    {
        throw std::out_of_range("Invalid face index");
    }

    const Index xFaces = (nx_ + 1) * ny_;
    if (face < xFaces)
    {
        const Index j = face / (nx_ + 1);
        const Index i = face % (nx_ + 1);
        if (i == 0)
        {
            return cellIndex(0, j);
        }
        return cellIndex(i - 1, j);
    }

    const Index yFaceId = face - xFaces;
    const Index j = yFaceId / nx_;
    const Index i = yFaceId % nx_;
    if (j == 0)
    {
        return cellIndex(i, 0);
    }
    return cellIndex(i, j - 1);
}

Index CartesianMesh::faceNeighbour(Index face) const
{
    if (face >= faceCount())
    {
        throw std::out_of_range("Invalid face index");
    }

    const Index xFaces = (nx_ + 1) * ny_;
    if (face < xFaces)
    {
        const Index j = face / (nx_ + 1);
        const Index i = face % (nx_ + 1);
        if (i == 0 || i == nx_)
        {
            return kInvalidCell;
        }
        return cellIndex(i, j);
    }

    const Index yFaceId = face - xFaces;
    const Index j = yFaceId / nx_;
    const Index i = yFaceId % nx_;
    if (j == 0 || j == ny_)
    {
        return kInvalidCell;
    }
    return cellIndex(i, j);
}

bool CartesianMesh::isBoundaryFace(Index face) const
{
    return faceNeighbour(face) == kInvalidCell;
}

IndexSpan CartesianMesh::cellFaces(Index cell) const
{
    if (cell >= cellCount())
    {
        throw std::out_of_range("Invalid cell index");
    }
    const Index start = cellFaceOffsets_[cell];
    return IndexSpan(cellFaceIds_.data() + start, cellFaceOffsets_[cell + 1] - start);
}

Scalar CartesianMesh::cellVolume(Index /*cell*/) const
{
    return volume_;
}

std::pair<Scalar, Scalar> CartesianMesh::cellCenter(Index cell) const
{
    auto [i, j] = cellIJ(cell);
    return cellCenter(i, j);
}

std::pair<Scalar, Scalar> CartesianMesh::cellCenter(Index i, Index j) const
{
    Scalar x = xMin_ + (static_cast<Scalar>(i) + 0.5) * dx_;
    Scalar y = yMin_ + (static_cast<Scalar>(j) + 0.5) * dy_;
    return { x, y };
}

Point2 CartesianMesh::cellCentroid(Index cell) const
{
    if (cell >= cellCount())
    {
        throw std::out_of_range("Invalid cell index");
    }
    const auto center = cellCenter(cell);
    return { center.first, center.second };
}

Point2 CartesianMesh::faceCentroid(Index face) const
{
    if (face >= faceCount())
    {
        throw std::out_of_range("Invalid face index");
    }

    const Index xFaces = (nx_ + 1) * ny_;
    if (face < xFaces)
    {
        const Index j = face / (nx_ + 1);
        const Index i = face % (nx_ + 1);
        return {
            xMin_ + static_cast<Scalar>(i) * dx_,
            yMin_ + (static_cast<Scalar>(j) + 0.5) * dy_
        };
    }

    const Index yFaceId = face - xFaces;
    const Index j = yFaceId / nx_;
    const Index i = yFaceId % nx_;
    return {
        xMin_ + (static_cast<Scalar>(i) + 0.5) * dx_,
        yMin_ + static_cast<Scalar>(j) * dy_
    };
}

Point2 CartesianMesh::faceAreaVector(Index face) const
{
    if (face >= faceCount())
    {
        throw std::out_of_range("Invalid face index");
    }

    const Index xFaces = (nx_ + 1) * ny_;
    if (face < xFaces)
    {
        const Index i = face % (nx_ + 1);
        if (i == 0)
        {
            return { -dy_, 0.0 };
        }
        return { dy_, 0.0 };
    }

    const Index j = (face - xFaces) / nx_;
    if (j == 0)
    {
        return { 0.0, -dx_ };
    }
    return { 0.0, dx_ };
}

Index CartesianMesh::patchCount() const
{
    return 4;
}

const std::string& CartesianMesh::patchName(Index patch) const
{
    if (patch >= patchCount())
    {
        throw std::out_of_range("Invalid patch index");
    }
    return patchNames_[patch];
}

IndexSpan CartesianMesh::patchFaces(Index patch) const
{
    if (patch >= patchCount())
    {
        throw std::out_of_range("Invalid patch index");
    }
    const Index start = patchFaceOffsets_[patch];
    return IndexSpan(patchFaceIds_.data() + start, patchFaceOffsets_[patch + 1] - start);
}

Index CartesianMesh::facePatch(Index face) const
{
    for (Index patch = 0; patch < patchCount(); ++patch)
    {
        const Index start = patchFaceOffsets_[patch];
        const Index count = patchFaceOffsets_[patch + 1] - start;
        for (Index local = 0; local < count; ++local)
        {
            if (patchFaceIds_[start + local] == face)
            {
                return patch;
            }
        }
    }
    throw std::out_of_range("Face is not on a patch");
}

} // namespace fvm::core
