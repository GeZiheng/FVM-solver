#include "Mesh.h"
#include <stdexcept>

namespace fvm::core {

CartesianMesh::CartesianMesh(Index nx, Index ny, Scalar xMin, Scalar yMin, Scalar xMax, Scalar yMax)
  : nx_(nx), ny_(ny), xMin_(xMin), yMin_(yMin), xMax_(xMax), yMax_(yMax)
{
  if (nx == 0 || ny == 0) {
    throw std::invalid_argument("Mesh dimensions must be positive");
  }
  dx_ = (xMax - xMin) / static_cast<Scalar>(nx);
  dy_ = (yMax - yMin) / static_cast<Scalar>(ny);
  volume_ = dx_ * dy_;
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

Index CartesianMesh::neighbor(Index cell, int face) const
{
  auto [i, j] = cellIJ(cell);
  switch (face) {
  case 0: // east
    if (i + 1 < nx_) return cellIndex(i + 1, j);
    break;
  case 1: // north
    if (j + 1 < ny_) return cellIndex(i, j + 1);
    break;
  case 2: // west
    if (i > 0) return cellIndex(i - 1, j);
    break;
  case 3: // south
    if (j > 0) return cellIndex(i, j - 1);
    break;
  default:
    throw std::invalid_argument("Invalid face index");
  }
  return cellCount(); // Boundary
}

Scalar CartesianMesh::faceArea(int face) const
{
  if (face == 0 || face == 2) return dy_; // east/west faces
  if (face == 1 || face == 3) return dx_; // north/south faces
  throw std::invalid_argument("Invalid face index");
}

std::pair<Scalar, Scalar> CartesianMesh::faceNormal(int face) const
{
  switch (face) {
  case 0: return { +1.0, 0.0 };
  case 1: return { 0.0, +1.0 };
  case 2: return { -1.0, 0.0 };
  case 3: return { 0.0, -1.0 };
  default: throw std::invalid_argument("Invalid face index");
  }
}

Scalar CartesianMesh::cellToCellDistance(Index /*cell*/, int face) const
{
  if (face == 0 || face == 2) return dx_;
  if (face == 1 || face == 3) return dy_;
  throw std::invalid_argument("Invalid face index");
}

Scalar CartesianMesh::cellToFaceDistance(int face) const
{
  if (face == 0 || face == 2) return dx_ / 2.0;
  if (face == 1 || face == 3) return dy_ / 2.0;
  throw std::invalid_argument("Invalid face index");
}

bool CartesianMesh::isBoundaryFace(Index cell, int face) const
{
  return neighbor(cell, face) == cellCount();
}

} // namespace fvm::core
