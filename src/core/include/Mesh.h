#pragma once

#include "Types.h"
#include <string>

namespace fvm::core
{

/// Abstract interface for a two-dimensional finite-volume mesh.
class Mesh
{
public:
    virtual ~Mesh() = default;

    virtual Index cellCount() const = 0;

    /// Interior and boundary faces in one unified numbering.
    virtual Index faceCount() const = 0;
    virtual Index faceOwner(Index face) const = 0;

    /// Returns kInvalidCell on a boundary face.
    virtual Index faceNeighbour(Index face) const = 0;
    virtual bool isBoundaryFace(Index face) const = 0;
    virtual IndexSpan cellFaces(Index cell) const = 0;
    virtual Scalar cellVolume(Index cell) const = 0;
    virtual Point2 cellCentroid(Index cell) const = 0;
    virtual Point2 faceCentroid(Index face) const = 0;

    /// Face area vector |S|*n, outward with respect to the owner cell.
    virtual Point2 faceAreaVector(Index face) const = 0;

    virtual Index patchCount() const = 0;
    virtual const std::string& patchName(Index patch) const = 0;
    virtual IndexSpan patchFaces(Index patch) const = 0;

    /// Valid for boundary faces only.
    virtual Index facePatch(Index face) const = 0;

    static constexpr Index kInvalidCell = static_cast<Index>(-1);
};

} // namespace fvm::core
