#pragma once

#include "Mesh.h"
#include "Types.h"
#include <string>

namespace fvm::core
{

/// Scalar flux stored on mesh faces (OpenFOAM-style surfaceScalarField).
/// Interior values follow the global face orientation: phi > 0 transports
/// from faceOwner to faceNeighbour.  Boundary values are owner-outward.
class FaceFluxField
{
public:
    FaceFluxField(const Mesh& mesh, const std::string& name = "")
        : mesh_(mesh)
        , name_(name)
        , phi_(mesh.faceCount())
    {
        setZero();
    }

    const Mesh& mesh() const
    {
        return mesh_;
    }

    const std::string& name() const
    {
        return name_;
    }

    Index faceCount() const
    {
        return phi_.size();
    }

    Scalar& operator[](Index face)
    {
        return phi_[face];
    }
    const Scalar& operator[](Index face) const
    {
        return phi_[face];
    }

    Vector& data()
    {
        return phi_;
    }
    const Vector& data() const
    {
        return phi_;
    }

    /// Signed outward flux of `cell` across one of its faces.
    Scalar outwardFlux(Index cell, Index face) const
    {
        return mesh_.faceOwner(face) == cell ? phi_[face] : -phi_[face];
    }

    /// Net outflow of a cell; zero to solver accuracy for a conservative
    /// flux field.
    Scalar cellImbalance(Index cell) const
    {
        Scalar sum = 0.0;
        for (const Index face : mesh_.cellFaces(cell))
        {
            sum += outwardFlux(cell, face);
        }
        return sum;
    }

    void setZero()
    {
        phi_.setZero();
    }

private:
    const Mesh& mesh_;
    std::string name_;
    Vector phi_;
};

} // namespace fvm::core
