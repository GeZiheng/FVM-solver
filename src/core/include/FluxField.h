#pragma once

#include "Mesh.h"
#include "Types.h"
#include <string>

namespace fvm::core
{

/**
 * @brief Scalar flux field stored on mesh faces (OpenFOAM-style
 * surfaceScalarField).
 *
 * Stores one scalar (typically the mass flux rho * u . n * S) per face,
 * including boundary faces. Sign convention: fluxes are positive along
 * the positive coordinate direction:
 *   - x(i, j): flux across the x-face i of row j (between cells i-1 and
 *     i), i in [0, nx], j in [0, ny);
 *   - y(i, j): flux across the y-face j of column i (between cells j-1
 *     and j), i in [0, nx), j in [0, ny].
 *
 * The outward flux of a cell therefore carries a sign: east/north faces
 * are +x/+y, west/south faces are -x/-y (see outwardFlux).
 */
class FaceFluxField
{
public:
    FaceFluxField(const CartesianMesh& mesh, const std::string& name = "")
        : mesh_(mesh)
        , name_(name)
        , x_((mesh.nx() + 1) * mesh.ny())
        , y_(mesh.nx() * (mesh.ny() + 1))
    {
        setZero();
    }

    const CartesianMesh& mesh() const
    {
        return mesh_;
    }
    const std::string& name() const
    {
        return name_;
    }

    Index xFaceCount() const
    {
        return x_.size();
    }
    Index yFaceCount() const
    {
        return y_.size();
    }

    // Linear index of the x-face (i, j), i in [0, nx], j in [0, ny).
    Index xIndex(Index i, Index j) const
    {
        return j * (mesh_.nx() + 1) + i;
    }
    // Linear index of the y-face (i, j), i in [0, nx), j in [0, ny].
    Index yIndex(Index i, Index j) const
    {
        return j * mesh_.nx() + i;
    }

    // Access by face (i, j) coordinates
    Scalar& x(Index i, Index j)
    {
        return x_[xIndex(i, j)];
    }
    const Scalar& x(Index i, Index j) const
    {
        return x_[xIndex(i, j)];
    }
    Scalar& y(Index i, Index j)
    {
        return y_[yIndex(i, j)];
    }
    const Scalar& y(Index i, Index j) const
    {
        return y_[yIndex(i, j)];
    }

    // Raw data access
    Vector& xData()
    {
        return x_;
    }
    const Vector& xData() const
    {
        return x_;
    }
    Vector& yData()
    {
        return y_;
    }
    const Vector& yData() const
    {
        return y_;
    }

    /**
     * @brief Signed outward flux of `cell` across local `face`
     * (0=east, 1=north, 2=west, 3=south).
     *
     * East/north faces take +x/+y, west/south faces take -x/-y.
     */
    Scalar outwardFlux(Index cell, int face) const
    {
        const auto [i, j] = mesh_.cellIJ(cell);
        switch (face)
        {
            case 0: // east
                return x(i + 1, j);
            case 1: // north
                return y(i, j + 1);
            case 2: // west
                return -x(i, j);
            default: // south
                return -y(i, j);
        }
    }

    /**
     * @brief Net outflow of a cell: sum_f outwardFlux(cell, f).
     *
     * Zero (up to solver accuracy) for a conservative flux field.
     */
    Scalar cellImbalance(Index cell) const
    {
        const auto [i, j] = mesh_.cellIJ(cell);
        return x(i + 1, j) - x(i, j) + y(i, j + 1) - y(i, j);
    }

    void setZero()
    {
        x_.setZero();
        y_.setZero();
    }

private:
    const CartesianMesh& mesh_;
    std::string name_;
    Vector x_;
    Vector y_;
};

} // namespace fvm::core
