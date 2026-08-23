#pragma once

#include "Mesh.h"
#include "Types.h"
#include <string>

namespace fvm::core
{

/**
 * @brief Scalar field defined on cell centers.
 */
class ScalarField
{
public:
    ScalarField(const CartesianMesh& mesh, const std::string& name = "")
        : mesh_(mesh)
        , name_(name)
        , data_(mesh.cellCount())
    {
        data_.setZero();
    }

    const CartesianMesh& mesh() const
    {
        return mesh_;
    }
    const std::string& name() const
    {
        return name_;
    }

    // Access by linear cell index
    Scalar& operator()(Index cell)
    {
        return data_[cell];
    }
    const Scalar& operator()(Index cell) const
    {
        return data_[cell];
    }

    // Access by (i, j) coordinates
    Scalar& operator()(Index i, Index j)
    {
        return data_[mesh_.cellIndex(i, j)];
    }
    const Scalar& operator()(Index i, Index j) const
    {
        return data_[mesh_.cellIndex(i, j)];
    }

    // Raw data access
    Vector& data()
    {
        return data_;
    }
    const Vector& data() const
    {
        return data_;
    }

    Index size() const
    {
        return data_.size();
    }

    void setZero()
    {
        data_.setZero();
    }
    void setConstant(Scalar value)
    {
        data_.setConstant(value);
    }

private:
    const CartesianMesh& mesh_;
    std::string name_;
    Vector data_;
};

/**
 * @brief 2D vector field (u, v components) on cell centers.
 */
class VectorField
{
public:
    VectorField(const CartesianMesh& mesh, const std::string& name = "")
        : mesh_(mesh)
        , name_(name)
        , u_(mesh, name + "_u")
        , v_(mesh, name + "_v")
    {}

    const CartesianMesh& mesh() const
    {
        return mesh_;
    }
    const std::string& name() const
    {
        return name_;
    }

    ScalarField& u()
    {
        return u_;
    }
    const ScalarField& u() const
    {
        return u_;
    }
    ScalarField& v()
    {
        return v_;
    }
    const ScalarField& v() const
    {
        return v_;
    }

    // Access by linear cell index
    std::pair<Scalar, Scalar> operator()(Index cell) const
    {
        return { u_(cell), v_(cell) };
    }

    void setZero()
    {
        u_.setZero();
        v_.setZero();
    }

private:
    const CartesianMesh& mesh_;
    std::string name_;
    ScalarField u_;
    ScalarField v_;
};

} // namespace fvm::core
