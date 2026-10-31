#pragma once

#include <Eigen/Core>
#include <cstddef>

namespace fvm::core
{

using Scalar = double;
using Index = std::size_t;
using Vector = Eigen::VectorXd;

/// A point in the 2D computational domain.
struct Point2
{
    Scalar x = 0.0;
    Scalar y = 0.0;
};

/// A non-owning view over a contiguous sequence of indices (C++17).
struct IndexSpan
{
    IndexSpan() = default;
    IndexSpan(const Index* data, std::size_t size)
        : data_(data)
        , size_(size)
    {
    }

    const Index* begin() const
    {
        return data_;
    }

    const Index* end() const
    {
        return data_ + size_;
    }

    std::size_t size() const
    {
        return size_;
    }

    bool empty() const
    {
        return size_ == 0;
    }

    const Index& operator[](std::size_t position) const
    {
        return data_[position];
    }

private:
    const Index* data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace fvm::core
