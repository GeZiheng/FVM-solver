#include "SparseMatrix.h"
#include <stdexcept>

namespace fvm::math
{

SparseMatrix::SparseMatrix(Index rows, Index cols)
    : mat_(rows, cols)
{}

void SparseMatrix::insert(Index row, Index col, Scalar value)
{
    if (finalized_)
    {
        throw std::runtime_error("Cannot insert into finalized sparse matrix");
    }
    triplets_.emplace_back(static_cast<Eigen::Index>(row),
        static_cast<Eigen::Index>(col),
        value);
}

void SparseMatrix::finalize()
{
    if (finalized_)
        return;
    mat_.setFromTriplets(triplets_.begin(), triplets_.end());
    triplets_.clear();
    finalized_ = true;
}

void SparseMatrix::setZero()
{
    mat_.setZero();
    triplets_.clear();
    finalized_ = false;
}

Index SparseMatrix::rows() const
{
    return static_cast<Index>(mat_.rows());
}

Index SparseMatrix::cols() const
{
    return static_cast<Index>(mat_.cols());
}

Index SparseMatrix::nnz() const
{
    return static_cast<Index>(mat_.nonZeros());
}

const Eigen::SparseMatrix<Scalar>& SparseMatrix::native() const
{
    if (!finalized_)
    {
        throw std::runtime_error(
            "SparseMatrix must be finalized before accessing native matrix");
    }
    return mat_;
}

} // namespace fvm::math
