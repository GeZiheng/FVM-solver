#pragma once

#include "Types.h"
#include <Eigen/Sparse>
#include <vector>

using fvm::core::Index;
using fvm::core::Scalar;

namespace fvm::math {

/*
 * @brief Lightweight wrapper around Eigen sparse matrix.
 *
 * Provides a triplet-based assembly API. Internal storage is hidden;
 * LinearSolver implementations access the native Eigen matrix via
 * a dedicated accessor.
 */
class SparseMatrix {
public:
  SparseMatrix(Index rows, Index cols);

  void insert(Index row, Index col, Scalar value);
  void finalize();
  void setZero();

  Index rows() const;
  Index cols() const;
  Index nnz() const;

  /**
   * @brief Access underlying Eigen matrix (for solver implementations).
   */
  const Eigen::SparseMatrix<Scalar>& native() const;

private:
  Eigen::SparseMatrix<Scalar> mat_;
  std::vector<Eigen::Triplet<Scalar>> triplets_;
  bool finalized_ = false;
};

} // namespace fvm::math
