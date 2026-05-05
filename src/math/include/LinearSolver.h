#pragma once

#include "SparseMatrix.h"
#include "Types.h"
#include <memory>
#include <string>

using fvm::core::Scalar;
using fvm::core::Vector;

namespace fvm::math {

struct SolverConfig {
  Scalar tolerance = 1e-6;
  int maxIterations = 1000;
  bool verbose = false;
};

/**
 * @brief Abstract interface for linear solvers.
 *
 * Implementations hide the backend library (Eigen, AMGCL, etc.).
 * Use factory functions to create concrete instances.
 */
class LinearSolver {
public:
  virtual ~LinearSolver() = default;

  /**
   * @brief Solve A * x = b.
   * @return Solution vector x.
   */
  virtual Vector solve(const SparseMatrix& A, const Vector& b) = 0;

  void setConfig(const SolverConfig& cfg) { config_ = cfg; }
  const SolverConfig& config() const { return config_; }

  int lastIterations() const { return lastIterations_; }
  Scalar lastResidual() const { return lastResidual_; }

protected:
  SolverConfig config_;
  int lastIterations_ = -1;
  Scalar lastResidual_ = -1.0;
};

// Factory functions ---------------------------------------------------------

std::unique_ptr<LinearSolver> createEigenBiCGSTAB(const SolverConfig& cfg = {});
std::unique_ptr<LinearSolver> createEigenCG(const SolverConfig& cfg = {});
std::unique_ptr<LinearSolver> createEigenSparseLU(const SolverConfig& cfg = {});

} // namespace fvm::math
