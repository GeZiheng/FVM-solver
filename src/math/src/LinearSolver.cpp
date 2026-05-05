#include "LinearSolver.h"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <iostream>
#include <stdexcept>

namespace fvm::math {

namespace {

// --------------------------------------------------------------------------
// Eigen BiCGSTAB
// --------------------------------------------------------------------------
class EigenBiCGSTABSolver : public LinearSolver {
public:
  Vector solve(const SparseMatrix& A, const Vector& b) override
  {
    const auto& native = A.native();
    if (native.rows() != b.size()) {
      throw std::invalid_argument("Matrix and vector dimensions mismatch");
    }

    Eigen::BiCGSTAB<Eigen::SparseMatrix<Scalar>, Eigen::DiagonalPreconditioner<Scalar>> solver;
    solver.setMaxIterations(config_.maxIterations);
    solver.setTolerance(config_.tolerance);
    solver.compute(native);

    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen BiCGSTAB: factorization failed");
    }

    Vector x = solver.solve(b);
    lastIterations_ = static_cast<int>(solver.iterations());
    lastResidual_ = solver.error();

    if (config_.verbose) {
      std::cout << "BiCGSTAB: iterations=" << lastIterations_
                << ", residual=" << lastResidual_ << std::endl;
    }

    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen BiCGSTAB: solve did not converge");
    }

    return x;
  }
};

// --------------------------------------------------------------------------
// Eigen Conjugate Gradient
// --------------------------------------------------------------------------
class EigenCGSolver : public LinearSolver {
public:
  Vector solve(const SparseMatrix& A, const Vector& b) override
  {
    const auto& native = A.native();
    if (native.rows() != b.size()) {
      throw std::invalid_argument("Matrix and vector dimensions mismatch");
    }

    Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar>, Eigen::Lower | Eigen::Upper,
                             Eigen::DiagonalPreconditioner<Scalar>>
      solver;
    solver.setMaxIterations(config_.maxIterations);
    solver.setTolerance(config_.tolerance);
    solver.compute(native);

    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen CG: factorization failed");
    }

    Vector x = solver.solve(b);
    lastIterations_ = static_cast<int>(solver.iterations());
    lastResidual_ = solver.error();

    if (config_.verbose) {
      std::cout << "CG: iterations=" << lastIterations_
                << ", residual=" << lastResidual_ << std::endl;
    }

    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen CG: solve did not converge");
    }

    return x;
  }
};

// --------------------------------------------------------------------------
// Eigen SparseLU (direct solver)
// --------------------------------------------------------------------------
class EigenSparseLUSolver : public LinearSolver {
public:
  Vector solve(const SparseMatrix& A, const Vector& b) override
  {
    const auto& native = A.native();
    if (native.rows() != b.size()) {
      throw std::invalid_argument("Matrix and vector dimensions mismatch");
    }

    Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
    solver.compute(native);

    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Eigen SparseLU: factorization failed");
    }

    Vector x = solver.solve(b);
    lastIterations_ = 1;
    lastResidual_ = (native * x - b).norm() / b.norm();

    if (config_.verbose) {
      std::cout << "SparseLU: residual=" << lastResidual_ << std::endl;
    }

    return x;
  }
};

} // anonymous namespace

// Factory implementations --------------------------------------------------

std::unique_ptr<LinearSolver> createEigenBiCGSTAB(const SolverConfig& cfg)
{
  auto solver = std::make_unique<EigenBiCGSTABSolver>();
  solver->setConfig(cfg);
  return solver;
}

std::unique_ptr<LinearSolver> createEigenCG(const SolverConfig& cfg)
{
  auto solver = std::make_unique<EigenCGSolver>();
  solver->setConfig(cfg);
  return solver;
}

std::unique_ptr<LinearSolver> createEigenSparseLU(const SolverConfig& cfg)
{
  auto solver = std::make_unique<EigenSparseLUSolver>();
  solver->setConfig(cfg);
  return solver;
}

} // namespace fvm::math
