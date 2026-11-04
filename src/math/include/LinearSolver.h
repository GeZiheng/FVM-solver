#pragma once

#include "SparseMatrix.h"
#include "Types.h"
#include <memory>
#include <string>

namespace fvm::math
{
using fvm::core::Scalar;
using fvm::core::Vector;

struct SolverConfig
{
    Scalar tolerance = 1e-6;
    int maxIterations = 1000;
    bool verbose = false;
    // "eigen" default; "amgcl-cg"/"amgcl-bicgstab" need FVM_BUILD_AMGCL=ON.
    std::string backend = "eigen";
};

/**
 * @brief Abstract interface for linear solvers.
 *
 * Implementations hide the backend library (Eigen, AMGCL, etc.).
 * Use factory functions to create concrete instances.
 */
class LinearSolver
{
public:
    virtual ~LinearSolver() = default;

    /**
     * @brief Solve A * x = b.
     * @return Solution vector x.
     */
    virtual Vector solve(const SparseMatrix& A, const Vector& b) = 0;

    void setConfig(const SolverConfig& cfg)
    {
        config_ = cfg;
    }
    const SolverConfig& config() const
    {
        return config_;
    }

    int lastIterations() const
    {
        return lastIterations_;
    }
    Scalar lastResidual() const
    {
        return lastResidual_;
    }

protected:
    SolverConfig config_;
    int lastIterations_ = -1;
    Scalar lastResidual_ = -1.0;
};

// Factory functions ---------------------------------------------------------

std::unique_ptr<LinearSolver> createEigenBiCGSTAB(const SolverConfig& cfg = {});
std::unique_ptr<LinearSolver> createEigenCG(const SolverConfig& cfg = {});
std::unique_ptr<LinearSolver> createEigenSparseLU(const SolverConfig& cfg = {});

// AMGCL-backed factories, available only when the project is configured
// with FVM_BUILD_AMGCL=ON (which defines FVM_HAVE_AMGCL).
#ifdef FVM_HAVE_AMGCL
std::unique_ptr<LinearSolver> createAmgclCG(const SolverConfig& cfg = {});
std::unique_ptr<LinearSolver> createAmgclBiCGSTAB(const SolverConfig& cfg = {});
#endif

} // namespace fvm::math
