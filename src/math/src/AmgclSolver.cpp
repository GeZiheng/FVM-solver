#ifdef FVM_HAVE_AMGCL
// AMGCL-backed linear solvers (Phase 5.2). The whole translation unit is
// compiled only when FVM_BUILD_AMGCL=ON defines FVM_HAVE_AMGCL, so the
// heavy amgcl headers never enter the default (Eigen-only) build.

#include "LinearSolver.h"

// amgcl's default parameter plumbing pulls in boost::property_tree through
// amgcl/util.hpp; the plain-struct params used below do not need it, so the
// overlay port stays Boost-free (eigen3 only). Must precede every amgcl
// include in this translation unit.
#define AMGCL_NO_BOOST

#include <amgcl/amg.hpp>
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/cg.hpp>

#include <iostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>

// Recognize fvm::core::Vector (Eigen::VectorXd) as a builtin-backend vector,
// so it can be passed to the solve call directly (RHS in, solution out).
AMGCL_USE_EIGEN_VECTORS_WITH_BUILTIN_BACKEND()

namespace fvm::math
{

namespace
{

// The amgcl builtin backend below is instantiated for double; keep Scalar
// pinned so the factories cannot silently mis-instantiate.
static_assert(std::is_same<Scalar, double>::value,
              "AMGCL builtin backend requires Scalar == double");

using AmgclBackend = amgcl::backend::builtin<Scalar>;

// --------------------------------------------------------------------------
// AMGCL Conjugate Gradient (SPD systems, e.g. the pressure equation)
// --------------------------------------------------------------------------
class AmgclCGSolver : public LinearSolver
{
public:
    Vector solve(const SparseMatrix& A, const Vector& b) override
    {
        const auto& native = A.native();
        if (native.rows() != native.cols() || native.rows() != b.size())
        {
            throw std::invalid_argument("Matrix and vector dimensions mismatch");
        }

        using Solver = amgcl::make_solver<
            amgcl::amg<AmgclBackend, amgcl::coarsening::smoothed_aggregation,
                       amgcl::relaxation::spai0>,
            amgcl::solver::cg<AmgclBackend>>;

        Solver::params prm;
        prm.solver.maxiter = static_cast<std::size_t>(config_.maxIterations);
        // amgcl's returned error is ||r||/||b||: the same relative-residual
        // semantics as SolverConfig::tolerance.
        prm.solver.tol = config_.tolerance;

        // One storage-order conversion ColMajor -> RowMajor; amgcl's builtin
        // backend copies the matrix row by row, which is direct on a
        // row-major matrix.
        Eigen::SparseMatrix<Scalar, Eigen::RowMajor> rowMajor(native);

        // Stateless like the Eigen wrappers: the preconditioner hierarchy is
        // rebuilt for every solve() from the matrix passed in.
        Solver solver(rowMajor, prm);
        Vector x = Vector::Zero(b.size());
        const auto info = solver(b, x);
        lastIterations_ = static_cast<int>(std::get<0>(info));
        lastResidual_ = std::get<1>(info);

        if (config_.verbose)
        {
            std::cout << "AMGCL CG: iterations=" << lastIterations_
                      << ", residual=" << lastResidual_ << std::endl;
        }

        if (lastResidual_ > config_.tolerance)
        {
            throw std::runtime_error("AMGCL CG: solve did not converge");
        }

        return x;
    }
};

// --------------------------------------------------------------------------
// AMGCL BiCGSTAB (non-symmetric systems, e.g. the momentum equation)
// --------------------------------------------------------------------------
class AmgclBiCGSTABSolver : public LinearSolver
{
public:
    Vector solve(const SparseMatrix& A, const Vector& b) override
    {
        const auto& native = A.native();
        if (native.rows() != native.cols() || native.rows() != b.size())
        {
            throw std::invalid_argument("Matrix and vector dimensions mismatch");
        }

        using Solver = amgcl::make_solver<
            amgcl::amg<AmgclBackend, amgcl::coarsening::smoothed_aggregation,
                       amgcl::relaxation::ilu0>,
            amgcl::solver::bicgstab<AmgclBackend>>;

        Solver::params prm;
        prm.solver.maxiter = static_cast<std::size_t>(config_.maxIterations);
        // amgcl's returned error is ||r||/||b||: the same relative-residual
        // semantics as SolverConfig::tolerance.
        prm.solver.tol = config_.tolerance;

        // One storage-order conversion ColMajor -> RowMajor; amgcl's builtin
        // backend copies the matrix row by row, which is direct on a
        // row-major matrix.
        Eigen::SparseMatrix<Scalar, Eigen::RowMajor> rowMajor(native);

        // Stateless like the Eigen wrappers: the preconditioner hierarchy is
        // rebuilt for every solve() from the matrix passed in.
        Solver solver(rowMajor, prm);
        Vector x = Vector::Zero(b.size());
        const auto info = solver(b, x);
        lastIterations_ = static_cast<int>(std::get<0>(info));
        lastResidual_ = std::get<1>(info);

        if (config_.verbose)
        {
            std::cout << "AMGCL BiCGSTAB: iterations=" << lastIterations_
                      << ", residual=" << lastResidual_ << std::endl;
        }

        if (lastResidual_ > config_.tolerance)
        {
            throw std::runtime_error("AMGCL BiCGSTAB: solve did not converge");
        }

        return x;
    }
};

} // anonymous namespace

// Factory implementations --------------------------------------------------

std::unique_ptr<LinearSolver> createAmgclCG(const SolverConfig& cfg)
{
    auto solver = std::make_unique<AmgclCGSolver>();
    solver->setConfig(cfg);
    return solver;
}

std::unique_ptr<LinearSolver> createAmgclBiCGSTAB(const SolverConfig& cfg)
{
    auto solver = std::make_unique<AmgclBiCGSTABSolver>();
    solver->setConfig(cfg);
    return solver;
}

} // namespace fvm::math

#endif // FVM_HAVE_AMGCL
