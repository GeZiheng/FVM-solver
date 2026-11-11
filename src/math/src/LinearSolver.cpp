#include "LinearSolver.h"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <iostream>
#include <stdexcept>

namespace fvm::math
{

namespace
{

// --------------------------------------------------------------------------
// Eigen BiCGSTAB
// --------------------------------------------------------------------------
class EigenBiCGSTABSolver : public LinearSolver
{
public:
    Vector solve(const SparseMatrix& A, const Vector& b) override
    {
        const auto& native = A.native();
        if (native.rows() != b.size())
        {
            throw std::
                invalid_argument("Matrix and vector dimensions mismatch");
        }

        Eigen::BiCGSTAB<Eigen::SparseMatrix<Scalar>,
            Eigen::DiagonalPreconditioner<Scalar>>
            solver;
        solver.setMaxIterations(config_.maxIterations);
        // Eigen 5.x BiCGSTAB stops on the ABSOLUTE residual norm and its
        // convergence flag compares mismatched quantities. Scale the tolerance
        // by |b| so that SolverConfig::tolerance keeps its documented meaning
        // (relative residual |Ax-b|/|b|), and check convergence ourselves below.
        const Scalar rhsNorm = b.stableNorm();
        solver.setTolerance(rhsNorm > 0.0 ? config_.tolerance * rhsNorm
                                          : config_.tolerance);
        solver.compute(native);

        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("Eigen BiCGSTAB: factorization failed");
        }

        Vector x = solver.solve(b);
        lastIterations_ = static_cast<int>(solver.iterations());
        lastResidual_ = solver.error(); // relative residual estimate |Ax-b|/|b|

        if (config_.verbose)
        {
            std::cout << "BiCGSTAB: iterations=" << lastIterations_
                      << ", residual=" << lastResidual_ << std::endl;
        }

        if (solver.info() == Eigen::NumericalIssue)
        {
            throw std::runtime_error("Eigen BiCGSTAB: numerical breakdown");
        }
        if (lastResidual_ > config_.tolerance)
        {
            throw std::runtime_error("Eigen BiCGSTAB: solve did not converge");
        }

        return x;
    }
};

// --------------------------------------------------------------------------
// Eigen BiCGSTAB with ILUT preconditioner
// --------------------------------------------------------------------------
class EigenBiCGSTABILUTSolver : public LinearSolver
{
public:
    Vector solve(const SparseMatrix& A, const Vector& b) override
    {
        const auto& native = A.native();
        if (native.rows() != b.size())
        {
            throw std::invalid_argument("Matrix and vector dimensions mismatch");
        }

        Eigen::BiCGSTAB<Eigen::SparseMatrix<Scalar>,
            Eigen::IncompleteLUT<Scalar>>
            solver; // fillfactor stays at the Eigen default
        solver.setMaxIterations(config_.maxIterations);
        // Same relative-residual contract as the plain BiCGSTAB wrapper:
        // Eigen 5.x BiCGSTAB stops on the ABSOLUTE residual norm and its
        // convergence flag compares mismatched quantities. Scale the
        // tolerance by |b| so that SolverConfig::tolerance keeps its
        // documented meaning (relative residual |Ax-b|/|b|), and check
        // convergence ourselves below.
        const Scalar rhsNorm = b.stableNorm();
        solver.setTolerance(rhsNorm > 0.0 ? config_.tolerance * rhsNorm
                                          : config_.tolerance);
        solver.compute(native);

        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error(
                "Eigen BiCGSTAB(ILUT): factorization failed");
        }

        Vector x = solver.solve(b);
        lastIterations_ = static_cast<int>(solver.iterations());
        lastResidual_ = solver.error(); // relative residual estimate |Ax-b|/|b|

        if (config_.verbose)
        {
            std::cout << "BiCGSTAB(ILUT): iterations=" << lastIterations_
                      << ", residual=" << lastResidual_ << std::endl;
        }

        if (solver.info() == Eigen::NumericalIssue)
        {
            throw std::runtime_error("Eigen BiCGSTAB(ILUT): numerical breakdown");
        }
        if (lastResidual_ > config_.tolerance)
        {
            throw std::runtime_error(
                "Eigen BiCGSTAB(ILUT): solve did not converge");
        }

        return x;
    }
};

// --------------------------------------------------------------------------
// Eigen Conjugate Gradient
// --------------------------------------------------------------------------
class EigenCGSolver : public LinearSolver
{
public:
    Vector solve(const SparseMatrix& A, const Vector& b) override
    {
        const auto& native = A.native();
        if (native.rows() != b.size())
        {
            throw std::
                invalid_argument("Matrix and vector dimensions mismatch");
        }

        Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar>,
            Eigen::Lower | Eigen::Upper,
            Eigen::DiagonalPreconditioner<Scalar>>
            solver;
        solver.setMaxIterations(config_.maxIterations);
        solver.setTolerance(config_.tolerance);
        solver.compute(native);

        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("Eigen CG: factorization failed");
        }

        Vector x = solver.solve(b);
        lastIterations_ = static_cast<int>(solver.iterations());
        lastResidual_ = solver.error();

        if (config_.verbose)
        {
            std::cout << "CG: iterations=" << lastIterations_
                      << ", residual=" << lastResidual_ << std::endl;
        }

        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("Eigen CG: solve did not converge");
        }

        return x;
    }
};

// --------------------------------------------------------------------------
// Eigen SparseLU (direct solver)
// --------------------------------------------------------------------------
class EigenSparseLUSolver : public LinearSolver
{
public:
    Vector solve(const SparseMatrix& A, const Vector& b) override
    {
        const auto& native = A.native();
        if (native.rows() != b.size())
        {
            throw std::
                invalid_argument("Matrix and vector dimensions mismatch");
        }

        Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
        solver.compute(native);

        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("Eigen SparseLU: factorization failed");
        }

        Vector x = solver.solve(b);
        lastIterations_ = 1;
        lastResidual_ = (native * x - b).norm() / b.norm();

        if (config_.verbose)
        {
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

std::unique_ptr<LinearSolver> createEigenBiCGSTABILUT(const SolverConfig& cfg)
{
    auto solver = std::make_unique<EigenBiCGSTABILUTSolver>();
    solver->setConfig(cfg);
    return solver;
}

// Dispatch: backend x method ------------------------------------------------

std::unique_ptr<LinearSolver> makeLinearSolver(const SolverConfig& cfg)
{
    if (cfg.backend == "eigen")
    {
        if (cfg.method == "bicgstab")
        {
            return createEigenBiCGSTAB(cfg);
        }
        if (cfg.method == "cg")
        {
            return createEigenCG(cfg);
        }
        if (cfg.method == "bicgstab_ilut")
        {
            return createEigenBiCGSTABILUT(cfg);
        }
    }
    else if (cfg.backend == "amgcl")
    {
#ifdef FVM_HAVE_AMGCL
        if (cfg.method == "cg")
        {
            return createAmgclCG(cfg);
        }
        if (cfg.method == "bicgstab")
        {
            return createAmgclBiCGSTAB(cfg);
        }
#else
        // Family-level gate, independent of the method: the OFF build
        // rejects amgcl before the method is even looked at.
        throw std::runtime_error(
            "backend 'amgcl' requires building the project with "
            "FVM_BUILD_AMGCL=ON");
#endif
    }
    else
    {
        throw std::invalid_argument(
            "unknown solver backend '" + cfg.backend
            + "': expected \"eigen\" or \"amgcl\"");
    }
    throw std::invalid_argument(
        "unknown solver method '" + cfg.method
        + "': expected \"bicgstab\", \"cg\" or \"bicgstab_ilut\" (eigen only)");
}

} // namespace fvm::math
