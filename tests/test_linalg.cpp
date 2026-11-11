#include "LinearSolver.h"
#include "SparseMatrix.h"
#include <cmath>
#include <doctest/doctest.h>

using namespace fvm::core;
using namespace fvm::math;

TEST_CASE("SparseMatrix construction and properties")
{
    SparseMatrix A(3, 3);
    A.insert(0, 0, 2.0);
    A.insert(0, 1, -1.0);
    A.insert(1, 0, -1.0);
    A.insert(1, 1, 2.0);
    A.insert(1, 2, -1.0);
    A.insert(2, 1, -1.0);
    A.insert(2, 2, 2.0);
    A.finalize();

    CHECK(A.rows() == 3);
    CHECK(A.cols() == 3);
    CHECK(A.nnz() == 7);
}

TEST_CASE("EigenSparseLU solves tridiagonal system")
{
    // 1D Poisson: -u'' = f with Dirichlet BCs
    // Discretized as: -u_{i-1} + 2u_i - u_{i+1} = h^2 * f_i
    int n = 5;
    SparseMatrix A(n, n);
    // Interior rows: standard 1D Poisson stencil
    for (int i = 1; i < n - 1; ++i)
    {
        A.insert(i, i, 2.0);
        A.insert(i, i - 1, -1.0);
        A.insert(i, i + 1, -1.0);
    }
    // Dirichlet BC rows: identity
    A.insert(0, 0, 1.0);
    A.insert(n - 1, n - 1, 1.0);
    A.finalize();

    Vector b(n);
    b.setZero();
    b[0] = 1.0;     // Dirichlet at left
    b[n - 1] = 0.0; // Dirichlet at right

    auto solver = createEigenSparseLU();
    Vector x = solver->solve(A, b);

    // Check boundary conditions
    CHECK(x[0] == doctest::Approx(1.0));
    CHECK(x[n - 1] == doctest::Approx(0.0));

    // Check intermediate values are monotonically decreasing
    for (int i = 1; i < n; ++i)
    {
        CHECK(x[i] <= x[i - 1]);
    }
}

TEST_CASE("EigenCG solves SPD system")
{
    // Same tridiagonal system (symmetric positive definite)
    int n = 10;
    SparseMatrix A(n, n);
    for (int i = 0; i < n; ++i)
    {
        A.insert(i, i, 2.0);
        if (i > 0)
            A.insert(i, i - 1, -1.0);
        if (i + 1 < n)
            A.insert(i, i + 1, -1.0);
    }
    A.finalize();

    Vector b = Vector::Random(n);

    auto solver = createEigenCG();
    Vector x = solver->solve(A, b);

    // Verify residual
    Vector residual = A.native() * x - b;
    CHECK(residual.norm() < 1e-10);
    CHECK(solver->lastIterations() > 0);
    CHECK(solver->lastIterations()
          < n); // Should converge in < n iterations for this simple system
}

TEST_CASE("EigenBiCGSTAB solves non-symmetric system")
{
    // Create a non-symmetric matrix
    int n = 5;
    SparseMatrix A(n, n);
    for (int i = 0; i < n; ++i)
    {
        A.insert(i, i, 3.0);
        if (i > 0)
            A.insert(i, i - 1, -1.0);
        if (i + 1 < n)
            A.insert(i, i + 1, -2.0); // asymmetric
    }
    A.finalize();

    Vector b = Vector::Random(n);

    auto solver = createEigenBiCGSTAB();
    Vector x = solver->solve(A, b);

    Vector residual = A.native() * x - b;
    CHECK(residual.norm() < 1e-10);
    CHECK(solver->lastIterations() > 0);
}

TEST_CASE("Solver dimension mismatch throws")
{
    SparseMatrix A(3, 3);
    A.insert(0, 0, 1.0);
    A.insert(1, 1, 1.0);
    A.insert(2, 2, 1.0);
    A.finalize();

    Vector b(2); // Wrong size

    auto solver = createEigenSparseLU();
    CHECK_THROWS_AS(solver->solve(A, b), std::invalid_argument);
}

TEST_CASE("SparseMatrix unfinalized access throws")
{
    SparseMatrix A(2, 2);
    A.insert(0, 0, 1.0);
    // Not finalized

    CHECK_THROWS_AS(A.native(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// makeLinearSolver dispatch (backend x method)
// ---------------------------------------------------------------------------

namespace
{

// SPD tridiagonal system (2 on the diagonal, -1 off): the shape the Eigen
// solvers in this file are exercised on.
SparseMatrix makeSpdTridiagonal(int n)
{
    SparseMatrix A(n, n);
    for (int i = 0; i < n; ++i)
    {
        A.insert(i, i, 2.0);
        if (i > 0)
            A.insert(i, i - 1, -1.0);
        if (i + 1 < n)
            A.insert(i, i + 1, -1.0);
    }
    A.finalize();
    return A;
}

// Non-symmetric tridiagonal system (asymmetric super-diagonal), the
// convection-type shape EigenBiCGSTAB is exercised on above.
SparseMatrix makeNonSymmetric(int n)
{
    SparseMatrix A(n, n);
    for (int i = 0; i < n; ++i)
    {
        A.insert(i, i, 3.0);
        if (i > 0)
            A.insert(i, i - 1, -1.0);
        if (i + 1 < n)
            A.insert(i, i + 1, -2.0); // asymmetric
    }
    A.finalize();
    return A;
}

Vector makeExactSolution(int n)
{
    Vector x(n);
    for (int i = 0; i < n; ++i)
    {
        x[i] = std::cos(0.7 * i) + 1.0;
    }
    return x;
}

} // namespace

TEST_CASE("makeLinearSolver default config matches direct factory")
{
    // Default SolverConfig is eigen x bicgstab.
    int n = 10;
    SparseMatrix A = makeSpdTridiagonal(n);

    Vector x_exact = makeExactSolution(n);
    Vector b = A.native() * x_exact;

    SolverConfig cfg; // backend "eigen", method "bicgstab"
    auto dispatched = makeLinearSolver(cfg);
    auto direct = createEigenBiCGSTAB(cfg);

    Vector xDispatch = dispatched->solve(A, b);
    Vector xDirect = direct->solve(A, b);

    // Same deterministic solver on the same input: identical results.
    CHECK((xDispatch - xDirect).norm() < 1e-12);

    const double solutionError = (xDispatch - x_exact).cwiseAbs().maxCoeff();
    CHECK(solutionError < 1e-6);
}

TEST_CASE("makeLinearSolver dispatches eigen cg and bicgstab_ilut")
{
    // SPD system for cg.
    int n = 10;
    SparseMatrix A = makeSpdTridiagonal(n);
    Vector x_exact = makeExactSolution(n);
    Vector b = A.native() * x_exact;

    SolverConfig cgCfg;
    cgCfg.method = "cg";
    cgCfg.tolerance = 1e-10;
    auto cgSolver = makeLinearSolver(cgCfg);
    Vector x = cgSolver->solve(A, b);

    CHECK((x - x_exact).cwiseAbs().maxCoeff() < 1e-6);
    CHECK(cgSolver->lastIterations() >= 0);

    // Non-symmetric (convection-type) system for bicgstab_ilut.
    int m = 8;
    SparseMatrix B = makeNonSymmetric(m);
    Vector x2_exact = makeExactSolution(m);
    Vector b2 = B.native() * x2_exact;

    SolverConfig ilutCfg;
    ilutCfg.method = "bicgstab_ilut";
    ilutCfg.tolerance = 1e-10;
    auto ilutSolver = makeLinearSolver(ilutCfg);
    Vector x2 = ilutSolver->solve(B, b2);

    CHECK((x2 - x2_exact).cwiseAbs().maxCoeff() < 1e-6);
    CHECK(ilutSolver->lastIterations() >= 0);
}

TEST_CASE("makeLinearSolver rejects unknown backend and method")
{
    SolverConfig badBackend;
    badBackend.backend = "hypre";
    CHECK_THROWS_AS(makeLinearSolver(badBackend), std::invalid_argument);

    SolverConfig badMethod;
    badMethod.method = "gmres";
    CHECK_THROWS_AS(makeLinearSolver(badMethod), std::invalid_argument);
}

TEST_CASE("makeLinearSolver amgcl backend gating")
{
    // The amgcl family exists only in the FVM_BUILD_AMGCL=ON build (which
    // defines FVM_HAVE_AMGCL): this case compiles in both configurations and
    // checks the family-level gate, like the file-level gating of
    // test_amgcl_solver.cpp.
#ifdef FVM_HAVE_AMGCL
    int n = 10;
    SparseMatrix A = makeSpdTridiagonal(n);
    Vector x_exact = makeExactSolution(n);
    Vector b = A.native() * x_exact;

    SolverConfig cfg;
    cfg.backend = "amgcl";
    cfg.method = "cg";
    cfg.tolerance = 1e-10;
    auto solver = makeLinearSolver(cfg);
    Vector x = solver->solve(A, b);

    CHECK((x - x_exact).cwiseAbs().maxCoeff() < 1e-6);

    // amgcl x bicgstab dispatches and solves (the SPD system works for both
    // Krylov methods); an unknown method is rejected in the ON build too.
    cfg.method = "bicgstab";
    auto solverBicg = makeLinearSolver(cfg);
    Vector xBicg = solverBicg->solve(A, b);
    CHECK((xBicg - x_exact).cwiseAbs().maxCoeff() < 1e-6);

    cfg.method = "bogus";
    CHECK_THROWS_AS(makeLinearSolver(cfg), std::invalid_argument);
#else
    // OFF build: amgcl is rejected with std::runtime_error before the method
    // is even looked at, for any method value.
    SolverConfig cfg;
    cfg.backend = "amgcl";
    cfg.method = "cg";
    CHECK_THROWS_AS(makeLinearSolver(cfg), std::runtime_error);

    cfg.method = "bicgstab_ilut";
    CHECK_THROWS_AS(makeLinearSolver(cfg), std::runtime_error);
#endif
}
