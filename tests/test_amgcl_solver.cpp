#ifdef FVM_HAVE_AMGCL
// AMGCL solver tests (Phase 5.2). Compiled only when FVM_BUILD_AMGCL=ON
// defines FVM_HAVE_AMGCL; the file contributes TEST_CASEs to the shared
// doctest binary (tests/main.cpp owns the single main()) and has no main()
// of its own, so it compiles to nothing in the default OFF configuration.

#include "LinearSolver.h"
#include "SparseMatrix.h"

#include <cmath>
#include <doctest/doctest.h>

namespace
{

// SPD tridiagonal system (2 on the diagonal, -1 off): the same shape the
// Eigen solvers are exercised on in test_linalg.cpp.
fvm::math::SparseMatrix makeSpdTridiagonal(int n)
{
    fvm::math::SparseMatrix A(n, n);
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

// Non-symmetric tridiagonal system (asymmetric super-diagonal), the same
// shape as test_linalg.cpp's EigenBiCGSTAB case.
fvm::math::SparseMatrix makeNonSymmetric(int n)
{
    fvm::math::SparseMatrix A(n, n);
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

} // namespace

TEST_CASE("AmgclCG solves SPD system")
{
    using namespace fvm::core;
    using namespace fvm::math;

    const int n = 10;
    SparseMatrix A = makeSpdTridiagonal(n);

    // Pick the exact solution first and form the matching right-hand side,
    // so the solution error can be checked directly.
    Vector x_exact(n);
    for (int i = 0; i < n; ++i)
    {
        x_exact[i] = std::cos(0.7 * i) + 1.0;
    }
    Vector b = A.native() * x_exact;

    SolverConfig cfg;
    cfg.tolerance = 1e-10;
    auto solver = createAmgclCG(cfg);
    Vector x = solver->solve(A, b);

    const double solutionError = (x - x_exact).cwiseAbs().maxCoeff();
    CHECK(solutionError < 1e-6);
    CHECK(solver->lastResidual() <= cfg.tolerance);
    CHECK(solver->lastIterations() >= 1);
}

TEST_CASE("AmgclBiCGSTAB solves non-symmetric system")
{
    using namespace fvm::core;
    using namespace fvm::math;

    const int n = 5;
    SparseMatrix A = makeNonSymmetric(n);

    Vector x_exact(n);
    for (int i = 0; i < n; ++i)
    {
        x_exact[i] = std::cos(0.7 * i) + 1.0;
    }
    Vector b = A.native() * x_exact;

    SolverConfig cfg;
    cfg.tolerance = 1e-10;
    const int maxIterations = 100;
    cfg.maxIterations = maxIterations;
    auto solver = createAmgclBiCGSTAB(cfg);
    Vector x = solver->solve(A, b);

    const double solutionError = (x - x_exact).cwiseAbs().maxCoeff();
    CHECK(solutionError < 1e-6);
    CHECK(solver->lastIterations() >= 1);
    CHECK(solver->lastIterations() <= maxIterations);
}

TEST_CASE("Amgcl solver dimension mismatch throws")
{
    using namespace fvm::core;
    using namespace fvm::math;

    SparseMatrix A(3, 3);
    A.insert(0, 0, 1.0);
    A.insert(1, 1, 1.0);
    A.insert(2, 2, 1.0);
    A.finalize();

    Vector b(2); // Wrong size

    auto solver = createAmgclCG();
    CHECK_THROWS_AS(solver->solve(A, b), std::invalid_argument);
}

#endif // FVM_HAVE_AMGCL
