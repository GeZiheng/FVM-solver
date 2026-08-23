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
