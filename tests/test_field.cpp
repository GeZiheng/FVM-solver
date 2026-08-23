#include "Field.h"
#include "Mesh.h"
#include <doctest/doctest.h>

using namespace fvm::core;

TEST_CASE("ScalarField construction and access")
{
    CartesianMesh mesh(3, 3, 0.0, 0.0, 1.0, 1.0);
    ScalarField phi(mesh, "phi");

    CHECK(phi.size() == mesh.cellCount());
    CHECK(phi.name() == "phi");

    phi(1, 2) = 3.14;
    CHECK(phi(mesh.cellIndex(1, 2)) == doctest::Approx(3.14));
}

TEST_CASE("ScalarField set operations")
{
    CartesianMesh mesh(2, 2, 0.0, 0.0, 1.0, 1.0);
    ScalarField phi(mesh, "phi");

    phi.setConstant(5.0);
    for (Index i = 0; i < mesh.cellCount(); ++i)
    {
        CHECK(phi(i) == doctest::Approx(5.0));
    }

    phi.setZero();
    for (Index i = 0; i < mesh.cellCount(); ++i)
    {
        CHECK(phi(i) == doctest::Approx(0.0));
    }
}

TEST_CASE("VectorField construction and access")
{
    CartesianMesh mesh(3, 3, 0.0, 0.0, 1.0, 1.0);
    VectorField vel(mesh, "velocity");

    CHECK(vel.u().size() == mesh.cellCount());
    CHECK(vel.v().size() == mesh.cellCount());

    vel.u()(1, 1) = 1.0;
    vel.v()(1, 1) = 2.0;

    auto [u, v] = vel(mesh.cellIndex(1, 1));
    CHECK(u == doctest::Approx(1.0));
    CHECK(v == doctest::Approx(2.0));
}

TEST_CASE("VectorField set zero")
{
    CartesianMesh mesh(2, 2, 0.0, 0.0, 1.0, 1.0);
    VectorField vel(mesh, "velocity");

    vel.u().setConstant(1.0);
    vel.v().setConstant(2.0);

    vel.setZero();

    for (Index i = 0; i < mesh.cellCount(); ++i)
    {
        auto [u, v] = vel(i);
        CHECK(u == doctest::Approx(0.0));
        CHECK(v == doctest::Approx(0.0));
    }
}
