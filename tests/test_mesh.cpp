#include "Mesh.h"
#include <cmath>
#include <doctest/doctest.h>

using namespace fvm::core;

TEST_CASE("CartesianMesh basic properties")
{
    Index nx = 4, ny = 3;
    Scalar xMin = 0.0, yMin = 0.0, xMax = 2.0, yMax = 1.5;

    CartesianMesh mesh(nx, ny, xMin, yMin, xMax, yMax);

    CHECK(mesh.nx() == nx);
    CHECK(mesh.ny() == ny);
    CHECK(mesh.cellCount() == nx * ny);
    CHECK(mesh.dx() == doctest::Approx(0.5));
    CHECK(mesh.dy() == doctest::Approx(0.5));
    CHECK(mesh.cellVolume(0) == doctest::Approx(0.25));
}

TEST_CASE("CartesianMesh cell center coordinates")
{
    Index nx = 2, ny = 2;
    CartesianMesh mesh(nx, ny, 0.0, 0.0, 2.0, 2.0);

    auto [x0, y0] = mesh.cellCenter(0, 0);
    CHECK(x0 == doctest::Approx(0.5));
    CHECK(y0 == doctest::Approx(0.5));

    auto [x1, y1] = mesh.cellCenter(1, 1);
    CHECK(x1 == doctest::Approx(1.5));
    CHECK(y1 == doctest::Approx(1.5));
}

TEST_CASE("CartesianMesh index mapping")
{
    Index nx = 3, ny = 4;
    CartesianMesh mesh(nx, ny, 0.0, 0.0, 1.0, 1.0);

    CHECK(mesh.cellIndex(0, 0) == 0);
    CHECK(mesh.cellIndex(2, 0) == 2);
    CHECK(mesh.cellIndex(0, 1) == 3);
    CHECK(mesh.cellIndex(2, 3) == 11);

    auto [i, j] = mesh.cellIJ(7);
    CHECK(i == 1);
    CHECK(j == 2);
}

TEST_CASE("CartesianMesh neighbors")
{
    Index nx = 3, ny = 3;
    CartesianMesh mesh(nx, ny, 0.0, 0.0, 1.0, 1.0);

    // Interior cell (1, 1)
    Index center = mesh.cellIndex(1, 1);
    CHECK(mesh.neighbor(center, 0) == mesh.cellIndex(2, 1)); // east
    CHECK(mesh.neighbor(center, 1) == mesh.cellIndex(1, 2)); // north
    CHECK(mesh.neighbor(center, 2) == mesh.cellIndex(0, 1)); // west
    CHECK(mesh.neighbor(center, 3) == mesh.cellIndex(1, 0)); // south

    // Boundary cell (0, 0)
    Index corner = mesh.cellIndex(0, 0);
    CHECK(mesh.neighbor(corner, 0) == mesh.cellIndex(1, 0));
    CHECK(mesh.neighbor(corner, 1) == mesh.cellIndex(0, 1));
    CHECK(mesh.neighbor(corner, 2) == mesh.cellCount()); // west boundary
    CHECK(mesh.neighbor(corner, 3) == mesh.cellCount()); // south boundary
}

TEST_CASE("CartesianMesh boundary detection")
{
    Index nx = 3, ny = 3;
    CartesianMesh mesh(nx, ny, 0.0, 0.0, 1.0, 1.0);

    Index corner = mesh.cellIndex(0, 0);
    CHECK(mesh.isBoundaryFace(corner, 2) == true);  // west
    CHECK(mesh.isBoundaryFace(corner, 3) == true);  // south
    CHECK(mesh.isBoundaryFace(corner, 0) == false); // east
    CHECK(mesh.isBoundaryFace(corner, 1) == false); // north
}

TEST_CASE("CartesianMesh face geometry")
{
    CartesianMesh mesh(4, 4, 0.0, 0.0, 1.0, 1.0);

    CHECK(mesh.faceArea(0) == doctest::Approx(0.25)); // east/west
    CHECK(mesh.faceArea(1) == doctest::Approx(0.25)); // north/south

    auto [nx0, ny0] = mesh.faceNormal(0);
    CHECK(nx0 == doctest::Approx(1.0));
    CHECK(ny0 == doctest::Approx(0.0));

    auto [nx2, ny2] = mesh.faceNormal(2);
    CHECK(nx2 == doctest::Approx(-1.0));
    CHECK(ny2 == doctest::Approx(0.0));
}
