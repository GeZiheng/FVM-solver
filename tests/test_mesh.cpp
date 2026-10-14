#include "CartesianMesh.h"
#include "BoundaryCondition.h"
#include <cmath>
#include <doctest/doctest.h>
#include <stdexcept>
#include <vector>

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

    // Interior cell (1, 1); cellFaces order is [east, north, west, south].
    Index center = mesh.cellIndex(1, 1);
    const IndexSpan faces = mesh.cellFaces(center);
    // The cell across a face: P is the owner of its east/north faces but
    // the neighbour of its west/south faces (owner is always west/south).
    const auto across = [&mesh](Index cell, Index face) -> Index
    {
        return (mesh.faceOwner(face) == cell) ? mesh.faceNeighbour(face)
                                              : mesh.faceOwner(face);
    };
    CHECK(across(center, faces[0]) == mesh.cellIndex(2, 1)); // east
    CHECK(across(center, faces[1]) == mesh.cellIndex(1, 2)); // north
    CHECK(across(center, faces[2]) == mesh.cellIndex(0, 1)); // west
    CHECK(across(center, faces[3]) == mesh.cellIndex(1, 0)); // south

    // Boundary cell (0, 0): west/south faces have no neighbour.
    Index corner = mesh.cellIndex(0, 0);
    const IndexSpan cornerFaces = mesh.cellFaces(corner);
    const auto acrossCorner = [&mesh](Index cell, Index face) -> Index
    {
        return (mesh.faceOwner(face) == cell) ? mesh.faceNeighbour(face)
                                              : mesh.faceOwner(face);
    };
    CHECK(acrossCorner(corner, cornerFaces[0]) == mesh.cellIndex(1, 0));
    CHECK(acrossCorner(corner, cornerFaces[1]) == mesh.cellIndex(0, 1));
    CHECK(mesh.faceNeighbour(cornerFaces[2]) == Mesh::kInvalidCell); // west
    CHECK(mesh.faceNeighbour(cornerFaces[3]) == Mesh::kInvalidCell); // south
}

TEST_CASE("CartesianMesh boundary detection")
{
    Index nx = 3, ny = 3;
    CartesianMesh mesh(nx, ny, 0.0, 0.0, 1.0, 1.0);

    Index corner = mesh.cellIndex(0, 0);
    const IndexSpan faces = mesh.cellFaces(corner);
    CHECK(mesh.isBoundaryFace(faces[2]) == true);  // west
    CHECK(mesh.isBoundaryFace(faces[3]) == true);  // south
    CHECK(mesh.isBoundaryFace(faces[0]) == false); // east
    CHECK(mesh.isBoundaryFace(faces[1]) == false); // north
}

TEST_CASE("CartesianMesh face geometry")
{
    CartesianMesh mesh(4, 4, 0.0, 0.0, 1.0, 1.0);

    // Interior x-face of cell (0, 0): |S| = dy, outward normal +x for the
    // owner (the west cell).
    const Point2 Se = mesh.faceAreaVector(mesh.xFace(1, 0));
    CHECK(std::hypot(Se.x, Se.y) == doctest::Approx(0.25)); // east/west
    CHECK(Se.x / 0.25 == doctest::Approx(1.0));
    CHECK(Se.y == doctest::Approx(0.0));

    // West boundary face of cell (0, 0): same magnitude, normal -x.
    const Point2 Sw = mesh.faceAreaVector(mesh.xFace(0, 0));
    CHECK(std::hypot(Sw.x, Sw.y) == doctest::Approx(0.25));
    CHECK(Sw.x / 0.25 == doctest::Approx(-1.0));
    CHECK(Sw.y == doctest::Approx(0.0));

}

TEST_CASE("BoundaryField is patch-based and validates patch indices")
{
    CartesianMesh mesh(3, 2, 0.0, 0.0, 3.0, 2.0);
    fvm::numerical::BoundaryField bc(mesh);

    CHECK(bc.patchCount() == mesh.patchCount());
    CHECK(&bc.mesh() == &mesh);

    bc.set(2, fvm::numerical::BCType::Dirichlet, 1.5); // west patch
    CHECK(bc.get(2).type == fvm::numerical::BCType::Dirichlet);
    CHECK(bc.get(2).value == doctest::Approx(1.5));
    // The compatibility constants index the same patch.
    CHECK(bc.get(fvm::numerical::BoundaryField::West).value
        == doctest::Approx(1.5));

    CHECK_THROWS_AS(
        bc.set(mesh.patchCount(), fvm::numerical::BCType::Neumann, 0.0),
        std::out_of_range);
    CHECK_THROWS_AS(bc.get(mesh.patchCount()), std::out_of_range);
}

TEST_CASE("CartesianMesh generic face topology")
{
    CartesianMesh cartesian(3, 2, 0.0, 0.0, 3.0, 2.0);
    const Mesh& base = cartesian;

    CHECK(base.faceCount() == 17);
    CHECK(cartesian.xFace(0, 0) == 0);
    CHECK(cartesian.xFace(3, 0) == 3);
    CHECK(cartesian.xFace(0, 1) == 4);
    CHECK(cartesian.xFace(3, 1) == 7);
    CHECK(cartesian.yFace(0, 0) == 8);
    CHECK(cartesian.yFace(2, 0) == 10);
    CHECK(cartesian.yFace(0, 1) == 11);
    CHECK(cartesian.yFace(2, 2) == 16);

    CHECK(base.faceOwner(cartesian.xFace(1, 0)) == cartesian.cellIndex(0, 0));
    CHECK(base.faceNeighbour(cartesian.xFace(1, 0)) == cartesian.cellIndex(1, 0));
    CHECK(base.faceOwner(cartesian.xFace(2, 1)) == cartesian.cellIndex(1, 1));
    CHECK(base.faceNeighbour(cartesian.xFace(2, 1)) == cartesian.cellIndex(2, 1));
    CHECK(base.faceOwner(cartesian.yFace(1, 1)) == cartesian.cellIndex(1, 0));
    CHECK(base.faceNeighbour(cartesian.yFace(1, 1)) == cartesian.cellIndex(1, 1));

    const Index westBoundary[] = { cartesian.xFace(0, 0), cartesian.xFace(0, 1) };
    const Index eastBoundary[] = { cartesian.xFace(3, 0), cartesian.xFace(3, 1) };
    const Index southBoundary[] = { cartesian.yFace(0, 0), cartesian.yFace(1, 0), cartesian.yFace(2, 0) };
    const Index northBoundary[] = { cartesian.yFace(0, 2), cartesian.yFace(1, 2), cartesian.yFace(2, 2) };

    for (Index face : westBoundary)
    {
        CHECK(base.faceNeighbour(face) == Mesh::kInvalidCell);
        CHECK(base.isBoundaryFace(face));
        CHECK(base.faceOwner(face) == cartesian.cellIndex(0, face / (cartesian.nx() + 1)));
    }
    for (Index face : eastBoundary)
    {
        CHECK(base.faceNeighbour(face) == Mesh::kInvalidCell);
        CHECK(base.isBoundaryFace(face));
        CHECK(base.faceOwner(face) == cartesian.cellIndex(cartesian.nx() - 1, face / (cartesian.nx() + 1)));
    }
    for (Index face : southBoundary)
    {
        CHECK(base.faceNeighbour(face) == Mesh::kInvalidCell);
        CHECK(base.isBoundaryFace(face));
        const Index xFaces = (cartesian.nx() + 1) * cartesian.ny();
        CHECK(base.faceOwner(face) == cartesian.cellIndex(face - xFaces, 0));
    }
    for (Index face : northBoundary)
    {
        CHECK(base.faceNeighbour(face) == Mesh::kInvalidCell);
        CHECK(base.isBoundaryFace(face));
        CHECK(base.faceOwner(face) == cartesian.cellIndex((face - (cartesian.nx() + 1) * cartesian.ny()) % cartesian.nx(), cartesian.ny() - 1));
    }
}

TEST_CASE("CartesianMesh generic face geometry")
{
    CartesianMesh cartesian(3, 2, 0.0, 0.0, 3.0, 2.0);
    const Mesh& base = cartesian;

    const Point2 xArea{ 1.0, 0.0 };
    const Point2 yArea{ 0.0, 1.0 };
    CHECK(base.faceAreaVector(cartesian.xFace(1, 0)).x == doctest::Approx(xArea.x));
    CHECK(base.faceAreaVector(cartesian.xFace(1, 0)).y == doctest::Approx(xArea.y));
    CHECK(base.faceAreaVector(cartesian.yFace(1, 1)).x == doctest::Approx(yArea.x));
    CHECK(base.faceAreaVector(cartesian.yFace(1, 1)).y == doctest::Approx(yArea.y));

    CHECK(base.faceAreaVector(cartesian.xFace(0, 0)).x == doctest::Approx(-1.0));
    CHECK(base.faceAreaVector(cartesian.xFace(3, 0)).x == doctest::Approx(1.0));
    CHECK(base.faceAreaVector(cartesian.yFace(0, 0)).y == doctest::Approx(-1.0));
    CHECK(base.faceAreaVector(cartesian.yFace(0, 2)).y == doctest::Approx(1.0));

    Point2 centroid = base.faceCentroid(cartesian.xFace(1, 0));
    CHECK(centroid.x == doctest::Approx(1.0));
    CHECK(centroid.y == doctest::Approx(0.5));
    centroid = base.faceCentroid(cartesian.yFace(2, 1));
    CHECK(centroid.x == doctest::Approx(2.5));
    CHECK(centroid.y == doctest::Approx(1.0));

    Point2 cellPoint = base.cellCentroid(cartesian.cellIndex(1, 1));
    CHECK(cellPoint.x == doctest::Approx(1.5));
    CHECK(cellPoint.y == doctest::Approx(1.5));
    CHECK(base.cellVolume(cartesian.cellIndex(1, 1)) == doctest::Approx(1.0));
}

TEST_CASE("CartesianMesh generic cell faces")
{
    CartesianMesh cartesian(3, 2, 0.0, 0.0, 3.0, 2.0);
    const Mesh& base = cartesian;

    const IndexSpan faces = base.cellFaces(cartesian.cellIndex(1, 0));
    REQUIRE(faces.size() == 4);
    CHECK(faces[0] == cartesian.xFace(2, 0));
    CHECK(faces[1] == cartesian.yFace(1, 1));
    CHECK(faces[2] == cartesian.xFace(1, 0));
    CHECK(faces[3] == cartesian.yFace(1, 0));

    for (Index cell = 0; cell < base.cellCount(); ++cell)
    {
        const IndexSpan cellFaces = base.cellFaces(cell);
        REQUIRE(cellFaces.size() == 4);
        const bool touchesEast = base.faceOwner(cellFaces[0]) == cell || base.faceNeighbour(cellFaces[0]) == cell;
        const bool touchesNorth = base.faceOwner(cellFaces[1]) == cell || base.faceNeighbour(cellFaces[1]) == cell;
        const bool touchesWest = base.faceOwner(cellFaces[2]) == cell || base.faceNeighbour(cellFaces[2]) == cell;
        const bool touchesSouth = base.faceOwner(cellFaces[3]) == cell || base.faceNeighbour(cellFaces[3]) == cell;
        CHECK(touchesEast);
        CHECK(touchesNorth);
        CHECK(touchesWest);
        CHECK(touchesSouth);
    }
}

TEST_CASE("CartesianMesh generic patches")
{
    CartesianMesh cartesian(3, 2, 0.0, 0.0, 3.0, 2.0);
    const Mesh& base = cartesian;

    REQUIRE(base.patchCount() == 4);
    CHECK(base.patchName(0) == "east");
    CHECK(base.patchName(1) == "north");
    CHECK(base.patchName(2) == "west");
    CHECK(base.patchName(3) == "south");

    const std::vector<Index> east = { cartesian.xFace(3, 0), cartesian.xFace(3, 1) };
    const std::vector<Index> north = { cartesian.yFace(0, 2), cartesian.yFace(1, 2), cartesian.yFace(2, 2) };
    const std::vector<Index> west = { cartesian.xFace(0, 0), cartesian.xFace(0, 1) };
    const std::vector<Index> south = { cartesian.yFace(0, 0), cartesian.yFace(1, 0), cartesian.yFace(2, 0) };
    const std::vector<std::vector<Index>> expected = { east, north, west, south };

    for (Index patch = 0; patch < base.patchCount(); ++patch)
    {
        const IndexSpan actual = base.patchFaces(patch);
        REQUIRE(actual.size() == expected[patch].size());
        for (Index local = 0; local < actual.size(); ++local)
        {
            CHECK(actual[local] == expected[patch][local]);
            CHECK(base.facePatch(actual[local]) == patch);
        }
    }
}

TEST_CASE("CartesianMesh generic mesh id validation")
{
    CartesianMesh cartesian(3, 2, 0.0, 0.0, 3.0, 2.0);
    const Mesh& base = cartesian;
    const Index invalidFace = base.faceCount();
    const Index invalidCell = base.cellCount();

    CHECK_THROWS_AS(base.faceOwner(invalidFace), std::out_of_range);
    CHECK_THROWS_AS(base.faceNeighbour(invalidFace), std::out_of_range);
    CHECK_THROWS_AS(base.isBoundaryFace(invalidFace), std::out_of_range);
    CHECK_THROWS_AS(base.cellFaces(invalidCell), std::out_of_range);
    CHECK_THROWS_AS(base.cellCentroid(invalidCell), std::out_of_range);
    CHECK_THROWS_AS(base.faceCentroid(invalidFace), std::out_of_range);
    CHECK_THROWS_AS(base.faceAreaVector(invalidFace), std::out_of_range);
    CHECK_THROWS_AS(base.patchName(4), std::out_of_range);
    CHECK_THROWS_AS(base.patchFaces(4), std::out_of_range);
    CHECK_THROWS_AS(base.facePatch(cartesian.xFace(1, 0)), std::out_of_range);
    CHECK_THROWS_AS(base.facePatch(invalidFace), std::out_of_range);
    CHECK_THROWS_AS(cartesian.xFace(4, 0), std::out_of_range);
    CHECK_THROWS_AS(cartesian.xFace(0, 2), std::out_of_range);
    CHECK_THROWS_AS(cartesian.yFace(3, 0), std::out_of_range);
    CHECK_THROWS_AS(cartesian.yFace(0, 3), std::out_of_range);
}
