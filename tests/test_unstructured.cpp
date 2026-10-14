#include "MeshGeomCache.h"
#include "UnstructuredMesh.h"

#include <cmath>
#include <doctest/doctest.h>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace fvm::core;

namespace
{
MeshInput makeSquareInput()
{
    MeshInput input;
    input.points = {
        { 0.0, 0.0 },
        { 1.0, 0.0 },
        { 1.0, 1.0 },
        { 0.0, 1.0 }
    };
    input.cells = {
        { 0, 1, 2, 3 }
    };
    input.patches = {
        { "east", { { 1, 2 } } },
        { "north", { { 2, 3 } } },
        { "west", { { 3, 0 } } },
        { "south", { { 0, 1 } } }
    };
    return input;
}
} // namespace

TEST_CASE("UnstructuredMesh triangle pair geometry")
{
    MeshInput input;
    input.points = {
        { 0.0, 0.0 },
        { 1.0, 0.0 },
        { 1.0, 1.0 },
        { 0.0, 1.0 }
    };
    input.cells = {
        { 0, 1, 2 },
        { 0, 2, 3 }
    };
    input.patches = {
        { "south", { { 0, 1 } } },
        { "east", { { 1, 2 } } },
        { "north", { { 2, 3 } } },
        { "west", { { 3, 0 } } }
    };

    const UnstructuredMesh mesh(input);
    const Mesh& base = mesh;
    REQUIRE(base.cellCount() == 2);
    REQUIRE(base.faceCount() == 5);

    Index interior = base.faceCount();
    Index boundaryCount = 0;
    for (Index face = 0; face < base.faceCount(); ++face)
    {
        if (base.isBoundaryFace(face))
        {
            ++boundaryCount;
        }
        else
        {
            interior = face;
        }
    }
    REQUIRE(boundaryCount == 4);
    REQUIRE(interior != base.faceCount());

    const bool forward = base.faceOwner(interior) == 0 && base.faceNeighbour(interior) == 1;
    const bool reverse = base.faceOwner(interior) == 1 && base.faceNeighbour(interior) == 0;
    CHECK((forward || reverse));

    CHECK(base.cellVolume(0) == doctest::Approx(0.5));
    CHECK(base.cellVolume(1) == doctest::Approx(0.5));
    CHECK(base.cellVolume(0) + base.cellVolume(1) == doctest::Approx(1.0));

    const Point2 firstCentroid = base.cellCentroid(0);
    const Point2 secondCentroid = base.cellCentroid(1);
    CHECK(firstCentroid.x == doctest::Approx(2.0 / 3.0));
    CHECK(firstCentroid.y == doctest::Approx(1.0 / 3.0));
    CHECK(secondCentroid.x == doctest::Approx(1.0 / 3.0));
    CHECK(secondCentroid.y == doctest::Approx(2.0 / 3.0));

    const Point2 area = base.faceAreaVector(interior);
    CHECK(std::sqrt(area.x * area.x + area.y * area.y) == doctest::Approx(std::sqrt(2.0)));
    const Point2 ownerCentroid = base.cellCentroid(base.faceOwner(interior));
    const Point2 neighbourCentroid = base.cellCentroid(base.faceNeighbour(interior));
    const Point2 distance {
        neighbourCentroid.x - ownerCentroid.x,
        neighbourCentroid.y - ownerCentroid.y
    };
    const Scalar cross = area.x * distance.y - area.y * distance.x;
    CHECK(std::abs(cross) == doctest::Approx(0.0));
    CHECK((area.x * distance.x + area.y * distance.y) > 0.0);

    for (Index cell = 0; cell < base.cellCount(); ++cell)
    {
        Point2 sum;
        for (const Index face : base.cellFaces(cell))
        {
            const Point2 faceArea = base.faceAreaVector(face);
            const Scalar orientation
                = base.faceOwner(face) == cell ? 1.0 : -1.0;
            sum.x += orientation * faceArea.x;
            sum.y += orientation * faceArea.y;
        }
        CHECK(sum.x == doctest::Approx(0.0));
        CHECK(sum.y == doctest::Approx(0.0));
    }

    CHECK(base.patchName(0) == "south");
    CHECK(base.patchName(1) == "east");
    CHECK(base.patchName(2) == "north");
    CHECK(base.patchName(3) == "west");
    for (Index patch = 0; patch < base.patchCount(); ++patch)
    {
        REQUIRE(base.patchFaces(patch).size() == 1);
        const Index face = base.patchFaces(patch)[0];
        CHECK(base.facePatch(face) == patch);
        CHECK(base.faceNeighbour(face) == Mesh::kInvalidCell);
    }

    const MeshGeomCache geometry(base);
    CHECK(geometry.interpWeight(interior) == doctest::Approx(0.5));
    CHECK(geometry.innerCoef(interior) == doctest::Approx(3.0));
    const Point2 correction = geometry.nonOrthCorrection(interior);
    CHECK(correction.x == doctest::Approx(0.0));
    CHECK(correction.y == doctest::Approx(0.0));
    CHECK(geometry.maxNonOrthogonality() == doctest::Approx(0.0));

    Index southFace = base.faceCount();
    for (const Index face : base.patchFaces(0))
    {
        southFace = face;
    }
    // With the specified formula, the triangular-cell boundary coefficient is
    // 1/(1/3), not 1/(1/2); delta is measured from the triangle centroid.
    CHECK(geometry.boundaryCoef(southFace) == doctest::Approx(3.0));

    CHECK_THROWS_AS(geometry.innerCoef(southFace), std::invalid_argument);
    CHECK_THROWS_AS(geometry.nonOrthCorrection(southFace), std::invalid_argument);
    CHECK_THROWS_AS(geometry.boundaryCoef(interior), std::invalid_argument);
    CHECK_THROWS_AS(geometry.interpWeight(base.faceCount()), std::out_of_range);
    CHECK_THROWS_AS(base.faceOwner(base.faceCount()), std::out_of_range);
    CHECK_THROWS_AS(base.cellFaces(base.cellCount()), std::out_of_range);
}

TEST_CASE("UnstructuredMesh non-orthogonal triangle pair")
{
    MeshInput input;
    input.points = {
        { 0.0, 0.0 },
        { 2.0, 0.0 },
        { 0.0, 1.0 },
        { 3.0, 1.0 }
    };
    input.cells = {
        { 0, 1, 2 },
        { 1, 3, 2 }
    };
    input.patches = {
        { "south", { { 0, 1 } } },
        { "east", { { 1, 3 } } },
        { "north", { { 3, 2 } } },
        { "west", { { 2, 0 } } }
    };

    const UnstructuredMesh mesh(input);
    const Mesh& base = mesh;
    const MeshGeomCache geometry(base);

    Index interior = base.faceCount();
    for (Index face = 0; face < base.faceCount(); ++face)
    {
        if (!base.isBoundaryFace(face))
        {
            interior = face;
        }
    }
    REQUIRE(interior != base.faceCount());

    const Point2 correction = geometry.nonOrthCorrection(interior);
    CHECK((std::abs(correction.x) + std::abs(correction.y)) > 0.0);

    const Point2 ownerCentroid = base.cellCentroid(base.faceOwner(interior));
    const Point2 neighbourCentroid = base.cellCentroid(base.faceNeighbour(interior));
    const Point2 distance {
        neighbourCentroid.x - ownerCentroid.x,
        neighbourCentroid.y - ownerCentroid.y
    };
    const Scalar dot = correction.x * distance.x + correction.y * distance.y;
    CHECK(std::abs(dot) < 1.0e-12);
    CHECK(geometry.innerCoef(interior) > 0.0);
    CHECK(geometry.maxNonOrthogonality() > 0.0);
    CHECK(geometry.maxNonOrthogonality() < 90.0);
}

TEST_CASE("UnstructuredMesh input validation and orientation")
{
    const UnstructuredMesh normalized(makeSquareInput());

    MeshInput clockwise = makeSquareInput();
    clockwise.cells[0] = { 0, 3, 2, 1 };
    const UnstructuredMesh normalizedFromClockwise(clockwise);
    CHECK(normalizedFromClockwise.cellCount() == normalized.cellCount());
    CHECK(normalizedFromClockwise.faceCount() == normalized.faceCount());
    CHECK(normalizedFromClockwise.cellVolume(0) == doctest::Approx(normalized.cellVolume(0)));

    MeshInput duplicatePatch = makeSquareInput();
    duplicatePatch.patches[3].edges.push_back({ 1, 0 });
    CHECK_THROWS_AS(UnstructuredMesh{duplicatePatch}, std::invalid_argument);

    MeshInput interiorPatch = makeSquareInput();
    interiorPatch.cells = {
        { 0, 1, 2 },
        { 0, 2, 3 }
    };
    interiorPatch.patches[2].edges[0] = { 0, 2 };
    CHECK_THROWS_AS(UnstructuredMesh{interiorPatch}, std::invalid_argument);

    MeshInput uncovered = makeSquareInput();
    uncovered.patches.pop_back();
    CHECK_THROWS_AS(UnstructuredMesh{uncovered}, std::invalid_argument);

    MeshInput zeroArea = makeSquareInput();
    zeroArea.points[1] = { 1.0, 0.0 };
    zeroArea.points[2] = { 2.0, 0.0 };
    zeroArea.points.push_back({ 0.0, 1.0 });
    zeroArea.cells[0] = { 0, 1, 2 };
    zeroArea.patches.clear();
    CHECK_THROWS_AS(UnstructuredMesh{zeroArea}, std::invalid_argument);
}
