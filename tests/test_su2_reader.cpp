#include "Su2Reader.h"
#include "MeshGeomCache.h"
#include "Mesh.h"
#include "UnstructuredMesh.h"

#include <doctest/doctest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace fvm::core;
using fvm::io::Su2Reader;

namespace
{
class TempSu2File
{
public:
    explicit TempSu2File(const std::string& contents)
        : path_(std::filesystem::current_path()
            / ("fvm_su2_test_"
                + std::to_string(++nextId_) + ".su2"))
    {
        std::ofstream file(path_);
        REQUIRE(file.good());
        file << contents;
    }

    ~TempSu2File()
    {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    TempSu2File(const TempSu2File&) = delete;
    TempSu2File& operator=(const TempSu2File&) = delete;

    const std::filesystem::path& path() const
    {
        return path_;
    }

private:
    std::filesystem::path path_;
    static std::atomic<int> nextId_;
};

std::atomic<int> TempSu2File::nextId_{ 0 };

const char* structuredGrid = R"(NDIME= 2
NPOIN= 12
1 0.0 0.0
2 1.0 0.0
3 2.0 0.0
4 3.0 0.0
5 0.0 1.0
6 1.0 1.0
7 2.0 1.0
8 3.0 1.0
9 0.0 2.0
10 1.0 2.0
11 2.0 2.0
12 3.0 2.0
NELEM= 6
9 1 2 6 5
9 2 3 7 6
9 3 4 8 7
9 5 6 10 9
9 6 7 11 10
9 7 8 12 11
NMARK= 4
MARKER_TAG= bottom
MARKER_ELEMS= 3
3 1 2
3 2 3
3 3 4
MARKER_TAG= right
MARKER_ELEMS= 2
3 4 8
3 8 12
MARKER_TAG= top
MARKER_ELEMS= 3
3 12 11
3 11 10
3 10 9
MARKER_TAG= left
MARKER_ELEMS= 2
3 9 5
3 5 1
)";

const char* mixedCells = R"(NDIME= 2
NPOIN= 6
1 0.0 0.0
2 1.0 0.0
3 0.0 1.0
4 1.0 1.0
5 2.0 0.0
6 2.0 1.0
NELEM= 3
5 1 2 4
5 1 4 3
9 2 4 6 5
NMARK= 4
MARKER_TAG= bottom
MARKER_ELEMS= 2
3 1 2
3 2 5
MARKER_TAG= right
MARKER_ELEMS= 1
3 5 6
MARKER_TAG= top
MARKER_ELEMS= 2
3 6 4
3 4 3
MARKER_TAG= left
MARKER_ELEMS= 1
3 3 1
)";

const char* sparseQuad = R"(NDIME= 2
NPOIN= 4
10 0.0 0.0
20 1.0 0.0
40 1.0 1.0
30 0.0 1.0
NELEM= 1
9 10 20 40 30
NMARK= 4
MARKER_TAG= south
MARKER_ELEMS= 1
3 10 20
MARKER_TAG= east
MARKER_ELEMS= 1
3 20 40
MARKER_TAG= north
MARKER_ELEMS= 1
3 40 30
MARKER_TAG= west
MARKER_ELEMS= 1
3 30 10
)";
} // namespace

TEST_CASE("Su2Reader reads a structured quad grid")
{
    const TempSu2File file(structuredGrid);
    const UnstructuredMesh mesh = Su2Reader::read(file.path().string());
    const Mesh& base = mesh;

    REQUIRE(base.cellCount() == 6);
    REQUIRE(base.faceCount() == 17);
    REQUIRE(base.patchCount() == 4);
    REQUIRE(base.patchName(0) == "bottom");
    REQUIRE(base.patchName(1) == "right");
    REQUIRE(base.patchName(2) == "top");
    REQUIRE(base.patchName(3) == "left");
    REQUIRE(base.patchFaces(0).size() == 3);
    REQUIRE(base.patchFaces(1).size() == 2);
    REQUIRE(base.patchFaces(2).size() == 3);
    REQUIRE(base.patchFaces(3).size() == 2);

    Scalar totalVolume = 0.0;
    for (Index cell = 0; cell < base.cellCount(); ++cell)
    {
        CHECK(base.cellVolume(cell) == doctest::Approx(1.0));
        totalVolume += base.cellVolume(cell);
    }
    CHECK(totalVolume == doctest::Approx(6.0));

    const MeshGeomCache geometry(base);
    CHECK(geometry.maxNonOrthogonality() == doctest::Approx(0.0));
}

TEST_CASE("Su2Reader normalizes a mixed mesh with a clockwise cell")
{
    const TempSu2File file(mixedCells);
    const UnstructuredMesh mesh = Su2Reader::read(file.path().string());
    const Mesh& base = mesh;

    REQUIRE(base.cellCount() == 3);
    CHECK(base.cellVolume(0) == doctest::Approx(0.5));
    CHECK(base.cellVolume(1) == doctest::Approx(0.5));
    CHECK(base.cellVolume(2) == doctest::Approx(1.0));
}

TEST_CASE("Su2Reader maps sparse one-based node ids")
{
    const TempSu2File file(sparseQuad);
    const UnstructuredMesh mesh = Su2Reader::read(file.path().string());
    const Mesh& base = mesh;

    REQUIRE(base.cellCount() == 1);
    CHECK(base.cellVolume(0) == doctest::Approx(1.0));
    CHECK(base.cellCentroid(0).x == doctest::Approx(0.5));
    CHECK(base.cellCentroid(0).y == doctest::Approx(0.5));
    REQUIRE(base.patchCount() == 4);
    REQUIRE(base.patchName(0) == "south");
    REQUIRE(base.patchName(1) == "east");
    REQUIRE(base.patchName(2) == "north");
    REQUIRE(base.patchName(3) == "west");
}

TEST_CASE("Su2Reader accepts optional trailing element ids")
{
    const TempSu2File file(R"(NDIME= 2
NPOIN= 4
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
NELEM= 2
5 1 2 3 41
5 1 3 4 7
NMARK= 4
MARKER_TAG= south
MARKER_ELEMS= 1
3 1 2
MARKER_TAG= east
MARKER_ELEMS= 1
3 2 3
MARKER_TAG= north
MARKER_ELEMS= 1
3 3 4
MARKER_TAG= west
MARKER_ELEMS= 1
3 4 1
)");
    const UnstructuredMesh mesh = Su2Reader::read(file.path().string());
    const Mesh& base = mesh;

    REQUIRE(base.cellCount() == 2);
    CHECK(base.cellVolume(0) == doctest::Approx(0.5));
    CHECK(base.cellVolume(1) == doctest::Approx(0.5));
    REQUIRE(base.patchCount() == 4);
}

TEST_CASE("Su2Reader rejects NDIME other than two")
{
    const TempSu2File file("NDIME= 3\n");
    CHECK_THROWS_AS(Su2Reader::read(file.path().string()), std::invalid_argument);
}

TEST_CASE("Su2Reader rejects an unsupported element type")
{
    const TempSu2File file(R"(NDIME= 2
NPOIN= 4
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
NELEM= 1
3 1 2 3
NMARK= 0
)");
    CHECK_THROWS_AS(Su2Reader::read(file.path().string()), std::invalid_argument);
}

TEST_CASE("Su2Reader rejects a marker edge on an interior edge")
{
    const TempSu2File file(R"(NDIME= 2
NPOIN= 4
1 0.0 0.0
2 1.0 0.0
3 0.0 1.0
4 1.0 1.0
NELEM= 2
5 1 2 4
5 1 4 3
NMARK= 4
MARKER_TAG= bad
MARKER_ELEMS= 1
3 1 4
MARKER_TAG= north
MARKER_ELEMS= 1
3 4 3
MARKER_TAG= west
MARKER_ELEMS= 1
3 3 1
MARKER_TAG= south
MARKER_ELEMS= 1
3 1 2
)");
    CHECK_THROWS_AS(Su2Reader::read(file.path().string()), std::invalid_argument);
}

TEST_CASE("Su2Reader rejects a boundary edge covered by no marker")
{
    const TempSu2File file(R"(NDIME= 2
NPOIN= 4
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
NELEM= 1
9 1 2 3 4
NMARK= 3
MARKER_TAG= south
MARKER_ELEMS= 1
3 1 2
MARKER_TAG= east
MARKER_ELEMS= 1
3 2 3
MARKER_TAG= north
MARKER_ELEMS= 1
3 3 4
)");
    CHECK_THROWS_AS(Su2Reader::read(file.path().string()), std::invalid_argument);
}

TEST_CASE("Su2Reader rejects an unknown node reference")
{
    const TempSu2File file(R"(NDIME= 2
NPOIN= 4
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
NELEM= 1
9 1 2 99 4
NMARK= 0
)");
    CHECK_THROWS_AS(Su2Reader::read(file.path().string()), std::invalid_argument);
}

TEST_CASE("Su2Reader rejects a duplicate node id")
{
    const TempSu2File file(R"(NDIME= 2
NPOIN= 4
1 0.0 0.0
1 1.0 0.0
3 1.0 1.0
4 0.0 1.0
NELEM= 0
NMARK= 0
)");
    CHECK_THROWS_AS(Su2Reader::read(file.path().string()), std::invalid_argument);
}

TEST_CASE("Su2Reader rejects a point-line count mismatch")
{
    const TempSu2File file(R"(NDIME= 2
NPOIN= 4
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
NELEM= 0
NMARK= 0
)");
    CHECK_THROWS_AS(Su2Reader::read(file.path().string()), std::invalid_argument);
}

TEST_CASE("Su2Reader rejects a marker element that is not a line")
{
    const TempSu2File file(R"(NDIME= 2
NPOIN= 4
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
NELEM= 1
9 1 2 3 4
NMARK= 1
MARKER_TAG= south
MARKER_ELEMS= 1
5 1 2
)");
    CHECK_THROWS_AS(Su2Reader::read(file.path().string()), std::invalid_argument);
}

TEST_CASE("Su2Reader rejects a marker edge with an unknown node")
{
    const TempSu2File file(R"(NDIME= 2
NPOIN= 4
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
NELEM= 1
9 1 2 3 4
NMARK= 1
MARKER_TAG= west
MARKER_ELEMS= 1
3 9 1
)");
    CHECK_THROWS_AS(Su2Reader::read(file.path().string()), std::invalid_argument);
}

TEST_CASE("Su2Reader rejects a non-manifold marker edge")
{
    const TempSu2File file(R"(NDIME= 2
NPOIN= 5
1 0.0 0.0
2 1.0 0.0
3 0.0 1.0
4 0.0 -1.0
5 1.0 1.0
NELEM= 3
5 1 2 3
5 1 2 4
5 1 2 5
NMARK= 1
MARKER_TAG= bad
MARKER_ELEMS= 1
3 1 2
)");
    CHECK_THROWS_AS(Su2Reader::read(file.path().string()), std::invalid_argument);
}

TEST_CASE("Su2Reader rejects a duplicate marker tag")
{
    const TempSu2File file(R"(NDIME= 2
NPOIN= 4
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
NELEM= 1
9 1 2 3 4
NMARK= 2
MARKER_TAG= south
MARKER_ELEMS= 1
3 1 2
MARKER_TAG= south
MARKER_ELEMS= 1
3 3 4
)");
    CHECK_THROWS_AS(Su2Reader::read(file.path().string()), std::invalid_argument);
}

TEST_CASE("Su2Reader rejects an empty element section")
{
    const TempSu2File file(R"(NDIME= 2
NPOIN= 4
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
NELEM= 0
NMARK= 0
)");
    CHECK_THROWS_AS(Su2Reader::read(file.path().string()), std::invalid_argument);
}

TEST_CASE("Su2Reader reports a non-existent file as unavailable")
{
    const auto path = std::filesystem::temp_directory_path()
        / "fvm_su2_test_missing_file.su2";
    CHECK_THROWS_AS(Su2Reader::read(path.string()), std::runtime_error);
}
