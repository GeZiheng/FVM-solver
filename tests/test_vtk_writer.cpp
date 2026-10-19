#include "Mesh.h"
#include "UnstructuredMesh.h"
#include "VtkWriter.h"

#include <doctest/doctest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace fvm::core;
using fvm::io::VtkWriter;

namespace
{
class TempVtkFile
{
public:
    TempVtkFile()
        : path_(std::filesystem::current_path()
            / ("fvm_vtk_test_"
                + std::to_string(++nextId_) + ".vtu"))
    {
        std::ofstream file(path_);
        REQUIRE(file.good());
    }

    ~TempVtkFile()
    {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    TempVtkFile(const TempVtkFile&) = delete;
    TempVtkFile& operator=(const TempVtkFile&) = delete;

    const std::filesystem::path& path() const
    {
        return path_;
    }

private:
    std::filesystem::path path_;
    static std::atomic<int> nextId_;
};

std::atomic<int> TempVtkFile::nextId_{ 0 };

MeshInput makeMixedCellInput()
{
    MeshInput input;
    input.points = {
        { 0, 0 },
        { 1, 0 },
        { 0, 1 },
        { 1, 1 },
        { 2, 0 },
        { 2, 1 }
    };
    input.cells = {
        { 0, 1, 3 },
        { 0, 3, 2 },
        { 1, 3, 5, 4 }
    };
    input.patches = {
        { "bottom", { { 0, 1 }, { 1, 4 } } },
        { "right", { { 4, 5 } } },
        { "top", { { 5, 3 }, { 3, 2 } } },
        { "left", { { 2, 0 } } }
    };
    return input;
}

bool contains(const std::string& contents, const std::string& expected)
{
    return contents.find(expected) != std::string::npos;
}

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream file(path);
    REQUIRE(file.good());
    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());
}
} // namespace

TEST_CASE("UnstructuredMesh exposes point and cell-loop nodes")
{
    const UnstructuredMesh mesh = makeStructuredMesh(2, 1, 0, 0, 1, 1);
    REQUIRE(mesh.pointCount() == 6);
    CHECK(mesh.point(0).x == 0.0);
    CHECK(mesh.point(0).y == 0.0);
    CHECK(mesh.point(1).x == 0.5);
    CHECK(mesh.point(1).y == 0.0);
    CHECK(mesh.point(4).x == 0.5);
    CHECK(mesh.point(4).y == 1.0);

    REQUIRE(mesh.cellCount() == 2);
    for (Index cell = 0; cell < mesh.cellCount(); ++cell)
    {
        CHECK(mesh.cellNodeCount(cell) == 4);
    }
    CHECK(mesh.cellNode(0, 0) == 0);
    CHECK(mesh.cellNode(0, 1) == 1);
    CHECK(mesh.cellNode(0, 2) == 4);
    CHECK(mesh.cellNode(0, 3) == 3);
}

TEST_CASE("VtkWriter writes an UnstructuredGrid with mixed cells")
{
    const UnstructuredMesh mesh(makeMixedCellInput());
    ScalarField temperature(mesh, "T");
    temperature(0) = 1.0;
    temperature(1) = 2.0;
    temperature(2) = 3.0;

    VectorField velocity(mesh, "U");
    velocity.u()(0) = 1.0;
    velocity.v()(0) = 2.0;
    velocity.u()(1) = 3.0;
    velocity.v()(1) = 4.0;
    velocity.u()(2) = 5.0;
    velocity.v()(2) = 6.0;

    const TempVtkFile file;
    VtkWriter::write(file.path().string(),
        mesh,
        { { "T", &temperature } },
        { { "U", &velocity } });
    const std::string contents = readFile(file.path());

    CHECK(contains(contents,
        "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
        "byte_order=\"LittleEndian\">"));
    CHECK(contains(contents, "<UnstructuredGrid>\n<Piece"));
    CHECK(contains(contents, "NumberOfPoints=\"6\""));
    CHECK(contains(contents, "NumberOfCells=\"3\""));
    CHECK(contains(contents, "0 0 0.0"));
    CHECK(contains(contents, "1 0 0.0"));
    CHECK(contains(contents, "2 0 0.0"));
    CHECK(contains(contents, "0 1 0.0"));
    CHECK(contains(contents, "1 1 0.0"));
    CHECK(contains(contents, "2 1 0.0"));
    CHECK(contains(contents, "0 1 3"));
    CHECK(contains(contents, "0 3 2"));
    CHECK(contains(contents, "4 5 3 1"));
    CHECK(contains(contents, "3 6 10"));
    CHECK(contains(contents, "5 5 9"));
    CHECK(contains(contents, "\n1\n2\n3\n"));
    CHECK(contains(contents, "1 2 0.0"));
    CHECK(contains(contents, "3 4 0.0"));
    CHECK(contains(contents, "5 6 0.0"));
}

TEST_CASE("VtkWriter escapes UnstructuredGrid field names")
{
    const UnstructuredMesh mesh = makeStructuredMesh(1, 1, 0, 0, 1, 1);
    ScalarField temperature(mesh, "T");
    temperature(0) = 1.0;

    const TempVtkFile file;
    VtkWriter::write(file.path().string(),
        mesh,
        { { "R&D \"<q>\"", &temperature } },
        {});
    const std::string contents = readFile(file.path());

    CHECK(contains(contents,
        "Name=\"R&amp;D &quot;&lt;q&gt;&quot;\""));
}

TEST_CASE("VtkWriter maps a polygon cell to VTK_POLYGON")
{
    MeshInput input;
    input.points = {
        { 0, 0 },
        { 2, 0 },
        { 3, 1 },
        { 1, 3 },
        { -1, 1 }
    };
    input.cells = { { 0, 1, 2, 3, 4 } };
    input.patches = {
        { "boundary",
            { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 0 } } }
    };

    const UnstructuredMesh mesh(input);
    const TempVtkFile file;
    VtkWriter::write(file.path().string(), mesh);
    const std::string contents = readFile(file.path());

    CHECK(contains(contents, "\n7\n"));
    CHECK(contains(contents, "\n5\n"));
}

TEST_CASE("VtkWriter reports a .vtu open failure")
{
    const UnstructuredMesh mesh = makeStructuredMesh(1, 1, 0, 0, 1, 1);
    const std::filesystem::path path =
        std::filesystem::path("fvm_missing_directory")
        / "fvm_vtk_test_failure.vtu";
    CHECK_THROWS_AS(VtkWriter::write(path.string(), mesh),
        std::runtime_error);
}
