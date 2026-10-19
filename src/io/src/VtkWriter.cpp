#include "VtkWriter.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fvm::io
{

namespace
{
std::string xmlEscape(const std::string& text)
{
    std::string escaped;
    escaped.reserve(text.size());
    for (const char character : text)
    {
        switch (character)
        {
        case '&':
            escaped += "&amp;";
            break;
        case '<':
            escaped += "&lt;";
            break;
        case '>':
            escaped += "&gt;";
            break;
        case '"':
            escaped += "&quot;";
            break;
        case '\'':
            escaped += "&apos;";
            break;
        default:
            escaped += character;
            break;
        }
    }
    return escaped;
}
} // namespace

void VtkWriter::write(const std::string& filename,
    const CartesianMesh& mesh,
    const std::vector<std::pair<std::string, const ScalarField*>>& scalarFields,
    const std::vector<std::pair<std::string, const VectorField*>>& vectorFields)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file for writing: "
                                 + filename);
    }

    Index nx = mesh.nx();
    Index ny = mesh.ny();
    Scalar dx = mesh.dx();
    Scalar dy = mesh.dy();
    Scalar originX = mesh.xMin();
    Scalar originY = mesh.yMin();

    // VTK ImageData is node-based, so dimensions are (nx+1, ny+1, 1)
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"ImageData\" version=\"1.0\" "
            "byte_order=\"LittleEndian\">\n";
    file << "<ImageData WholeExtent=\"0 " << nx << " 0 " << ny << " 0 0\"\n";
    file << "             Origin=\"" << originX << " " << originY << " 0.0\"\n";
    file << "             Spacing=\"" << dx << " " << dy << " 1.0\">\n";
    file << "<Piece Extent=\"0 " << nx << " 0 " << ny << " 0 0\">\n";

    // Cell data
    file << "<CellData>\n";

    // Scalar fields
    for (const auto& [name, field] : scalarFields)
    {
        if (!field)
            continue;
        file << "  <DataArray type=\"Float64\" Name=\"" << xmlEscape(name)
             << "\" format=\"ascii\">\n";
        for (Index j = 0; j < ny; ++j)
        {
            for (Index i = 0; i < nx; ++i)
            {
                file << (*field)(mesh.cellIndex(i, j));
                if (i < nx - 1)
                    file << " ";
            }
            file << "\n";
        }
        file << "  </DataArray>\n";
    }

    // Vector fields
    for (const auto& [name, field] : vectorFields)
    {
        if (!field)
            continue;
        file << "  <DataArray type=\"Float64\" Name=\"" << xmlEscape(name)
             << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (Index j = 0; j < ny; ++j)
        {
            for (Index i = 0; i < nx; ++i)
            {
                auto [u, v] = (*field)(mesh.cellIndex(i, j));
                file << u << " " << v << " 0.0";
                if (i < nx - 1)
                    file << " ";
            }
            file << "\n";
        }
        file << "  </DataArray>\n";
    }

    file << "</CellData>\n";
    file << "</Piece>\n";
    file << "</ImageData>\n";
    file << "</VTKFile>\n";

    file.close();
}

void VtkWriter::write(const std::string& filename,
    const UnstructuredMesh& mesh,
    const std::vector<std::pair<std::string, const ScalarField*>>& scalarFields,
    const std::vector<std::pair<std::string, const VectorField*>>& vectorFields)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file for writing: "
                                 + filename);
    }

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
            "byte_order=\"LittleEndian\">\n";
    file << "<UnstructuredGrid>\n";
    file << "<Piece NumberOfPoints=\"" << mesh.pointCount()
         << "\" NumberOfCells=\"" << mesh.cellCount() << "\">\n";
    file << "<Points>\n";
    file << "  <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
            "format=\"ascii\">\n";
    for (Index point = 0; point < mesh.pointCount(); ++point)
    {
        const fvm::core::Point2 coordinates = mesh.point(point);
        file << coordinates.x << " " << coordinates.y << " 0.0\n";
    }
    file << "  </DataArray>\n";
    file << "</Points>\n";

    file << "<Cells>\n";
    file << "  <DataArray type=\"Int64\" Name=\"connectivity\" "
            "format=\"ascii\">\n";
    for (Index cell = 0; cell < mesh.cellCount(); ++cell)
    {
        for (Index local = 0; local < mesh.cellNodeCount(cell); ++local)
        {
            if (local != 0)
                file << " ";
            file << mesh.cellNode(cell, local);
        }
        file << "\n";
    }
    file << "  </DataArray>\n";

    file << "  <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    Index offset = 0;
    for (Index cell = 0; cell < mesh.cellCount(); ++cell)
    {
        offset += mesh.cellNodeCount(cell);
        if (cell != 0)
            file << " ";
        file << offset;
    }
    file << "\n";
    file << "  </DataArray>\n";

    file << "  <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (Index cell = 0; cell < mesh.cellCount(); ++cell)
    {
        const Index nodeCount = mesh.cellNodeCount(cell);
        const int cellType = nodeCount == 3 ? 5 : nodeCount == 4 ? 9 : 7;
        if (cell != 0)
            file << " ";
        file << cellType;
    }
    file << "\n";
    file << "  </DataArray>\n";
    file << "</Cells>\n";

    file << "<CellData>\n";
    for (const auto& [name, field] : scalarFields)
    {
        if (!field)
            continue;
        file << "  <DataArray type=\"Float64\" Name=\"" << xmlEscape(name)
             << "\" format=\"ascii\">\n";
        for (Index cell = 0; cell < mesh.cellCount(); ++cell)
        {
            file << (*field)(cell) << "\n";
        }
        file << "  </DataArray>\n";
    }

    for (const auto& [name, field] : vectorFields)
    {
        if (!field)
            continue;
        file << "  <DataArray type=\"Float64\" Name=\"" << xmlEscape(name)
             << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (Index cell = 0; cell < mesh.cellCount(); ++cell)
        {
            const auto [u, v] = (*field)(cell);
            file << u << " " << v << " 0.0\n";
        }
        file << "  </DataArray>\n";
    }
    file << "</CellData>\n";

    file << "</Piece>\n";
    file << "</UnstructuredGrid>\n";
    file << "</VTKFile>\n";

    file.close();
}

} // namespace fvm::io
