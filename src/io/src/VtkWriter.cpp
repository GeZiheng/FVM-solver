#include "VtkWriter.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fvm::io
{

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
        file << "  <DataArray type=\"Float64\" Name=\"" << name
             << "\" format=\"ascii\">\n";
        for (Index j = 0; j < ny; ++j)
        {
            for (Index i = 0; i < nx; ++i)
            {
                file << (*field)(i, j);
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
        file << "  <DataArray type=\"Float64\" Name=\"" << name
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

} // namespace fvm::io
