#pragma once

#include "UnstructuredMesh.h"
#include <string>

namespace fvm::io
{
using fvm::core::UnstructuredMesh;

/// Reads a two-dimensional SU2 ASCII mesh file into a validated mesh.
class Su2Reader
{
public:
    /// @param filename Path to an SU2 mesh file (NDIME= 2, ASCII).
    /// @throws std::runtime_error if the file cannot be opened.
    /// @throws std::invalid_argument on any format or topology error.
    static UnstructuredMesh read(const std::string& filename);
};
} // namespace fvm::io
