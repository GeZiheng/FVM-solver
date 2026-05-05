#pragma once

#include "Field.h"
#include "Mesh.h"
#include <string>
#include <utility>
#include <vector>

using fvm::core::CartesianMesh;
using fvm::core::Index;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::VectorField;

namespace fvm::io {

/**
 * @brief Write fields to VTK ImageData (.vti) format.
 *
 * Assumes a uniform Cartesian mesh, which maps directly to vtkImageData.
 */
class VtkWriter {
public:
  /**
   * @brief Write scalar and vector fields to a .vti file.
   * @param filename Output file path (should end in .vti).
   * @param mesh The Cartesian mesh.
   * @param scalarFields List of (name, field) pairs to write as scalars.
   * @param vectorFields List of (name, field) pairs to write as vectors.
   */
  static void write(const std::string& filename,
                    const CartesianMesh& mesh,
                    const std::vector<std::pair<std::string, const ScalarField*>>& scalarFields = {},
                    const std::vector<std::pair<std::string, const VectorField*>>& vectorFields = {});
};

} // namespace fvm::io
