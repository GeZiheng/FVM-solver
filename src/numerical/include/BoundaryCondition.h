#pragma once

#include <array>
#include <stdexcept>

#include "Types.h"

using fvm::core::Scalar;

namespace fvm::numerical {

/**
 * @brief Boundary condition type.
 *
 * - Dirichlet: fixed value phi_b at the boundary face.
 * - Neumann  : fixed outward-normal derivative d(phi)/dn at the boundary face.
 */
enum class BCType { Dirichlet,
                    Neumann };

struct BoundaryCondition {
  BCType type = BCType::Neumann;
  Scalar value = 0.0;
};

/**
 * @brief Boundary conditions for the four sides of a rectangular domain.
 *
 * Side indices follow the mesh face convention:
 *   0 = east, 1 = north, 2 = west, 3 = south.
 *
 * Default: zero-flux (Neumann, g = 0) on all sides.
 */
class BoundaryField {
public:
  BoundaryField() = default;

  void set(int side, BCType type, Scalar value)
  {
    checkSide(side);
    bcs_[static_cast<size_t>(side)] = BoundaryCondition{ type, value };
  }

  const BoundaryCondition& get(int side) const
  {
    checkSide(side);
    return bcs_[static_cast<size_t>(side)];
  }

  // Side index constants (match CartesianMesh face convention)
  static constexpr int East = 0;
  static constexpr int North = 1;
  static constexpr int West = 2;
  static constexpr int South = 3;

private:
  static void checkSide(int side)
  {
    if (side < 0 || side > 3) {
      throw std::invalid_argument("Boundary side must be in [0, 3]");
    }
  }

  std::array<BoundaryCondition, 4> bcs_{};
};

} // namespace fvm::numerical
