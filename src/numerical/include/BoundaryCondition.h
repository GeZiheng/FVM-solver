#pragma once

#include <stdexcept>
#include <vector>

#include "Mesh.h"
#include "Types.h"

namespace fvm::numerical
{
using fvm::core::Index;
using fvm::core::Scalar;

/**
 * @brief Boundary condition type.
 *
 * - Dirichlet: fixed value phi_b at the boundary face.
 * - Neumann  : fixed outward-normal derivative d(phi)/dn at the boundary face.
 */
enum class BCType
{
    Dirichlet,
    Neumann
};

struct BoundaryCondition
{
    BCType type = BCType::Neumann;
    Scalar value = 0.0;
};

/**
 * @brief Boundary conditions indexed by mesh patches.
 *
 * CartesianMesh creates patches in the order
 *   0 = east, 1 = north, 2 = west, 3 = south.
 *
 * Default: zero-flux (Neumann, g = 0) on all sides.
 */
class BoundaryField
{
public:
    explicit BoundaryField(const fvm::core::Mesh& mesh)
        : mesh_(mesh)
        , bcs_(mesh.patchCount())
    {
    }

    void set(Index patch, BCType type, Scalar value)
    {
        checkPatch(patch);
        bcs_[static_cast<std::size_t>(patch)] = BoundaryCondition{ type, value };
    }

    const BoundaryCondition& get(Index patch) const
    {
        checkPatch(patch);
        return bcs_[static_cast<std::size_t>(patch)];
    }

    Index patchCount() const
    {
        return bcs_.size();
    }

    const fvm::core::Mesh& mesh() const
    {
        return mesh_;
    }

    // Valid for the CartesianMesh patch order.
    static constexpr Index East = 0;
    static constexpr Index North = 1;
    static constexpr Index West = 2;
    static constexpr Index South = 3;

private:
    void checkPatch(Index patch) const
    {
        if (patch >= mesh_.patchCount())
        {
            throw std::out_of_range("Boundary patch index is out of range");
        }
    }

    const fvm::core::Mesh& mesh_;
    std::vector<BoundaryCondition> bcs_;
};

} // namespace fvm::numerical
