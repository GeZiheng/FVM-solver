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

    /**
     * @brief Per-face boundary values.
     *
     * values[k] is the value at the k-th face of the patch in
     * mesh.patchFaces(patch) traversal order. The patch-level type is set;
     * the patch-level constant value is left untouched and is ignored by
     * valueAt() for every face of this patch.
     *
     * Dirichlet values are face values phi_b; Neumann values are face
     * normal derivatives d(phi)/dn.
     */
    void setValues(Index patch, BCType type, std::vector<Scalar> values)
    {
        checkPatch(patch);
        const auto faces = mesh_.patchFaces(patch);
        if (values.size() != static_cast<std::size_t>(faces.size()))
        {
            throw std::invalid_argument(
                "per-face value count must equal the patch face count");
        }
        if (faceValues_.empty())
        {
            faceValues_.assign(static_cast<std::size_t>(mesh_.faceCount()),
                0.0);
            faceHasValue_.assign(
                static_cast<std::size_t>(mesh_.faceCount()), 0);
        }
        for (std::size_t k = 0; k < values.size(); ++k)
        {
            const auto face = static_cast<Index>(faces[k]);
            faceValues_[static_cast<std::size_t>(face)] = values[k];
            faceHasValue_[static_cast<std::size_t>(face)] = 1;
        }
        bcs_[static_cast<std::size_t>(patch)].type = type;
    }

    const BoundaryCondition& get(Index patch) const
    {
        checkPatch(patch);
        return bcs_[static_cast<std::size_t>(patch)];
    }

    /**
     * @brief Resolve the boundary value at a boundary face.
     *
     * Returns the per-face value when setValues() was used for the face's
     * patch, otherwise the patch constant. Equivalent to get(patch).value
     * for every boundary face when setValues() was never called.
     *
     * Interior faces are rejected (std::invalid_argument).
     */
    Scalar valueAt(Index face) const
    {
        if (!mesh_.isBoundaryFace(face))
        {
            throw std::invalid_argument("valueAt requires a boundary face");
        }
        const auto slot = static_cast<std::size_t>(face);
        if (!faceValues_.empty() && faceHasValue_[slot] != 0)
        {
            return faceValues_[slot];
        }
        return bcs_[static_cast<std::size_t>(mesh_.facePatch(face))].value;
    }

    /// Whether setValues() has been used for any face of this patch.
    bool hasFaceValues(Index patch) const
    {
        checkPatch(patch);
        if (faceValues_.empty())
        {
            return false;
        }
        for (const Index face : mesh_.patchFaces(patch))
        {
            if (faceHasValue_[static_cast<std::size_t>(face)] != 0)
            {
                return true;
            }
        }
        return false;
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
    std::vector<Scalar> faceValues_;
    std::vector<unsigned char> faceHasValue_;
};

} // namespace fvm::numerical
