#pragma once

#include "UnstructuredMesh.h"

#include <string>

namespace fvm::numerical
{

inline fvm::core::MeshInput makeObliquePair()
{
    fvm::core::MeshInput input;
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
    return input;
}

inline fvm::core::MeshInput makeShearedTriMesh(fvm::core::Index n,
    fvm::core::Scalar shear = 0.5)
{
    fvm::core::MeshInput input;
    const fvm::core::Index lineCount = n + 1;
    input.points.reserve(lineCount * lineCount);
    for (fvm::core::Index j = 0; j < lineCount; ++j)
    {
        const fvm::core::Scalar y = static_cast<fvm::core::Scalar>(j) / n;
        for (fvm::core::Index i = 0; i < lineCount; ++i)
        {
            const fvm::core::Scalar x = static_cast<fvm::core::Scalar>(i) / n;
            input.points.push_back({ x + shear * y, y });
        }
    }

    input.cells.reserve(2 * n * n);
    for (fvm::core::Index j = 0; j < n; ++j)
    {
        for (fvm::core::Index i = 0; i < n; ++i)
        {
            const fvm::core::Index bl = j * lineCount + i;
            const fvm::core::Index br = bl + 1;
            const fvm::core::Index tl = bl + lineCount;
            const fvm::core::Index tr = tl + 1;
            const bool interiorQuad = i > 0 && i < n - 1 && j > 0 && j < n - 1;
            if (interiorQuad)
            {
                input.cells.push_back({ bl, br, tr });
                input.cells.push_back({ bl, tr, tl });
            }
            else
            {
                input.cells.push_back({ bl, br, tr, tl });
            }
        }
    }

    for (fvm::core::Index j = 0; j < n; ++j)
    {
        const fvm::core::Index bl = j * lineCount + (n - 1);
        const fvm::core::Index br = bl + 1;
        const fvm::core::Index tl = bl + lineCount;
        const fvm::core::Index tr = tl + 1;
        input.patches.push_back({ "e" + std::to_string(j), { { br, tr } } });
    }
    for (fvm::core::Index i = 0; i < n; ++i)
    {
        const fvm::core::Index bl = (n - 1) * lineCount + i;
        const fvm::core::Index br = bl + 1;
        const fvm::core::Index tl = bl + lineCount;
        const fvm::core::Index tr = tl + 1;
        input.patches.push_back({ "n" + std::to_string(i), { { tl, tr } } });
    }
    for (fvm::core::Index j = 0; j < n; ++j)
    {
        const fvm::core::Index bl = j * lineCount;
        const fvm::core::Index br = bl + 1;
        const fvm::core::Index tl = bl + lineCount;
        const fvm::core::Index tr = tl + 1;
        input.patches.push_back({ "w" + std::to_string(j), { { bl, tl } } });
    }
    for (fvm::core::Index i = 0; i < n; ++i)
    {
        const fvm::core::Index bl = i;
        const fvm::core::Index br = bl + 1;
        const fvm::core::Index tl = bl + lineCount;
        const fvm::core::Index tr = tl + 1;
        input.patches.push_back({ "s" + std::to_string(i), { { bl, br } } });
    }

    return input;
}

} // namespace fvm::numerical
