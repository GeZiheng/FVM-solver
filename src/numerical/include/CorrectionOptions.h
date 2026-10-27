#pragma once

#include "MeshGeomCache.h"

namespace fvm::numerical
{

/// The split family lives in core next to the MeshGeomCache accessors that
/// consume it; re-export it so numerical users keep one name.
using fvm::core::NonOrthSplit;

/// Discretization corrections for non-orthogonal / skewed meshes.  All terms
/// are exact zeros on orthogonal meshes, so the defaults are safe everywhere.
struct CorrectionOptions
{
    bool nonOrthogonalDiffusion = true;  ///< Gates the explicit cross terms
                                         ///< only: diffusion X and pressure
                                         ///< crossP (the implicit alpha
                                         ///< always follows nonOrthSplit).
    bool skewInterpolation = true;       ///< Skew-corrected linear interpolation.
    int nNonOrthogonalCorrectors = 0;    ///< Pressure solve sub-loop count.
    /// Interior (alpha, k) pair selection.  Both splits give the same
    /// converged solution where the face gradient satisfies
    /// G.d = (phiN - phiO) -- exact for linear fields.  For general fields
    /// the fixed points differ at O(h^2) through
    /// (alphaOr - alphaMin)*(G.d - dPhi): the choice changes both the
    /// iteration path and, at O(h^2), the fixed point.
    NonOrthSplit nonOrthSplit = NonOrthSplit::MinimumCorrection;
};

} // namespace fvm::numerical
