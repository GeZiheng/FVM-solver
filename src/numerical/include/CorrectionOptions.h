#pragma once

namespace fvm::numerical
{

/// Discretization corrections for non-orthogonal / skewed meshes.  All terms
/// are exact zeros on orthogonal meshes, so the defaults are safe everywhere.
struct CorrectionOptions
{
    bool nonOrthogonalDiffusion = true;  ///< Explicit interior cross-diffusion.
    bool skewInterpolation = true;       ///< Skew-corrected linear interpolation.
    int nNonOrthogonalCorrectors = 0;    ///< Pressure solve sub-loop count.
};

} // namespace fvm::numerical
