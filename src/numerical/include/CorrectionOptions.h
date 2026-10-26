#pragma once

namespace fvm::numerical
{

/// Interior-face non-orthogonal flux split family.  Both families give the
/// same converged solution where the face gradient satisfies
/// G.d = (phiN - phiO) -- exact for linear fields.  For general fields the
/// fixed points differ at O(h^2) through
/// (alphaOr - alphaMin)*(G.d - dPhi).  The choice therefore changes both
/// the iteration path and, at O(h^2), the fixed point.
enum class NonOrthSplit
{
    MinimumCorrection,
    OverRelaxed
};

/// Discretization corrections for non-orthogonal / skewed meshes.  All terms
/// are exact zeros on orthogonal meshes, so the defaults are safe everywhere.
struct CorrectionOptions
{
    bool nonOrthogonalDiffusion = true;  ///< Gates the explicit cross terms
                                         ///< only (the implicit alpha always
                                         ///< follows nonOrthSplit).
    bool skewInterpolation = true;       ///< Skew-corrected linear interpolation.
    int nNonOrthogonalCorrectors = 0;    ///< Pressure solve sub-loop count.
    /// Interior (alpha, k) pair selection.
    NonOrthSplit nonOrthSplit = NonOrthSplit::MinimumCorrection;
};

} // namespace fvm::numerical
