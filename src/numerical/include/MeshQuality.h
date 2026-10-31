#pragma once

#include "Mesh.h"
#include "Types.h"

#include <string>

namespace fvm::numerical
{
using fvm::core::Mesh;
using fvm::core::Scalar;

/// Mesh-quality warning text for the solver drivers: empty when the mesh is
/// within both thresholds.  Returned as a string instead of logging directly
/// so tests can exercise the thresholds without capturing stderr.
/// `maxNonOrthDeg` gates the deferred-correction convergence warning (largest
/// interior non-orthogonality angle in degrees); `maxSkew` gates the
/// interpolation-accuracy warning (largest interior |s_f|/|d_f|).  Both
/// warnings are advisory; the solve continues unchanged.
std::string meshQualityWarning(const Mesh& mesh,
    Scalar maxNonOrthDeg = 70.0,
    Scalar maxSkew = 1.0);

} // namespace fvm::numerical
