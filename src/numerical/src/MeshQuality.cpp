#include "MeshQuality.h"

#include "MeshGeomCache.h"

#include <sstream>

namespace fvm::numerical
{
using fvm::core::MeshGeomCache;

std::string meshQualityWarning(const Mesh& mesh,
    Scalar maxNonOrthDeg,
    Scalar maxSkew)
{
    const MeshGeomCache geometry(mesh);
    std::ostringstream message;

    if (geometry.maxNonOrthogonality() > maxNonOrthDeg)
    {
        message << "warning: max non-orthogonality exceeds "
                << maxNonOrthDeg
                << " degrees; deferred corrections may converge slowly\n";
    }
    if (geometry.maxSkewness() > maxSkew)
    {
        message << "warning: max face skewness |s|/|d| exceeds " << maxSkew
                << "; interpolation corrections may lose accuracy\n";
    }
    return message.str();
}

} // namespace fvm::numerical
