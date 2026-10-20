#include "Diffusion.h"
#include "GridOperators.h"
#include "MeshGeomCache.h"

#include <cmath>
#include <vector>

namespace fvm::numerical
{
using fvm::core::MeshGeomCache;
using fvm::core::Point2;

void assembleDiffusion(const Mesh& mesh,
    const ScalarField& gamma,
    const BoundaryField& bc,
    SparseMatrix& A,
    Vector& b,
    const ScalarField* explicitPhi,
    const CorrectionOptions& corrections)
{
    const MeshGeomCache geometry(mesh);
    const bool applyCrossDiffusion
        = explicitPhi != nullptr && corrections.nonOrthogonalDiffusion;
    const std::vector<Point2> gradients = applyCrossDiffusion
        ? leastSquaresGradients(mesh, *explicitPhi, bc)
        : std::vector<Point2>();

    for (Index face = 0; face < mesh.faceCount(); ++face)
    {
        const Index owner = mesh.faceOwner(face);

        if (!mesh.isBoundaryFace(face))
        {
            const Index neighbour = mesh.faceNeighbour(face);
            const Scalar gammaF = 0.5 * (gamma(owner) + gamma(neighbour));
            const Scalar diffusion = gammaF * geometry.innerCoef(face);

            A.insert(owner, owner, diffusion);
            A.insert(neighbour, neighbour, diffusion);
            A.insert(owner, neighbour, -diffusion);
            A.insert(neighbour, owner, -diffusion);

            if (applyCrossDiffusion)
            {
                const Scalar g = geometry.interpWeight(face);
                const Point2 gradF{ g * gradients[owner].x
                        + (1.0 - g) * gradients[neighbour].x,
                    g * gradients[owner].y
                        + (1.0 - g) * gradients[neighbour].y };
                const Point2 correction = geometry.nonOrthCorrection(face);
                const Scalar crossDiffusion = gammaF
                    * (correction.x * gradF.x + correction.y * gradF.y);

                b(owner) += crossDiffusion;
                b(neighbour) -= crossDiffusion;
            }
        }
        else
        {
            const BoundaryCondition& cond = bc.get(mesh.facePatch(face));

            if (cond.type == BCType::Dirichlet)
            {
                const Scalar diffusion
                    = gamma(owner) * geometry.boundaryCoef(face);
                A.insert(owner, owner, diffusion);
                b(owner) += diffusion * cond.value;
            }
            else
            {
                const Point2 area = mesh.faceAreaVector(face);
                const Scalar faceArea
                    = std::sqrt(area.x * area.x + area.y * area.y);
                b(owner) += gamma(owner) * cond.value * faceArea;
            }
        }
    }
}

} // namespace fvm::numerical
