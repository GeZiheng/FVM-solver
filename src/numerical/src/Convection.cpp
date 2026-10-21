#include "Convection.h"
#include "GridOperators.h"
#include "MeshGeomCache.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace fvm::numerical
{
using fvm::core::MeshGeomCache;
using fvm::core::Point2;

FaceFluxField interpolateCellVelocityFlux(const Mesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    const CorrectionOptions& corrections)
{
    FaceFluxField flux(mesh, "phi");
    const MeshGeomCache geometry(mesh);
    const std::vector<Point2> uGradients
        = corrections.skewInterpolation
        ? leastSquaresGradients(mesh,
            velocity.u(),
            BoundaryField(mesh))
        : std::vector<Point2>();
    const std::vector<Point2> vGradients
        = corrections.skewInterpolation
        ? leastSquaresGradients(mesh,
            velocity.v(),
            BoundaryField(mesh))
        : std::vector<Point2>();

    for (Index face = 0; face < mesh.faceCount(); ++face)
    {
        const Point2 area = mesh.faceAreaVector(face);

        if (mesh.isBoundaryFace(face))
        {
            const auto [u, v] = velocity(mesh.faceOwner(face));
            flux[face] = rho * (u * area.x + v * area.y);
        }
        else
        {
            const Scalar g = geometry.interpWeight(face);
            const Index owner = mesh.faceOwner(face);
            const Index neighbour = mesh.faceNeighbour(face);
            const auto [uOwner, vOwner] = velocity(owner);
            const auto [uNeighbour, vNeighbour] = velocity(neighbour);
            const Scalar uf = g * uOwner + (1.0 - g) * uNeighbour;
            const Scalar vf = g * vOwner + (1.0 - g) * vNeighbour;
            Scalar uFace = uf;
            Scalar vFace = vf;
            if (corrections.skewInterpolation)
            {
                const Point2 skew = geometry.skewVector(face);
                const Point2 uFGrad{ g * uGradients[owner].x
                        + (1.0 - g) * uGradients[neighbour].x,
                    g * uGradients[owner].y
                        + (1.0 - g) * uGradients[neighbour].y };
                const Point2 vFGrad{ g * vGradients[owner].x
                        + (1.0 - g) * vGradients[neighbour].x,
                    g * vGradients[owner].y
                        + (1.0 - g) * vGradients[neighbour].y };
                uFace += uFGrad.x * skew.x + uFGrad.y * skew.y;
                vFace += vFGrad.x * skew.x + vFGrad.y * skew.y;
            }
            flux[face] = rho * (uFace * area.x + vFace * area.y);
        }
    }

    return flux;
}

void computeMassFlux(const Mesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    FaceFluxField& flux,
    const CorrectionOptions& corrections)
{
    const MeshGeomCache geometry(mesh);
    const std::vector<Point2> uGradients
        = corrections.skewInterpolation
        ? leastSquaresGradients(mesh, velocity.u(), bcU)
        : std::vector<Point2>();
    const std::vector<Point2> vGradients
        = corrections.skewInterpolation
        ? leastSquaresGradients(mesh, velocity.v(), bcV)
        : std::vector<Point2>();

    for (Index face = 0; face < mesh.faceCount(); ++face)
    {
        const Point2 area = mesh.faceAreaVector(face);
        const Index owner = mesh.faceOwner(face);
        Scalar uf;
        Scalar vf;

        if (mesh.isBoundaryFace(face))
        {
            const auto patch = mesh.facePatch(face);
            const auto [uOwner, vOwner] = velocity(owner);

            const BoundaryCondition& uCond = bcU.get(patch);
            uf = (uCond.type == BCType::Dirichlet) ? uCond.value : uOwner;
            const BoundaryCondition& vCond = bcV.get(patch);
            vf = (vCond.type == BCType::Dirichlet) ? vCond.value : vOwner;
        }
        else
        {
            const Scalar g = geometry.interpWeight(face);
            const Index neighbour = mesh.faceNeighbour(face);
            const auto [uOwner, vOwner] = velocity(owner);
            const auto [uNeighbour, vNeighbour] = velocity(neighbour);
            uf = g * uOwner + (1.0 - g) * uNeighbour;
            vf = g * vOwner + (1.0 - g) * vNeighbour;

            if (corrections.skewInterpolation)
            {
                const Point2 skew = geometry.skewVector(face);
                const Point2 uFGrad{ g * uGradients[owner].x
                        + (1.0 - g) * uGradients[neighbour].x,
                    g * uGradients[owner].y
                        + (1.0 - g) * uGradients[neighbour].y };
                const Point2 vFGrad{ g * vGradients[owner].x
                        + (1.0 - g) * vGradients[neighbour].x,
                    g * vGradients[owner].y
                        + (1.0 - g) * vGradients[neighbour].y };
                uf += uFGrad.x * skew.x + uFGrad.y * skew.y;
                vf += vFGrad.x * skew.x + vFGrad.y * skew.y;
            }
        }

        flux[face] = rho * (uf * area.x + vf * area.y);
    }
}

void checkFluxCompatibility(const FaceFluxField& flux, Scalar relTol)
{
    const Mesh& mesh = flux.mesh();
    Scalar net = 0.0;
    Scalar scale = 0.0;

    for (Index face = 0; face < mesh.faceCount(); ++face)
    {
        if (!mesh.isBoundaryFace(face))
            continue;

        const Scalar outward = flux[face];
        net += outward;
        scale += std::abs(outward);
    }

    if (std::abs(net) > relTol * scale)
    {
        throw std::runtime_error(
            "checkFluxCompatibility: net boundary outflow "
            + std::to_string(net)
            + " exceeds tolerance (relative to total boundary flux "
            + std::to_string(scale)
            + "); the pressure-correction equation is incompatible. "
              "Check that the velocity boundary conditions balance "
            + "(e.g. inlet flux = outlet flux).");
    }
}

bool hasDirichletPressure(const BoundaryField& bcP)
{
    for (Index patch = 0; patch < bcP.patchCount(); ++patch)
    {
        if (bcP.get(patch).type == BCType::Dirichlet)
            return true;
    }
    return false;
}

void initializeMassFlux(const Mesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const BoundaryField& bcP,
    FaceFluxField& flux,
    const CorrectionOptions& corrections)
{
    computeMassFlux(mesh, velocity, rho, bcU, bcV, flux, corrections);

    // Pure-Neumann p is only compatible when the boundary fluxes balance;
    // catch unbalanced velocity BCs here instead of solving an inconsistent
    // system.
    if (!hasDirichletPressure(bcP))
        checkFluxCompatibility(flux);
}

void assembleConvection(const Mesh& mesh,
    const FaceFluxField& flux,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    SparseMatrix& A,
    Vector& b,
    const ScalarField* explicitPhi,
    const CorrectionOptions& corrections)
{
    const MeshGeomCache geometry(mesh);
    const bool applySkew = explicitPhi != nullptr
        && corrections.skewInterpolation
        && scheme == ConvectionScheme::Central;
    const std::vector<Point2> gradients = applySkew
        ? leastSquaresGradients(mesh, *explicitPhi, bc)
        : std::vector<Point2>();

    for (Index face = 0; face < mesh.faceCount(); ++face)
    {
        const Index owner = mesh.faceOwner(face);
        const Scalar F = flux[face];

        if (!mesh.isBoundaryFace(face))
        {
            const Index neighbour = mesh.faceNeighbour(face);
            const Scalar g = geometry.interpWeight(face);

            if (scheme == ConvectionScheme::Upwind)
            {
                if (F >= 0.0)
                { // flow owner -> neighbour: face value = phi_owner
                    A.insert(owner, owner, F);
                    A.insert(neighbour, owner, -F);
                }
                else
                { // flow neighbour -> owner: face value = phi_neighbour
                    A.insert(owner, neighbour, F);
                    A.insert(neighbour, neighbour, -F);
                }
            }
            else
            { // Central: face value = g*phi_o + (1-g)*phi_n
                A.insert(owner, owner, F * g);
                A.insert(owner, neighbour, F * (1.0 - g));
                A.insert(neighbour, owner, -F * g);
                A.insert(neighbour, neighbour, -F * (1.0 - g));

                if (applySkew)
                {
                    const Point2 skew = geometry.skewVector(face);
                    const Point2 gradF{ g * gradients[owner].x
                            + (1.0 - g) * gradients[neighbour].x,
                        g * gradients[owner].y
                            + (1.0 - g) * gradients[neighbour].y };
                    const Scalar correction
                        = gradF.x * skew.x + gradF.y * skew.y;
                    b(owner) -= F * correction;
                    b(neighbour) += F * correction;
                }
            }
        }
        else if (F >= 0.0)
        {
            // Outflow: extrapolate the owner value (upwind).
            A.insert(owner, owner, F);
        }
        else
        {
            const BoundaryCondition& cond = bc.get(mesh.facePatch(face));
            if (cond.type == BCType::Dirichlet)
            {
                b(owner) -= F * cond.value; // known incoming flux -> RHS
            }
            else
            {
                A.insert(owner, owner, F); // zero-gradient: phi_face = phi_P
            }
        }
    }
}

} // namespace fvm::numerical
