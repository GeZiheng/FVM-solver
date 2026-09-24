#include "Convection.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace fvm::numerical
{

namespace
{

// Fill the flux field: interior faces use the arithmetic mean of the
// adjacent cell velocities, boundary faces the adjacent cell velocity.
void interpolateVelocityIntoFlux(const CartesianMesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    FaceFluxField& flux)
{
    const Index nx = mesh.nx();
    const Index ny = mesh.ny();
    const Scalar Sx = mesh.faceArea(BoundaryField::East);
    const Scalar Sy = mesh.faceArea(BoundaryField::North);

    for (Index j = 0; j < ny; ++j)
    {
        for (Index i = 0; i <= nx; ++i)
        {
            Scalar uf;
            if (i == 0)
                uf = velocity.u()(0, j);
            else if (i == nx)
                uf = velocity.u()(nx - 1, j);
            else
                uf = 0.5 * (velocity.u()(i - 1, j) + velocity.u()(i, j));
            flux.x(i, j) = rho * uf * Sx;
        }
    }
    for (Index j = 0; j <= ny; ++j)
    {
        for (Index i = 0; i < nx; ++i)
        {
            Scalar vf;
            if (j == 0)
                vf = velocity.v()(i, 0);
            else if (j == ny)
                vf = velocity.v()(i, ny - 1);
            else
                vf = 0.5 * (velocity.v()(i, j - 1) + velocity.v()(i, j));
            flux.y(i, j) = rho * vf * Sy;
        }
    }
}

} // namespace

FaceFluxField interpolateCellVelocityFlux(const CartesianMesh& mesh,
    const VectorField& velocity,
    Scalar rho)
{
    FaceFluxField flux(mesh, "phi");
    interpolateVelocityIntoFlux(mesh, velocity, rho, flux);
    return flux;
}

void computeMassFlux(const CartesianMesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    FaceFluxField& flux)
{
    interpolateVelocityIntoFlux(mesh, velocity, rho, flux);
    const Index nx = mesh.nx();
    const Index ny = mesh.ny();
    const Scalar Sx = mesh.faceArea(BoundaryField::East);
    const Scalar Sy = mesh.faceArea(BoundaryField::North);

    // Boundary faces with a Dirichlet velocity BC take the prescribed
    // value for the normal component.
    const BoundaryCondition& west = bcU.get(BoundaryField::West);
    const BoundaryCondition& east = bcU.get(BoundaryField::East);
    if (west.type == BCType::Dirichlet || east.type == BCType::Dirichlet)
    {
        for (Index j = 0; j < ny; ++j)
        {
            if (west.type == BCType::Dirichlet)
                flux.x(0, j) = rho * west.value * Sx;
            if (east.type == BCType::Dirichlet)
                flux.x(nx, j) = rho * east.value * Sx;
        }
    }
    const BoundaryCondition& south = bcV.get(BoundaryField::South);
    const BoundaryCondition& north = bcV.get(BoundaryField::North);
    if (south.type == BCType::Dirichlet || north.type == BCType::Dirichlet)
    {
        for (Index i = 0; i < nx; ++i)
        {
            if (south.type == BCType::Dirichlet)
                flux.y(i, 0) = rho * south.value * Sy;
            if (north.type == BCType::Dirichlet)
                flux.y(i, ny) = rho * north.value * Sy;
        }
    }
}

void checkFluxCompatibility(const FaceFluxField& flux, Scalar relTol)
{
    const CartesianMesh& mesh = flux.mesh();
    const Index nx = mesh.nx();
    const Index ny = mesh.ny();

    Scalar net = 0.0;
    Scalar scale = 0.0;
    for (Index j = 0; j < ny; ++j)
    {
        const Scalar wOut = -flux.x(0, j);
        const Scalar eOut = flux.x(nx, j);
        net += wOut + eOut;
        scale += std::abs(wOut) + std::abs(eOut);
    }
    for (Index i = 0; i < nx; ++i)
    {
        const Scalar sOut = -flux.y(i, 0);
        const Scalar nOut = flux.y(i, ny);
        net += sOut + nOut;
        scale += std::abs(sOut) + std::abs(nOut);
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
              "(e.g. inlet flux = outlet flux).");
    }
}

void assembleConvection(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    SparseMatrix& A,
    Vector& b)
{
    const Index nCells = mesh.cellCount();

    for (Index P = 0; P < nCells; ++P)
    {
        const auto [iP, jP] = mesh.cellIJ(P);
        for (int face = 0; face < 4; ++face)
        {
            const Index N = mesh.neighbor(P, face);

            if (N != nCells)
            {
                // Interior face: process only east/north to avoid double
                // counting. F is positive from P to N (stored convention).
                if (face != BoundaryField::East && face != BoundaryField::North)
                    continue;

                const Scalar F = (face == BoundaryField::East)
                                     ? flux.x(iP + 1, jP)
                                     : flux.y(iP, jP + 1);

                if (scheme == ConvectionScheme::Upwind)
                {
                    if (F > 0.0)
                    { // flow P -> N: face value = phi_P
                        A.insert(P, P, F);
                        A.insert(N, P, -F);
                    }
                    else
                    { // flow N -> P: face value = phi_N
                        A.insert(N, N, -F);
                        A.insert(P, N, F);
                    }
                }
                else
                { // Central: face value = 0.5 * (phi_P + phi_N)
                    const Scalar h = 0.5 * F;
                    A.insert(P, P, h);
                    A.insert(P, N, h);
                    A.insert(N, P, -h);
                    A.insert(N, N, -h);
                }
            }
            else
            {
                // Boundary face: outward flux from the field.
                const Scalar Fb = flux.outwardFlux(P, face);

                if (Fb > 0.0)
                {
                    // Outflow: extrapolate cell value (upwind).
                    A.insert(P, P, Fb);
                }
                else
                {
                    // Inflow.
                    const BoundaryCondition& cond = bc.get(face);
                    if (cond.type == BCType::Dirichlet)
                    {
                        b(P) -= Fb * cond.value; // known incoming flux -> RHS
                    }
                    else
                    {
                        A.insert(P, P, Fb); // zero-gradient: phi_face = phi_P
                    }
                }
            }
        }
    }
}

} // namespace fvm::numerical
