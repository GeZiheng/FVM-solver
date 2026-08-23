#include "Convection.h"

namespace fvm::numerical
{

namespace
{

// Mass flux through an interior face (positive along the outward normal of P).
Scalar interiorFaceFlux(const CartesianMesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    Index P,
    Index N,
    int face)
{
    const auto [nx, ny] = mesh.faceNormal(face);
    const auto [uPx, uPy] = velocity(P);
    const auto [uNx, uNy] = velocity(N);
    const Scalar uf = 0.5 * (uPx + uNx) * nx + 0.5 * (uPy + uNy) * ny;
    return rho * uf * mesh.faceArea(face);
}

// Mass flux through a boundary face, using the cell-center velocity.
Scalar boundaryFaceFlux(const CartesianMesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    Index P,
    int face)
{
    const auto [nx, ny] = mesh.faceNormal(face);
    const auto [uPx, uPy] = velocity(P);
    const Scalar ub = uPx * nx + uPy * ny;
    return rho * ub * mesh.faceArea(face);
}

} // namespace

void assembleConvection(const CartesianMesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    SparseMatrix& A,
    Vector& b)
{
    const Index nCells = mesh.cellCount();

    for (Index P = 0; P < nCells; ++P)
    {
        for (int face = 0; face < 4; ++face)
        {
            const Index N = mesh.neighbor(P, face);

            if (N != nCells)
            {
                // Interior face: process only east/north to avoid double
                // counting.
                if (face != BoundaryField::East && face != BoundaryField::North)
                    continue;

                const Scalar F
                    = interiorFaceFlux(mesh, velocity, rho, P, N, face);

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
                // Boundary face.
                const Scalar Fb
                    = boundaryFaceFlux(mesh, velocity, rho, P, face);

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
