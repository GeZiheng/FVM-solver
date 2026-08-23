#include "Diffusion.h"

namespace fvm::numerical {

namespace {

// Handle a boundary face of cell P on the given side.
void applyBoundaryFace(const CartesianMesh& mesh,
                       const ScalarField& gamma,
                       const BoundaryField& bc,
                       Index P,
                       int face,
                       SparseMatrix& A,
                       Vector& b)
{
  const BoundaryCondition& cond = bc.get(face);
  const Scalar Sf = mesh.faceArea(face);
  const Scalar gP = gamma(P);

  if (cond.type == BCType::Dirichlet) {
    // Diffusive flux through boundary face: D_b * (phi_P - phi_b)
    const Scalar Db = gP * Sf / mesh.cellToFaceDistance(face);
    A.insert(P, P, Db);
    b(P) += Db * cond.value;
  } else {
    // Neumann: -gamma * d(phi)/dn * S_f is a known flux -> move to RHS.
    // cond.value is the outward-normal derivative g.
    b(P) += gP * cond.value * Sf;
  }
}

} // namespace

void assembleDiffusion(const CartesianMesh& mesh,
                       const ScalarField& gamma,
                       const BoundaryField& bc,
                       SparseMatrix& A,
                       Vector& b)
{
  const Index nCells = mesh.cellCount();

  for (Index P = 0; P < nCells; ++P) {
    for (int face = 0; face < 4; ++face) {
      const Index N = mesh.neighbor(P, face);

      if (N != nCells) {
        // Interior face: process only east/north to avoid double counting.
        if (face != BoundaryField::East && face != BoundaryField::North)
          continue;

        const Scalar gf = 0.5 * (gamma(P) + gamma(N));
        const Scalar Df = gf * mesh.faceArea(face) / mesh.cellToCellDistance(P, face);

        A.insert(P, P, Df);
        A.insert(N, N, Df);
        A.insert(P, N, -Df);
        A.insert(N, P, -Df);
      } else {
        // Boundary face (west/south handled here since they are not
        // reached from any other cell).
        applyBoundaryFace(mesh, gamma, bc, P, face, A, b);
      }
    }
  }
}

} // namespace fvm::numerical
