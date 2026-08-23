#include "TransportEquation.h"
#include <stdexcept>

namespace fvm::numerical {

EquationSystem assembleTransport(const CartesianMesh& mesh,
                                 const VectorField& velocity,
                                 const ScalarField& gamma,
                                 Scalar rho,
                                 ConvectionScheme scheme,
                                 const BoundaryField& bc,
                                 const ScalarField* Sc,
                                 const ScalarField* Sp)
{
  EquationSystem sys(mesh.cellCount());

  assembleDiffusion(mesh, gamma, bc, sys.A, sys.b);
  assembleConvection(mesh, velocity, rho, scheme, bc, sys.A, sys.b);

  const Index nCells = mesh.cellCount();
  const Scalar vol = mesh.cellVolume(0);

  if (Sc) {
    for (Index c = 0; c < nCells; ++c) {
      sys.b(c) += (*Sc)(c) * vol;
    }
  }

  if (Sp) {
    for (Index c = 0; c < nCells; ++c) {
      if ((*Sp)(c) > 0.0) {
        throw std::invalid_argument(
          "assembleTransport: Sp must be non-positive (Patankar linearization)");
      }
      sys.A.insert(c, c, -(*Sp)(c) * vol);
    }
  }

  return sys;
}

} // namespace fvm::numerical
