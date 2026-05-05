#include <cmath>
#include <iostream>

#include "Field.h"
#include "Mesh.h"
#include "VtkWriter.h"

using namespace fvm::core;
using namespace fvm::io;

int main()
{
  // Create a 20x20 uniform Cartesian mesh on [0, 1] x [0, 1]
  Index nx = 20, ny = 20;
  Scalar xMin = 0.0, yMin = 0.0, xMax = 1.0, yMax = 1.0;

  CartesianMesh mesh(nx, ny, xMin, yMin, xMax, yMax);
  std::cout << "Mesh: " << nx << " x " << ny << " cells\n";
  std::cout << "Cell volume: " << mesh.cellVolume(0) << "\n";

  // Create a scalar field: Gaussian bump at center
  ScalarField phi(mesh, "phi");
  Scalar cx = (xMin + xMax) / 2.0;
  Scalar cy = (yMin + yMax) / 2.0;
  Scalar sigma = 0.1;

  for (Index j = 0; j < ny; ++j) {
    for (Index i = 0; i < nx; ++i) {
      auto [x, y] = mesh.cellCenter(i, j);
      Scalar r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      phi(i, j) = std::exp(-r2 / (2.0 * sigma * sigma));
    }
  }

  // Create a vector field: rotational velocity
  VectorField velocity(mesh, "velocity");
  for (Index j = 0; j < ny; ++j) {
    for (Index i = 0; i < nx; ++i) {
      auto [x, y] = mesh.cellCenter(i, j);
      velocity.u()(i, j) = -(y - cy);
      velocity.v()(i, j) = (x - cx);
    }
  }

  // Write to VTK
  std::string filename = "gaussian_bump.vti";
  VtkWriter::write(filename, mesh,
                   { { "phi", &phi } },
                   { { "velocity", &velocity } });

  std::cout << "Wrote " << filename << "\n";
  std::cout << "Open with ParaView to visualize.\n";

  return 0;
}
