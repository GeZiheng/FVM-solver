#include "BoundaryCondition.h"
#include "CartesianMesh.h"
#include "Field.h"
#include "GridOperators.h"
#include "ObliqueMeshFixtures.h"
#include "UnstructuredMesh.h"

#include <algorithm>
#include <cmath>
#include <doctest/doctest.h>
#include <vector>

using namespace fvm::core;
using namespace fvm::numerical;

namespace
{
struct LinearField
{
    static Scalar value(const Point2& point)
    {
        return 1.0 + 2.0 * point.x - 3.0 * point.y;
    }

    static Scalar derivativeX()
    {
        return 2.0;
    }

    static Scalar derivativeY()
    {
        return -3.0;
    }
};

struct LinearXField
{
    static Scalar value(const Point2& point)
    {
        return 1.0 + 2.0 * point.x;
    }

    static Scalar derivativeX()
    {
        return 2.0;
    }

    static Scalar derivativeY()
    {
        return 0.0;
    }
};

Scalar maxGradientError(const Mesh& mesh,
    const std::vector<Point2>& gradients,
    const Scalar expectedX,
    const Scalar expectedY)
{
    Scalar error = 0.0;
    for (const Point2& gradient : gradients)
    {
        error = std::max(error,
            std::hypot(gradient.x - expectedX, gradient.y - expectedY));
    }
    return error;
}
} // namespace

TEST_CASE("leastSquaresGradients is exact on a Cartesian mesh")
{
    const CartesianMesh mesh(4, 3, 0.0, 0.0, 2.0, 1.5);
    ScalarField phi(mesh, "phi");
    for (Index cell = 0; cell < mesh.cellCount(); ++cell)
    {
        phi(cell) = LinearXField::value(mesh.cellCentroid(cell));
    }

    BoundaryField bc(mesh);
    bc.set(BoundaryField::East, BCType::Dirichlet, 5.0);
    bc.set(BoundaryField::West, BCType::Dirichlet, 1.0);
    bc.set(BoundaryField::North, BCType::Neumann, 0.0);
    bc.set(BoundaryField::South, BCType::Neumann, 0.0);

    const std::vector<Point2> gradients
        = leastSquaresGradients(mesh, phi, bc);
    REQUIRE(gradients.size() == mesh.cellCount());
    CHECK(maxGradientError(mesh,
        gradients,
        LinearXField::derivativeX(),
        LinearXField::derivativeY())
        <= 1.0e-12);

    for (Index cell = 0; cell < mesh.cellCount(); ++cell)
    {
        CHECK(cellGradient(mesh, phi, bc, cell, 0)
            == doctest::Approx(LinearXField::derivativeX()).epsilon(1.0e-12));
        CHECK(cellGradient(mesh, phi, bc, cell, 1)
            == doctest::Approx(LinearXField::derivativeY()).epsilon(1.0e-12));
    }
}

TEST_CASE("leastSquaresGradients is exact on an oblique triangle pair")
{
    const UnstructuredMesh mesh(makeObliquePair());
    const Mesh& base = mesh;
    ScalarField phi(base, "phi");
    for (Index cell = 0; cell < base.cellCount(); ++cell)
    {
        phi(cell) = LinearField::value(base.cellCentroid(cell));
    }

    BoundaryField bc(base);
    for (Index patch = 0; patch < bc.patchCount(); ++patch)
    {
        for (const Index face : base.patchFaces(patch))
        {
            bc.set(patch,
                BCType::Dirichlet,
                LinearField::value(base.faceCentroid(face)));
        }
    }

    const std::vector<Point2> gradients
        = leastSquaresGradients(base, phi, bc);
    REQUIRE(gradients.size() == base.cellCount());
    CHECK(maxGradientError(base,
        gradients,
        LinearField::derivativeX(),
        LinearField::derivativeY())
        <= 1.0e-12);

    Scalar greenGaussError = 0.0;
    for (Index cell = 0; cell < base.cellCount(); ++cell)
    {
        const Scalar errorX = cellGradient(base, phi, bc, cell, 0)
            - LinearField::derivativeX();
        const Scalar errorY = cellGradient(base, phi, bc, cell, 1)
            - LinearField::derivativeY();
        greenGaussError = std::max(greenGaussError,
            std::hypot(errorX, errorY));
    }
    CHECK(greenGaussError > 1.0e-6);
}

TEST_CASE("leastSquaresGradients honors exact Neumann data")
{
    const UnstructuredMesh mesh(makeObliquePair());
    const Mesh& base = mesh;
    ScalarField phi(base, "phi");
    for (Index cell = 0; cell < base.cellCount(); ++cell)
    {
        phi(cell) = LinearField::value(base.cellCentroid(cell));
    }

    BoundaryField bc(base);
    for (Index patch = 0; patch < bc.patchCount(); ++patch)
    {
        for (const Index face : base.patchFaces(patch))
        {
            const Point2 area = base.faceAreaVector(face);
            const Scalar areaNorm = std::hypot(area.x, area.y);
            const Scalar normalDerivative
                = (LinearField::derivativeX() * area.x
                      + LinearField::derivativeY() * area.y)
                / areaNorm;
            bc.set(patch, BCType::Neumann, normalDerivative);
        }
    }

    const std::vector<Point2> gradients
        = leastSquaresGradients(base, phi, bc);
    REQUIRE(gradients.size() == base.cellCount());
    CHECK(maxGradientError(base,
        gradients,
        LinearField::derivativeX(),
        LinearField::derivativeY())
        <= 1.0e-12);
}
