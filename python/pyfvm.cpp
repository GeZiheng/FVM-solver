// pyfvm: pybind11 bindings for the fvm_core library.
//
// Exposes just what a Python case script needs: mesh + field classes with
// zero-copy numpy views, boundary conditions, the SIMPLE/PISO drivers, a
// high-level scalar-transport solver (SparseMatrix/EquationSystem stay in
// C++), the flux helpers, and VTK output. Python names are snake_case.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

#include "BoundaryCondition.h"
#include "Convection.h"
#include "Field.h"
#include "FluxField.h"
#include "LinearSolver.h"
#include "Mesh.h"
#include "Piso.h"
#include "Simple.h"
#include "TimeScheme.h"
#include "TransportEquation.h"
#include "VtkWriter.h"

namespace py = pybind11;

namespace fvm::python
{

using fvm::core::CartesianMesh;
using fvm::core::FaceFluxField;
using fvm::core::Index;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::Vector;
using fvm::core::VectorField;
using fvm::io::VtkWriter;
using fvm::math::SolverConfig;
using fvm::numerical::BCType;
using fvm::numerical::BoundaryCondition;
using fvm::numerical::BoundaryField;
using fvm::numerical::ConvectionScheme;
using fvm::numerical::PisoConfig;
using fvm::numerical::PisoResult;
using fvm::numerical::PisoStepInfo;
using fvm::numerical::SimpleConfig;
using fvm::numerical::SimpleResiduals;
using fvm::numerical::SimpleResult;
using fvm::numerical::TimeScheme;

/// Zero-copy writable 1-D numpy view over an Eigen vector; `owner` (the
/// field object) is kept alive by the array's base.
py::array_t<Scalar> vectorView(const py::object& owner, Vector& v)
{
    return py::array_t<Scalar>({ static_cast<py::ssize_t>(v.size()) },
        { static_cast<py::ssize_t>(sizeof(Scalar)) },
        v.data(),
        owner);
}

/// Copy of an Eigen vector as a new numpy array.
py::array_t<Scalar> vectorCopy(const Vector& v)
{
    return py::array_t<Scalar>({ static_cast<py::ssize_t>(v.size()) },
        { static_cast<py::ssize_t>(sizeof(Scalar)) },
        v.data());
}

/// High-level steady scalar transport: assemble, solve with BiCGSTAB,
/// return the solution as a numpy array (row-major cell order, j * nx + i).
py::array_t<Scalar> solveTransport(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    const ScalarField& gamma,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    const SolverConfig& solverConfig,
    const ScalarField* Sc,
    const ScalarField* Sp)
{
    auto sys = numerical::assembleTransport(mesh, flux, gamma, scheme, bc, Sc, Sp);
    sys.A.finalize();
    auto solver = math::createEigenBiCGSTAB(solverConfig);
    Vector x;
    {
        py::gil_scoped_release release;
        x = solver->solve(sys.A, sys.b);
    }
    return vectorCopy(x);
}

/// High-level transient scalar transport: one implicit theta step.
py::array_t<Scalar> solveTransientTransport(const CartesianMesh& mesh,
    const FaceFluxField& flux,
    const ScalarField& gamma,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    const ScalarField& phiOld,
    Scalar dt,
    TimeScheme timeScheme,
    const SolverConfig& solverConfig,
    const ScalarField* Sc,
    const ScalarField* Sp)
{
    auto sys = numerical::assembleTransientTransport(
        mesh, flux, gamma, scheme, bc, phiOld, dt, timeScheme, Sc, Sp);
    sys.A.finalize();
    auto solver = math::createEigenBiCGSTAB(solverConfig);
    Vector x;
    {
        py::gil_scoped_release release;
        x = solver->solve(sys.A, sys.b);
    }
    return vectorCopy(x);
}

// The drivers run long enough to be worth releasing the GIL, but pybind11's
// call_guard would still hold it released during the return-value conversion
// (which creates Python objects and needs the GIL). Release it manually
// around the C++ call only.
SimpleResult runSimple(const CartesianMesh& mesh,
    Scalar rho,
    Scalar mu,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const BoundaryField& bcP,
    const SimpleConfig& config,
    VectorField& velocity,
    ScalarField& pressure,
    FaceFluxField& flux)
{
    py::gil_scoped_release release;
    return numerical::solveSimple(
        mesh, rho, mu, bcU, bcV, bcP, config, velocity, pressure, flux);
}

PisoResult runPiso(const CartesianMesh& mesh,
    Scalar rho,
    Scalar mu,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const BoundaryField& bcP,
    const PisoConfig& config,
    VectorField& velocity,
    ScalarField& pressure,
    FaceFluxField& flux)
{
    py::gil_scoped_release release;
    return numerical::solvePiso(
        mesh, rho, mu, bcU, bcV, bcP, config, velocity, pressure, flux);
}

/// VtkWriter::write with Python dicts {name: field}.
void writeVti(const std::string& filename,
    const CartesianMesh& mesh,
    const py::dict& scalarFields,
    const py::dict& vectorFields)
{
    std::vector<std::pair<std::string, const ScalarField*>> scalars;
    for (const auto& item : scalarFields)
    {
        scalars.emplace_back(py::cast<std::string>(item.first),
            py::cast<const ScalarField*>(item.second));
    }
    std::vector<std::pair<std::string, const VectorField*>> vectors;
    for (const auto& item : vectorFields)
    {
        vectors.emplace_back(py::cast<std::string>(item.first),
            py::cast<const VectorField*>(item.second));
    }
    VtkWriter::write(filename, mesh, scalars, vectors);
}

void bindCore(py::module_& m)
{
    py::class_<CartesianMesh>(m, "CartesianMesh")
        .def(py::init<Index, Index, Scalar, Scalar, Scalar, Scalar>(),
            py::arg("nx"), py::arg("ny"),
            py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"))
        .def("nx", &CartesianMesh::nx)
        .def("ny", &CartesianMesh::ny)
        .def("cell_count", &CartesianMesh::cellCount)
        .def("dx", &CartesianMesh::dx)
        .def("dy", &CartesianMesh::dy)
        .def("cell_volume", &CartesianMesh::cellVolume, py::arg("cell"))
        .def("cell_center",
            py::overload_cast<Index>(&CartesianMesh::cellCenter, py::const_),
            py::arg("cell"))
        .def("cell_index", &CartesianMesh::cellIndex, py::arg("i"), py::arg("j"))
        .def("cell_ij", &CartesianMesh::cellIJ, py::arg("cell"));

    // Fields hold a mesh reference: keep the mesh alive while the field lives.
    py::class_<ScalarField>(m, "ScalarField")
        .def(py::init<const CartesianMesh&, const std::string&>(),
            py::arg("mesh"), py::arg("name") = "", py::keep_alive<1, 2>())
        .def("name", &ScalarField::name)
        .def("size", &ScalarField::size)
        .def("set_zero", &ScalarField::setZero)
        .def("set_constant", &ScalarField::setConstant, py::arg("value"))
        .def("to_numpy", [](const py::object& self)
            { return vectorView(self, self.cast<ScalarField&>().data()); },
            "Writable 1-D numpy view of the cell data (cell index j * nx + i).")
        .def("value", [](const ScalarField& f, Index i, Index j)
            { return f(i, j); }, py::arg("i"), py::arg("j"))
        .def("set", [](ScalarField& f, Index i, Index j, Scalar v)
            { f(i, j) = v; }, py::arg("i"), py::arg("j"), py::arg("value"));

    py::class_<VectorField>(m, "VectorField")
        .def(py::init<const CartesianMesh&, const std::string&>(),
            py::arg("mesh"), py::arg("name") = "", py::keep_alive<1, 2>())
        .def("name", &VectorField::name)
        .def("set_zero", &VectorField::setZero)
        .def_property_readonly("u",
            static_cast<ScalarField& (VectorField::*)()>(&VectorField::u),
            py::return_value_policy::reference_internal)
        .def_property_readonly("v",
            static_cast<ScalarField& (VectorField::*)()>(&VectorField::v),
            py::return_value_policy::reference_internal);

    py::class_<FaceFluxField>(m, "FaceFluxField")
        .def(py::init<const CartesianMesh&, const std::string&>(),
            py::arg("mesh"), py::arg("name") = "", py::keep_alive<1, 2>())
        .def("name", &FaceFluxField::name)
        .def("x_face_count", &FaceFluxField::xFaceCount)
        .def("y_face_count", &FaceFluxField::yFaceCount)
        .def("set_zero", &FaceFluxField::setZero)
        .def("x", [](const FaceFluxField& f, Index i, Index j)
            { return f.x(i, j); }, py::arg("i"), py::arg("j"))
        .def("y", [](const FaceFluxField& f, Index i, Index j)
            { return f.y(i, j); }, py::arg("i"), py::arg("j"))
        .def("to_numpy_x", [](const py::object& self)
            { return vectorView(self, self.cast<FaceFluxField&>().xData()); },
            "Writable 1-D numpy view of the x-face fluxes, index j * (nx + 1) + i.")
        .def("to_numpy_y", [](const py::object& self)
            { return vectorView(self, self.cast<FaceFluxField&>().yData()); },
            "Writable 1-D numpy view of the y-face fluxes, index j * nx + i.")
        .def("cell_imbalance", &FaceFluxField::cellImbalance, py::arg("cell"));
}

void bindNumerical(py::module_& m)
{
    py::enum_<BCType>(m, "BCType")
        .value("Dirichlet", BCType::Dirichlet)
        .value("Neumann", BCType::Neumann)
        .export_values();

    py::enum_<ConvectionScheme>(m, "ConvectionScheme")
        .value("Upwind", ConvectionScheme::Upwind)
        .value("Central", ConvectionScheme::Central)
        .export_values();

    py::enum_<TimeScheme>(m, "TimeScheme")
        .value("Euler", TimeScheme::Euler)
        .value("CrankNicolson", TimeScheme::CrankNicolson)
        .export_values();

    py::class_<BoundaryCondition>(m, "BoundaryCondition")
        .def(py::init<>())
        .def_readwrite("type", &BoundaryCondition::type)
        .def_readwrite("value", &BoundaryCondition::value);

    auto pyBoundaryField = py::class_<BoundaryField>(m, "BoundaryField")
        .def(py::init<>())
        .def("set", &BoundaryField::set,
            py::arg("side"), py::arg("type"), py::arg("value"))
        .def("get", &BoundaryField::get, py::arg("side"),
            py::return_value_policy::copy);
    pyBoundaryField.attr("East") = py::int_(BoundaryField::East);
    pyBoundaryField.attr("North") = py::int_(BoundaryField::North);
    pyBoundaryField.attr("West") = py::int_(BoundaryField::West);
    pyBoundaryField.attr("South") = py::int_(BoundaryField::South);

    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("tolerance", &SolverConfig::tolerance)
        .def_readwrite("max_iterations", &SolverConfig::maxIterations)
        .def_readwrite("verbose", &SolverConfig::verbose);

    // -- SIMPLE -------------------------------------------------------------
    py::class_<SimpleResiduals>(m, "SimpleResiduals")
        .def_readonly("continuity", &SimpleResiduals::continuity)
        .def_readonly("u", &SimpleResiduals::u)
        .def_readonly("v", &SimpleResiduals::v)
        .def_readonly("pressure", &SimpleResiduals::pressure);

    py::class_<SimpleConfig>(m, "SimpleConfig")
        .def(py::init<>())
        .def_readwrite("max_iterations", &SimpleConfig::maxIterations)
        .def_readwrite("tolerance", &SimpleConfig::tolerance)
        .def_readwrite("relaxation_u", &SimpleConfig::relaxationU)
        .def_readwrite("relaxation_p", &SimpleConfig::relaxationP)
        .def_readwrite("scheme", &SimpleConfig::scheme)
        .def_readwrite("solver_config", &SimpleConfig::solverConfig)
        .def_readwrite("verbose", &SimpleConfig::verbose);

    py::class_<SimpleResult>(m, "SimpleResult")
        .def_readonly("converged", &SimpleResult::converged)
        .def_readonly("iterations", &SimpleResult::iterations)
        .def_readonly("history", &SimpleResult::history);

    m.def("solve_simple", &runSimple,
        py::arg("mesh"), py::arg("rho"), py::arg("mu"),
        py::arg("bc_u"), py::arg("bc_v"), py::arg("bc_p"),
        py::arg("config"), py::arg("velocity"), py::arg("pressure"),
        py::arg("flux"),
        "Steady incompressible Navier-Stokes via SIMPLE (collocated, "
        "Rhie-Chow). velocity/pressure are in/out, flux is rebuilt.");

    // -- PISO ---------------------------------------------------------------
    py::class_<PisoStepInfo>(m, "PisoStepInfo")
        .def_readonly("time", &PisoStepInfo::time)
        .def_readonly("continuity", &PisoStepInfo::continuity)
        .def_readonly("max_speed", &PisoStepInfo::maxSpeed);

    py::class_<PisoConfig>(m, "PisoConfig")
        .def(py::init<>())
        .def_readwrite("dt", &PisoConfig::dt)
        .def_readwrite("n_steps", &PisoConfig::nSteps)
        .def_readwrite("n_correctors", &PisoConfig::nCorrectors)
        .def_readwrite("time_scheme", &PisoConfig::timeScheme)
        .def_readwrite("scheme", &PisoConfig::scheme)
        .def_readwrite("solver_config", &PisoConfig::solverConfig)
        .def_readwrite("verbose", &PisoConfig::verbose);

    py::class_<PisoResult>(m, "PisoResult")
        .def_readonly("steps", &PisoResult::steps)
        .def_readonly("history", &PisoResult::history);

    m.def("solve_piso", &runPiso,
        py::arg("mesh"), py::arg("rho"), py::arg("mu"),
        py::arg("bc_u"), py::arg("bc_v"), py::arg("bc_p"),
        py::arg("config"), py::arg("velocity"), py::arg("pressure"),
        py::arg("flux"),
        "Transient incompressible Navier-Stokes via PISO. "
        "velocity/pressure/flux are in/out and carried between steps.");

    // -- Flux helpers ---------------------------------------------------------
    m.def("interpolate_cell_velocity_flux",
        &numerical::interpolateCellVelocityFlux,
        py::arg("mesh"), py::arg("velocity"), py::arg("rho"),
        py::keep_alive<0, 1>(),
        "Face mass flux from a cell-centred velocity (no BCs; for transport "
        "with a prescribed velocity).");
    m.def("compute_mass_flux", &numerical::computeMassFlux,
        py::arg("mesh"), py::arg("velocity"), py::arg("rho"),
        py::arg("bc_u"), py::arg("bc_v"), py::arg("flux"));
    m.def("initialize_mass_flux", &numerical::initializeMassFlux,
        py::arg("mesh"), py::arg("velocity"), py::arg("rho"),
        py::arg("bc_u"), py::arg("bc_v"), py::arg("bc_p"), py::arg("flux"));
    m.def("check_flux_compatibility", &numerical::checkFluxCompatibility,
        py::arg("flux"), py::arg("rel_tol") = 1e-10);
    m.def("has_dirichlet_pressure", &numerical::hasDirichletPressure,
        py::arg("bc_p"));

    // -- Scalar transport (high level) ---------------------------------------
    m.def("solve_transport", &solveTransport,
        py::arg("mesh"), py::arg("flux"), py::arg("gamma"),
        py::arg("scheme"), py::arg("bc"),
        py::arg("solver_config") = SolverConfig{},
        py::arg("Sc") = nullptr, py::arg("Sp") = nullptr,
        "Steady convection-diffusion-source solve; returns the solution as a "
        "numpy array (cell order j * nx + i).");
    m.def("solve_transient_transport", &solveTransientTransport,
        py::arg("mesh"), py::arg("flux"), py::arg("gamma"),
        py::arg("scheme"), py::arg("bc"), py::arg("phi_old"),
        py::arg("dt"), py::arg("time_scheme"),
        py::arg("solver_config") = SolverConfig{},
        py::arg("Sc") = nullptr, py::arg("Sp") = nullptr,
        "One implicit theta-scheme time step of the scalar transport "
        "equation; returns the new solution as a numpy array.");
}

void bindAll(py::module_& m)
{
    bindCore(m);
    bindNumerical(m);

    m.def("write_vti", &writeVti,
        py::arg("filename"), py::arg("mesh"),
        py::arg("scalars") = py::dict(), py::arg("vectors") = py::dict(),
        "Write scalar/vector fields (dicts of name -> field) to a .vti file.");
}

} // namespace fvm::python

PYBIND11_MODULE(pyfvm, m)
{
    m.doc() = "Python bindings for the FVM-solver (2D Cartesian FVM: "
              "convection-diffusion transport, SIMPLE, PISO).";
    fvm::python::bindAll(m);
}
