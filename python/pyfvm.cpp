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
#include <stdexcept>
#include <utility>
#include <vector>

#include "BoundaryCondition.h"
#include "Convection.h"
#include "Field.h"
#include "FluxField.h"
#include "LinearSolver.h"
#include "CartesianMesh.h"
#include "Piso.h"
#include "Simple.h"
#include "Su2Reader.h"
#include "TimeScheme.h"
#include "TransportEquation.h"
#include "UnstructuredMesh.h"
#include "VtkWriter.h"

namespace py = pybind11;

namespace fvm::python
{

using fvm::core::CartesianMesh;
using fvm::core::FaceFluxField;
using fvm::core::Index;
using fvm::core::Mesh;
using fvm::core::Scalar;
using fvm::core::ScalarField;
using fvm::core::UnstructuredMesh;
using fvm::core::Vector;
using fvm::core::VectorField;
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

} // namespace fvm::python

namespace pybind11::detail {
template <>
struct type_caster<fvm::core::Point2>
{
    PYBIND11_TYPE_CASTER(fvm::core::Point2, _("tuple[float, float]"));

    bool load(handle src, bool convert)
    {
        make_caster<std::pair<fvm::core::Scalar, fvm::core::Scalar>>
            pairCaster;
        if (!pairCaster.load(src, convert))
        {
            return false;
        }
        const auto coordinates = static_cast<
            std::pair<fvm::core::Scalar, fvm::core::Scalar>>(pairCaster);
        value.x = coordinates.first;
        value.y = coordinates.second;
        return true;
    }

    static handle cast(const fvm::core::Point2& src,
        return_value_policy policy,
        handle parent)
    {
        return make_caster<
            std::pair<fvm::core::Scalar, fvm::core::Scalar>>::cast(
            std::make_pair(src.x, src.y), policy, parent);
    }
};
} // namespace pybind11::detail

namespace fvm::python
{

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

/// Binding-level guard: every field stores the mesh it was created on, but
/// the free functions below take the mesh as a separate argument. A field
/// from another mesh must be rejected before it reaches the C++ core (the
/// core has no such check; Release builds would silently read out of
/// bounds). std::invalid_argument maps to Python ValueError.
template <typename FieldT>
void requireSameMesh(const Mesh& mesh, const FieldT& field, const char* what)
{
    if (&field.mesh() != &mesh)
        throw std::invalid_argument(std::string("mesh mismatch: '") + what
            + "' was created on a different mesh");
}

/// Binding-level bounds guard: Release builds carry no Eigen assertions, so
/// an out-of-range index coming from Python must be checked here to raise
/// IndexError instead of crashing the interpreter.
void requireIndex(bool ok)
{
    if (!ok)
        throw py::index_error("index out of range");
}

/// High-level steady scalar transport: assemble, solve with BiCGSTAB,
/// return the solution as a numpy array (row-major cell order, j * nx + i).
py::array_t<Scalar> solveTransport(const Mesh& mesh,
    const FaceFluxField& flux,
    const ScalarField& gamma,
    ConvectionScheme scheme,
    const BoundaryField& bc,
    const SolverConfig& solverConfig,
    const ScalarField* Sc,
    const ScalarField* Sp)
{
    requireSameMesh(mesh, flux, "flux");
    requireSameMesh(mesh, gamma, "gamma");
    requireSameMesh(mesh, bc, "bc");
    if (Sc)
        requireSameMesh(mesh, *Sc, "Sc");
    if (Sp)
        requireSameMesh(mesh, *Sp, "Sp");
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
py::array_t<Scalar> solveTransientTransport(const Mesh& mesh,
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
    requireSameMesh(mesh, flux, "flux");
    requireSameMesh(mesh, gamma, "gamma");
    requireSameMesh(mesh, phiOld, "phi_old");
    requireSameMesh(mesh, bc, "bc");
    if (Sc)
        requireSameMesh(mesh, *Sc, "Sc");
    if (Sp)
        requireSameMesh(mesh, *Sp, "Sp");
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
SimpleResult runSimple(const Mesh& mesh,
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
    requireSameMesh(mesh, velocity, "velocity");
    requireSameMesh(mesh, pressure, "pressure");
    requireSameMesh(mesh, flux, "flux");
    requireSameMesh(mesh, bcU, "bc_u");
    requireSameMesh(mesh, bcV, "bc_v");
    requireSameMesh(mesh, bcP, "bc_p");
    py::gil_scoped_release release;
    return numerical::solveSimple(
        mesh, rho, mu, bcU, bcV, bcP, config, velocity, pressure, flux);
}

PisoResult runPiso(const Mesh& mesh,
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
    requireSameMesh(mesh, velocity, "velocity");
    requireSameMesh(mesh, pressure, "pressure");
    requireSameMesh(mesh, flux, "flux");
    requireSameMesh(mesh, bcU, "bc_u");
    requireSameMesh(mesh, bcV, "bc_v");
    requireSameMesh(mesh, bcP, "bc_p");
    py::gil_scoped_release release;
    return numerical::solvePiso(
        mesh, rho, mu, bcU, bcV, bcP, config, velocity, pressure, flux);
}

FaceFluxField interpolateCellVelocityFlux(const Mesh& mesh,
    const VectorField& velocity,
    Scalar rho)
{
    requireSameMesh(mesh, velocity, "velocity");
    return numerical::interpolateCellVelocityFlux(mesh, velocity, rho);
}

void computeMassFlux(const Mesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    FaceFluxField& flux)
{
    requireSameMesh(mesh, velocity, "velocity");
    requireSameMesh(mesh, flux, "flux");
    requireSameMesh(mesh, bcU, "bc_u");
    requireSameMesh(mesh, bcV, "bc_v");
    numerical::computeMassFlux(mesh, velocity, rho, bcU, bcV, flux);
}

void initializeMassFlux(const Mesh& mesh,
    const VectorField& velocity,
    Scalar rho,
    const BoundaryField& bcU,
    const BoundaryField& bcV,
    const BoundaryField& bcP,
    FaceFluxField& flux)
{
    requireSameMesh(mesh, velocity, "velocity");
    requireSameMesh(mesh, flux, "flux");
    requireSameMesh(mesh, bcU, "bc_u");
    requireSameMesh(mesh, bcV, "bc_v");
    requireSameMesh(mesh, bcP, "bc_p");
    numerical::initializeMassFlux(mesh, velocity, rho, bcU, bcV, bcP, flux);
}

/// io::writeVti with Python dicts {name: field}.
void writeVti(const std::string& filename,
    const CartesianMesh& mesh,
    const py::dict& scalarFields,
    const py::dict& vectorFields)
{
    std::vector<std::pair<std::string, const ScalarField*>> scalars;
    for (const auto& item : scalarFields)
    {
        auto name = py::cast<std::string>(item.first);
        const ScalarField* field = py::cast<const ScalarField*>(item.second);
        requireSameMesh(mesh, *field, name.c_str());
        scalars.emplace_back(std::move(name), field);
    }
    std::vector<std::pair<std::string, const VectorField*>> vectors;
    for (const auto& item : vectorFields)
    {
        auto name = py::cast<std::string>(item.first);
        const VectorField* field = py::cast<const VectorField*>(item.second);
        requireSameMesh(mesh, *field, name.c_str());
        vectors.emplace_back(std::move(name), field);
    }
    io::writeVti(filename, mesh, scalars, vectors);
}

/// io::writeVtu with Python dicts {name: field}.
void writeVtu(const std::string& filename,
    const UnstructuredMesh& mesh,
    const py::dict& scalarFields,
    const py::dict& vectorFields)
{
    std::vector<std::pair<std::string, const ScalarField*>> scalars;
    for (const auto& item : scalarFields)
    {
        auto name = py::cast<std::string>(item.first);
        const ScalarField* field = py::cast<const ScalarField*>(item.second);
        requireSameMesh(mesh, *field, name.c_str());
        scalars.emplace_back(std::move(name), field);
    }
    std::vector<std::pair<std::string, const VectorField*>> vectors;
    for (const auto& item : vectorFields)
    {
        auto name = py::cast<std::string>(item.first);
        const VectorField* field = py::cast<const VectorField*>(item.second);
        requireSameMesh(mesh, *field, name.c_str());
        vectors.emplace_back(std::move(name), field);
    }
    io::writeVtu(filename, mesh, scalars, vectors);
}

void bindCore(py::module_& m)
{
    py::class_<Mesh>(m, "Mesh")
        .def("cell_count", &Mesh::cellCount)
        .def("face_count", &Mesh::faceCount)
        .def("face_owner", [](const Mesh& m, Index face)
            {
                requireIndex(face < m.faceCount());
                return m.faceOwner(face);
            }, py::arg("face"))
        .def("face_neighbour", [](const Mesh& m, Index face)
            {
                requireIndex(face < m.faceCount());
                return m.faceNeighbour(face);
            }, py::arg("face"))
        .def("is_boundary_face", [](const Mesh& m, Index face)
            {
                requireIndex(face < m.faceCount());
                return m.isBoundaryFace(face);
            }, py::arg("face"))
        .def("cell_faces", [](const Mesh& m, Index cell)
            {
                requireIndex(cell < m.cellCount());
                const auto faces = m.cellFaces(cell);
                return std::vector<Index>(faces.begin(), faces.end());
            }, py::arg("cell"))
        .def("cell_volume", [](const Mesh& m, Index cell)
            {
                requireIndex(cell < m.cellCount());
                return m.cellVolume(cell);
            }, py::arg("cell"))
        .def("cell_centroid", [](const Mesh& m, Index cell)
            {
                requireIndex(cell < m.cellCount());
                return m.cellCentroid(cell);
            }, py::arg("cell"))
        .def("face_centroid", [](const Mesh& m, Index face)
            {
                requireIndex(face < m.faceCount());
                return m.faceCentroid(face);
            }, py::arg("face"))
        .def("face_area_vector", [](const Mesh& m, Index face)
            {
                requireIndex(face < m.faceCount());
                return m.faceAreaVector(face);
            }, py::arg("face"))
        .def("patch_count", &Mesh::patchCount)
        .def("patch_name", [](const Mesh& m, Index patch)
            {
                requireIndex(patch < m.patchCount());
                return m.patchName(patch);
            }, py::arg("patch"))
        .def("patch_faces", [](const Mesh& m, Index patch)
            {
                requireIndex(patch < m.patchCount());
                const auto faces = m.patchFaces(patch);
                return std::vector<Index>(faces.begin(), faces.end());
            }, py::arg("patch"))
        .def("face_patch", [](const Mesh& m, Index face)
            {
                requireIndex(face < m.faceCount());
                return m.facePatch(face);
            }, py::arg("face"));

    py::class_<CartesianMesh, Mesh>(m, "CartesianMesh")
        .def(py::init<Index, Index, Scalar, Scalar, Scalar, Scalar>(),
            py::arg("nx"), py::arg("ny"),
            py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"))
        .def("nx", &CartesianMesh::nx)
        .def("ny", &CartesianMesh::ny)
        .def("cell_count", &CartesianMesh::cellCount)
        .def("dx", &CartesianMesh::dx)
        .def("dy", &CartesianMesh::dy)
        .def("cell_volume", [](const CartesianMesh& m, Index cell)
            {
                requireIndex(cell >= 0 && cell < m.cellCount());
                return m.cellVolume(cell);
            }, py::arg("cell"))
        .def("cell_center", [](const CartesianMesh& m, Index cell)
            {
                requireIndex(cell >= 0 && cell < m.cellCount());
                return m.cellCenter(cell);
            }, py::arg("cell"))
        .def("cell_index", [](const CartesianMesh& m, Index i, Index j)
            {
                requireIndex(i >= 0 && i < m.nx() && j >= 0 && j < m.ny());
                return m.cellIndex(i, j);
            }, py::arg("i"), py::arg("j"))
        .def("cell_ij", [](const CartesianMesh& m, Index cell)
            {
                requireIndex(cell >= 0 && cell < m.cellCount());
                return m.cellIJ(cell);
            }, py::arg("cell"))
        .def("x_face", &CartesianMesh::xFace, py::arg("i"), py::arg("j"))
        .def("y_face", &CartesianMesh::yFace, py::arg("i"), py::arg("j"));

    py::class_<UnstructuredMesh, Mesh>(m, "UnstructuredMesh")
        .def("point_count", &UnstructuredMesh::pointCount)
        .def("point", [](const UnstructuredMesh& mesh, Index point)
            {
                requireIndex(point < mesh.pointCount());
                return mesh.point(point);
            }, py::arg("point"))
        .def("cell_node_count", [](const UnstructuredMesh& mesh, Index cell)
            {
                requireIndex(cell < mesh.cellCount());
                return mesh.cellNodeCount(cell);
            }, py::arg("cell"))
        .def("cell_node", [](const UnstructuredMesh& mesh, Index cell,
                Index local)
            {
                requireIndex(cell < mesh.cellCount());
                requireIndex(local < mesh.cellNodeCount(cell));
                return mesh.cellNode(cell, local);
            }, py::arg("cell"), py::arg("local"));

    // Fields hold a mesh reference: keep the mesh alive while the field lives.
    py::class_<ScalarField>(m, "ScalarField")
        .def(py::init<const Mesh&, const std::string&>(),
            py::arg("mesh"), py::arg("name") = "", py::keep_alive<1, 2>())
        .def("name", &ScalarField::name)
        .def("size", &ScalarField::size)
        .def("set_zero", &ScalarField::setZero)
        .def("set_constant", &ScalarField::setConstant, py::arg("value"))
        .def("to_numpy", [](const py::object& self)
            { return vectorView(self, self.cast<ScalarField&>().data()); },
            "Writable 1-D numpy view of the cell data (cell index "
            "j * nx + i; use mesh.cell_index(i, j) for structured access).");

    py::class_<VectorField>(m, "VectorField")
        .def(py::init<const Mesh&, const std::string&>(),
            py::arg("mesh"), py::arg("name") = "", py::keep_alive<1, 2>())
        .def("name", &VectorField::name)
        .def("size", &VectorField::size)
        .def("set_zero", &VectorField::setZero)
        .def("set_constant", &VectorField::setConstant, py::arg("value"))
        .def_property_readonly("u",
            static_cast<ScalarField& (VectorField::*)()>(&VectorField::u),
            py::return_value_policy::reference_internal)
        .def_property_readonly("v",
            static_cast<ScalarField& (VectorField::*)()>(&VectorField::v),
            py::return_value_policy::reference_internal);

    py::class_<FaceFluxField>(m, "FaceFluxField")
        .def(py::init<const Mesh&, const std::string&>(),
            py::arg("mesh"), py::arg("name") = "", py::keep_alive<1, 2>())
        .def("name", &FaceFluxField::name)
        .def("face_count", &FaceFluxField::faceCount)
        .def("set_zero", &FaceFluxField::setZero)
        .def("phi", [](const FaceFluxField& f, Index face)
            {
                requireIndex(face < f.faceCount());
                return f[face];
            }, py::arg("face"))
        .def("set_phi", [](FaceFluxField& f, Index face, Scalar value)
            {
                requireIndex(face < f.faceCount());
                f[face] = value;
            }, py::arg("face"), py::arg("value"))
        .def("to_numpy", [](const py::object& self)
            { return vectorView(self, self.cast<FaceFluxField&>().data()); },
            "Writable 1-D numpy view of the face fluxes (global face order).")
        .def("cell_imbalance", [](const FaceFluxField& f, Index cell)
            {
                requireIndex(cell >= 0 && cell < f.mesh().cellCount());
                return f.cellImbalance(cell);
            }, py::arg("cell"));
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
        .def(py::init<const Mesh&>(), py::arg("mesh"),
            py::keep_alive<1, 2>())
        .def("set", &BoundaryField::set,
            py::arg("side"), py::arg("type"), py::arg("value"))
        .def("get", &BoundaryField::get, py::arg("side"),
            py::return_value_policy::copy);
    pyBoundaryField.def("patch_count", &BoundaryField::patchCount);
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
    m.def("interpolate_cell_velocity_flux", &interpolateCellVelocityFlux,
        py::arg("mesh"), py::arg("velocity"), py::arg("rho"),
        py::keep_alive<0, 1>(),
        "Face mass flux from a cell-centred velocity (no BCs; for transport "
        "with a prescribed velocity).");
    m.def("compute_mass_flux", &computeMassFlux,
        py::arg("mesh"), py::arg("velocity"), py::arg("rho"),
        py::arg("bc_u"), py::arg("bc_v"), py::arg("flux"));
    m.def("initialize_mass_flux", &initializeMassFlux,
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
    m.def("read_su2", [](const std::string& filename)
        { return io::readSu2(filename); },
        py::arg("path"),
        "Read a 2D SU2 ASCII mesh into a validated UnstructuredMesh.");
    m.def("write_vtu", &writeVtu,
        py::arg("filename"), py::arg("mesh"),
        py::arg("scalars") = py::dict(), py::arg("vectors") = py::dict(),
        "Write scalar/vector fields (dicts of name -> field) to a .vtu file.");
}

} // namespace fvm::python

PYBIND11_MODULE(pyfvm, m)
{
    m.doc() = "Python bindings for the FVM-solver (2D Cartesian FVM: "
              "convection-diffusion transport, SIMPLE, PISO). Meshes may be "
              "Cartesian or SU2-read unstructured meshes. Fluxes use a "
              "single per-face array; BoundaryField requires a mesh.";
    fvm::python::bindAll(m);
}
