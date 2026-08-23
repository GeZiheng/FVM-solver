# AGENTS.md

## Project State
- Phase 1 complete: project skeleton, core data structures, linear algebra wrapper, VTK I/O, and unit tests implemented and passing.
- Phase 2 complete: FVM discretization operators (diffusion, convection UD/CD, source-term linearization, boundary conditions) implemented. All 31 test cases pass, including grid-convergence order checks (UD ~ O(h), CD/diffusion ~ O(h^2)).
- Build system verified working (CMake + vcpkg).

## Architecture

```
src/
  core/
    include/         — public API headers
      Types.h        — Scalar, Index, Vector aliases (namespace fvm::core)
      Mesh.h         — CartesianMesh (2D uniform Cartesian)
      Field.h        — ScalarField, VectorField (cell-centered)
    src/
      Mesh.cpp
      Field.cpp
  math/
    include/
      SparseMatrix.h — triplet-based assembly, wraps Eigen sparse matrix
      LinearSolver.h — abstract interface + Eigen BiCGSTAB/CG/SparseLU
    src/
      SparseMatrix.cpp
      LinearSolver.cpp
  io/
    include/
      VtkWriter.h    — VTK ImageData (.vti) output
    src/
      VtkWriter.cpp
  numerical/
    include/
      BoundaryCondition.h  — BCType{Dirichlet, Neumann} + BoundaryField (4 sides, header-only)
      Diffusion.h          — -div(gamma grad phi) assembly
      Convection.h         — div(rho u phi) assembly, ConvectionScheme{Upwind, Central}
      TransportEquation.h  — full conv-diff-source assembly -> EquationSystem{A, b}
    src/
      Diffusion.cpp
      Convection.cpp
      TransportEquation.cpp
  app/
    main.cpp         — steady convection-diffusion demo (recirculating flow, hot/cold walls)

tests/
  main.cpp           — doctest entry point
  test_mesh.cpp      — mesh geometry tests
  test_field.cpp     — field accessor tests
  test_linalg.cpp    — sparse matrix & linear solver tests
  test_diffusion.cpp — diffusion operator: exact solutions, SPD, BC handling, O(h^2)
  test_convection.cpp— convection operator: UD O(h) / CD O(h^2), boundedness, conservation
  test_transport.cpp — combined assembly, source terms (Sc + Sp*phi), conservation

docs/                 — per-module code documentation (Chinese)
  core.md            — Types, CartesianMesh, ScalarField/VectorField
  math.md            — SparseMatrix (triplet assembly), LinearSolver backends
  io.md              — VtkWriter (.vti output)
  numerical.md       — BCs, diffusion/convection assembly, transport equation
  app.md             — fvm_solver demo walkthrough
```

When changing a module's classes/design/key algorithms, update the corresponding docs/*.md file. Math formulas in docs are written in LaTeX (`$...$` inline, `$$...$$` display); matrix-assembly statements that mirror code (e.g., `A(P,P) += D_f`) stay in code blocks.

## Naming Conventions
- **Strict sub-namespaces**: `fvm::core`, `fvm::math`, `fvm::io`, `fvm::numerical`
- Cross-module dependencies use `using` declarations for brevity (e.g., `math` uses `fvm::core::Scalar`)
- Each module has separate `include/` (public headers) and `src/` (implementation) directories

## Build Instructions

**Prerequisites:** CMake >= 3.20, vcpkg (with `VCPKG_ROOT` environment variable set).

**IMPORTANT for agents: ALWAYS use the `build-and-test` custom tool to configure, build, and test this project. Do NOT run `cmake --build`, `cmake -S/-B`, `ctest`, or `ninja` via the bash tool — those commands are denied by permission rules.**
- Default invocation (no args): Release mode, builds `fvm_solver` + `fvm_tests`, runs ctest.
- `config`: `"Debug"` | `"Release"` (default `"Release"`).
- `target`: `"all"` | `"fvm_solver"` | `"fvm_tests"` (default `"all"`).
- `run`: `"none"` | `"tests"` | `"solver"` | `"both"` (default `"tests"`) — what to run after a successful build.
- When the user directly requests a build/test, confirm the options with the user via the `question` tool first (unless the user already stated them explicitly). When building as part of a code modification workflow, proceed directly with the defaults without asking.

Manual commands (for humans, outside opencode):

```bash
# Configure (vcpkg installs eigen3, doctest automatically)
cmake -B build -S . -G Ninja -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build build --config Release

# Run tests
ctest --test-dir build --output-on-failure

# Run example
./build/Release/fvm_solver
```

**Windows (PowerShell):**
```powershell
cmake -B build -S . -G Ninja -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"
cmake --build build --config Release
ctest --test-dir build --output-on-failure
.\build\Release\fvm_solver.exe
```

## Key Design Decisions

- **LinearSolver is abstract** with factory functions (`createEigenBiCGSTAB`, `createEigenCG`, `createEigenSparseLU`). Future backends (AMGCL, Hypre) only need new subclasses + factories.
- **SparseMatrix wraps Eigen** internally but exposes only `insert`, `finalize`, `native()`. If swapping backends, change `native()` return type or add internal accessors.
- **Mesh uses cell-centered storage**. Face indexing: 0=east, 1=north, 2=west, 3=south.
- **VTK output uses `.vti` (ImageData)** — native match for Cartesian grids, opens directly in ParaView.
- **Module isolation**: Each module (core/math/io/numerical) has its own namespace and directory structure with `include/` + `src/`.
- **Discretization assembly accumulates**: `assembleDiffusion`/`assembleConvection` add into (A, b) without zeroing; `assembleTransport` owns the full assembly and returns an unfinalized `EquationSystem`. Boundary side indices match the mesh face convention (0=E, 1=N, 2=W, 3=S).
- **Source terms follow Patankar**: S(phi) = Sc + Sp*phi per unit volume; Sp must be <= 0 (treated implicitly, enforced by exception).
- **SolverConfig::tolerance means relative residual** |Ax-b|/|b|. The BiCGSTAB wrapper scales Eigen's absolute stopping tolerance by |b| and checks convergence itself (Eigen 5.x compares mismatched absolute/relative quantities internally — do not rely on `solver.info()` for BiCGSTAB convergence).

## Dependencies
- `eigen3` — sparse linear algebra
- `doctest` — unit testing (header-only)

## Next Phase (Phase 3)
SIMPLE algorithm for incompressible steady-state NS. Reuse `assembleConvection` face-flux computation for Rhie-Chow interpolation.
