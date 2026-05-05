# AGENTS.md

## Project State
- Phase 1 complete: project skeleton, core data structures, linear algebra wrapper, VTK I/O, and unit tests implemented and passing.
- Build system verified working (CMake + vcpkg). All 16 test cases pass.

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
  app/
    main.cpp         — example program

tests/
  main.cpp           — doctest entry point
  test_mesh.cpp      — mesh geometry tests
  test_field.cpp     — field accessor tests
  test_linalg.cpp    — sparse matrix & linear solver tests
```

## Naming Conventions
- **Strict sub-namespaces**: `fvm::core`, `fvm::math`, `fvm::io`
- Cross-module dependencies use `using` declarations for brevity (e.g., `math` uses `fvm::core::Scalar`)
- Each module has separate `include/` (public headers) and `src/` (implementation) directories

## Build Instructions

**Prerequisites:** CMake >= 3.20, vcpkg (with `VCPKG_ROOT` environment variable set).

```bash
# Configure (vcpkg installs eigen3, doctest automatically)
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build build --config Release

# Run tests
ctest --test-dir build --output-on-failure

# Run example
./build/Release/fvm_solver
```

**Windows (PowerShell):**
```powershell
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"
cmake --build build --config Release
ctest --test-dir build --output-on-failure
.\build\Release\fvm_solver.exe
```

## Key Design Decisions

- **LinearSolver is abstract** with factory functions (`createEigenBiCGSTAB`, `createEigenCG`, `createEigenSparseLU`). Future backends (AMGCL, Hypre) only need new subclasses + factories.
- **SparseMatrix wraps Eigen** internally but exposes only `insert`, `finalize`, `native()`. If swapping backends, change `native()` return type or add internal accessors.
- **Mesh uses cell-centered storage**. Face indexing: 0=east, 1=north, 2=west, 3=south.
- **VTK output uses `.vti` (ImageData)** — native match for Cartesian grids, opens directly in ParaView.
- **Module isolation**: Each module (core/math/io) has its own namespace and directory structure with `include/` + `src/`.

## Dependencies
- `eigen3` — sparse linear algebra
- `doctest` — unit testing (header-only)

## Next Phase (Phase 2)
Implement FVM discretization operators:
- Diffusion term (Laplacian)
- Convection term (UD, CD)
- Boundary condition handlers

Then Phase 3: SIMPLE algorithm for incompressible steady-state NS.
