# AGENTS.md

## Project State
- Phase 1 complete: project skeleton, core data structures, linear algebra wrapper, VTK I/O, and unit tests implemented and passing.
- Phase 2 complete: FVM discretization operators (diffusion, convection UD/CD, source-term linearization, boundary conditions) implemented. All 31 test cases pass, including grid-convergence order checks (UD ~ O(h), CD/diffusion ~ O(h^2)).
- Phase 3 complete: SIMPLE algorithm for incompressible steady-state NS (collocated grid + Rhie-Chow interpolation, Patankar under-relaxation, pressure-correction equation with reference-cell elimination). All 36 test cases pass, including Poiseuille channel flow (quantitative parabolic profile + flow rate) and lid-driven cavity at Re=100.
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
      Simple.h             — SIMPLE algorithm: SimpleConfig/SimpleResult, assembleMomentum, solveSimple
    src/
      Diffusion.cpp
      Convection.cpp
      TransportEquation.cpp
      Simple.cpp           — momentum predictor, Rhie-Chow fluxes, pressure-correction eq, corrections
  app/
    main.cpp         — two demos: steady convection-diffusion (recirculating flow, hot/cold walls)
                       and lid-driven cavity at Re=100 via SIMPLE (cavity.vti)

tests/
  main.cpp           — doctest entry point
  test_mesh.cpp      — mesh geometry tests
  test_field.cpp     — field accessor tests
  test_linalg.cpp    — sparse matrix & linear solver tests
  test_diffusion.cpp — diffusion operator: exact solutions, SPD, BC handling, O(h^2)
  test_convection.cpp— convection operator: UD O(h) / CD O(h^2), boundedness, conservation
  test_transport.cpp — combined assembly, source terms (Sc + Sp*phi), conservation
  test_simple.cpp    — momentum assembly/under-relaxation identities, Poiseuille, cavity Re=100

docs/                 — per-module code documentation (Chinese)
  core.md            — Types, CartesianMesh, ScalarField/VectorField
  math.md            — SparseMatrix (triplet assembly), LinearSolver backends
  io.md              — VtkWriter (.vti output)
  numerical.md       — BCs, diffusion/convection assembly, transport equation, SIMPLE
  app.md             — fvm_solver demo walkthrough (convection-diffusion + lid-driven cavity)
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
- **SIMPLE (collocated, Rhie-Chow)**: momentum assembly reuses the convection/diffusion operators; the pressure-gradient source uses the Gauss form (sum_f p_f n S_f) so Dirichlet pressure boundary values drive the flow. Under-relaxed diagonal `a_P` is read from a finalized matrix copy; `d = vol/a_P` feeds the Rhie-Chow face fluxes (x-faces use the u-equation diagonal, y-faces the v-equation one).
- **Pressure-correction sign convention**: with F_f = F*_f - C(p'_N - p'_P), continuity gives A p' = **-**massImbalance. Getting this sign wrong creates positive feedback and blows up the velocity field.
- **Pure-Neumann pressure is handled by reference-cell elimination** (drop cell 0, p'_0 = 0 — its equation is redundant since imbalances sum to zero), NOT a penalty diagonal; this keeps the CG system well-conditioned SPD. With any Dirichlet pressure side, no elimination is needed.

## Dependencies
- `eigen3` — sparse linear algebra
- `doctest` — unit testing (header-only)

## Next Phase (Phase 4)
Candidate directions: unsteady terms (theta scheme), non-orthogonal/skew mesh support, or AMGCL/Hypre solver backends for larger meshes.
