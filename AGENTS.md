# AGENTS.md

## Project State
- Phase 1 complete: project skeleton, core data structures, linear algebra wrapper, VTK I/O, and unit tests implemented and passing.
- Phase 2 complete: FVM discretization operators (diffusion, convection UD/CD, source-term linearization, boundary conditions) implemented. All 31 test cases pass, including grid-convergence order checks (UD ~ O(h), CD/diffusion ~ O(h^2)).
- Phase 3 complete: SIMPLE algorithm for incompressible steady-state NS (collocated grid + Rhie-Chow interpolation, Patankar under-relaxation, pressure-correction equation with reference-cell elimination). All 36 test cases pass, including Poiseuille channel flow (quantitative parabolic profile + flow rate) and lid-driven cavity at Re=100.
- Persistent conservative flux field (OpenFOAM-style `phi`): `FaceFluxField` in core; convection/transport/momentum assemblers take the flux field as their only convection input (velocity-based convenience interfaces removed; use `interpolateCellVelocityFlux` / `computeMassFlux` to build one). `solveSimple` maintains the flux across iterations (predicted F* written during assembly, corrected by p' afterwards) and returns it via an out parameter — the converged flux is conservative to pressure-solver accuracy. All 43 test cases pass, including per-cell flux-conservation checks.
- SIMPLE iteration split into reusable `predictMomentum`/`correctPressure` free functions (UEqn.H/pEqn.H-style, ready for PISO reuse); closed-domain flux compatibility enforced via `checkFluxCompatibility` (OpenFOAM adjustPhi-style) at `solveSimple` entry.
- Build system verified working (CMake + vcpkg).

## Architecture

```
src/
  core/
    include/         — public API headers
      Types.h        — Scalar, Index, Vector aliases (namespace fvm::core)
      Mesh.h         — CartesianMesh (2D uniform Cartesian)
      Field.h        — ScalarField, VectorField (cell-centered)
      FluxField.h    — FaceFluxField (face-stored flux, positive along +x/+y, header-only)
    src/
      Mesh.cpp
      Field.cpp
      FluxField.cpp
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
      Convection.h         — div(F phi) assembly from FaceFluxField, ConvectionScheme{Upwind, Central},
                             flux constructors interpolateCellVelocityFlux / computeMassFlux,
                             closed-domain checkFluxCompatibility
      TransportEquation.h  — full conv-diff-source assembly from FaceFluxField -> EquationSystem{A, b}
      Simple.h             — SIMPLE algorithm: SimpleConfig/SimpleResult, assembleMomentum,
                             predictMomentum/correctPressure (UEqn.H/pEqn.H-style split, reusable by PISO),
                             solveSimple (persistent flux out parameter)
    src/
      Diffusion.cpp
      Convection.cpp
      TransportEquation.cpp
      Simple.cpp           — momentum predictor, Rhie-Chow fluxes, pressure-correction eq, corrections
                             (as free functions; solveSimple loops over them)
  app/
    main.cpp         — two demos: steady convection-diffusion (recirculating flow, hot/cold walls)
                       and lid-driven cavity at Re=100 via SIMPLE (cavity.vti)

tests/
  main.cpp           — doctest entry point
  test_mesh.cpp      — mesh geometry tests
  test_field.cpp     — field accessor tests
  test_flux.cpp      — flux field layout/sign conventions, flux constructors, conservative SIMPLE flux
  test_linalg.cpp    — sparse matrix & linear solver tests
  test_diffusion.cpp — diffusion operator: exact solutions, SPD, BC handling, O(h^2)
  test_convection.cpp— convection operator: UD O(h) / CD O(h^2), boundedness, conservation
  test_transport.cpp — combined assembly, source terms (Sc + Sp*phi), conservation
  test_simple.cpp    — momentum assembly/under-relaxation identities, Poiseuille, cavity Re=100,
                       closed-domain flux-compatibility check

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

**Prerequisites:** CMake >= 3.20, vcpkg (with `VCPKG_ROOT` set as a Windows user environment variable).

**IMPORTANT for agents:** if your environment provides the `build-and-test` MCP tool (`build_and_test`), ALWAYS use it to configure, build, and test this project — do NOT shell out to `cmake`, `ctest`, or `ninja` yourself (agent configs may deny those commands). The server implementation lives in `.agents/mcp/build_and_test_server.py` and is shared across agents; it is registered for Codex in `.codex/config.toml` and for opencode through the `mcp` block in `opencode.json`. If the MCP tool is NOT available in your environment (other agents, humans), use the manual commands below instead.
- Default invocation (no args): Release mode, builds `fvm_solver` + `fvm_tests`, runs ctest.
- `config`: `"Debug"` | `"Release"` (default `"Release"`).
- `target`: `"all"` | `"fvm_solver"` | `"fvm_tests"` (default `"all"`).
- `run`: `"none"` | `"tests"` | `"solver"` | `"both"` (default `"tests"`) — what to run after a successful build.
- When the user directly requests a build/test, confirm the options (config, target, run) with the user first (unless the user already stated them explicitly). When building as part of a code modification workflow, proceed directly with the defaults without asking.
- The tool configures into `build/<config>-agent` (one directory per config, shared by all agents), not into `build/`.
- The server runs **outside** the agent's per-command sandbox, which is required here: vcpkg writes under `$VCPKG_ROOT`, outside this repository, so running the CMake commands from a sandboxed shell fails at the vcpkg step. The server reads `VCPKG_ROOT` from the process environment and falls back to the persisted Windows environment (HKCU, then HKLM).

Manual commands (for humans, or when the MCP tool is unavailable):

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
- **Flux is a first-class citizen** (OpenFOAM-style): convection/transport/momentum assemblers take a `FaceFluxField` (positive along +x/+y; boundary outward fluxes signed accordingly) instead of a velocity field. Build one with `interpolateCellVelocityFlux` (given velocity field, boundary from cell velocity) or `computeMassFlux` (velocity-BC aware, in-place).
- **Source terms follow Patankar**: S(phi) = Sc + Sp*phi per unit volume; Sp must be <= 0 (treated implicitly, enforced by exception).
- **SolverConfig::tolerance means relative residual** |Ax-b|/|b|. The BiCGSTAB wrapper scales Eigen's absolute stopping tolerance by |b| and checks convergence itself (Eigen 5.x compares mismatched absolute/relative quantities internally — do not rely on `solver.info()` for BiCGSTAB convergence).
- **SIMPLE (collocated, Rhie-Chow)**: momentum assembly reuses the convection/diffusion operators; the pressure-gradient source uses the Gauss form (sum_f p_f n S_f) so Dirichlet pressure boundary values drive the flow. Under-relaxed diagonal `a_P` is read from a finalized matrix copy; `d = vol/a_P` feeds the Rhie-Chow face fluxes (x-faces use the u-equation diagonal, y-faces the v-equation one). The persistent flux field is overwritten with the predicted F* during p' assembly and corrected in place afterwards (`F_f -= C(p'_N - p'_P)`; Dirichlet-p boundary faces `±= Cb·p'_P`), so it stays conservative to the pressure solver's accuracy and feeds the next momentum assembly.
- **Pressure-correction sign convention**: with F_f = F*_f - C(p'_N - p'_P), continuity gives A p' = **-**massImbalance. Getting this sign wrong creates positive feedback and blows up the velocity field.
- **Pure-Neumann pressure is handled by reference-cell elimination** (drop cell 0, p'_0 = 0 — its equation is redundant since imbalances sum to zero), NOT a penalty diagonal; this keeps the CG system well-conditioned SPD. With any Dirichlet pressure side, no elimination is needed. Closed-domain (pure-Neumann-p) compatibility is enforced by `checkFluxCompatibility` at `solveSimple` entry (net boundary outflow must be ~0, OpenFOAM adjustPhi-style), throwing on unbalanced velocity BCs.
- **SIMPLE iteration is split into reusable primitives**: `predictMomentum` (UEqn.H-style) and `correctPressure` (pEqn.H-style) are free functions; `solveSimple` just loops over them. PISO/unsteady solvers should reuse them instead of duplicating the corrector logic.

## Dependencies
- `eigen3` — sparse linear algebra
- `doctest` — unit testing (header-only)

## Next Phase (Phase 4)
Agreed roadmap (in order):
1. **Unsteady terms + PISO** (theta-scheme time integration; reuse `predictMomentum`/`correctPressure` per time step: one momentum predictor + multiple pressure correctors, no under-relaxation).
2. **`pyfvm` Python bindings** (pybind11 via vcpkg, optional build target) — case setup becomes a Python script (initial fields/source terms/post-processing in numpy), replacing any JSON-config idea; `fvm_solver` exe stays as a smoke demo. Start only after the C++ solver API stabilizes (post-PISO).
3. **Arbitrary mesh input** (Gmsh `.msh` reader first) — the main motivation for the Python front-end; may come with non-orthogonal/skew mesh support.

Deferred/rejected: per-case executables under `examples/` (too cumbersome), JSON case config (redundant once Python scripting exists), per-module CMakeLists/tests/docs split (revisit only if a module is reused externally or build times degrade). AMGCL/Hypre backends remain a candidate for larger meshes.
