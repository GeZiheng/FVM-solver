# AGENTS.md

## Project State
- **Status**: phases 1–4-1 complete; all 50 test cases (3271 assertions) pass. Everything described below is implemented and verified.
- **Capabilities**: 2D uniform-Cartesian FVM (core/io), Eigen-backed sparse solvers (math), steady SIMPLE and transient PISO for incompressible NS on a collocated grid with Rhie–Chow interpolation, and theta-scheme transient scalar transport. A persistent `FaceFluxField` (`phi`) is the only convection input of the assemblers.
- **Solver structure**: the iteration is split into reusable `predictMomentum` / `correctPressure` primitives (`Momentum.h/.cpp`, `Pressure.h/.cpp`); `Simple`/`Piso` are thin driver loops. Extend those primitives rather than duplicating the loops.
- **PISO**: working; `tests/test_piso.cpp` is enabled. The invariant that cost the most to find: refresh `uHat = H/a_P` from the current velocity before every corrector after the first (OpenFOAM's per-corrector `HbyA = rAU*UEqn.H()`). A frozen `uHat` degenerates to `nCorrectors = 1`, which diverges in OpenFOAM too. Conclusions and the OpenFOAM-14 cross-check numbers: `docs/numerical.md` ("PISO 与 OpenFOAM 的对照结论"); process, pitfalls and the ruled-out list: `.agents/skills/openfoam-crosscheck/`.

## Skills (`.agents/skills/`, shared by Codex and opencode)

Both agents load the project skills from `.agents/skills` (Codex: project skill root; opencode: `skills.paths` in `opencode.json`). Use them:

- **`refactor-code`** — every planned change to this project's code goes through its four phases (plan → edit → review → docs). Phase 1 must present a plan and wait for the user's confirmation before any code is touched.
- **`openfoam-crosscheck`** — when a numerical/discretization detail needs a reference-implementation check: build same-parameter cases in the WSL OpenFOAM-14 (`~/OpenFOAM/OpenFOAM-14`), compare cell-by-cell/face-by-face, read OpenFOAM source when the data alone cannot explain the difference, and record conclusions with their evidence. Reusable tools: `scripts/make_of_case.sh` and `scripts/of_log_summary.py`; `references/` holds the OpenFOAM-14 fact index (including the ruled-out list) and the pitfalls. WSL access needs escalation in Codex.

## Architecture

```
src/
  core/                — Types (Scalar/Index/Vector), CartesianMesh (2D uniform), ScalarField/
                         VectorField (cell-centered), FaceFluxField (face-stored phi, header-only)
  math/                — SparseMatrix (triplet assembly over Eigen), LinearSolver (abstract +
                         Eigen CG / BiCGSTAB / SparseLU factories)
  io/                  — VtkWriter (.vti ImageData output)
  numerical/include/
    BoundaryCondition.h  — BCType{Dirichlet,Neumann} + BoundaryField (4 sides, header-only)
    GridOperators.h      — cellGradient (Gauss)
    Diffusion.h          — -div(gamma grad phi) assembly
    Convection.h         — div(F phi) from FaceFluxField; ConvectionScheme{Upwind,Central};
                           flux constructors interpolateCellVelocityFlux / computeMassFlux;
                           closed-domain checkFluxCompatibility
    TransportEquation.h  — steady conv-diff-source assembly + assembleTransientTransport (theta)
    TimeScheme.h         — TimeScheme{Euler,CrankNicolson}, TimeTerm (header-only)
    Momentum.h           — MomentumAssembly, assembleMomentum, MomentumPrediction,
                           predictMomentum, refreshUHat
    Pressure.h           — CorrectorResult, correctPressure
    Simple.h / Piso.h    — driver configs/results; solveSimple / solvePiso
  numerical/src/         — the implementations above. Momentum.cpp holds the Rhie-Chow data
                           (computeUHat -> uHat = H/a_P, d = V/a_P); Pressure.cpp holds the p'
                           equation, reference-cell elimination and the flux/U/p corrections.
  app/main.cpp           — demos: steady convection-diffusion; lid-driven cavity Re=100 (cavity.vti)

tests/                   — doctest: mesh, field, flux, linalg, diffusion, convection, transport,
                           transient (time order), simple (Poiseuille, cavity Re=100),
                           piso (impulsive Couette, transient Poiseuille)
docs/                    — module docs (Chinese): core.md, math.md, io.md, numerical.md, app.md
```

## Naming Conventions
- **Strict sub-namespaces**: `fvm::core`, `fvm::math`, `fvm::io`, `fvm::numerical`; each module has its own `include/` + `src/`.
- Cross-module dependencies use `using` declarations for brevity (e.g. `math` uses `fvm::core::Scalar`).
- **Dependency rule**: `core` → nothing; `math` → core (+ Eigen); `io` → core; `numerical` → core + math. Inside `numerical`: `Momentum` → Diffusion/Convection/GridOperators/TransportEquation/TimeScheme; `Pressure` → Momentum/GridOperators; `Simple`/`Piso` → Momentum/Pressure/Convection (they own the linear solvers). Do not introduce cycles.
- **Source files stay pure ASCII** (MSVC C4819); docs are Chinese, code comments English.

## Build Instructions

**Prerequisites:** CMake >= 3.20, vcpkg, Ninja. Two things must be in the **environment** (add them as persistent Windows variables *before* launching an agent — a running agent will not pick up later changes):
- **`VCPKG_ROOT`** — the vcpkg installation.
- **`Path`** — must contain the directory holding `ninja.exe`.

**IMPORTANT for agents:** if your environment provides the `build-and-test` MCP tool (`build_and_test`), ALWAYS use it — do NOT shell out to `cmake`/`ctest`/`ninja` (agent configs deny them). It configures into `build/<config>-agent`, and runs **outside** the agent sandbox, which is required because vcpkg writes under `$VCPKG_ROOT` (outside this repository). Options: `config` Release/Debug, `target` all/fvm_solver/fvm_tests, `run` none/tests/solver/both (defaults: Release, all, tests). As part of a code-modification workflow, proceed with the defaults; when the user asks for a build/test directly, confirm the options first. The server lives in `.agents/mcp/build_and_test_server.py`, registered for Codex in `.codex/config.toml` and for opencode in `opencode.json`.

**IMPORTANT for agents (OpenFOAM comparison runs):** if your environment provides the `openfoam` MCP tool, use it instead of shelling into WSL by hand — it also runs outside the sandbox (no per-call escalation) and returns structured JSON: `run_case` (fresh timestamped case dir with the configuration baked into the name; patches `endTime`/`deltaT`/`writeInterval`/`nCorrectors`; runs `foamRun`; returns `case_dir`, `log_path`, per-step residuals/Courant/continuity, divergence flag; default root `~/OpenFOAM/gzh1057-14/run`), `summarize_log`, `list_cases`. The server (`.agents/mcp/openfoam_server.py`) wraps the `openfoam-crosscheck` skill scripts, which remain the single implementation and the fallback when the tool is unavailable.

Manual commands (humans, or when the MCP tool is unavailable):

```bash
cmake -B build -S . -G Ninja -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
ctest --test-dir build --output-on-failure
./build/Release/fvm_solver        # demos
```

## Key Design Decisions

- **LinearSolver is abstract** (`createEigenBiCGSTAB` / `createEigenCG` / `createEigenSparseLU`); new backends (AMGCL, Hypre) only need a subclass + factory. **SparseMatrix wraps Eigen** and exposes only `insert`, `finalize`, `native()`.
- **Mesh is cell-centered**; face indices are 0=east, 1=north, 2=west, 3=south (all boundary arrays use the same order). VTK output is `.vti` ImageData.
- **Assembly accumulates**: `assembleDiffusion`/`assembleConvection` add into `(A, b)` without zeroing; `assembleTransport` owns the full assembly and returns an unfinalized `EquationSystem`.
- **Flux is first-class** (OpenFOAM `phi`): assemblers take a `FaceFluxField` (positive along +x/+y, boundary outward fluxes signed) instead of a velocity field; build it with `interpolateCellVelocityFlux` or `computeMassFlux`.
- **Source terms follow Patankar**: `S(phi) = Sc + Sp*phi` per unit volume, `Sp <= 0` enforced by exception.
- **`SolverConfig::tolerance` is a relative residual** `|Ax-b|/|b|`. The BiCGSTAB wrapper scales Eigen's absolute tolerance by `|b|` and checks convergence itself — do not rely on `solver.info()` for it.
- **SIMPLE (collocated + Rhie-Chow)**: momentum reuses the convection/diffusion operators; the pressure-gradient source is the Gauss form `sum_f p_f n S_f` (so Dirichlet pressure values drive the flow). The relaxed diagonal is read from a finalized matrix copy; `d = V/a_P` feeds the Rhie-Chow faces (x-faces use the u-equation diagonal, y-faces the v-equation one). The persistent flux is overwritten with the predicted `F*` during p' assembly and corrected in place (`F_f -= C(p'_N - p'_P)`; Dirichlet-p boundaries `±= Cb*p'_P`).
- **Pressure-correction sign convention**: with `F_f = F*_f - C(p'_N - p'_P)`, continuity gives `A p' = -massImbalance`. Getting this sign wrong creates positive feedback and blows up.
- **Boundary face fluxes must be same-sourced as the interior**: the predicted boundary flux is the Rhie-Chow flux at the old pressure — pressure-free `uHat` plus the face-normal pressure gradient (`Cb = rho d_P S_f / d_Pb`), with no pressure term on zero-gradient-p sides. Using `uStar` there re-adds the fixed boundary pressure drop every corrector and the iteration diverges.
- **Pure-Neumann pressure uses reference-cell elimination** (drop cell 0, `p'_0 = 0`; its equation is redundant), not a penalty diagonal — this keeps the CG system well-conditioned SPD. Closed domains must satisfy `checkFluxCompatibility` at entry (adjustPhi-style, throws on unbalanced velocity BCs), and `correctPressure` rebalances the predicted boundary fluxes on every corrector (pure-Neumann only) so the p' system stays solvable. No-op for all-Dirichlet-velocity domains (e.g. the cavity).
- **Time integration is an implicit theta scheme** (`TimeScheme` + `TimeTerm{rho, dt, scheme}`; `dt <= 0` means steady). The ddt term is assembled inside the shared momentum operator, so SIMPLE and PISO share it; `solveSimple` passes `TimeTerm{}` (bit-for-bit unchanged). `assembleTransientTransport` uses the same formula.
- **Transient momentum requires `relaxation == 1`** (under-relaxation is a steady SIMPLE device), and the ddt diagonal `rho V/dt` is part of `a_P`, so `d = V/a_P` is transient-consistent (OpenFOAM `rAU = 1/UEqn.A()`).
- **`correctPressure` has a `cumulativeVelocityCorrection` flag**: SIMPLE (false) uses `u = u* - d grad(p')`; PISO (true, requires `relaxationP == 1`) reconstructs `u = uHat - d grad(p)` from the full pressure so repeated correctors accumulate. **The pressure update must be a separate pass over all cells before the velocity reconstruction** — the cumulative reconstruction reads `grad(p)` of the whole field, so updating in the same loop corrupts it by orders of magnitude.
- **SIMPLE/PISO reuse the same predictor/corrector** (`solveSimple` = steady loop; `solvePiso` = one predictor + `nCorrectors` correctors, no under-relaxation). Do not duplicate the corrector logic. **PISO must refresh the Rhie-Chow data between correctors**: `refreshUHat(pred, velocity)` before every corrector after the first. Skipping it makes the loop reach its own fixed point after one sweep (algebraically `nCorrectors = 1`) and the run diverges — as OpenFOAM does with a single corrector.

## Dependencies
- `eigen3` — sparse linear algebra
- `doctest` — unit testing (header-only)

## Next Phase (Phase 4)
Agreed roadmap, in order:
1. ✅ **Unsteady terms + PISO** — theta-scheme time integration, transient scalar transport (verified time order), the shared `Momentum`/`Pressure`/`GridOperators` primitives, and `solvePiso` (one predictor + `nCorrectors`, per-corrector `refreshUHat`). `tests/test_piso.cpp` enabled. Details: `docs/numerical.md`; process and ruled-out list: `.agents/skills/openfoam-crosscheck/`.
2. **`pyfvm` Python bindings** (pybind11 via vcpkg, optional target) — case setup and post-processing from Python/numpy, replacing any JSON-config idea; the `fvm_solver` exe stays a smoke demo. PISO is complete, so the API-stability gate is satisfied; review the solver API once before starting.
3. **Arbitrary mesh input** (Gmsh `.msh` first) — the main motivation for the Python front-end; may come with non-orthogonal/skew mesh support.

Deferred/rejected: per-case executables under `examples/` (cumbersome), JSON case config (redundant with Python scripting), per-module CMakeLists/tests/docs split (revisit only if a module is reused externally or build times degrade). AMGCL/Hypre backends remain candidates for larger meshes.
