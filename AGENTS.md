# AGENTS.md

## Project State
- **Status**: phases 1–4-1 complete; all 50 test cases (3271 assertions) pass. Everything described below is implemented and verified.
- **Capabilities**: 2D uniform-Cartesian FVM (core/io), Eigen-backed sparse solvers (math), steady SIMPLE and transient PISO for incompressible NS on a collocated grid with Rhie–Chow interpolation, and theta-scheme transient scalar transport. A persistent `FaceFluxField` (`phi`) is the only convection input of the assemblers.
- **Solver structure**: the iteration is split into reusable `predictMomentum` / `correctPressure` primitives (`Momentum.h/.cpp`, `Pressure.h/.cpp`); `Simple`/`Piso` are thin driver loops. Extend those primitives rather than duplicating the loops.
- **PISO**: working; `tests/test_piso.cpp` is enabled. Two invariants dominated the implementation — the per-corrector `refreshUHat` and the absolute-pressure form; both are spelled out under **Key Design Decisions** below and reproduce the OpenFOAM single-step fingerprints. Cross-check numbers and the ruled-out list: `docs/numerical.md`; run comparisons with the `openfoam` MCP server.

## Skills (`.agents/skills/`, shared by Codex and opencode)

Both agents load the project skills from `.agents/skills` (Codex: project skill root; opencode: `skills.paths` in `opencode.json`). Use them:

- **`refactor-code`** — every planned change to this project's code goes through its four phases (plan → edit → review → docs). Phase 1 must present a plan and wait for the user's confirmation before any code is touched.

There is no OpenFOAM skill any more: the cross-check mechanics live in the `openfoam` MCP server (below) and the conclusions live in `docs/numerical.md`.

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
                           initializeMassFlux (+ hasDirichletPressure), checkFluxCompatibility
    TransportEquation.h  — steady conv-diff-source assembly + assembleTransientTransport (theta)
    TimeScheme.h         — TimeScheme{Euler,CrankNicolson}, TimeTerm (header-only)
    Momentum.h           — MomentumAssembly, assembleMomentum, MomentumPrediction,
                           predictMomentum, refreshUHat
    Pressure.h           — CorrectorResult, correctPressure
    Simple.h / Piso.h    — driver configs/results; solveSimple / solvePiso
  numerical/src/         — the implementations above. Momentum.cpp holds the Rhie-Chow data
                           (computeUHat -> uHat = H/a_P, d = V/a_P); Pressure.cpp holds the
                           absolute-pressure equation, reference-cell elimination and the
                           flux/U/p reconstruction.
  app/main.cpp           — demos: steady convection-diffusion; lid-driven cavity Re=100 (cavity.vti)

tests/                   — doctest: mesh, field, flux, linalg, diffusion, convection, transport,
                           transient (time order), simple (Poiseuille, cavity Re=100),
                           piso (impulsive Couette, transient Poiseuille)
docs/                    — module docs (Chinese): core.md, math.md, io.md, numerical.md, app.md
```

## Naming Conventions
- **Strict sub-namespaces**: `fvm::core`, `fvm::math`, `fvm::io`, `fvm::numerical`; each module has its own `include/` + `src/`.
- Cross-module dependencies use `using` declarations for brevity (e.g. `math` uses `fvm::core::Scalar`). Declare them **inside the module's own namespace**, never at global scope: a global-scope `using` injects the alias into every translation unit that includes the header, hides missing includes (each header must list the aliases it actually uses), and collides with Eigen/pybind11 internals (measured: 269 MSVC C4459 warnings, all traced back to these declarations). A using-declaration imports one name only — it does not nest the other namespace.
- **Dependency rule**: `core` → nothing; `math` → core (+ Eigen); `io` → core; `numerical` → core + math. Inside `numerical`: `Momentum` → Diffusion/Convection/GridOperators/TransportEquation/TimeScheme; `Pressure` → Momentum/GridOperators; `Simple`/`Piso` → Momentum/Pressure/Convection (they own the linear solvers). Do not introduce cycles.
- **Source files stay pure ASCII** (MSVC C4819); docs are Chinese, code comments English.

## Build Instructions

**Prerequisites:** CMake >= 3.20, vcpkg, Ninja. Two things must be in the **environment** (add them as persistent Windows variables *before* launching an agent — a running agent will not pick up later changes):
- **`VCPKG_ROOT`** — the vcpkg installation.
- **`Path`** — must contain the directory holding `ninja.exe`.

**IMPORTANT for agents:** use the `build-and-test` MCP tool (see **MCP Servers** below) instead of shelling out to `cmake`/`ctest`/`ninja`. It configures into `build/<config>-agent` and runs **outside** the agent sandbox, which is required because vcpkg writes under `$VCPKG_ROOT` (outside this repository). Options: `config` Release/Debug, `target` all/fvm_solver/fvm_tests, `run` none/tests/solver/both (defaults: Release, all, tests). As part of a code-modification workflow, proceed with the defaults; when the user asks for a build/test directly, confirm the options first.

Manual commands (humans, or when the MCP tool is unavailable):

```bash
cmake -B build -S . -G Ninja -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
ctest --test-dir build --output-on-failure
./build/Release/fvm_solver        # demos
```

## MCP Servers

Two project-scoped servers, both plain-stdlib Python speaking newline-delimited JSON-RPC over stdio: `build-and-test` (`.agents/mcp/build_and_test_server.py`) and `openfoam` (`.agents/mcp/openfoam_server.py`). They are registered for Codex in `.codex/config.toml` and for opencode in `opencode.json`, and both run **outside** the agent sandbox — that is what makes WSL and vcpkg access work without per-call escalation, and it is the reason they exist. Prefer them; fall back to the manual commands above only when a server is unavailable.

### build-and-test (`.agents/mcp/build_and_test_server.py`)

One tool, `build_and_test(config, target, run)`: configure (`cmake -G Ninja`) + build + optionally run ctest and/or the demo. Always use it instead of shelling out to `cmake`/`ctest`/`ninja` — the opencode bash permission block denies those commands, and in Codex this instruction is the rule. Options, defaults (Release / all / tests) and when to confirm them with the user: see **Build Instructions** above.

### openfoam (`.agents/mcp/openfoam_server.py`)

The single entry point for reference-implementation checks; it returns structured JSON. Parameters live in the tool schemas, so only "when to use which" is repeated here:

- `run_case` — fresh timestamped case dir with the configuration baked into the name (patches `endTime`/`deltaT`/`writeInterval`/`nCorrectors`), starts `foamRun` in the background and returns immediately with `status: "running"`. The run stays alive only while the MCP server process lives — do not restart the agent session mid-run.
- `run_status` — poll a background run (`running` / `finished` / `failed`) plus a partial or complete per-step summary; poll this after `run_case` until it leaves `running`.
- `summarize_log` — per-step table from a solver log (`res_p` of the *second* p solve is the fingerprint that the extra corrector is really advancing).
- `extract_fields` — read a time directory's fields and map them onto this project's layout (`grid[j][i]`, `x_faces`/`y_faces`, per-patch values with owner cells and a side guess); `nx`/`ny` must equal the mesh's own cell counts (`nCells`).
- `find_source` / `read_source` — grep and read the OpenFOAM-14 tree (`~/OpenFOAM/OpenFOAM-14`) when the data alone cannot explain a difference. Note: OF-14 has no `pEqn.H`/`UEqn.H` for the modern solvers; the incompressible pressure correction is `applications/modules/incompressibleFluid/correctPressure.C`.
- `list_cases` — existing case directories under the root.

Discipline for a comparison (this is what keeps conclusions valid): fix the contract first (mesh, properties, schemes, every BC, `dt`, solver tolerances, algorithm settings such as `nCorrectors`); compare a **single step** first (`end_time == dt`) and check `p(0,0)` and the inlet `phi/S` to solver-tolerance level; one configuration per directory (never reuse a directory with different settings); run both `nCorrectors = 1` and `= 2`; never loosen a test tolerance to hide a difference. Record the conclusion (with its evidence and any ruled-out hypotheses) in `docs/numerical.md`, not in this file.

In Codex every `openfoam` tool except `list_cases` is `approval_mode = "approve"`, so expect a prompt. The helper scripts the server drives (`make_of_case.sh`, `of_log_summary.py`) live in `.agents/mcp/openfoam/` — the fallback for humans when the MCP tool is unavailable. Nothing in this workflow ever modifies the OpenFOAM source tree or an existing case directory.

## Key Design Decisions

- **LinearSolver is abstract** (`createEigenBiCGSTAB` / `createEigenCG` / `createEigenSparseLU`); new backends (AMGCL, Hypre) only need a subclass + factory. **SparseMatrix wraps Eigen** and exposes only `insert`, `finalize`, `native()`.
- **Mesh is cell-centered**; face indices are 0=east, 1=north, 2=west, 3=south (all boundary arrays use the same order). VTK output is `.vti` ImageData.
- **Assembly accumulates**: `assembleDiffusion`/`assembleConvection` add into `(A, b)` without zeroing; `assembleTransport` owns the full assembly and returns an unfinalized `EquationSystem`.
- **Flux is first-class** (OpenFOAM `phi`): assemblers take a `FaceFluxField` (positive along +x/+y, boundary outward fluxes signed) instead of a velocity field; build it with `interpolateCellVelocityFlux` or `computeMassFlux`.
- **Source terms follow Patankar**: `S(phi) = Sc + Sp*phi` per unit volume, `Sp <= 0` enforced by exception.
- **`SolverConfig::tolerance` is a relative residual** `|Ax-b|/|b|`. The BiCGSTAB wrapper scales Eigen's absolute tolerance by `|b|` and checks convergence itself — do not rely on `solver.info()` for it.
- **SIMPLE (collocated + Rhie-Chow)**: momentum reuses the convection/diffusion operators; the pressure-gradient source is the Gauss form `sum_f p_f n S_f` (so Dirichlet pressure values drive the flow). The relaxed diagonal is read from a finalized matrix copy; `d = V/a_P` feeds the Rhie-Chow faces (x-faces use the u-equation diagonal, y-faces the v-equation one). During assembly the persistent flux is overwritten with the pressure-free `phiHbyA = rho S uHat_f` and then corrected in place (`F_f = phiHbyA_f - C(p_N - p_P)`; Dirichlet-p boundaries `- Cb(p_b - p_P)`), exactly OpenFOAM's `phi = phiHbyA - pEqn.flux()`. The per-face coefficients `C`/`Cb` are derived once during the assembly and stored per face; the correction reads them back instead of recomputing them, so the equation and its correction cannot drift apart.
- **Pressure equation is assembled for the absolute pressure**: `A p = b` with `b = -sum_f outward(phiHbyA_f) + sum_Dirichlet C_b p_b`. `p' = p - p_old` solves the same system as the old incremental form (`b - A p_old = -m`); the absolute form avoids forming that cancelling difference and needs no mirror (`Dirichlet -> 0`) pressure BCs. Getting the RHS sign wrong creates positive feedback and blows up.
- **Boundary face fluxes must be same-sourced as the interior**: the boundary `phiHbyA` uses the pressure-free `uHat` (or the Dirichlet velocity value) with no pressure term; the Dirichlet-p value enters the RHS as `C_b p_b` and the pressure term is applied to the flux afterwards. Using `uStar` (cell-centred pressure gradient) there is inconsistent with the equation and diverges.
- **Pure-Neumann pressure uses reference-cell elimination**: drop cell 0 and pin it to its previous value (`p_0 = p_old(0)`, the old `p'_0 = 0`), moving the dropped column into its neighbours' RHS — this keeps the CG system well-conditioned SPD. Closed domains must satisfy `checkFluxCompatibility` at entry (adjustPhi-style, throws on unbalanced velocity BCs), and `correctPressure` rebalances `phiHbyA` on every corrector (pure-Neumann only, with the same adjustment reaching the RHS) so the system stays solvable. No-op for all-Dirichlet-velocity domains (e.g. the cavity).
- **Time integration is an implicit theta scheme** (`TimeScheme` + `TimeTerm{rho, dt, scheme}`; `dt <= 0` means steady). The ddt term is assembled inside the shared momentum operator, so SIMPLE and PISO share it; `solveSimple` passes `TimeTerm{}` (bit-for-bit unchanged). `assembleTransientTransport` uses the same formula.
- **Transient momentum requires `relaxation == 1`** (under-relaxation is a steady SIMPLE device), and the ddt diagonal `rho V/dt` is part of `a_P`, so `d = V/a_P` is transient-consistent (OpenFOAM `rAU = 1/UEqn.A()`).
- **`correctPressure` rebuilds the flux and the velocity from the *unrelaxed* solved pressure**: `u = uHat - d grad(p_solved)` using the physical p BCs. For SIMPLE this equals the old `u* - d grad(p')` (since `uHat = u* + d grad(p_old)`), so the stored pressure may be relaxed (`p += relaxationP*(p_solved - p_old)`) while the velocity takes the full correction; for PISO `relaxationP == 1`, so the reconstruction accumulates across correctors. Note OpenFOAM's steady path instead calls `p.relax()` *before* `U = HbyA - rAU*grad(p)` (its velocity inherits the relaxation, its flux does not); we keep the Patankar variant. There is no `cumulativeVelocityCorrection` flag any more: absolute pressure plus the unrelaxed reconstruction covers both drivers, and no ordering constraint between the pressure update and the velocity reconstruction remains (`p_solved` is an immutable local field).
- **SIMPLE/PISO reuse the same predictor/corrector** (`solveSimple` = steady loop; `solvePiso` = one predictor + `nCorrectors` correctors, no under-relaxation). Do not duplicate the corrector logic. **PISO must refresh the Rhie-Chow data between correctors**: `refreshUHat(pred, velocity)` before every corrector after the first. Skipping it makes the loop reach its own fixed point after one sweep (algebraically `nCorrectors = 1`) and the run diverges — as OpenFOAM does with a single corrector. `solvePiso` therefore rejects `nCorrectors < 2`; the OpenFOAM-side cross-check still exercises `nCorrectors = 1` against the reference solver, which is a different code path.

## Dependencies
- `eigen3` — sparse linear algebra
- `doctest` — unit testing (header-only)

## Next Phase (Phase 4)
Agreed roadmap, in order:
1. ✅ **Unsteady terms + PISO** — done: transient theta scheme (time order verified), transient transport, the shared `Momentum`/`Pressure`/`GridOperators` primitives and `solvePiso`; `tests/test_piso.cpp` enabled. Cross-check numbers and the ruled-out list: `docs/numerical.md`.
2. **`pyfvm` Python bindings** (pybind11 via vcpkg, optional target) — case setup and post-processing from Python/numpy, replacing any JSON-config idea; the `fvm_solver` exe stays a smoke demo. PISO is complete, so the API-stability gate is satisfied; review the solver API once before starting.
3. **Arbitrary mesh input** (Gmsh `.msh` first) — the main motivation for the Python front-end; may come with non-orthogonal/skew mesh support.

Deferred/rejected: per-case executables under `examples/` (cumbersome), JSON case config (redundant with Python scripting), per-module CMakeLists/tests/docs split (revisit only if a module is reused externally or build times degrade). AMGCL/Hypre backends remain candidates for larger meshes.
