# AGENTS.md

## Project State
- **Status**: phases 1–4-2 complete; 4.3a mesh abstraction (dual mesh representations, zero numerical change) done. C++ suite: 63 test cases (4695 assertions) pass; binding suite (`pyfvm_tests`, 14 tests) passes. Everything described below is implemented and verified.
- **Capabilities**: user-facing capabilities and the repository layout live in `README.md`. The implementation uses dual mesh representations behind one interface, mesh-generic operators, Eigen-backed sparse solvers, persistent `FaceFluxField` (`phi`) as the only convection input, and optional `pyfvm` bindings; there is no C++ demo executable.
- **Solver structure**: the iteration is split into reusable `predictMomentum` / `correctPressure` primitives (`Momentum.h/.cpp`, `Pressure.h/.cpp`); `Simple`/`Piso` are thin driver loops. Extend those primitives rather than duplicating the loops.
- **PISO**: working; `tests/test_piso.cpp` is enabled. Two invariants dominated the implementation — the per-corrector `refreshUHat` and the absolute-pressure form; both are spelled out under **Key Design Decisions** below and reproduce the OpenFOAM single-step fingerprints. Cross-check numbers and the ruled-out list: `docs/numerical.md`; run comparisons with the `openfoam` MCP server.

## Skills (`.agents/skills/`, shared by Codex and OpenCode)

Both agents load the project skills from `.agents/skills` (Codex: project skill root; OpenCode: `skills.paths` in `opencode.json`). Use them:

- **`refactor-code`** — every planned change to this project's code goes through its four phases (plan → edit → review → docs). Phase 1 must present a plan and wait for the user's confirmation before any code is touched.

There is no OpenFOAM skill any more: the cross-check mechanics live in the `openfoam` MCP server (below) and the conclusions live in `docs/numerical.md`.

## Codex Subagent Policy

The project parent uses `glm-5.3` (configured in `.codex/config.toml`; the ZAI provider remains user-level).  The `glm-5.3-flash` custom agents in `.codex/agents/` are the **default executors** for bounded, low-reasoning work.  Once a task falls into a class below, delegate it automatically — no extra user approval is needed beyond the plan confirmation the workflow already requires:

| Task class | Default agent | Mode |
| --- | --- | --- |
| Bounded code edits under an approved plan (refactor Phase 2) | `fvm-implementer` | serialized, disjoint write scopes |
| Structural/style review of a bounded diff (Phase 3 only) | `fvm-reviewer` | may run in parallel |
| Documentation updates for an approved change (Phase 4) | `fvm-docs` | serialized |
| Build / ctest / binding / OpenFOAM log triage | `fvm-test-triage` | may run in parallel |
| Read-only fact-finding answering one specific question | `fvm-explorer` | may run in parallel |

Skip delegation and let the parent do the work directly when the round trip would cost more than the task itself: single-file or single-hunk fixes, questions answerable by reading one or two files, or tasks that would need repeated parent-child clarification.  Delegation pays off when it saves parent context (large material distilled into a summary) or moves mechanical multi-file edits onto the flash model.

**Delegation shape (persistent implementer session)**: one approved plan maps
to one persistent `fvm-implementer` session.  Spawn it once with the full
step-group overview plus the first group's detailed brief, feed each later
group as a queued message, quickly review each group's diff (`git diff --stat`
first, then the touched hunks) before feeding the next one, and close the
session when the plan is done — never respawn the implementer per step, and
never `fork_context` it (see Token discipline).  The gate review is the
parent's own quick pass and is never delegated: `fvm-reviewer` runs once, on
the whole diff, in Phase 3 only, and a stalled group goes through the fuse
rule — triage it yourself or via `fvm-test-triage`, never through a
mid-Phase-2 reviewer (including a pipelined deep review of one group while
the next one runs).  Solver-sensitive slices (SIMPLE/PISO, pressure
correction, Rhie–Chow, flux reconstruction, boundary flux logic) stay
delegable, but only with a **design-complete brief**: formulas, sign/index
conventions, boundary handling, and every call site are fixed by the approved
plan.  A detail the brief does not cover is a stop-and-report question, not a
decision the child makes.

Single source of truth for delegation: this section (routing table, skip threshold, delegation shape, parent-owned responsibilities) is the only home of the policy.  `.codex/agents/*.toml` own each agent's runtime behaviour (the mechanism layer), and the `refactor-code` skill only maps workflow phases to agents without restating the policy.

The parent owns requirements analysis, decomposition, architecture, numerical
design decisions, conflict resolution, final integration, and final review of
solver-sensitive code (SIMPLE/PISO, pressure correction, Rhie–Chow, flux
reconstruction, and boundary flux logic).  A flash child must escalate
ambiguity, repeated test failures, numerical-risk findings, or cross-module
changes instead of deciding them itself.  Keep child outputs compact and never
let parallel children edit overlapping files.

### Token discipline (the parent context is the scarce resource)

- Run one refactor phase per parent session.  A persistent implementer session may span all step groups of that one phase (see Delegation shape); state handed across phases or across agent sessions goes through `AGENTS.md` / `docs/`, never through conversation history.
- Spawn flash sub-agents with a fresh context and a self-contained brief; never combine `fork_context` with `agent_type` (forked agents inherit the parent model and its whole context).
- Reuse one live implementer session across the step groups of a plan (feed the groups as queued messages) instead of respawning it per group; close sub-agents once their work is done — completed agents keep occupying a concurrency slot.
- Sub-agent reports must stay compact (a few hundred words); long material is written to a scratch file (e.g. `.agents/notes/<topic>.md`) and the report returns only the summary plus the path.
- Prefer an `fvm-explorer` distillation over the parent reading whole files; when the parent must read, read headers, signatures or narrow line windows, and run `git diff --stat` before any full diff.
- `build_and_test` reports success compactly by default (`quiet=true`); pass `quiet=false` only when triaging a failure needs the full transcript.

## Architecture

- The code is organized into `core`, `math`, `io`, and `numerical` modules, with optional `pyfvm` bindings and `doctest` coverage in `tests/`.
- The numerical stack is mesh-generic; `Momentum`/`Pressure` are reusable primitives and `Simple`/`Piso` are thin drivers.
- The strict inter-module dependency rule is kept in **Naming Conventions** below.

Full directory layout: see README.md (Project Structure).

## Naming Conventions
- **Strict sub-namespaces**: `fvm::core`, `fvm::math`, `fvm::io`, `fvm::numerical`; each module has its own `include/` + `src/`.
- Cross-module dependencies use `using` declarations for brevity (e.g. `math` uses `fvm::core::Scalar`). Declare them **inside the module's own namespace**, never at global scope: a global-scope `using` injects the alias into every translation unit that includes the header, hides missing includes (each header must list the aliases it actually uses), and collides with Eigen/pybind11 internals (measured: 269 MSVC C4459 warnings, all traced back to these declarations). A using-declaration imports one name only — it does not nest the other namespace.
- **Dependency rule**: `core` → nothing; `math` → core (+ Eigen); `io` → core; `numerical` → core + math. Inside `numerical`: `Momentum` → Diffusion/Convection/GridOperators/TransportEquation/TimeScheme; `Pressure` → Momentum/GridOperators; `Simple`/`Piso` → Momentum/Pressure/Convection (they own the linear solvers). Do not introduce cycles.
- **Source files stay pure ASCII** (MSVC C4819); docs are Chinese, code comments English.

## Build Instructions

Human-facing prerequisites and the canonical manual CMake/CTest commands are in README.md (Build & Test); use them only when the MCP tool is unavailable.

**IMPORTANT for agents:** use the `build-and-test` MCP tool (see **MCP Servers** below) instead of shelling out to `cmake`/`ctest`/`ninja`. It configures into `build/<config>-agent` and runs **outside** the agent sandbox, which is required because vcpkg writes under `$VCPKG_ROOT` (outside this repository). Options: `config` Release/Debug, `target` all/fvm_tests/pyfvm, `run` none/tests, `python` true/false, `quiet` true/false (defaults: Release, all, tests, false, true). `python=true` builds the `pyfvm` bindings (`-DFVM_BUILD_PYTHON=ON`) into the separate `build/<config>-agent-py` directory, and ctest there additionally runs `pyfvm_tests` (the Python binding tests). The result is structured JSON: `quiet=true` (the default) keeps success compact, and failures carry `first_error`, a bounded `tail` and `log_path` (the untruncated transcript goes to `build_and_test.log` in the build directory). As part of a code-modification workflow, proceed with the defaults unless the workflow step specifies otherwise (refactor Phase 2's grouped validation is the exception; the `refactor-code` skill owns that granularity); when the user asks for a build/test directly, confirm the options first. Running a Python case script needs no sandbox escape — just `python python/examples/<name>.py`.

## MCP Servers

Two project-scoped servers, both plain-stdlib Python speaking newline-delimited JSON-RPC over stdio: `build-and-test` (`.agents/mcp/build_and_test_server.py`) and `openfoam` (`.agents/mcp/openfoam_server.py`). They are registered for Codex in `.codex/config.toml` and for OpenCode in `opencode.json`, and both run **outside** the agent sandbox — that is what makes WSL and vcpkg access work without per-call escalation, and it is the reason they exist. Prefer them; fall back to the manual commands in README.md (Build & Test) only when a server is unavailable.
Harness internals (file map, environment variables, verification and extension rules): README.md, "Agent 开发环境（harness）".

### build-and-test (`.agents/mcp/build_and_test_server.py`)

One tool, `build_and_test(config, target, run, python, quiet)`: configure (`cmake -G Ninja`) + build + optionally run ctest. Always use it instead of shelling out to `cmake`/`ctest`/`ninja` — the OpenCode bash permission block denies those commands, and in Codex this instruction is the rule. Options, defaults (Release / all / tests / false / quiet=true) and when to confirm them with the user: see **Build Instructions** above. `quiet=true` (default) keeps success reports compact to protect the caller's context; failures return structured diagnostics (`first_error`, a bounded `tail`, `log_path`), and the untruncated transcript is written to `build_and_test.log` in the build directory. It deliberately does **not** run arbitrary scripts: Python case scripts run fine inside the agent sandbox, and regression coverage for the bindings lives in `pyfvm_tests`.

### openfoam (`.agents/mcp/openfoam_server.py`)

The single entry point for reference-implementation checks; it returns structured JSON. Parameters live in the tool schemas, so only "when to use which" is repeated here:

- `run_case` — fresh timestamped case dir with the configuration baked into the name (patches `endTime`/`deltaT`/`writeInterval`/`nCorrectors`), starts `foamRun` in the background and returns immediately with `status: "running"`. The run stays alive only while the MCP server process lives — do not restart the agent session mid-run.
- `run_status` — poll a background run (`running` / `finished` / `failed` / `stale`; `stale` means the launcher died before writing `.exitcode`, e.g. after a server restart — rerun the case) plus a partial or complete per-step summary; poll this after `run_case` until the state is terminal.
- `summarize_log` — per-step table from a solver log (`res_p` of the *second* p solve is the fingerprint that the extra corrector is really advancing).
- `extract_fields` — read a time directory's fields and map them onto this project's layout (`grid[j][i]`, `x_faces`/`y_faces`, per-patch values with owner cells and a side guess); `nx`/`ny` must equal the mesh's own cell counts (`nCells`).
- `find_source` / `read_source` — grep and read the OpenFOAM-14 tree (`~/OpenFOAM/OpenFOAM-14`) when the data alone cannot explain a difference. Note: OF-14 has no `pEqn.H`/`UEqn.H` for the modern solvers; the incompressible pressure correction is `applications/modules/incompressibleFluid/correctPressure.C`.
- `list_cases` — existing case directories under the root.

Hardening: `run_case` confines its `template`/`root` arguments to allowed roots — the default run root plus `FVM_OF_ALLOWED_ROOTS` (":"-separated absolute WSL paths) — canonicalised with `realpath` inside WSL, so a confused agent cannot copy a template from (or write a case to) an arbitrary location. `extract_fields` is ascii-only: a `writeFormat binary` case is rejected explicitly (plus a per-field NUL/replacement-character heuristic) instead of returning garbage. The dictionary parsing and field remapping live in the pure module `openfoam_fields.py` (no WSL, no I/O), so they can be unit-tested directly.

Discipline for a comparison (this is what keeps conclusions valid): fix the contract first (mesh, properties, schemes, every BC, `dt`, solver tolerances, algorithm settings such as `nCorrectors`); compare a **single step** first (`end_time == dt`) and check `p(0,0)` and the inlet `phi/S` to solver-tolerance level; one configuration per directory (never reuse a directory with different settings); run both `nCorrectors = 1` and `= 2`; never loosen a test tolerance to hide a difference. Record the conclusion (with its evidence and any ruled-out hypotheses) in `docs/numerical.md`, not in this file.

In Codex only `run_case` requires approval; `build_and_test` is auto-approved (its writes are confined to the repository `build/` tree and the vcpkg cache, and the persistent-implementer workflow relies on unattended intermediate compile checks), and the read-only `openfoam` tools are auto-approved. Read-only subagents are additionally instructed never to call `build_and_test`/`run_case` (`.codex/agents/*.toml`). The helper scripts the server drives (`make_of_case.sh`, `of_log_summary.py`) live in `.agents/mcp/openfoam/` — the fallback for humans when the MCP tool is unavailable. Nothing in this workflow ever modifies the OpenFOAM source tree or an existing case directory.

## Key Design Decisions

- **LinearSolver is abstract** (`createEigenBiCGSTAB` / `createEigenCG` / `createEigenSparseLU`); new backends (AMGCL, Hypre) only need a subclass + factory. **SparseMatrix wraps Eigen** and exposes only `insert`, `finalize`, `native()`.
- **Mesh abstraction is face-first**: `fvm::core::Mesh` exposes unified interior/boundary face ids and patch topology; `CartesianMesh` and `UnstructuredMesh` implement it. Cells remain cell-centered. Numerical code sees only `const Mesh&` — `src/numerical` must not include `CartesianMesh.h` — while `CartesianMesh` keeps its structured convenience layer for tests and bindings.
- **Mesh/flux conventions are owner-relative**: face area vector `S` points outward from the owner; interior `phi > 0` means owner→neighbour and boundary `phi` is owner-outward. Cartesian patches remain east/north/west/south (0/1/2/3). VTK output remains `.vti` ImageData (Cartesian meshes only; `.vtu` is planned for 4.3b).
- **Dual-representation regression discipline**: `makeStructuredMesh` builds the same structured mesh through `UnstructuredMesh`; `tests/test_mesh_dual.cpp` locks geometry, operators, coupled momentum assembly, and end-to-end SIMPLE equivalence so both implementations must answer identically.
- **Assembly accumulates**: `assembleDiffusion`/`assembleConvection` add into `(A, b)` without zeroing; `assembleTransport` owns the full assembly and returns an unfinalized `EquationSystem`.
- **Flux is first-class** (OpenFOAM `phi`): assemblers take a `FaceFluxField` (interior flux positive owner→neighbour; boundary flux owner-outward) instead of a velocity field; build it with `interpolateCellVelocityFlux` or `computeMassFlux`.
- **Source terms follow Patankar**: `S(phi) = Sc + Sp*phi` per unit volume, `Sp <= 0` enforced by exception.
- **`SolverConfig::tolerance` is a relative residual** `|Ax-b|/|b|`. The BiCGSTAB wrapper scales Eigen's absolute tolerance by `|b|` and checks convergence itself — do not rely on `solver.info()` for it.
- **SIMPLE (collocated + Rhie-Chow)**: momentum reuses the convection/diffusion operators; the pressure-gradient source is the Gauss form `sum_f p_f S_f` (so Dirichlet pressure values drive the flow). The relaxed diagonal is read from a finalized matrix copy; per-cell `d = V/a_P` is interpolated to faces and projected onto `S_f` (on Cartesian meshes this reduces to the u-equation `d` on x-faces and the v-equation `d` on y-faces). During assembly the persistent flux is overwritten with the pressure-free `phiHbyA = rho (uHat_f . S_f)` and then corrected in place (`F_f = phiHbyA_f - C(p_N - p_P)`; Dirichlet-p boundaries `- Cb(p_b - p_P)`), exactly OpenFOAM's `phi = phiHbyA - pEqn.flux()`. The per-face coefficients `C`/`Cb` are derived once during the assembly and stored per face; the correction reads them back instead of recomputing them, so the equation and its correction cannot drift apart.
- **Pressure equation is assembled for the absolute pressure**: `A p = b` with `b = -sum_f outward(phiHbyA_f) + sum_Dirichlet C_b p_b`. `p' = p - p_old` solves the same system as the old incremental form (`b - A p_old = -m`); the absolute form avoids forming that cancelling difference and needs no mirror (`Dirichlet -> 0`) pressure BCs. Getting the RHS sign wrong creates positive feedback and blows up.
- **Boundary face fluxes must be same-sourced as the interior**: the boundary `phiHbyA` uses the pressure-free `uHat` (or the Dirichlet velocity value) with no pressure term; the Dirichlet-p value enters the RHS as `C_b p_b` and the pressure term is applied to the flux afterwards. Using `uStar` (cell-centred pressure gradient) there is inconsistent with the equation and diverges.
- **Pure-Neumann pressure uses reference-cell elimination**: drop cell 0 and pin it to its previous value (`p_0 = p_old(0)`, the old `p'_0 = 0`), moving the dropped column into its neighbours' RHS — this keeps the CG system well-conditioned SPD. Closed domains must satisfy `checkFluxCompatibility` at entry (adjustPhi-style, throws on unbalanced velocity BCs), and `correctPressure` rebalances `phiHbyA` on every corrector (pure-Neumann only, with the same adjustment reaching the RHS) so the system stays solvable. No-op for all-Dirichlet-velocity domains (e.g. the cavity).
- **Time integration is an implicit theta scheme** (`TimeScheme` + `TimeTerm{rho, dt, scheme}`; `dt <= 0` means steady). The ddt term is assembled inside the shared momentum operator, so SIMPLE and PISO share it; `solveSimple` passes `TimeTerm{}` (bit-for-bit unchanged). `assembleTransientTransport` uses the same formula.
- **Transient momentum requires `relaxation == 1`** (under-relaxation is a steady SIMPLE device), and the ddt diagonal `rho V/dt` is part of `a_P`, so `d = V/a_P` is transient-consistent (OpenFOAM `rAU = 1/UEqn.A()`).
- **`correctPressure` rebuilds the flux and the velocity from the *unrelaxed* solved pressure**: `u = uHat - d grad(p_solved)` using the physical p BCs. For SIMPLE this equals the old `u* - d grad(p')` (since `uHat = u* + d grad(p_old)`), so the stored pressure may be relaxed (`p += relaxationP*(p_solved - p_old)`) while the velocity takes the full correction; for PISO `relaxationP == 1`, so the reconstruction accumulates across correctors. Note OpenFOAM's steady path instead calls `p.relax()` *before* `U = HbyA - rAU*grad(p)` (its velocity inherits the relaxation, its flux does not); we keep the Patankar variant. There is no `cumulativeVelocityCorrection` flag any more: absolute pressure plus the unrelaxed reconstruction covers both drivers, and no ordering constraint between the pressure update and the velocity reconstruction remains (`p_solved` is an immutable local field).
- **SIMPLE/PISO reuse the same predictor/corrector** (`solveSimple` = steady loop; `solvePiso` = one predictor + `nCorrectors` correctors, no under-relaxation). Do not duplicate the corrector logic. **PISO must refresh the Rhie-Chow data between correctors**: `refreshUHat(pred, velocity)` before every corrector after the first. Skipping it makes the loop reach its own fixed point after one sweep (algebraically `nCorrectors = 1`) and the run diverges — as OpenFOAM does with a single corrector. `solvePiso` therefore rejects `nCorrectors < 2`; the OpenFOAM-side cross-check still exercises `nCorrectors = 1` against the reference solver, which is a different code path.

## Next Phase (Phase 4)
Next is 4.3b: Gmsh `.msh` reader + `.vtu` writer + `pyfvm.read_gmsh`; 4.3c then adds non-orthogonal correction + validation. Application tests for the example cases are agreed in principle but not scheduled.
Full roadmap: see README.md.
