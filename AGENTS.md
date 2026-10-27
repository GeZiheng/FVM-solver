# AGENTS.md

## Project State

- **Status**: phases 1–4-2, 4.3a, 4.3b and 4.3c (mesh abstraction, dual representations, SU2 mesh reader, `.vtu` writer + `pyfvm.read_su2`, non-orthogonal/skewness corrections + LSQ gradients + optional interior `NonOrthSplit`) complete; 100 C++ test cases and `pyfvm_tests` (22) pass. The OpenFOAM oblique trapezoidal NS cross-check passes at converged corrector counts (L2 4.5e-4 at NC=8; the NC=2 gap is mainly corrector lag, and the remaining general-field split dependence is O(h²), see `docs/numerical.md`).
- **Capabilities & layout**: see `README.md`. No C++ demo executable.
- **Solver structure**: reusable `predictMomentum` / `correctPressure` primitives (`Momentum`/`Pressure`); `Simple`/`Piso` are thin drivers — extend the primitives, not the loops.
- **PISO**: working (`tests/test_piso.cpp`); invariants and cross-check evidence: **Key Design Decisions** + `docs/numerical.md` (via the `openfoam` MCP server).

## Skills (`.agents/skills/`, shared by Codex and OpenCode)

Both agents load the project skills from `.agents/skills` (Codex: project skill root; OpenCode: `skills.paths` in `opencode.json`). Use them:

- **`refactor-code`** — every planned change to this project's code goes through its four phases (plan → edit → review → docs). Phase 1 must present a plan and wait for the user's confirmation before any code is touched.

There is no OpenFOAM skill: cross-check mechanics live in the `openfoam` MCP server, conclusions in `docs/numerical.md`.

## Codex Subagent Policy

The project parent uses `glm-5.3` (`.codex/config.toml`; the ZAI provider stays user-level). The `glm-5.3-flash` agents in `.codex/agents/` execute work that **passes the delegation gates** or **meets the read-only triggers** below; everything else is parent-direct, with no approval beyond the plan confirmation the workflow already requires:

| Task class | Default agent | Mode |
| --- | --- | --- |
| Gate-passing code-edit groups under an approved plan (refactor Phase 2) | `fvm-implementer` | serialized, disjoint write scopes |
| Structural/style review of the whole diff (Phase 3 only) | `fvm-reviewer` | may run in parallel |
| Gate-passing documentation updates (Phase 4) | `fvm-docs` | serialized |
| Build / ctest / binding / OpenFOAM log triage (trigger-gated) | `fvm-test-triage` | may run in parallel |
| Read-only fact-finding answering one specific question (trigger-gated) | `fvm-explorer` | may run in parallel |

**Delegation gates** — one round trip costs a measured floor of ~40 s and ~1k parent-side coordination tokens (`.agents/notes/fvm-implementer-benchmark.md`); the gates keep that off small work. The Phase 1 plan labels every step group's executor (parent / `fvm-implementer`) according to:

- **Size gate** (initial values; re-tune only with new benchmark data): delegate only if the group touches >= 3 files, or >= 2 files with >= 100 changed lines, or applies one pattern at >= 3 call sites (porter class — the brief carries a worked before/after example).
- **Design gate**: formulas, sign/index conventions, boundary handling, and every call site are fixed by the plan; an uncovered detail is a stop-and-report question, not a child decision, and a paste-ready brief is pasted by the parent instead of delegated.
- **Value gate**: delegation buys parallelism or parent-context relief, not latency — if the parent would only idle-wait on the child, it executes directly.

The parent may cross a gate either way with a one-line justification; `fvm-docs` passes the same gates. The gates measure edits — read-only agents use the triggers below instead: normal in-policy delegations, never gate overrides, still logged.

**Read-only triggers** — `fvm-explorer`: >= 2 independent read-only questions fan out while the parent advances the critical path (Phase 1 planning is the natural slot), or the answer needs a whole-file / cross-module / multi-doc sweep (> ~150 lines) where only the distilled conclusion matters; blocking questions are parent-direct, and solver-sensitive numerical-design reading is first-hand (headers, signatures, narrow windows otherwise). `fvm-test-triage`: the structured report (`first_error` / `tail` / `summarize_log`) is not self-explanatory and full-log or multi-cycle triage would flood parent context, triage parallels parent work, or the Phase-2 fuse rule routes a stalled group there; a single obvious failure read straight from the report is parent-direct. Each dispatch must outweigh the fixed overhead above; when in doubt, parent-direct.

**Session shape** — one approved plan, one persistent `fvm-implementer` session: spawn once with the step-group overview plus the first brief, feed later groups as queued messages, review each group's diff (`git diff --stat`, then the touched hunks) before the next, close when the plan is done — never respawn per group. Give flash sub-agents fresh context and a self-contained brief, never `fork_context` (forking inherits the parent model and whole context), and close every sub-agent once done — completed agents keep occupying a concurrency slot.

**Reviewer rule** — `fvm-reviewer` is the ungated exception: exactly once per diff, on the whole diff, Phase 3 only. Phase-2 gate reviews are the parent's own quick pass, never delegated; a stalled group goes to the fuse rule (triage it yourself or via `fvm-test-triage`), never to a mid-Phase-2 reviewer or a pipelined per-group deep review.

**Briefs, reports** — briefs carry scope files, interface/invariants, the call-site list, the validation instruction, and the trivial details the child may decide. Reports: <= 150 words — changed files, validation status, deviations, risks — never restating the plan; long material goes to a scratch file (`.agents/notes/<topic>.md`), the report returns summary plus path. Batching trial: up to 3 disjoint porter-class groups per message, one combined report; otherwise one brief per gate review.

**Parent-owned work** — requirements analysis, decomposition, architecture, numerical design decisions, conflict resolution, final integration, and final review of solver-sensitive code (SIMPLE/PISO, pressure correction, Rhie–Chow, flux reconstruction, boundary flux logic); such slices stay delegable only with a design-complete brief (design gate). A flash child escalates ambiguity, repeated test failures, numerical-risk findings, or cross-module changes instead of deciding them; parallel children never edit overlapping files.

**Measured non-levers** (do not re-litigate without new data): reasoning-effort overrides (A/B: 40 s vs 41 s), respawning the implementer per group, removing stop-and-report.

Single source of truth for delegation: this section (routing table, gates, read-only triggers, session shape, parent-owned work). `.codex/agents/*.toml` own runtime behaviour; the `refactor-code` skill only maps workflow phases to agents, never restating policy or gate numbers.

### Token discipline (the parent context is the scarce resource)

- Run one refactor phase per parent session; state handed across phases or across agent sessions goes through `AGENTS.md` / `docs/`, never through conversation history.
- Prefer an `fvm-explorer` distillation over the parent reading whole files; when the parent must read, read headers, signatures or narrow line windows, and run `git diff --stat` before any full diff.
- `build_and_test` reports success compactly by default (`quiet=true`); pass `quiet=false` only when triaging a failure needs the full transcript.

## Architecture

- The code is organized into `core`, `math`, `io`, and `numerical` modules, with optional `pyfvm` bindings and `doctest` coverage in `tests/`.
- The numerical stack is mesh-generic; `Momentum`/`Pressure` are reusable primitives and `Simple`/`Piso` are thin drivers.
- The strict inter-module dependency rule is kept in **Naming Conventions** below.

Full directory layout: see README.md (Project Structure).

## Naming Conventions

- **Strict sub-namespaces**: `fvm::core`, `fvm::math`, `fvm::io`, `fvm::numerical`; each module has its own `include/` + `src/`.
- Cross-module dependencies use `using` declarations for brevity (e.g. `math` uses `fvm::core::Scalar`). Declare them **inside the module's own namespace**, never at global scope: a global-scope `using` injects the alias into every translation unit that includes the header, hides missing includes (each header must list the aliases it actually uses), and collides with Eigen/pybind11 internals (measured: 269 MSVC C4459 warnings). A using-declaration imports one name only — it does not nest the other namespace.
- **Dependency rule**: `core` → nothing; `math` → core (+ Eigen); `io` → core; `numerical` → core + math. Inside `numerical`: `Momentum` → Diffusion/Convection/GridOperators/TransportEquation/TimeScheme; `Pressure` → Momentum/GridOperators; `Simple`/`Piso` → Momentum/Pressure/Convection (they own the linear solvers). Do not introduce cycles.
- **Source files stay pure ASCII** (MSVC C4819); docs are Chinese, code comments English.

## Build Instructions

Human-facing prerequisites and the canonical manual CMake/CTest commands are in README.md (Build & Test); use them only when the MCP tool is unavailable.

**IMPORTANT for agents:** use the `build-and-test` MCP tool instead of shelling out to `cmake`/`ctest`/`ninja` — it runs **outside** the agent sandbox (vcpkg writes under `$VCPKG_ROOT`). Options: `config` Release/Debug, `target` all/fvm_tests/pyfvm, `run` none/tests, `python` true/false, `quiet` true/false (defaults: Release, all, tests, false, true). `python=true` builds the bindings into the separate `build/<config>-agent-py` directory; ctest there also runs `pyfvm_tests`. In a code-modification workflow use the defaults unless the step says otherwise (refactor Phase 2's grouped validation is the exception; the `refactor-code` skill owns that granularity); confirm options first when the user asks for a build/test directly. Python case scripts need no sandbox escape.

## MCP Servers

Two project-scoped stdio JSON-RPC servers (plain-stdlib Python): `build-and-test` and `openfoam`, registered in `.codex/config.toml` / `opencode.json` and running **outside** the agent sandbox — that is what makes WSL and vcpkg access work. Prefer them; manual fallback commands: README.md (Build & Test). Harness internals (file map, env vars, verification rules): README.md, "Agent 开发环境（harness）".

### build-and-test (`.agents/mcp/build_and_test_server.py`)

One tool, `build_and_test(...)`: configure + build + optional ctest — always use it instead of `cmake`/`ctest`/`ninja` (the OpenCode bash permission block denies those anyway). Options and defaults: **Build Instructions** above; `quiet=true` keeps success compact, failures return `first_error` / `tail` / `log_path` (full transcript in `build_and_test.log`). It deliberately runs no arbitrary scripts — binding regression lives in `pyfvm_tests`.

### openfoam (`.agents/mcp/openfoam_server.py`)

The single entry point for OpenFOAM-14 reference checks (structured JSON; parameters live in the tool schemas). "When to use which":

- `run_case` — fresh timestamped case dir, starts `foamRun` in the background; the run lives only as long as the MCP server process — never restart the agent session mid-run.
- `run_status` — poll until terminal (`finished` / `failed` / `stale`; `stale` = launcher died, rerun the case).
- `summarize_log` — per-step residual table; `res_p` of the *second* p solve is the fingerprint that the extra corrector is really advancing.
- `extract_fields` — map a time directory onto this project's layout (ascii cases only; `writeFormat binary` rejected); `nx`/`ny` must equal the mesh's own `nCells`.
- `find_source` / `read_source` — grep/read the OF-14 tree (incompressible pressure correction: `applications/modules/incompressibleFluid/correctPressure.C`).
- `list_cases` — existing case directories under the root.

Comparison discipline (full protocol: `docs/numerical.md` §对比纪律): fix the contract first; **single step** first (`end_time == dt`, `p(0,0)` + inlet `phi/S` to solver tolerance); one configuration per directory; `nCorrectors = 1` and `= 2`; never loosen tolerances.

Only `run_case` requires approval; `build_and_test` is auto-approved; the read-only `openfoam` tools are auto-approved. Hardening and side effects: README "Agent 开发环境（harness）". Helper scripts (`.agents/mcp/openfoam/`) are the human fallback. Never modify the OpenFOAM source tree or an existing case directory.

## Key Design Decisions

Rules only; derivations and evidence: `docs/<module>.md` (mainly `docs/numerical.md`). Keep in sync with design changes.

- **LinearSolver is abstract** (`createEigenBiCGSTAB` / `createEigenCG` / `createEigenSparseLU`); new backends (AMGCL, Hypre) need only a subclass + factory. **SparseMatrix wraps Eigen** and exposes only `insert`, `finalize`, `native()` (`docs/math.md`).
- **Mesh abstraction is face-first**: `fvm::core::Mesh` exposes unified interior/boundary face ids and patch topology; `CartesianMesh` and `UnstructuredMesh` implement it; cells stay cell-centered. Numerical code sees only `const Mesh&` — `src/numerical` must not include `CartesianMesh.h` — while `CartesianMesh` keeps its structured convenience layer (`docs/core.md`).
- **Mesh/flux conventions are owner-relative**: `S` points outward from the owner; interior `phi > 0` means owner→neighbour and boundary `phi` is owner-outward; patches stay east/north/west/south (0/1/2/3); structured VTK output uses `.vti` ImageData, unstructured uses `.vtu` UnstructuredGrid (node counts 3/4/other → VTK types 5/9/7; field names XML-escaped) (`docs/core.md`).
- **Dual-representation regression discipline**: `makeStructuredMesh` + `tests/test_mesh_dual.cpp` lock geometry, operators, coupled momentum assembly, and end-to-end SIMPLE equivalence (`docs/numerical.md` §4.3a).
- **SU2 mesh input is strict and marker-ordered**: `io::readSu2` maps a 2D SU2 ASCII file onto `MeshInput` and delegates topology validation to `UnstructuredMesh`; marker file order is the patch order, and every off-contract variant is rejected (`docs/io.md`).
- **Assembly accumulates**: `assembleDiffusion`/`assembleConvection` add into `(A, b)` without zeroing; `assembleTransport` owns the full assembly and returns an unfinalized `EquationSystem` (`docs/numerical.md`).
- **Flux is first-class** (OpenFOAM `phi`): assemblers take a `FaceFluxField`, never a velocity field; build it with `interpolateCellVelocityFlux` or `computeMassFlux` (`docs/core.md`).
- **Source terms follow Patankar**: `S(phi) = Sc + Sp*phi` per unit volume, `Sp <= 0` enforced by exception (`docs/numerical.md`).
- **`SolverConfig::tolerance` is a relative residual** `|Ax-b|/|b|`; the BiCGSTAB wrapper scales Eigen's tolerance by `|b|` and checks convergence itself — never rely on `solver.info()` (`docs/math.md`).
- **SIMPLE (collocated + Rhie–Chow)**: Gauss-form pressure source `sum_f p_f S_f`; relaxed diagonal read from a finalized matrix copy; `d = V/a_P` interpolated to faces and projected on `S_f`; the persistent flux is overwritten with the pressure-free `phiHbyA` and corrected in place; per-face `C`/`Cb` are derived once, stored per face, and read back by the correction (`docs/numerical.md` §Pressure).
- **Pressure equation is assembled for the absolute pressure**: `A p = b`, `b = -sum_f outward(phiHbyA_f) + sum_Dirichlet C_b p_b` — a wrong RHS sign creates positive feedback and blows up (`docs/numerical.md`).
- **Boundary face fluxes are same-sourced as the interior**: boundary `phiHbyA` uses the pressure-free `uHat` (or the Dirichlet velocity value), `p_b` enters the RHS, and the pressure term is applied to the flux afterwards — never use `uStar` there (`docs/numerical.md`).
- **Pure-Neumann pressure uses reference-cell elimination**: drop cell 0 and pin `p_0 = p_old(0)` (keeps CG SPD); closed domains must pass `checkFluxCompatibility` at entry; `correctPressure` rebalances `phiHbyA` on every corrector (`docs/numerical.md`).
- **Time integration is an implicit theta scheme** (`TimeTerm{rho, dt, scheme}`; `dt <= 0` means steady); the ddt term lives inside the shared momentum operator, and `solveSimple` passes `TimeTerm{}` (`docs/numerical.md` §TimeScheme).
- **Transient momentum requires `relaxation == 1`**; the ddt diagonal `rho V/dt` is part of `a_P`, so `d = V/a_P` is transient-consistent (`docs/numerical.md`).
- **`correctPressure` reconstructs flux and velocity from the *unrelaxed* solved pressure** (`u = uHat - d grad(p_solved)`, physical p BCs): stored `p` may be relaxed while the velocity takes the full correction; there is no `cumulativeVelocityCorrection` flag and no ordering constraint (`docs/numerical.md` §修正与收敛判据).
- **SIMPLE/PISO reuse the same predictor/corrector, and PISO must refresh the Rhie–Chow data between correctors**: `refreshUHat(pred, velocity)` before every corrector after the first; `solvePiso` rejects `nCorrectors < 2` (`docs/numerical.md` §Piso).
- **Non-orthogonal/skewness corrections are on by default and are identically zero on Cartesian meshes**: diffusion cross-term `X = gamma*(k·G)` is RHS-only (matrix symmetry/SPD unchanged, no boundary correction); pressure uses `crossP = rho*(dUf*R.x*S.x + dVf*R.y*S.y)` with `nNonOrthogonalCorrectors+1` solves per corrector and only the final sub-iteration writing flux; skew correction applies only to Central convection and interior mass-flux interpolation; LSQ gradients supply correction gradients. `nonOrthSplit` always selects the interior `(alpha,k)` pair (`MinimumCorrection` by default, `OverRelaxed` optional); `nonOrthogonalDiffusion` gates only the explicit cross terms (diffusion `X` and pressure `crossP`). The two interior splits match exactly only when `G·d = Δφ` (LSQ-exact linear fields); otherwise their fixed points differ at O(h²). Boundaries stay over-relaxed in both modes because they have no explicit completion term. The Dirichlet-p boundary coefficient `C_b` is over-relaxed (`rho*(dU*Sx^2 + dV*Sy^2)/max(S.delta, 0.05|S||delta|)`), consistent with the diffusion boundary `beta_f`; on non-orthogonal meshes a single-step cross-check is only meaningful at converged corrector counts (split-dependent intermediate states, `docs/numerical.md` §斜交网格对照). On rotated-orthogonal meshes pressure `crossP` keeps an anisotropy-completion term whenever `dUf != dVf`, so never skip it based on `maxNonOrthogonality ≈ 0` (diffusion `X` and skew terms are exactly zero there, `docs/numerical.md` §压力方程). Drivers warn via `meshQualityWarning` (non-orthogonality > 70°, skewness `|s|/|d| > 1`) and reject negative `nNonOrthogonalCorrectors` (`docs/numerical.md` §4.3c).

## Next Phase (Phase 4)

Next: optionally add a Gmsh reader sharing the `MeshInput` mapping. Application tests for the example cases are agreed in principle but not scheduled.
Full roadmap: see README.md.
