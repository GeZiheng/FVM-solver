# FVM-solver

这是一个面向学习的 2D FVM 求解器，支持 structured 与 unstructured polygon mesh，并通过统一的 `Mesh` 接口提供 dual mesh representations。数值 operator 都是 mesh-generic（接收 `const Mesh&`），底层使用 Eigen-backed sparse solvers。

求解器实现了 incompressible Navier–Stokes 的 steady SIMPLE 与 transient PISO，采用 collocated grid 和 Rhie–Chow interpolation；scalar transport 支持隐式 theta-scheme。对流输入统一使用持久化的 `FaceFluxField`（`phi`）。算例通过可选的 `pyfvm` Python bindings（pybind11、numpy-interop）以脚本方式编写；项目不再提供 C++ demo executable。

## 功能特性

- 2D FVM：支持 structured 或 unstructured polygon mesh；dual mesh representations 隐藏在同一个接口之后。
- mesh-generic numerical operators：核心数值装配代码只依赖抽象 `Mesh` 接口。
- Eigen-backed sparse solvers：提供 sparse matrix assembly 和多种线性求解器后端。
- incompressible Navier–Stokes：steady SIMPLE 与 transient PISO，基于 collocated grid 与 Rhie–Chow interpolation。
- transient scalar transport：隐式 theta-scheme。
- flux-first 数据模型：持久化 `FaceFluxField`（`phi`）是 assembler 的唯一对流输入。
- Python 算例脚本：通过可选 `pyfvm` bindings 编写；没有 C++ demo executable。

## 项目结构

```
src/
  core/                — Types (Scalar/Index/Vector/Point2/IndexSpan)
    Mesh.h               — abstract face-first Mesh base (unified face ids, patches)
    CartesianMesh.h/.cpp — uniform 2D implementation + structured convenience API
    UnstructuredMesh.h/.cpp — polygon mesh from MeshInput + makeStructuredMesh
    MeshGeomCache.h/.cpp — cached interpolation/diffusion/non-orthogonality coefficients
    Field.h/FluxField.h  — mesh-generic ScalarField/VectorField (cell-centered) and
                           FaceFluxField (single per-face phi array, header-only)
  math/                — SparseMatrix (triplet assembly over Eigen), LinearSolver (abstract +
                         Eigen CG / BiCGSTAB / SparseLU factories)
  io/                  — VtkWriter (.vti ImageData output)
  numerical/include/    — mesh-generic APIs (operators and drivers take const Mesh&)
    BoundaryCondition.h  — BCType{Dirichlet,Neumann} + mesh-backed BoundaryField
    GridOperators.h      — cellGradient (generalized Gauss)
    Diffusion.h          — -div(gamma grad phi) assembly from MeshGeomCache
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
tests/                   — doctest: mesh, unstructured, mesh_dual (Cartesian vs
                           makeStructuredMesh equivalence), field, flux, linalg, diffusion,
                           convection, transport, transient (time order), simple
                           (Poiseuille, cavity Re=100), piso (impulsive Couette,
                           transient Poiseuille)
python/                  — pyfvm bindings (optional, -DFVM_BUILD_PYTHON=ON): pyfvm.cpp (pybind11
                           layer, snake_case API), _path.py (sys.path bootstrap), examples/
                           (case scripts: convection_diffusion.py, cavity.py, impulsive_couette.py),
                           tests/test_pyfvm.py (binding-contract tests = ctest entry pyfvm_tests),
                           overlay-ports/python3 (empty vcpkg port: link against system Python)
docs/                    — module docs (Chinese): core.md, math.md, io.md, numerical.md, python.md
```

## 构建与测试

### 前提

- CMake >= 3.20
- vcpkg
- Ninja

需要在环境中准备好：

- `VCPKG_ROOT` — vcpkg 安装目录。
- `Path` — 必须包含 `ninja.exe` 所在目录。

这两项应在启动 agent 之前配置为持久 Windows 环境变量；已经运行的 agent 不会感知其后的变更。

### C++ 构建与测试

```powershell
cmake -B build -S . -G Ninja "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

### Python bindings（可选）

启用 `-DFVM_BUILD_PYTHON=ON` 后会添加 `pyfvm` module 与 `pyfvm_tests` ctest entry：

```powershell
cmake -B build/release-py -S . -G Ninja "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release -DFVM_BUILD_PYTHON=ON
cmake --build build/release-py
ctest --test-dir build/release-py --output-on-failure
```

binding 的 vcpkg feature、overlay port、解释器版本锁定与产物定位机制见 [`docs/python.md`](docs/python.md)。

### 运行算例

```powershell
python python/examples/<name>.py
```

## 示例

三个算例是原 C++ demo 与 PISO 测试的 Python 版，运行后 vti 写到 `python/examples/output/<算例名>/`，与当前工作目录无关：

- **convection_diffusion.py**：单位方域稳态对流-扩散。无散度回流速度场（流函数 ψ = sin(πx)·sin(πy) 派生），γ = 0.01（Pe ~ 100），西墙 T=1、东墙 T=0、南北绝热。演示 `solve_transport` 与 numpy 填场。
- **cavity.py**：顶盖驱动方腔流 Re=100（64×64，SIMPLE，relaxation_u=0.7 / relaxation_p=0.3），约 1000 次外迭代收敛；输出 `cavity.vti` 可在 ParaView 看主涡与角隅二次涡。
- **impulsive_couette.py**：瞬态 Couette 启动（PISO，ν=0.1，dt=0.005 × 100 步到 t=0.5），与精确 Fourier 级数解对比打印误差。

## 依赖

- `eigen3` — sparse linear algebra
- `doctest` — unit testing (header-only)
- `pybind11` — Python bindings (optional; only via the vcpkg `python` manifest feature when `FVM_BUILD_PYTHON=ON`)

## 路线图

已商定的路线图，按顺序：

1. ✅ **非定常项 + PISO** — 已完成：瞬态 theta scheme（时间精度已验证）、瞬态输运、共享的 `Momentum`/`Pressure`/`GridOperators` 原语与 `solvePiso`；`tests/test_piso.cpp` 已启用。交叉验证数值与已排除假设清单见 [`docs/numerical.md`](docs/numerical.md)。
2. ✅ **`pyfvm` Python bindings** — 已完成：pybind11（vcpkg `python` feature、`FVM_BUILD_PYTHON=ON`、可选），覆盖 mesh/fields/BC/SIMPLE/PISO/transport/vti 的 snake_case API 与 numpy 零拷贝视图；算例脚本在 `python/examples/`；绑定契约测试在 `python/tests/`（ctest 条目 `pyfvm_tests`）；C++ `fvm_solver` demo 可执行文件已移除——算例一律为 Python 脚本。细节见 [`docs/python.md`](docs/python.md)。
3. **任意网格输入** — 4.3a（mesh abstraction、双网格表示、零数值变化）已完成。下一步 4.3b：Gmsh `.msh` reader + `.vtu` writer + `pyfvm.read_gmsh`；随后 4.3c：non-orthogonal correction + validation。
4. **示例算例的 application tests** — 把 `python/examples/` 脚本作为独立测试层运行（与 unit tests 分开），例如注册为带标签的 ctest 条目。已原则同意，未排期。

Deferred/rejected：`examples/` 下的 per-case 可执行文件（繁琐）、JSON case 配置（与 Python 脚本重复）、按模块拆分 CMakeLists/tests/docs（仅当模块被外部复用或构建时间恶化时再议）。AMGCL/Hypre 后端仍是更大网格的候选方案。

## Agent 开发环境（harness）

本仓库的 agent harness 由两个项目级 MCP server 组成：`build-and-test` 和 `openfoam`。两者都运行在 agent 沙箱之外，这是 WSL 与 vcpkg 操作无需逐次提权就能工作的原因。使用时机、工具参数与比较纪律见 [`AGENTS.md`](AGENTS.md) 的 MCP Servers 小节。

harness 的单一出处分为三层：

- 委派策略（routing table、skip threshold、parent-owned responsibilities）：`AGENTS.md` 的 Codex Subagent Policy。
- Agent runtime 行为（每个子 agent 的自我简报）：`.codex/agents/*.toml`。
- 工作流阶段映射（哪个重构阶段委派给哪个 agent）：`.agents/skills/refactor-code/SKILL.md`；它只引用策略、不复述。

同一条规则出现在第二层即是缺陷：应更新 owning 层并让引用处指向它，而不是维护两份事实。

### 文件图

```
.agents/mcp/
  build_and_test_server.py   — build-and-test MCP server（CMake/Ninja/ctest 封装）
  openfoam_server.py         — openfoam MCP server（WSL OpenFOAM-14 驱动）
  openfoam_fields.py         — 纯 ascii 解析 + 字段重映射（无 WSL、无 subprocess、无 I/O）
  openfoam/                  — make_of_case.sh、of_log_summary.py 等辅助脚本；tests/ 计划中
.agents/skills/refactor-code/SKILL.md — 四阶段代码改动工作流（中文）
.codex/config.toml          — MCP server 注册 + 审批模式
.codex/agents/*.toml        — explorer / implementer / reviewer / test-triage / docs（glm-5.3-flash）
opencode.json               — OpenCode：skill 路径、同一组 MCP server、bash 权限块
```

两个 agent 注册并访问同一份 server 实现。

### 环境与配置要点

- `VCPKG_ROOT` 与 `Path` 中的 `ninja.exe` 必须在 agent 会话启动前就绪；前提与构建命令见上文“构建与测试”。
- `FVM_OF_ALLOWED_ROOTS` 是冒号分隔的绝对 WSL 路径，用于扩大 `run_case` 允许的 `template`/`root` 根；候选路径在 WSL 内经 `realpath` 规范化，符号链接与 `..` 无法逃逸。该检查不是 TOCTOU-free 的；威胁模型是混淆的 agent，不是本地竞争者。
- 工具副作用：`build_and_test` 写 `build/<config>-agent[-py]/`，`run_case` 在 WSL 中创建 case 目录并启动 `foamRun`，其余工具全部只读。ascii-only 限制见 [`AGENTS.md`](AGENTS.md) 的 MCP Servers 小节。
- `build_and_test` 返回结构化 JSON：成功含 `ok`/`config`/`build_dir`/`stage`/`tests`/`report`；失败附 `first_error`（按 CMake > ninja `FAILED:` > MSVC `error C<n>` > `error:` > ctest 的优先级取首行）、截断的 `tail` 与 `log_path`——完整未截断 transcript 只写入构建目录的 `build_and_test.log`。
- `run_status` 的状态为 `running` / `finished` / `failed` / `stale` / `unknown`；`stale` 表示启动器进程已死且没有 `.exitcode`（例如 server 重启），需要重跑 case。启动器 PID 记录在 case 目录的 `.launcher_pid`（`kill -0` + `/proc/<pid>/cmdline` 含 case 路径双重校验，防 PID 复用）。
- 审批模式记录在 `.codex/config.toml`：当前仅 `run_case` 与 `build_and_test` 需要审批，`openfoam` 的只读工具默认免批；三个只读子 agent（explorer/reviewer/test-triage）的 toml 另有"禁调 `build_and_test`/`run_case`"的指令。
- openfoam 字段映射假定结构网格：`nx*ny` 必须等于网格单元数，`grid[j][i] = cell j*nx+i`，面通量从 OpenFOAM 的 owner→neighbour 约定转换到本项目的 `+x/+y` 约定。

### 手工验证

harness 自身还没有自动化测试（计划覆盖字典解析器、`extract_fields` 重映射、`of_log_summary.py` fixture，以及一个 CI job）。当前手工验证：

```powershell
python -m py_compile .agents\mcp\openfoam_server.py .agents\mcp\openfoam_fields.py
# 纯函数冒烟：直接导入模块，不需要 WSL
python -c "import sys; sys.path.insert(0, r'.agents/mcp'); import openfoam_fields, openfoam_server; print('import ok')"
```

真实数据回归：对一个已有 case（如 16x16 的 `mcptest_*`）以 `nx=ny=16` 运行 `extract_fields`，要求零警告。

### 扩展规则

- 新的 MCP 工具与其 schema 一起加进所属 server；可复用的 OpenFOAM 文本解析放进 `openfoam_fields.py`，且必须保持无 WSL/subprocess/I/O，以维持可单测性。
- server 保持纯标准库 Python、stdio 上的换行分隔 JSON-RPC；`.codex/config.toml` 与 `opencode.json` 注册的入口（`python .agents/mcp/<server>.py`）必须始终可用。
- harness 源码仅 ASCII；面向 agent 的文档跟随所在文件的语言（中文以外的术语保留英文）。

### 已知局限 / 计划项

- 只读子 agent 的本地命令受沙箱约束，但 MCP 工具在沙箱外运行；已通过 toml 指令禁调 `build_and_test`/`run_case`（软约束），硬性的 per-agent 工具白名单待 Codex 机制验证后再引入。

## 文档索引

- [`docs/core.md`](docs/core.md) — core 模块：基础类型、mesh abstraction 与 field/flux 表示。
- [`docs/math.md`](docs/math.md) — math 模块：sparse matrix assembly 与 linear solver backends。
- [`docs/io.md`](docs/io.md) — io 模块：VTK `.vti` 输出。
- [`docs/numerical.md`](docs/numerical.md) — numerical 模块：算子、SIMPLE/PISO 设计与 OpenFOAM cross-check 结论。
- [`docs/python.md`](docs/python.md) — `pyfvm` bindings：机制、API、约定与测试分工。
- 代码约定与 agent 工作流见 [`AGENTS.md`](AGENTS.md)。
