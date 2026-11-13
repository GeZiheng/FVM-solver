# FVM-solver

这是一个面向学习的 2D FVM 求解器，支持 structured 与 unstructured polygon mesh，并通过统一的 `Mesh` 接口提供 dual mesh representations。数值 operator 都是 mesh-generic（接收 `const Mesh&`），底层使用 Eigen-backed sparse solvers。

求解器实现了 incompressible Navier–Stokes 的 steady SIMPLE 与 transient PISO，采用 collocated grid 和 Rhie–Chow interpolation；scalar transport 支持隐式 theta-scheme。对流输入统一使用持久化的 `FaceFluxField`（`phi`）。算例通过可选的 `pyfvm` Python bindings（pybind11、numpy-interop）以脚本方式编写；项目不再提供 C++ demo executable。

## 功能特性

- 2D FVM：支持 structured 或 unstructured polygon mesh；dual mesh representations 隐藏在同一个接口之后。
- mesh-generic numerical operators：核心数值装配代码只依赖抽象 `Mesh` 接口。
- Eigen-backed sparse solvers：提供 sparse matrix assembly 和多种线性求解器后端。
- incompressible Navier–Stokes：steady SIMPLE 与 transient PISO，基于 collocated grid 与 Rhie–Chow interpolation；斜交网格支持 non-orthogonal diffusion/pressure correction、可选择的内部 non-orthogonal split（minimum / over-relaxed）、skew-corrected interpolation 与 LSQ gradients。
- transient scalar transport：隐式 theta-scheme。
- flux-first 数据模型：持久化 `FaceFluxField`（`phi`）是 assembler 的唯一对流输入。
- Python 算例脚本：通过可选 `pyfvm` bindings 编写；没有 C++ demo executable。

## 项目结构

```text
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
  io/                  — readSu2 (.su2 mesh input) + writeVti/writeVtu (.vti/.vtu output)
  numerical/include/    — mesh-generic APIs (operators and drivers take const Mesh&)
    BoundaryCondition.h  — BCType{Dirichlet,Neumann} + mesh-backed BoundaryField
    GridOperators.h      — cellGradient (generalized Gauss), leastSquaresGradients
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
tests/                   — doctest: mesh, unstructured, su2_reader (SU2 format
                           contract), mesh_dual (Cartesian vs
                           makeStructuredMesh equivalence), field, flux, linalg, diffusion,
                           convection, transport, transient (time order), simple
                           (Poiseuille, cavity Re=100), piso (impulsive Couette,
                           transient Poiseuille), vtk_writer
python/                  — pyfvm bindings (optional, -DFVM_BUILD_PYTHON=ON): pyfvm.cpp (pybind11
                           layer, snake_case API), _path.py (sys.path bootstrap), examples/
                           (case scripts and shared SU2 generators: convection_diffusion.py,
                           cavity.py, impulsive_couette.py, mesh_su2.py,
                           backward_facing_step.py, cylinder.py),
                           tests/test_pyfvm.py (binding-contract tests = ctest entry pyfvm_tests),
                           rom/ (offline ROM pipeline: snapshot npz contract v1, POD, exact DMD;
                           numpy-only core, no pyfvm import),
                           overlay-ports/python3 (empty vcpkg port: link against system Python)
ports/                   — vcpkg overlay port for amgcl (header-only, locked 1.5.0 +
                           SHA512, depends only on eigen3; linked when FVM_BUILD_AMGCL=ON)
docs/                    — module docs (Chinese): core.md, math.md, io.md, numerical.md,
                           python.md, tests.md
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

pyfvm 测试与算例脚本运行时依赖 numpy：`python -m pip install -r python/requirements.txt`。

其中 application 算例精度回归可用标签单独运行：

```powershell
ctest --test-dir build/release-py -L application --output-on-failure
```

binding 的 vcpkg feature、overlay port、解释器版本锁定与产物定位机制见 [`docs/python.md`](docs/python.md)。

### AMGCL 线性求解后端（可选）

追加 `-DFVM_BUILD_AMGCL=ON`（默认 OFF）会启用 AMGCL 线性求解后端：新增 `createAmgclCG`（压力类 SPD）/ `createAmgclBiCGSTAB`（动量类非对称）两个工厂（编译定义 `FVM_HAVE_AMGCL`），依赖经仓库内 overlay port `ports/amgcl/`（`vcpkg-configuration.json`）接入；OFF 构建不执行任何 amgcl 接线，既有 CI 行为不变。Python 侧经 `SolverConfig` 两个正交字段选择：`backend`（`"eigen"` 默认 | `"amgcl"`）× `method`（`"bicgstab"` 默认 | `"cg"`）。

### 运行算例

```powershell
python python/examples/<name>.py
```

## 示例

基础算例与解析基准算例运行后 vti/vtu 写到 `python/examples/output/<算例名>/`，与当前工作目录无关；瞬态算例（Couette、Stokes、Taylor–Green、涡街）在非测试模式另输出多帧序列 `<stem>_0000.<ext>…`（ParaView 按文件序列打开即得流场演化动画），默认约每 50 步一帧、长算例自动拉伸封顶 ~50 帧（`--frames-every K`，0 关闭）：

- **convection_diffusion.py**：单位方域稳态对流-扩散。无散度回流速度场（流函数 ψ = sin(πx)·sin(πy) 派生），γ = 0.01（Pe ~ 100），西墙 T=1、东墙 T=0、南北绝热。演示 `solve_transport` 与 numpy 填场。
- **cavity.py**：顶盖驱动方腔流 `--re {100,400,1000}`（默认 100；Re≤400 用 64×64，Re=1000 用 128×128，SIMPLE + Central）；对照 Ghia et al. (1982) 中心线基准并报告 L∞/L2，`--test` 执行精度断言。输出 `cavity_re<Re>.vti` 可在 ParaView 看主涡与角隅二次涡。
- **impulsive_couette.py**：瞬态 Couette 启动（PISO，ν=0.1，dt=0.005 × 100 步到 t=0.5），与精确 Fourier 级数解对比打印误差；逐步调用模式下输出 3 帧序列（t=0/0.25/0.5）展示边界层成长。
- **poiseuille.py**：平面 Poiseuille 通道（L=10/H=1、Re=20、320×32，SIMPLE + Central；入口速度 Dirichlet、出口压力 Dirichlet 0），检验充分发展抛物线剖面与压降斜率。
- **stokes_oscillating.py**：Stokes 第二问题振荡平板（顶板速度 `cos(ωt)`，PISO Euler），在周期稳态后对照半无限域解析剖面；附带逐步调用与长调用等价性检查。非测试模式输出帧序列（默认每 50 步）。
- **piso_nonortho_channel.py**：4.3c 梯形收缩通道单步 PISO 算例，用于复现非正交/skewness 修正对照；`--no-corrections` 关闭修正，`--correctors N` 调整 corrector 数。

## 依赖

- `eigen3` — sparse linear algebra
- `doctest` — unit testing (header-only)
- `pybind11` — Python bindings (optional; only via the vcpkg `python` manifest feature when `FVM_BUILD_PYTHON=ON`)
- `amgcl` — AMG-preconditioned linear solver backend (optional; only linked when `FVM_BUILD_AMGCL=ON`, via the in-repo overlay port `ports/amgcl/` — version locked 1.5.0 + SHA512, depends only on `eigen3`)

## 路线图

### 已完成

Phase 1–3 为路线图之前的基础：Phase 1 项目骨架（`Mesh`/`Field` 数据定义、基于 Eigen 的线性方程组求解、VTK 输出后处理）；Phase 2 动量方程组装（对流/扩散/边界条件离散格式）；Phase 3 SIMPLE 循环与 OpenFOAM 风格的持久面通量。

- **4.1 非定常项 + PISO**：隐式 theta 时间格式（二阶精度已验证）、瞬态输运、共享 `Momentum`/`Pressure`/`GridOperators` 原语与 `solvePiso`；与 OpenFOAM-14 的对照与结论见 [`docs/numerical.md`](docs/numerical.md)。
- **4.2 `pyfvm` Python bindings**：pybind11 snake_case API + numpy 零拷贝视图；算例脚本在 `python/examples/`，契约测试为 `pyfvm_tests`；细节见 [`docs/python.md`](docs/python.md)。
- **4.3 任意网格输入与非正交修正（a–c）**：face-first mesh abstraction 双表示（`CartesianMesh`/`UnstructuredMesh`）、SU2 reader + `.vtu` 输出、非正交/偏斜修正 + LSQ 梯度 + 可选 split；斜交梯形 NS 的 OpenFOAM 对照在 NC=8 时 L2 `4.5e-4`（[`docs/numerical.md`](docs/numerical.md)）。Gmsh reader 搁置（SU2-first；重启时共享 `MeshInput` 映射层）。
- **4.4 Application tests（阶段 A–B）**：cavity Re=100/400/1000（Ghia）、平面 Poiseuille、Stokes 振荡平板、后向台阶 Re=50/100、圆柱 Re=20/40（SU2 → `UnstructuredMesh` → SIMPLE → `.vtu`）；ctest `-L application` 共 8 项。

当前测试基线（计数、按构建模式门控的细节）见 [`AGENTS.md`](AGENTS.md) Project State。

### Phase 5（2026-11-01 商定，分批交付中）

按 5.1 → 5.2 → 5.3 顺序执行。各批次交付日期与执行状态见 [`AGENTS.md`](AGENTS.md) Next Phase；✅ 条目只保留完成标记：

- ✅ **5.1 `BoundaryField` 逐面 Dirichlet 值（Option B）** — 按面 Dirichlet 值进入装配与压力边界系数，解锁空间变化 BC（Taylor–Green 精确解、抛物线入口）。先行——5.3 多数算例依赖它。设计要点见 [`docs/numerical.md`](docs/numerical.md) §BoundaryCondition。
- ✅ **5.2 AMGCL 线性求解后端** — `FVM_BUILD_AMGCL=ON` 默认关闭、既有 CI 不变。选型依据（vs Hypre）、overlay port 与工厂细节见 [`docs/math.md`](docs/math.md)。
- **5.3 更大规模/复杂算例**（按复杂度递增）：
  1. ✅ 方腔 Re=3200 @ 256×256（Ghia 对照；AMG vs BiCGSTAB-ILUT 性能 showcase——性能数据记录进 [`docs/numerical.md`](docs/numerical.md)，CI 只断言正确性）；
  2. ✅ 圆柱 Re=20/40 稳态改矩形域 `[-10,25]×[-10,10]`（O-4 多块生成器，`polar_ogrid` 退役；对 Fornberg / Dennis & Chang 带重校准）— 2026-11-10 完成；网格研究、文献锚偏移讨论与 **Re=20 核心加密稳定性边界**见 [`docs/numerical.md`](docs/numerical.md) 圆柱节；
  3. ✅ 后向台阶 Re=389 充分发展（抛物线）入口（Armaly 1983 上带）— 2026-11-10 完成；Armaly 约定 `μ=2/389` 对齐后实测 x_r/h=8.10（2D 文献簇 8.2–8.6 / 实验 7.9），n=16→24 漂移 0.09%；
  4. ✅ 斜网格抛物线入口 Poiseuille（等截面斜切通道、两端逐面抛物线速度 + 全 Neumann 解析压力）— 2026-11-03 随 5.1 提前完成；实测斜面 Dirichlet-p 的 `Cb` 无补全偏移，见 [`docs/numerical.md`](docs/numerical.md)；
  5. ✅ Taylor–Green 涡（瞬态 PISO、精确 BC、64²→128²→256² 网格收敛验证二阶精度）— 2026-11-10 完成；固定小 dt 下 64²→128² 实测空间阶 2.31；时间方向受 dt²/h 类混合项限制不单调（不断言），误差分解矩阵见 [`docs/numerical.md`](docs/numerical.md)；
  6. ✅ 圆柱 Re=100 卡门涡街（同矩形域生成器，瞬态 PISO + Central；断言 St（文献带 ≈0.164–0.167）、Cd_mean、Cl 振幅；ctest 跑缩短版，完整收敛研究由脚本参数承担）— 2026-11-10 完成；CI 缩短版实测 St=0.159 / Cd_mean=1.369 / Cl_amp=0.351 全落 2D 文献簇。

大算例（方腔 256²、涡街）标 `application` 但不进 `application_default`，保护默认 CI 时长。

Deferred/rejected：`examples/` per-case 可执行文件、JSON case 配置、按模块拆分 CMakeLists/tests/docs（条件同前）。meshgen MCP server 仍是条件触发候选（`mesh_su2` 成长为小型网格库且出现跨会话需求时再做薄封装）。修正限制器与边界显式补全按 [`docs/numerical.md`](docs/numerical.md) 登记的触发条件执行。

### Phase 6（2026-11-12 商定）：ROM

按 6.1 → 6.2 → 6.3 顺序执行（6.4 可选）。执行状态见 [`AGENTS.md`](AGENTS.md) Next Phase：

- ✅ **6.1 ROM 快照导出 + POD/DMD 离线管线** — `python/rom/` 包（`SnapshotCollector`/版本化 npz 快照契约 v1、均值中心化 POD、exact DMD、可选绘图；核心仅 numpy，不 import pyfvm）+ `test_rom.py` 7 用例（自动进 `pyfvm_tests`，TG 冒烟端到端）+ 探索驱动脚本 `taylor_green_rom.py`/`vortex_shedding_rom.py`（不注册 application 层）；C++ 内核与 CMake 零改动。DMD St 只诊断不断言；契约与算法细节见 [`docs/rom.md`](docs/rom.md)，脚本用法见 [`docs/python.md`](docs/python.md)。
- ✅ **6.2 pyfvm 绑定 A/b → 侵入式 Galerkin 离线投影** — `pyfvm.assemble_momentum`（稳态/瞬态装配以 COO 三元组返回，避免 scipy 硬依赖；C++ 内核零改动）；theta 分离恒等式 `A_tr = θ·A_sp + diag(ρV/dt)` 由契约测试锁定，设计与证据见 [`docs/rom.md`](docs/rom.md)（2026-11-13 完成）。
- **6.3 在线 ROM 积分（RK4）与对照 FOM 轨迹验证**。
- **6.4（可选/暂缓）**：压力重构、Re 参数化 ROM。
- **条件触发候选**（Phase 6 后续工作，出现稳定跨会话需求时再立项）：**meshgen MCP**（`mesh_su2` 薄封装，条件同上文 Phase 5 登记项）；**ROM 分析 MCP**（`python/rom/` 的 POD/DMD/快照分析工具封装；含长运行的 `generate_snapshots` 类工具须按 `run_case` 的后台运行/审批语义设计；6.2 开工时一并评估）。

## Agent 开发环境（harness）

本仓库的 agent harness 由三个项目级 MCP server 组成：`build-and-test`、`build-and-test-wsl` 与 `openfoam`。三者都运行在 agent 沙箱之外，这是 WSL 与 vcpkg 操作无需逐次提权就能工作的原因。`build-and-test-wsl` 用 WSL 的 `python3` 启动**同一份** `build_and_test_server.py`（`wsl.exe -e bash -lc "exec python3 .agents/mcp/build_and_test_server.py"`），用于在本地复现 Linux 构建路径（如 CI 才会暴露的 `-fPIC` 链接问题）。使用时机、工具参数与比较纪律见 [`AGENTS.md`](AGENTS.md) 的 MCP Servers 小节。

harness 的单一出处分为三层：

- 委派策略（routing table、skip threshold、parent-owned responsibilities）：`AGENTS.md` 的 Subagent Policy (Codex & ZCode)。
- Agent runtime 行为（每个子 agent 的自我简报）：`.codex/agents/*.toml` 与其 ZCode 移植 `.zcode/agents/*.md`（同一简报；ZCode 版用 frontmatter 的 `tools` 白名单替代 Codex 的 sandbox_mode，只读 agent 因此硬性拿不到 MCP 工具）。
- 工作流阶段映射（哪个重构阶段委派给哪个 agent）：`.agents/skills/refactor-code/SKILL.md`；它只引用策略、不复述。

同一条规则出现在第二层即是缺陷：应更新 owning 层并让引用处指向它，而不是维护两份事实。

### 文件图

```text
.agents/mcp/
  build_and_test_server.py   — build-and-test MCP server（CMake/Ninja/ctest 封装；Windows 与 WSL 双注册）
  openfoam_server.py         — openfoam MCP server（WSL OpenFOAM-14 驱动）
  openfoam_fields.py         — 纯 ascii 解析 + 字段重映射（无 WSL、无 subprocess、无 I/O）
  openfoam/                  — make_of_case.sh、of_log_summary.py 等辅助脚本；tests/ 计划中
.agents/skills/refactor-code/SKILL.md — 四阶段代码改动工作流（中文）
.codex/config.toml          — MCP server 注册 + 审批模式
.codex/agents/*.toml        — explorer / implementer / reviewer / test-triage / docs（glm-5.3-flash）
.zcode/config.json          — ZCode：同一组 MCP server（`mcp.servers`）
.zcode/agents/*.md          — 上述五个子 agent 的 ZCode 版（GLM-5.3-Flash；只读 agent 带 tools 白名单）
opencode.json               — OpenCode：skill 路径、同一组 MCP server、V2 permissions 数组（shell deny + run_case 审批）
```

三个 agent（Codex、OpenCode、ZCode）注册并访问同一份 server 实现；skills 三者共用 `.agents/skills/`（ZCode 原生扫描该目录）。

### 环境与配置要点

- `VCPKG_ROOT` 与 `Path` 中的 `ninja.exe` 必须在 agent 会话启动前就绪；前提与构建命令见上文“构建与测试”。
- `build-and-test-wsl` 的前提在 WSL 默认发行版内：`python3`、`cmake`、`ninja`、C++ 编译器与 Linux vcpkg（env `VCPKG_ROOT`，须在 `~/.profile` 导出——`bash -lc` 看不到 `~/.bashrc` 交互守卫后的 export）；仓库须位于 Windows 盘符下（`wsl.exe` 把 cwd 翻译为 `/mnt/<drive>/...`）。其构建目录为 `build/<config>-agent[-py]-wsl/`（与 Windows 实例隔离，异步 marker 带 `host` 字段互不误回收；无 `host` 字段的旧格式 marker 视为异宿主、不回收）；可选 env `FVM_BUILD_ROOT` 把构建根移到 WSL ext4（`/mnt/<drive>` 上构建偏慢），此时 `log_path` 为绝对路径。
- `FVM_OF_ALLOWED_ROOTS` 是冒号分隔的绝对 WSL 路径，用于扩大 `run_case` 允许的 `template`/`root` 根；候选路径在 WSL 内经 `realpath` 规范化，符号链接与 `..` 无法逃逸。该检查不是 TOCTOU-free 的；威胁模型是混淆的 agent，不是本地竞争者。
- 工具副作用：`build_and_test` 写 `build/<config>-agent[-py]/`（WSL 实例写 `build/<config>-agent[-py]-wsl/`，或 `FVM_BUILD_ROOT` 下的同名目录；异步模式另写 `ctest_async.log` 与 `mcp_ctest.json`，无 `host` 字段的旧格式 marker 视为异宿主、不回收），`run_case` 在 WSL 中创建 case 目录并启动 `foamRun`，其余工具全部只读。ascii-only 限制见 [`AGENTS.md`](AGENTS.md) 的 MCP Servers 小节。
- `build_and_test` 返回结构化 JSON：成功含 `ok`/`config`/`build_dir`/`stage`/`tests`/`report`；失败附 `first_error`（按 CMake > ninja `FAILED:` > MSVC `error C<n>` > `error:` > ctest 的优先级取首行）、截断的 `tail` 与 `log_path`——完整未截断 transcript 只写入构建目录的 `build_and_test.log`。`run=application/all` 默认后台执行（`wait=false`，规避客户端 ~300s 工具调用超时）：立即返回 `stage=test-async`，随后轮询 `test_status` 直到 `finished`/`timeout`/`stale`，终态字段与同步路径同构。`run=application` 默认跑 `application_default` 标签子集（5 条，成员在 `CMakeLists.txt`，与 CI 默认一致）；`full=true` 跑整层，或用 `tests` 正则传给 `ctest -R`（子串匹配需 `$` 锚定；空匹配经 `--no-tests=error` 显式失败；与 `full` 互斥）。
- `build_and_test` 另有可选参数 `cmake_args`（字符串，默认空）：仅接受 `-D<NAME>=<VALUE>` 形式的 token（空白分隔、逐个校验），合法则追加到 cmake configure 命令末尾，非法 token 直接报错——ON 模式选项（如 `-DFVM_BUILD_AMGCL=ON`）的验证通道。
- `run_status` 的状态为 `running` / `finished` / `failed` / `stale` / `unknown`；`stale` 表示启动器进程已死且没有 `.exitcode`（例如 server 重启），需要重跑 case。启动器 PID 记录在 case 目录的 `.launcher_pid`（`kill -0` + `/proc/<pid>/cmdline` 含 case 路径双重校验，防 PID 复用）。
- 审批模式记录在 `.codex/config.toml`：当前仅 `run_case` 需要审批；`build_and_test`（含 WSL 实例）免批（副作用仅限仓库内 `build/`、`FVM_BUILD_ROOT` 覆盖指定的构建根与各自 vcpkg 缓存，且持久 implementer 工作流依赖无人工介入的中间编译检查），`openfoam` 的只读工具默认免批；三个只读子 agent（explorer/reviewer/test-triage）的 toml 另有"禁调 `build_and_test`/`run_case`"的指令。OpenCode 侧在 `opencode.json` 对齐同一契约：V2 `permissions` 数组中 `openfoam_run_case` 为 `ask`、其余 openfoam 工具走默认放行，shell 规则 deny 一切 `cmake`/`ctest`/`ninja`/`fvm_tests` 直调（强制走 `build-and-test` MCP 工具）。ZCode 侧在 `.zcode/config.json` 注册同一组 server 并自动连接，但没有 per-tool 审批与 shell deny 机制：`run_case` 需确认改由 AGENTS.md 的行为约定承担（ZCode 会话中调用 `run_case` 前先征得用户同意），shell 直调 cmake/ctest 的禁令同样以 AGENTS.md 指令为准。
- openfoam 字段映射假定结构网格：`nx*ny` 必须等于网格单元数，`grid[j][i] = cell j*nx+i`，面通量从 OpenFOAM 的 owner→neighbour 约定转换到本项目的 `+x/+y` 约定。

### 手工验证

harness 自身还没有自动化测试（计划覆盖字典解析器、`extract_fields` 重映射、`of_log_summary.py` fixture，以及一个 CI job）。当前手工验证：

```powershell
python -m py_compile .agents\mcp\build_and_test_server.py .agents\mcp\openfoam_server.py .agents\mcp\openfoam_fields.py
# 纯函数冒烟：直接导入模块，不需要 WSL
python -c "import sys; sys.path.insert(0, r'.agents/mcp'); import openfoam_fields, openfoam_server; print('import ok')"
```

marker 宿主隔离自检（伪造异宿主 running marker，确认本宿主忽略）：

```powershell
python -c "import sys, json; sys.path.insert(0, r'.agents/mcp'); import build_and_test_server as b; d=b.REPO_ROOT/'build'/'probe-agent-wsl'; d.mkdir(parents=True, exist_ok=True); (d/'mcp_ctest.json').write_text(json.dumps({'state':'running','host':'wsl'}), encoding='utf-8'); print('foreign marker ignored:', b._find_running_marker() is None)"
```

build-and-test 的分层与异步路径（`run=unit` 同步、`run=application` 后台 + `test_status` 轮询、并发启动拒绝、未轮询句柄回收、`stale` 检测）通过 stdio JSON-RPC 驱动脚本手工验证，脚本不入库（位于 gitignored 的 `build/` 下）。

真实数据回归：对一个已有 case（如 16x16 的 `mcptest_*`）以 `nx=ny=16` 运行 `extract_fields`，要求零警告。

### 扩展规则

- 新的 MCP 工具与其 schema 一起加进所属 server；可复用的 OpenFOAM 文本解析放进 `openfoam_fields.py`，且必须保持无 WSL/subprocess/I/O，以维持可单测性。
- server 保持纯标准库 Python、stdio 上的换行分隔 JSON-RPC；`.codex/config.toml`、`opencode.json` 与 `.zcode/config.json` 注册的入口（`python .agents/mcp/<server>.py`）必须始终可用。
- harness 源码仅 ASCII；面向 agent 的文档跟随所在文件的语言（中文以外的术语保留英文）。

### 已知局限 / 计划项

- 只读子 agent 的本地命令受沙箱约束，但 MCP 工具在沙箱外运行；Codex 侧已通过 toml 指令禁调 `build_and_test`/`run_case`（软约束），硬性的 per-agent 工具白名单待 Codex 机制验证后再引入。ZCode 侧已通过 `.zcode/agents/*.md` 的 `tools: [Read, Bash]` 白名单硬性裁剪（MCP 工具与写工具均不可见），Bash 只读语义仍靠 agent 指令约束。
- `build-and-test-wsl` 首次 vcpkg 引导在慢网络下可能超过 configure 的 1800s 超时（实测：vcpkg 自带的 Linux cmake ~62MB 直连 ~8KB/s 即触发）。该引导是一次性的，且文件可预置：在 Windows 侧（有代理）下载同名文件放入 `~/vcpkg/downloads/` 即被 vcpkg 按 SHA 校验直接采用（本次的 cmake/doctest/eigen/pybind11/patchelf 均以此方式完成）；`VCPKG_DOWNLOADS` 亦可指向共享缓存。已缓存后 WSL 构建与 Windows 同量级。

## 文档索引

- [`docs/core.md`](docs/core.md) — core 模块：基础类型、mesh abstraction 与 field/flux 表示。
- [`docs/math.md`](docs/math.md) — math 模块：sparse matrix assembly 与 linear solver backends。
- [`docs/io.md`](docs/io.md) — io 模块：VTK `.vti`/`.vtu` 输出。
- [`docs/numerical.md`](docs/numerical.md) — numerical 模块：算子、SIMPLE/PISO 设计与 OpenFOAM cross-check 结论。
- [`docs/python.md`](docs/python.md) — `pyfvm` bindings：机制、API、约定与测试分工。
- [`docs/tests.md`](docs/tests.md) — 测试层：ctest 三层架构、注册点、分层验证粒度与已知坑。
- [`docs/rom.md`](docs/rom.md) — ROM：快照契约 v1、POD/exact DMD 算法、已验证结论与 Phase 6 后续批次。
- 代码约定与 agent 工作流见 [`AGENTS.md`](AGENTS.md)。
