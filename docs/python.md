# python 模块说明（pyfvm 绑定）

`python/` 提供 pybind11 绑定模块 `pyfvm`，让具体算例用 Python 脚本编写：建网格、读 SU2 网格、numpy 填场、设边界、调 SIMPLE/PISO/输运求解、读回 numpy 后处理、写 VTK（vti/vtu）。绑定层只做转发，不含任何数值逻辑；`fvm_core` 的 C++ API 不因绑定而改动。

## 目录结构

```
python/
  pyfvm.cpp              — 全部绑定（单文件；Python 侧命名 snake_case）
  _path.py               — sys.path 引导：按当前解释器标签扫描 build/*/python 定位 pyfvm 扩展
  examples/              — 算例脚本 + 共享 helper _common.py（见下）
  rom/                   — ROM 离线管线（SnapshotCollector/npz 契约 v1、POD、exact DMD、
                           可选绘图 rom/plot.py；核心仅 numpy，不 import pyfvm；
                           契约细节与算法见 docs/rom.md）
  tests/test_pyfvm.py    — 绑定契约测试（unittest；ctest 条目 pyfvm_tests）
  tests/test_rom.py      — ROM 管线测试（unittest；随 pyfvm_tests 的 discover 自动收编）
  overlay-ports/python3/ — 空 vcpkg overlay 端口（机制见下）
```

## 构建

默认关闭，需要用 `-DFVM_BUILD_PYTHON=ON` 显式开启；完整的 configure/build/CTest 命令以 README.md 的“构建与测试”为 canonical 出处。

测试与算例脚本的 Python 运行时依赖（目前仅 numpy）声明在 `python/requirements.txt`，CI 与本地共用该文件。

机制要点：

- **vcpkg feature**：`pybind11` 声明在 `vcpkg.json` 的 `"python"` feature 里；`-DFVM_BUILD_PYTHON=ON` 会在 `project()` 之前自动设置 `VCPKG_MANIFEST_FEATURES=python`，默认构建不拉取 pybind11。
- **空 overlay 端口**：pybind11 端口声明依赖 vcpkg 的 `python3` 端口（会源码构建 CPython 3.12，但我们用不上）。`python/overlay-ports/python3` 是一个空端口，满足依赖图但不装任何东西；扩展模块始终对着**系统 Python** 编译。
- **解释器选择**：扩展模块与 Python 版本绑定（cp314 的 `.pyd` 不能在 3.12 上 import）。configure 优先用 `-DPython_EXECUTABLE=...`，否则取 PATH 上第一个 `python3.14`/`python3`/`python`。pybind11 ≥ 3.0 才支持 Python 3.14（当前 vcpkg 端口 3.0.1）。
- **产物**：`build/<dir>/python/pyfvm*.pyd`（Linux 为 `.so`）；`_path.py` 优先采用环境变量 `FVM_PYFVM_DIR`，该目录必须已含当前解释器标签的扩展，否则抛 `RuntimeError`；未设置时扫描 `build/*/python`，只接受当前平台的原生扩展后缀（Windows `.pyd`、Linux `.so`，另一平台产物永不参选——否则共享 build 树时 WSL 的 `.so` 会在 Windows 上遮蔽 `.pyd`）且匹配当前解释器标签（Windows `cp3XX`、Linux `cpython-3XX`）的扩展，并在多个匹配目录间取其中文件 mtime 最新者。自动定位失败时抛 `RuntimeError`，错误信息列出解释器版本、标签模式和已扫描目录。绑定测试直接 `import _path` 后即可 `import pyfvm`；算例脚本经 `examples/_common.py` 引导，由它内部完成 `import _path`。多构建树并存时手工复跑测试须钉定该环境变量，见 `docs/tests.md` 已知坑。

## API 一览

命名约定：Python 侧全部 snake_case（如 `maxIterations` → `max_iterations`）。

core：

- `Mesh` 是多态基类绑定：`cell_count()` / `face_count()` / `face_owner(face)` / `face_neighbour(face)` / `is_boundary_face(face)` / `cell_faces(cell)` / `cell_volume(cell)` / `cell_centroid(cell)` / `face_centroid(face)` / `face_area_vector(face)` / `patch_count()` / `patch_name(patch)` / `patch_faces(patch)` / `face_patch(face)`。`face_neighbour()` 在边界面上返回无效单元哨兵 `SIZE_MAX`（即 C++ `Mesh::kInvalidCell` 的值），不要当作单元索引用。
- `CartesianMesh(nx, ny, x_min, y_min, x_max, y_max)` 继承 `Mesh`，保留 `nx()` / `ny()` / `cell_count()` / `dx()` / `dy()` / `cell_volume(c)` / `cell_center(c)` / `cell_index(i, j)` / `cell_ij(c)`，并新增 `x_face(i, j)` / `y_face(i, j)` 返回全局面编号。
- `UnstructuredMesh` 继承 `Mesh`，没有 Python 构造器，由 `read_su2(path)` 获得；方法为 `point_count()`、`point(i)`（返回 `(x, y)`）、`cell_node_count(c)` 与 `cell_node(c, local)`。
- 自定义 `Point2` type caster 使 `cell_centroid` / `face_centroid` / `face_area_vector` 返回 `(x, y)` tuple。
- `ScalarField(mesh, name="")`、`VectorField(mesh, name="")`（`.u` / `.v` 返回 `ScalarField`；`size()` 返回单元数，`set_constant(v)` 同时设置两个分量）、`FaceFluxField(mesh, name="")`。
- `field.to_numpy()`：**零拷贝可写一维视图**，按全局单元编号存储；笛卡尔网格仍为 `j*nx+i`，可 `reshape(ny, nx)`。二维 `(i,j)` 读写接口已移除，用 `field.to_numpy()[mesh.cell_index(i,j)]`。
- `flux.face_count()` / `flux.phi(face)` / `flux.set_phi(face, value)` / `flux.to_numpy()` / `flux.set_zero()` / `flux.cell_imbalance(cell)`：通量是单一按全局面编号的数组（零拷贝视图为 `to_numpy()`）。内部面 `phi>0` 表示 owner → neighbour，边界面为 owner 外向通量；`x()/y()/to_numpy_x()/to_numpy_y()` 已移除。

numerical：

- 边界：`BoundaryField(mesh)` + `.set(patch, type, value)` / `.get(patch)` / `.patch_count()`；patch 常量 `BoundaryField.East/North/West/South` 只对应笛卡尔 patch 顺序，越界 patch 抛 `IndexError`；类型 `BCType.Dirichlet/Neumann`。默认全部 patch 零梯度 Neumann。逐面值用 `.set_values(patch, type, values)` 设置：`values` 为 1-D numpy 数组，长度必须等于该 patch 面数，顺序遵循 `mesh.patch_faces(patch)` 遍历序；非 1-D 或长度不符抛 `ValueError`，patch 越界抛 `IndexError`。`.value_at(face)` 优先返回逐面值，否则回退 patch 常数。逐面值的含义与 C++ 侧一致：Dirichlet 为面值，Neumann 为法向导数。
- 枚举：`ConvectionScheme.Upwind/Central`、`TimeScheme.Euler/CrankNicolson`、`NonOrthSplit.MinimumCorrection/OverRelaxed`。
- `SolverConfig`：`tolerance`（相对残差）/ `max_iterations` / `verbose` / `backend`（`"eigen"` 默认 | `"amgcl"`，后者需 `FVM_BUILD_AMGCL=ON` 构建，用 `pyfvm.HAVE_AMGCL` 布尔探测）/ `method`（`"bicgstab"` 默认 | `"cg"` | `"bicgstab_ilut"`，最后一个仅 eigen 族支持）。未知组合抛 `ValueError`；OFF 构建选 amgcl 抛 `RuntimeError`（与方法无关）。
- `CorrectionOptions`：`non_orthogonal_diffusion=True`、`skew_interpolation=True`、`n_non_orthogonal_correctors=0`、`non_orth_split=NonOrthSplit.MinimumCorrection`。笛卡尔网格上修正量恒为零，默认行为保持零数值变化（旋转正交网格上压力 `crossP` 仍含各向异性补全项，见 `docs/numerical.md` 压力方程一节）。`n_non_orthogonal_correctors` 是压力求解子循环附加数（0 表示一次压力求解），负值会被拒绝。`non_orth_split` 永远控制内部面 $(\alpha,k)$；`non_orthogonal_diffusion` 只门控显式交叉项，因此“关闭修正 + OverRelaxed”表示 over 隐式、无显式修正。示例：`corrections.non_orth_split = NonOrthSplit.OverRelaxed`。
- `solve_simple(mesh, rho, mu, bc_u, bc_v, bc_p, config, velocity, pressure, flux)` → `SimpleResult`（`converged` / `iterations` / `history`）；`SimpleConfig`：`max_iterations` / `tolerance` / `relaxation_u` / `relaxation_p` / `scheme` / `solver_config`（动量方程线性求解器）/ `pressure_solver_config`（压力方程线性求解器，默认 eigen×cg）/ `corrections` / `verbose`。**行为变化（5.3-1）**：两个 solver config 完全独立，压力**不再继承** `solver_config` 的容差/迭代数——依赖旧行为的脚本必须显式设置 `pressure_solver_config`（仓库 `python/examples/` 全部脚本已同步）。
- `solve_piso(...)`（签名同上）→ `PisoResult`（`steps` / `history`）；`PisoConfig`：`dt` / `n_steps` / `n_correctors`（≥2）/ `time_scheme` / `scheme` / `solver_config`（动量）/ `pressure_solver_config`（压力，独立分派同上）/ `corrections` / `verbose`。
- `solve_transport(mesh, flux, gamma, scheme, bc, solver_config=..., Sc=None, Sp=None, phi_explicit=None, corrections=...)`：高层稳态输运（装配 + BiCGSTAB），返回 numpy 解向量。稳态交叉扩散需要 `phi_explicit` 提供冻结的显式场；未提供时该修正保持惰性。`solve_transient_transport(..., phi_old, dt, time_scheme, ..., corrections=...)` 为单个隐式 theta 时间步，`corrections` 沿用同一选项。
- 通量工具：`interpolate_cell_velocity_flux(mesh, velocity, rho, corrections=...)`、`compute_mass_flux(..., corrections=...)`、`initialize_mass_flux(..., corrections=...)`、`check_flux_compatibility(flux, rel_tol=1e-10)`、`has_dirichlet_pressure(bc_p)`。
- `assemble_momentum(mesh, flux, velocity, mu, scheme, bc, component, pressure, bc_p, relaxation=1.0, rho=1.0, dt=0.0, time_scheme=TimeScheme.Euler, corrections=...)` → `MomentumSystem`（Phase 6.2，侵入式 Galerkin 的算子出口）：对给定状态装配单个动量分量（`component` 0=u/1=v）而不求解，返回 finalized 矩阵的 COO 三元组（`rows`/`cols`/`vals`，阶数 `n`=单元数）、RHS `b`、（松弛后）对角 `diag` 与无压力梯度源 RHS `rhs_no_pressure`。`dt=0` 装配稳态算子；`dt>0` 装配一步隐式 theta（要求 `relaxation=1`，违反与非法 `component` 均抛 `ValueError`）。算子分离恒等式（$A_{\mathrm{tr}} = \theta\,A_{\mathrm{sp}} + \mathrm{diag}(\rho V/\mathrm{dt})$，ROM 的 $M\,\dot{a}$ 提取依据）与压力项语义见 `docs/rom.md`。

上述求解器与通量工具保持既有位置参数兼容；4.3c 只增加 `CorrectionOptions` 字段/枚举和 trailing 默认参数。

io：

- `write_vti(filename, mesh, scalars={"name": ScalarField}, vectors={"name": VectorField})`。
- `read_su2(path)` 返回 `UnstructuredMesh`；SU2 格式错误抛 `ValueError`，文件打开失败抛 `RuntimeError`。共享的 SU2 文本生成 helper 位于 `examples/mesh_su2.py`。
- `write_vtu(filename, mesh, scalars={"name": ScalarField}, vectors={"name": VectorField})` 写非结构网格结果；传入其他 mesh 的场抛 `ValueError`。

约定与陷阱：

- **生命周期**：C++ 场类持有网格引用，绑定用 `keep_alive` 让场对象钉住网格，正常使用无需关心。
- **网格一致性**：所有求解/装配/输出入口都会检查场对象与 `BoundaryField` 是否创建在传入的同一个 mesh 上；不一致抛 `ValueError`，消息中给出字段名（`write_vti` 用字典键，`Sc` / `Sp` 仅在非 `None` 时检查）。
- **索引范围**：`Mesh` 的 cell/face/patch 查询、`flux.phi/set_phi`、`flux.cell_imbalance` 分别要求索引小于 `cell_count/face_count/patch_count`；`CartesianMesh.cell_index` 要求 `i∈[0,nx)`、`j∈[0,ny)`，`x_face` 要求 `i∈[0,nx]`、`j∈[0,ny)`，`y_face` 要求 `i∈[0,nx)`、`j∈[0,ny]`。越界抛 `IndexError`；负数因 `Index` 是无符号类型，在 pybind11 参数转换阶段就抛 `TypeError`，不会进入上述范围检查。
- **GIL 与并发写**：长求解（`solve_simple` / `solve_piso` / 两个 `solve_*transport`）在 C++ 段内手动释放 GIL、返回前重新持有。求解运行期间，其他 Python 线程通过任何 `to_numpy()` 视图写字段或通量都是数据竞争（未定义行为）；单线程算例不受影响。不要用 pybind11 的 `call_guard<gil_scoped_release>` 包返回 numpy 数组或绑定对象的函数——返回值转换也发生在 GIL 释放区里，会段错误（本项目实测踩过）。
- **异常**：C++ 的 `std::exception`（非法 patch、`Sp > 0`、闭域通量不平衡等）自动映射为 Python 异常。

## 示例

基础与解析基准 Python 算例的用途、参数与运行方式见 README.md 的“示例”。

- `cavity.py` 支持 `--re {100,400,1000,3200}`，用 Ghia et al. (1982) 中心线数据做 L∞/L2 对照，`--test` 启用精度断言。`--re 3200`（Phase 5.3-1）为 256×256 大算例：外容差放宽至 2e-5、迭代上限 3000、本机冷态约 80 min，Ghia 3200 u 表的 y=0.4531 文献伪值按勘误先例修正并从度量剔除（见 `docs/numerical.md`）。`--backend/--method` 与 `--p-backend/--p-method` 分别指定动量/压力线性求解器（AMGCL 性能 showcase 用，对照数据记录在 `docs/numerical.md`）。
- `poiseuille.py` 检验平面 Poiseuille 的充分发展抛物线剖面与压降斜率。
- `stokes_oscillating.py` 检验 Stokes 第二问题的周期稳态解；非测试模式全程每 50 步写一帧（`--frames-every K`，0 关闭），序列覆盖起振到周期稳态。
- `mesh_su2.py` 是 Phase B 算例共享的纯 Python SU2 文本生成 helper，提供 `l_shaped_bfs_grid(n)`、`cylinder_rectangle_grid(...)`（O-4 多块：极坐标核心 + 8 个张量块，块间共享坐标池逐点共形）与 `sheared_channel(n, shear)`；生成文本经 `examples/_common.py` 的 `read_su2_text` 临时落盘后交给 `read_su2`。
- `backward_facing_step.py` 求解扩张比 2、入口段 10h 的后向台阶：Re=50/100（均匀入口，Upwind，历史校准）与 5.3-3 的 Re=389（逐面抛物线入口 `u=6s(1−s)`、Central；**Armaly 1983 约定** `Re=U_bulk·2h/ν` 即 `μ=2/389`，实测 x_r/h=8.10 与 2D 文献簇 8.2–8.6 / 实验 7.9 对齐），断言再附长度的文献量级带与 Re 趋势，网格研究数据记录在脚本注释中。
- `cylinder.py`（5.3-2 矩形域版）求解 `[-10,25]×[-10,10]` 圆柱绕流（Re=20/40，稳态 SIMPLE + Central，O-4 多块网格，`--grid {coarse,medium,fine}` 三档；fine 只加密外围——Re=20 下核心加密有稳定性边界，见 `docs/numerical.md` 圆柱节），逐面积分 Cd、测量尾迹长度并断言文献阈值带；`Cl` 仅作对称性诊断打印（校准值 ~±2e-4），不断言。
- `skewed_poiseuille.py` 是 5.1 逐面 Dirichlet 的定量示例（Phase 5.3 #4 提前完成）：等截面斜切平行四边形通道（端面倾角 atan(0.3)），两端逐面抛物线速度 `u=4y(1-y)`，压力全 Neumann 解析值 + 参考单元定规（刻意避开斜面 Dirichlet-p 的 `Cb` 无补全偏移，见 `docs/numerical.md`），`nNonOrthogonalCorrectors=1`。断言：入口通量与离散解析和精确相等、下游剖面 vs 抛物线 O(h) 带（n=16 校准 1.412e-2，带 3e-2；n=32 实测 6.784e-3，比值 2.1）。
- `taylor_green.py`（5.3-5）是精确边界瞬态算例：`[0,π]²` 衰减单模 TG，四边逐面 Dirichlet 精确 u/v/p（t^{n+1} 时刻），逐步 PISO + CN；固定小 dt（0.00125）的 64²→128² 空间收敛研究断言二阶（实测比值 4.96），`--full` 附加 256² 点（128→256 停滞、只报告不断言）；时间方向非单调（dt²/h 类混合项），证据矩阵见 `docs/numerical.md` Taylor–Green 节。非测试模式仅在 64² 档输出帧序列（`--frames-every`，默认 50）。
- `vortex_shedding.py`（5.3-6）是 Re=100 卡门涡街：复用 cylinder.py 网格/BC/力系数，确定性反对称种子，后 60% 时段窗口测 St/Cd_mean/Cl 振幅；CI 缩短版（coarse、dt=0.01、T=250，约 11 min）断言三带，medium 完整研究由脚本参数承担；非测试模式输出帧序列（`--frames-every`，默认 50，长算例间隔自动拉伸、总量 ≤~50 帧）。

- `taylor_green_rom.py`（Phase 6.1）是 TG 算例的 ROM 探索驱动：复用 `taylor_green.py` 的构造逐步采集快照（`--nx/--dt/--end-time/--snap-every`，默认 64²、dt=0.0025、T=2.0），落盘版本化 npz 到 `output/taylor_green_rom/`（契约 v1：`format_version/case/params/t/u/v/p/cx/cy`，行=快照，见 `python/rom/snapshots.py`），打印 POD 能量谱与 err(r) 表（`--ranks`）；`--modes N` 另写前 N 阶 POD 模态 + 均值场 `.vti`；`--test` 只跑机器级断言（契约成立、全秩重构误差≈0），不写 VTK/npz。
- `vortex_shedding_rom.py`（Phase 6.1）是涡街算例的 ROM 探索驱动：复用 `vortex_shedding.py` 的网格/BC/种子构造采集快照，瞬态丢弃窗口 `--window-frac`（默认 0.6，与 St 测量窗一致；1.0 表示全时段）后做窗口 POD 谱 + exact DMD，打印 DMD 主导频率换算的 St vs 文献带 0.164–0.167——**只诊断不断言**（DMD 主导模态启发式不入 CI）；npz 落盘 `output/vortex_shedding_rom/`；`--test` 仅断言机器级不变量（契约 key、全秩 POD 重构误差≈0、DMD 谱有限）。两 ROM 脚本均**不注册** application 层（先例：`impulsive_couette`）。
- `taylor_green_galerkin.py`（Phase 6.3）是 TG 算例的 Galerkin ROM 驱动：复用 `taylor_green.py` 的构造逐步采集快照 → u/v 中心化 POD（`--rank`，默认 8）→ `rom.galerkin.GalerkinRom` 显式 RK4 从 t=0 积分（`--dt-rom` 缺省取 FOM dt），**双变体**对照——零压（纯速度 ROM，量化结构性压力泄漏）与 oracle（经 pressure 注入口在阶段时刻供入精确压力，隔离机构误差）；输出 err(t) 表（两变体 vs 精确解与 FOM 轨迹）+ 单步一致性检查（一步 ROM RK4 vs 一步 FOM PISO，"小而非相等"）；`--frames` 写重构场 .vti；`--test`（16²、dt=0.01、T=0.3）断言状态往返、`M_r` SPD、单步一致性与短程零压误差带三点。实测证据见 `docs/rom.md` Galerkin 节。
- `vortex_shedding_galerkin.py`（Phase 6.3）是涡街算例的 Galerkin ROM **诊断**驱动：窗口 POD（`--rank`，默认 16）→ ROM 从窗口起点积分（静态 BC，无 bc_refresh）→ a₁/a₂ 系数 rfft 谱换算 ROM St，对照 FOM St 与文献带打印——**只诊断不断言**（纯速度 ROM 无压力约束，极限环保持无保证）；`--test` 仅断言机器级不变量（推进完成、系数有限、投影往返）。两 Galerkin 脚本均**不注册** application 层（沿用 6.1 先例）。

`python/examples/piso_nonortho_channel.py` 是 4.3c 斜交算例：由 bilinear 映射生成 $16\times16$ 梯形收缩通道 SU2 网格（顶边收窄到 $[0.3,0.7]$），运行单步压力驱动 PISO，打印 $p(0,0)$、入口单位长度通量与连续性误差。默认开启修正并使用两个 PISO corrector；`--no-corrections` 关闭三项修正，`--correctors N` 覆盖 corrector 数。该脚本用于复现 `docs/numerical.md` “斜交网格对照（未决）”中的算例设置。

时间相关但空间均匀的壁面值，标准用法是在每个时间步先更新 `BoundaryField.set(...)`，再把 `PisoConfig.n_steps` 置为 1 并调用 `solve_piso`（Stokes 振荡平板示例）。逐步调用与一次长调用语义等价：Couette 100 步两种调用方式的终态最大差为 `7.7e-12`（判据 `1e-9`）。`impulsive_couette.py` 即采用逐步调用模式，并按 `frame_interval(100)`=50 输出 3 帧序列（t=0/0.25/0.5）。

## 测试分工

### 新增算例脚本

新算例按以下约定加入：

- **脚本模式**：`argparse` 参数化（如 `--re`、`--backend`），`--test` 开关启用精度断言；不带 `--test` 时只打印结果供人工检查。精度阈值锁定在脚本内。
- **共享胶水**：sys.path/`_path` 引导、VTK 输出（`write_results`）、SU2 临时落盘（`read_su2_text`）、SIMPLE 收敛状态行（`print_simple_status`）与 `--test` PASS/FAIL 尾部（`report_test`）一律用 `examples/_common.py`，不在脚本内重写。瞬态多帧输出用 `write_frame`（写 `stem_{index:04d}.<ext>`，ParaView 按文件序列识别）+ `frame_interval`（默认每 50 步一帧；步数已知的算例间隔自动拉伸、序列总量 ≤~50 帧），序列含 t=0 初值帧与终态帧（末步去重），`--frames-every K` 控制（0 关闭），`--test` 模式不写帧。
- **阈值校准**：余量一般不小于 2×（现有例外与各条目的校准值见下文）；网格研究数据记录在脚本注释或 `docs/numerical.md`。
- **注册**：在根 `CMakeLists.txt` 用 `fvm_add_application_test(name script --test)` 添加，label/TIMEOUT 机制见 `docs/tests.md`；是否进 `application_default` 按"每物理家族/代码路径一条"取舍。
- **网格**：非结构算例复用 `examples/mesh_su2.py` 的 SU2 文本生成 helper，不自建重复生成逻辑。
- **证据**：算例设置、阈值校准与文献对照记录进 `docs/numerical.md`（或脚本注释）；手工复跑测试的坑见 `docs/tests.md`。

### 现有条目

`python/tests/test_pyfvm.py` 是**绑定契约测试**：numpy 零拷贝与索引顺序、网格几何、BC 默认值与非法参数、纯扩散线性精确解（`solve_transport` 全通路）、SIMPLE/PISO 粗网格冒烟、`write_vti` 冒烟。

`python/tests/test_rom.py` 是 **ROM 管线测试**（Phase 6.1，7 个用例，分三层语义）：纯代数单元（低秩矩阵 POD 精确重构/中心化、双正弦 DMD 频率恢复）、契约式（npz 保存/加载往返 + 非法快照栈/未知 `format_version` 拒绝）、TG 32² 端到端冒烟（快照契约成立、全秩 POD 重构误差、能量集中阈值）。`rom` 包不 import pyfvm，代数/契约测试无需扩展构建；pyfvm 缺失时仅 TG 冒烟 `skipIf` 跳过，其余照跑。两文件同被 `python -m unittest discover -s python/tests` 收编进 `pyfvm_tests`。

application 层由 13 个带 `application` 标签的 ctest 条目组成：阶段 A 的 `pyfvm_app_cavity_re100` / `_re400` / `_re1000` / `pyfvm_app_poiseuille` / `pyfvm_app_stokes_oscillating`，阶段 B 的 `pyfvm_app_backward_step` / `pyfvm_app_cylinder_re40` / `pyfvm_app_cylinder_re20`，Phase 5.3 的 `pyfvm_app_skewed_poiseuille`、`pyfvm_app_backward_step_re389`、`pyfvm_app_taylor_green`、`pyfvm_app_vortex_shedding`，以及 5.3-1 的 `pyfvm_app_cavity_re3200`（256×256，约 80 min，`TIMEOUT 10800`，**不在默认子集**；用 `ctest -L application` 运行），负责整算例精度回归；精度阈值锁定在示例脚本中。阈值校准余量一般不小于 2×，Stokes 为 1.4×。阶段 B 中，BFS 的 Re=50/100 再附长度带分别为 `[2,4]` / `[3,7]`，另要求 `x_r(100) >= 1.5*x_r(50)`；n=8/16/24 校准值为 3.841/4.374/4.569（Re=100，表观阶约 2.5、外推约 4.7），对应 20% 自洽护栏的实测余量约 4.5×；5.3-3 的 Re=389（抛物线入口，Armaly 约定 `μ=2/389`）断言 `x_r/h∈[7.2,9.0]`（n=16/24 实测 8.103/8.095，漂移 0.09%）。Taylor–Green（5.3-5）断言 64²→128² 空间阶 ∈[1.7,2.5] 与 64² 绝对 L2(u) ≤4e-4（实测阶 2.31）。涡街（5.3-6）断言 `St∈[0.145,0.185]`、`Cd_mean∈[1.20,1.55]`、`Cl_amp∈[0.25,0.45]`（coarse 实测 0.159/1.369/0.351）。圆柱（5.3-2 矩形域重校准）Re=40 断言 `Cd∈[1.45,1.72]`、`L_w/D∈[1.80,2.65]`，Re=20 断言 `Cd∈[1.95,2.30]`；medium 档实测 Cd(40)=1.606、Cd(20)=2.164（coarse→medium 漂移 0.6%/0.4%，fine 外围加密对 Cd 影响 <2e-4），完整网格研究与文献锚偏移讨论见 `docs/numerical.md` 圆柱节；单条约 65 s（TIMEOUT 600）。application 层串行全跑约 3.5–4 小时（`cavity_re3200` 约 80 min、`vortex_shedding` 约 11 min、`backward_step_re389` 约 19 min、`taylor_green` 约 2.5 min，其余更快）；持续满载后各条目可能变慢 2–15×，个别 TIMEOUT 余量薄。默认子集为 `application_default` 标签的 5 条（`cavity_re100`、`stokes_oscillating`、`backward_step`、`cylinder_re40`、`cylinder_re20`，成员在 `CMakeLists.txt` 注册处维护）：CI（GitHub Actions `CMake Build and Test` 的 linux 腿，带 `FVM_BUILD_PYTHON` 构建）每个 PR 跑单元 + 绑定 + 该子集，`workflow_dispatch` 的 `run_full` 输入改为全部 13 条；本地 MCP `build_and_test(run=application)` 同样默认该子集，`full=true` 跑整层。该层不改变 `fvm_unit_tests` 与 `pyfvm_tests` 的语义。
