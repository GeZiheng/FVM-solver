# python 模块说明（pyfvm 绑定）

`python/` 提供 pybind11 绑定模块 `pyfvm`，让具体算例用 Python 脚本编写：建网格、读 SU2 网格、numpy 填场、设边界、调 SIMPLE/PISO/输运求解、读回 numpy 后处理、写 VTK（vti/vtu）。绑定层只做转发，不含任何数值逻辑；`fvm_core` 的 C++ API 不因绑定而改动。

## 目录结构

```
python/
  pyfvm.cpp              — 全部绑定（单文件；Python 侧命名 snake_case）
  _path.py               — sys.path 引导：按当前解释器标签扫描 build/*/python 定位 pyfvm 扩展
  examples/              — 算例脚本（见下）
  tests/test_pyfvm.py    — 绑定契约测试（unittest；ctest 条目 pyfvm_tests）
  overlay-ports/python3/ — 空 vcpkg overlay 端口（机制见下）
```

## 构建

默认关闭，需要用 `-DFVM_BUILD_PYTHON=ON` 显式开启；完整的 configure/build/CTest 命令以 README.md 的“构建与测试”为 canonical 出处。

机制要点：

- **vcpkg feature**：`pybind11` 声明在 `vcpkg.json` 的 `"python"` feature 里；`-DFVM_BUILD_PYTHON=ON` 会在 `project()` 之前自动设置 `VCPKG_MANIFEST_FEATURES=python`，默认构建不拉取 pybind11。
- **空 overlay 端口**：pybind11 端口声明依赖 vcpkg 的 `python3` 端口（会源码构建 CPython 3.12，但我们用不上）。`python/overlay-ports/python3` 是一个空端口，满足依赖图但不装任何东西；扩展模块始终对着**系统 Python** 编译。
- **解释器选择**：扩展模块与 Python 版本绑定（cp314 的 `.pyd` 不能在 3.12 上 import）。configure 优先用 `-DPython_EXECUTABLE=...`，否则取 PATH 上第一个 `python3.14`/`python3`/`python`。pybind11 ≥ 3.0 才支持 Python 3.14（当前 vcpkg 端口 3.0.1）。
- **产物**：`build/<dir>/python/pyfvm*.pyd`（Linux 为 `.so`）；`_path.py` 优先采用环境变量 `FVM_PYFVM_DIR`，该目录必须已含当前解释器标签的扩展，否则抛 `RuntimeError`；未设置时扫描 `build/*/python`，只接受当前解释器标签（Windows `cp3XX`、Linux `cpython-3XX`）的扩展，并在多个匹配目录间取其中文件 mtime 最新者。自动定位失败时抛 `RuntimeError`，错误信息列出解释器版本、标签模式和已扫描目录。示例和测试直接 `import _path` 后即可 `import pyfvm`。

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

- 边界：`BoundaryField(mesh)` + `.set(patch, type, value)` / `.get(patch)` / `.patch_count()`；patch 常量 `BoundaryField.East/North/West/South` 只对应笛卡尔 patch 顺序，越界 patch 抛 `IndexError`；类型 `BCType.Dirichlet/Neumann`。默认全部 patch 零梯度 Neumann。
- 枚举：`ConvectionScheme.Upwind/Central`、`TimeScheme.Euler/CrankNicolson`、`NonOrthSplit.MinimumCorrection/OverRelaxed`。
- `SolverConfig`：`tolerance`（相对残差）/ `max_iterations` / `verbose`。
- `CorrectionOptions`：`non_orthogonal_diffusion=True`、`skew_interpolation=True`、`n_non_orthogonal_correctors=0`、`non_orth_split=NonOrthSplit.MinimumCorrection`。笛卡尔网格上修正量恒为零，默认行为保持零数值变化（旋转正交网格上压力 `crossP` 仍含各向异性补全项，见 `docs/numerical.md` 压力方程一节）。`n_non_orthogonal_correctors` 是压力求解子循环附加数（0 表示一次压力求解），负值会被拒绝。`non_orth_split` 永远控制内部面 $(\alpha,k)$；`non_orthogonal_diffusion` 只门控显式交叉项，因此“关闭修正 + OverRelaxed”表示 over 隐式、无显式修正。示例：`corrections.non_orth_split = NonOrthSplit.OverRelaxed`。
- `solve_simple(mesh, rho, mu, bc_u, bc_v, bc_p, config, velocity, pressure, flux)` → `SimpleResult`（`converged` / `iterations` / `history`）；`SimpleConfig`：`max_iterations` / `tolerance` / `relaxation_u` / `relaxation_p` / `scheme` / `solver_config` / `corrections` / `verbose`。
- `solve_piso(...)`（签名同上）→ `PisoResult`（`steps` / `history`）；`PisoConfig`：`dt` / `n_steps` / `n_correctors`（≥2）/ `time_scheme` / `scheme` / `solver_config` / `corrections` / `verbose`。
- `solve_transport(mesh, flux, gamma, scheme, bc, solver_config=..., Sc=None, Sp=None, phi_explicit=None, corrections=...)`：高层稳态输运（装配 + BiCGSTAB），返回 numpy 解向量。稳态交叉扩散需要 `phi_explicit` 提供冻结的显式场；未提供时该修正保持惰性。`solve_transient_transport(..., phi_old, dt, time_scheme, ..., corrections=...)` 为单个隐式 theta 时间步，`corrections` 沿用同一选项。
- 通量工具：`interpolate_cell_velocity_flux(mesh, velocity, rho, corrections=...)`、`compute_mass_flux(..., corrections=...)`、`initialize_mass_flux(..., corrections=...)`、`check_flux_compatibility(flux, rel_tol=1e-10)`、`has_dirichlet_pressure(bc_p)`。

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

- `cavity.py` 支持 `--re {100,400,1000}`，用 Ghia et al. (1982) 中心线数据做 L∞/L2 对照，`--test` 启用精度断言。
- `poiseuille.py` 检验平面 Poiseuille 的充分发展抛物线剖面与压降斜率。
- `stokes_oscillating.py` 检验 Stokes 第二问题的周期稳态解。
- `mesh_su2.py` 是 Phase B 算例共享的纯 Python SU2 文本生成 helper，提供 `l_shaped_bfs_grid(n)` 与 `polar_ogrid(...)`；生成文本由算例临时落盘后交给 `read_su2`。
- `backward_facing_step.py` 求解扩张比 2、入口段 10h 的后向台阶（Re=50/100，Upwind），断言再附长度的文献量级带与 Re 趋势，网格研究数据记录在脚本注释中。
- `cylinder.py` 求解 96×64 极坐标 O-grid 圆柱绕流（Re=20/40，稳态 SIMPLE + Central），逐面积分 Cd、测量尾迹长度并断言文献阈值带；`Cl` 仅作对称性诊断打印（校准值约 `-5e-5`），不断言。

`python/examples/piso_nonortho_channel.py` 是 4.3c 斜交算例：由 bilinear 映射生成 $16\times16$ 梯形收缩通道 SU2 网格（顶边收窄到 $[0.3,0.7]$），运行单步压力驱动 PISO，打印 $p(0,0)$、入口单位长度通量与连续性误差。默认开启修正并使用两个 PISO corrector；`--no-corrections` 关闭三项修正，`--correctors N` 覆盖 corrector 数。该脚本用于复现 `docs/numerical.md` “斜交网格对照（未决）”中的算例设置。

时间相关但空间均匀的壁面值，标准用法是在每个时间步先更新 `BoundaryField.set(...)`，再把 `PisoConfig.n_steps` 置为 1 并调用 `solve_piso`（Stokes 振荡平板示例）。逐步调用与一次长调用语义等价：Couette 100 步两种调用方式的终态最大差为 `7.7e-12`（判据 `1e-9`）。

## 测试分工

`python/tests/test_pyfvm.py` 是**绑定契约测试**：numpy 零拷贝与索引顺序、网格几何、BC 默认值与非法参数、纯扩散线性精确解（`solve_transport` 全通路）、SIMPLE/PISO 粗网格冒烟、`write_vti` 冒烟。

application 层由 8 个带 `application` 标签的 ctest 条目组成：阶段 A 的 `pyfvm_app_cavity_re100` / `_re400` / `_re1000` / `pyfvm_app_poiseuille` / `pyfvm_app_stokes_oscillating`，以及阶段 B 的 `pyfvm_app_backward_step` / `pyfvm_app_cylinder_re40` / `pyfvm_app_cylinder_re20`（用 `ctest -L application` 运行），负责整算例精度回归；精度阈值锁定在示例脚本中。阈值校准余量一般不小于 2×，Stokes 为 1.4×。阶段 B 中，BFS 的 Re=50/100 再附长度带分别为 `[2,4]` / `[3,7]`，另要求 `x_r(100) >= 1.5*x_r(50)`；n=8/16/24 校准值为 3.841/4.374/4.569（Re=100，表观阶约 2.5、外推约 4.7），对应 20% 自洽护栏的实测余量约 4.5×。圆柱 Re=40 断言 `Cd∈[1.40,1.72]`、`L_w/D∈[1.8,2.8]`，Re=20 断言 `Cd∈[1.80,2.30]`；96×64→128×80 的 Cd 漂移实测 0.17–0.21%（Cd(40)=1.559、Cd(20)=2.090）。application 层串行全跑约 1 小时；本机冷态实测 backward_step 229s、cavity_re1000 368s、stokes 38s，持续满载后各条目可能变慢 2–15×，个别 TIMEOUT 余量薄。该层不改变 `fvm_unit_tests` 与 `pyfvm_tests` 的语义。
