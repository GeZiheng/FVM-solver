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
- 枚举：`ConvectionScheme.Upwind/Central`、`TimeScheme.Euler/CrankNicolson`。
- `SolverConfig`：`tolerance`（相对残差）/ `max_iterations` / `verbose`。
- `solve_simple(mesh, rho, mu, bc_u, bc_v, bc_p, config, velocity, pressure, flux)` → `SimpleResult`（`converged` / `iterations` / `history`）；`SimpleConfig`：`max_iterations` / `tolerance` / `relaxation_u` / `relaxation_p` / `scheme` / `solver_config` / `verbose`。
- `solve_piso(...)`（签名同上）→ `PisoResult`（`steps` / `history`）；`PisoConfig`：`dt` / `n_steps` / `n_correctors`（≥2）/ `time_scheme` / `scheme` / `solver_config` / `verbose`。
- `solve_transport(mesh, flux, gamma, scheme, bc, solver_config=..., Sc=None, Sp=None)`：高层稳态输运（装配 + BiCGSTAB），返回 numpy 解向量；`solve_transient_transport(..., phi_old, dt, time_scheme, ...)` 为单个隐式 theta 时间步。
- 通量工具：`interpolate_cell_velocity_flux(mesh, velocity, rho)`、`compute_mass_flux(...)`、`initialize_mass_flux(...)`、`check_flux_compatibility(flux, rel_tol=1e-10)`、`has_dirichlet_pressure(bc_p)`。

上述求解器/通量工具的 Python 签名与 4.3a 之前保持不变；变化只在网格、场与 `BoundaryField` 的构造和索引方式。

io：

- `write_vti(filename, mesh, scalars={"name": ScalarField}, vectors={"name": VectorField})`。
- `read_su2(path)` 返回 `UnstructuredMesh`；SU2 格式错误抛 `ValueError`，文件打开失败抛 `RuntimeError`。
- `write_vtu(filename, mesh, scalars={"name": ScalarField}, vectors={"name": VectorField})` 写非结构网格结果；传入其他 mesh 的场抛 `ValueError`。

约定与陷阱：

- **生命周期**：C++ 场类持有网格引用，绑定用 `keep_alive` 让场对象钉住网格，正常使用无需关心。
- **网格一致性**：所有求解/装配/输出入口都会检查场对象与 `BoundaryField` 是否创建在传入的同一个 mesh 上；不一致抛 `ValueError`，消息中给出字段名（`write_vti` 用字典键，`Sc` / `Sp` 仅在非 `None` 时检查）。
- **索引范围**：`Mesh` 的 cell/face/patch 查询、`flux.phi/set_phi`、`flux.cell_imbalance` 分别要求索引小于 `cell_count/face_count/patch_count`；`CartesianMesh.cell_index` 要求 `i∈[0,nx)`、`j∈[0,ny)`，`x_face` 要求 `i∈[0,nx]`、`j∈[0,ny)`，`y_face` 要求 `i∈[0,nx)`、`j∈[0,ny]`。越界抛 `IndexError`；负数因 `Index` 是无符号类型，在 pybind11 参数转换阶段就抛 `TypeError`，不会进入上述范围检查。
- **GIL 与并发写**：长求解（`solve_simple` / `solve_piso` / 两个 `solve_*transport`）在 C++ 段内手动释放 GIL、返回前重新持有。求解运行期间，其他 Python 线程通过任何 `to_numpy()` 视图写字段或通量都是数据竞争（未定义行为）；单线程算例不受影响。不要用 pybind11 的 `call_guard<gil_scoped_release>` 包返回 numpy 数组或绑定对象的函数——返回值转换也发生在 GIL 释放区里，会段错误（本项目实测踩过）。
- **异常**：C++ 的 `std::exception`（非法 patch、`Sp > 0`、闭域通量不平衡等）自动映射为 Python 异常。

## 示例

三个 Python 算例的用途、参数与运行方式见 README.md 的“示例”。

## 测试分工

`python/tests/test_pyfvm.py` 是**绑定契约测试**：numpy 零拷贝与索引顺序、网格几何、BC 默认值与非法参数、纯扩散线性精确解（`solve_transport` 全通路）、SIMPLE/PISO 粗网格冒烟、`write_vti` 冒烟。物理与数值正确性由 C++ doctest（`tests/`）独家负责，两侧不重复断言。算例级的持续验证（examples 作为 application tests）后续单独做，不混入 unit test。
