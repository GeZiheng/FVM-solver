# python 模块说明（pyfvm 绑定）

`python/` 提供 pybind11 绑定模块 `pyfvm`，让具体算例用 Python 脚本编写：建网格、numpy 填场、设边界、调 SIMPLE/PISO/输运求解、读回 numpy 后处理、写 vti。绑定层只做转发，不含任何数值逻辑；`fvm_core` 的 C++ API 不因绑定而改动。

## 目录结构

```
python/
  pyfvm.cpp              — 全部绑定（单文件；Python 侧命名 snake_case）
  _path.py               — sys.path 引导：扫描 build/*/python 找到编译出的 pyfvm 扩展
  examples/              — 算例脚本（见下）
  tests/test_pyfvm.py    — 绑定契约测试（unittest；ctest 条目 pyfvm_tests）
  overlay-ports/python3/ — 空 vcpkg overlay 端口（见"构建"）
```

## 构建

默认关闭，需要显式开启：

```powershell
cmake -B build/release-py -S . -G Ninja `
      -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake `
      -DCMAKE_BUILD_TYPE=Release -DFVM_BUILD_PYTHON=ON
cmake --build build/release-py
ctest --test-dir build/release-py --output-on-failure   # fvm_unit_tests + pyfvm_tests
```

机制要点：

- **vcpkg feature**：`pybind11` 声明在 `vcpkg.json` 的 `"python"` feature 里；`-DFVM_BUILD_PYTHON=ON` 会在 `project()` 之前自动设置 `VCPKG_MANIFEST_FEATURES=python`，默认构建不拉取 pybind11。
- **空 overlay 端口**：pybind11 端口声明依赖 vcpkg 的 `python3` 端口（会源码构建 CPython 3.12，但我们用不上）。`python/overlay-ports/python3` 是一个空端口，满足依赖图但不装任何东西；扩展模块始终对着**系统 Python** 编译。
- **解释器选择**：扩展模块与 Python 版本绑定（cp314 的 `.pyd` 不能在 3.12 上 import）。configure 优先用 `-DPython_EXECUTABLE=...`，否则取 PATH 上第一个 `python3.14`/`python3`/`python`。pybind11 ≥ 3.0 才支持 Python 3.14（当前 vcpkg 端口 3.0.1）。
- **产物**：`build/<dir>/python/pyfvm*.pyd`；`_path.py` 按 `build/*/python` 扫描定位，示例和测试直接 `import _path` 后即可 `import pyfvm`。

## API 一览

命名约定：Python 侧全部 snake_case（如 `maxIterations` → `max_iterations`）。

core：

- `CartesianMesh(nx, ny, x_min, y_min, x_max, y_max)`：`nx()` / `ny()` / `cell_count()` / `dx()` / `dy()` / `cell_volume(c)` / `cell_center(c)` / `cell_index(i, j)` / `cell_ij(c)`。
- `ScalarField(mesh, name="")`、`VectorField(mesh, name="")`（`.u` / `.v` 返回 `ScalarField`）、`FaceFluxField(mesh, name="")`。
- `f.to_numpy()`：**零拷贝可写视图**，cell 顺序 `j*nx+i`，可直接 `reshape(ny, nx)`；`f.value(i, j)` / `f.set(i, j, v)` 按单元读写；`set_zero()` / `set_constant(v)`。
- `flux.to_numpy_x()` / `flux.to_numpy_y()`：面通量视图（x 面索引 `j*(nx+1)+i`，y 面 `j*nx+i`）；`flux.x(i, j)` / `flux.y(i, j)` / `flux.cell_imbalance(c)`。

numerical：

- 边界：`BoundaryField()` + `.set(side, type, value)` / `.get(side)`；边常量 `BoundaryField.East/North/West/South`；类型 `BCType.Dirichlet/Neumann`。默认四边零梯度 Neumann。
- 枚举：`ConvectionScheme.Upwind/Central`、`TimeScheme.Euler/CrankNicolson`。
- `SolverConfig`：`tolerance`（相对残差）/ `max_iterations` / `verbose`。
- `solve_simple(mesh, rho, mu, bc_u, bc_v, bc_p, config, velocity, pressure, flux)` → `SimpleResult`（`converged` / `iterations` / `history`）；`SimpleConfig`：`max_iterations` / `tolerance` / `relaxation_u` / `relaxation_p` / `scheme` / `solver_config` / `verbose`。
- `solve_piso(...)`（签名同上）→ `PisoResult`（`steps` / `history`）；`PisoConfig`：`dt` / `n_steps` / `n_correctors`（≥2）/ `time_scheme` / `scheme` / `solver_config` / `verbose`。
- `solve_transport(mesh, flux, gamma, scheme, bc, solver_config=..., Sc=None, Sp=None)`：高层稳态输运（装配 + BiCGSTAB），返回 numpy 解向量；`solve_transient_transport(..., phi_old, dt, time_scheme, ...)` 为单个隐式 theta 时间步。
- 通量工具：`interpolate_cell_velocity_flux(mesh, velocity, rho)`、`compute_mass_flux(...)`、`initialize_mass_flux(...)`、`check_flux_compatibility(flux, rel_tol=1e-10)`、`has_dirichlet_pressure(bc_p)`。

io：

- `write_vti(filename, mesh, scalars={"name": ScalarField}, vectors={"name": VectorField})`。

约定与陷阱：

- **生命周期**：C++ 场类持有网格引用，绑定用 `keep_alive` 让场对象钉住网格，正常使用无需关心。
- **GIL**：长求解（`solve_simple` / `solve_piso` / 两个 `solve_*transport`）在 C++ 段内手动释放 GIL、返回前重新持有。不要用 pybind11 的 `call_guard<gil_scoped_release>` 包返回 numpy 数组或绑定对象的函数——返回值转换也发生在 GIL 释放区里，会段错误（本项目实测踩过）。
- **异常**：C++ 的 `std::exception`（非法边界 side、`Sp > 0`、闭域通量不平衡等）自动映射为 Python 异常。

## 示例（python/examples/）

三个算例即原 C++ demo 与 PISO 测试的 Python 版，直接 `python python/examples/<name>.py` 运行，vti 写到当前目录：

- **convection_diffusion.py**：单位方域稳态对流-扩散。无散度回流速度场（流函数 ψ = sin(πx)·sin(πy) 派生），γ = 0.01（Pe ~ 100），西墙 T=1、东墙 T=0、南北绝热。演示 `solve_transport` 与 numpy 填场。
- **cavity.py**：顶盖驱动方腔流 Re=100（64×64，SIMPLE，relaxation_u=0.7 / relaxation_p=0.3），约 1000 次外迭代收敛；输出 `cavity.vti` 可在 ParaView 看主涡与角隅二次涡。
- **impulsive_couette.py**：瞬态 Couette 启动（PISO，ν=0.1，dt=0.005 × 100 步到 t=0.5），与精确 Fourier 级数解对比打印误差。

## 测试分工

`python/tests/test_pyfvm.py` 是**绑定契约测试**：numpy 零拷贝与索引顺序、网格几何、BC 默认值与非法参数、纯扩散线性精确解（`solve_transport` 全通路）、SIMPLE/PISO 粗网格冒烟、`write_vti` 冒烟。物理与数值正确性由 C++ doctest（`tests/`）独家负责，两侧不重复断言。算例级的持续验证（examples 作为 application tests）后续单独做，不混入 unit test。
