# 测试层说明（ctest 三层）

项目测试分三层，各自职责独立、互不替代。本文是测试层的**机制与约定**出处：注册点、分层验证粒度、已知坑。条目名单、阈值校准值与耗时实测属于 python 模块的内容，以 `docs/python.md` §测试分工 为唯一出处；构建/测试工具（build-and-test MCP 与手动命令）的调用方式以 `AGENTS.md` Build Instructions 及 `README.md` 为准，本文不重复。

## 三层架构

| 层 | ctest 条目 | 职责 | 载体 |
|---|---|---|---|
| C++ 单元测试 | `fvm_unit_tests` | 模块级正确性：算子、装配、求解器、网格、IO | doctest，单可执行文件 `fvm_tests` |
| 绑定契约 | `pyfvm_tests` | pyfvm 绑定契约：零拷贝视图、索引顺序、异常映射、粗网格冒烟 | Python unittest（`python/tests/test_pyfvm.py`） |
| 整算例回归 | `pyfvm_app_*`（`application` 标签） | 完整算例 vs 解析/文献基准的精度断言 | `python/examples/` 脚本 + `--test` |

三层语义互不覆盖：单元层不跑整算例，application 层不重复绑定契约；application 层的存在不改变 `fvm_unit_tests` 与 `pyfvm_tests` 的语义。

## C++ 单元测试约定

- **单可执行文件**：所有测试编进一个 `fvm_tests`；`tests/main.cpp` 仅含 `DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`，ctest 条目只有 `fvm_unit_tests` 一个。
- **文件组织**：`tests/test_<主题>.cpp` 一文件一主题（`test_diffusion.cpp`、`test_piso.cpp`…）。新增测试文件在根 `CMakeLists.txt` 的 `fvm_tests` 源列表注册；`test_amgcl_solver.cpp` 由 `FVM_BUILD_AMGCL` 门控追加，OFF 构建不参与编译。
- **风格**：`TEST_CASE("描述性名称")` 逐例命名，断言用 `CHECK` / `REQUIRE`；测试源文件与项目源文件同样保持**纯 ASCII**（MSVC C4819）。
- **共享夹具**：`tests/ObliqueMeshFixtures.h` 提供斜交网格夹具（`makeObliquePair`、`makeShearedTriMesh`），供多个非正交相关测试复用。
- **双表示纪律**：网格抽象（`CartesianMesh` vs `UnstructuredMesh`）的等价性回归锚定在 `tests/test_mesh_dual.cpp`（其 `makeStructuredMesh` 为文件内局部 helper，在 `UnstructuredMesh` 上复现笛卡尔网格）；设计动机与证据见 `docs/numerical.md` §4.3a。

## pyfvm_tests

- 条目在根 `CMakeLists.txt` 的 `FVM_BUILD_PYTHON=ON` 分支内：命令为系统解释器 `python -m unittest discover -s python/tests`，并已用 `ENVIRONMENT FVM_PYFVM_DIR=${CMAKE_BINARY_DIR}/python` 钉定扩展目录——**ctest 证据不受多构建树影响**（见下文已知坑）。
- 测试文件两个：`test_pyfvm.py`（绑定契约，37 用例，含 6.2 的 `TestAssembleMomentum`）与 `test_rom.py`（Phase 6.1 ROM 管线 + 6.3 Galerkin，11 用例，自动被 discover 收编）：POD 代数 ×2、DMD 双正弦合成信号 ×1、npz 契约往返 + 错误路径 ×3、Taylor–Green 32² 端到端冒烟 ×1（以上 6.1，实测约 +0.6 s）、`TestGalerkinRomTaylorGreen` ×4（16² 短程：全秩状态往返、`M_r` SPD、单步 ROM RK4 vs FOM PISO 一致性、零压短程误差带——整文件约 1 s 量级）。`rom` 包核心不依赖 pyfvm（`rom.galerkin` 惰性 import）——pyfvm 缺失时仅 TG 冒烟与 Galerkin 用例跳过，代数/契约用例照跑；内容范围详见 `docs/python.md` §测试分工。
- AMGCL 子集的构建模式门控见 `docs/python.md` §测试分工。

## application 层

- **注册**：根 `CMakeLists.txt` 的 `fvm_add_application_test(name script ...)` 函数，条目名 `pyfvm_app_<name>`，命令为 `python examples/<script>.py <args...>`，默认 `LABELS application`、`TIMEOUT 600`；个别长算例另行覆盖（如 `cavity_re3200` 为 10800）。
- **label 语义**：`application` = 整层（`ctest -L application`）；`application_default` = CI PR 与 MCP `run=application` 默认跑的子集。成员在注册处维护，**名单与取舍理由以 `docs/python.md` §测试分工 为唯一出处**，CMake 注释要求两边保持同步。
- **阈值**：精度阈值锁定在各示例脚本内，由 `--test` 启用断言；新增算例的完整约定见 `docs/python.md` §测试分工"新增算例脚本"。

## 分层验证粒度

改动类型 → 验证要求（工具参数与默认值见 `AGENTS.md` Build Instructions）：

| 改动类型 | 验证要求 |
|---|---|
| 纯 C++ 模块（core/math/io/numerical） | `run=unit`（`ctest -LE application`：`fvm_unit_tests` + `pyfvm_tests`） |
| `python/`、pybind11 绑定、`pyfvm_tests` | 在上述基础上**必须加一轮 `python=true`** 的 build-and-test（绑定测试只在 python 构建树的 ctest 里运行） |
| 新增/修改算例或其阈值 | application 子集（`run=application`）；全层用 `full=true`，指定条目用 `tests` 正则 |
| AMGCL 门控特性 | 以 `cmake_args="-DFVM_BUILD_AMGCL=ON"` 单独验证；多构建树并存时的注意事项见下文 |

## 已知坑（操作性）

1. **多构建树 mtime 遮蔽**：`_path.py` 在未设 `FVM_PYFVM_DIR` 时按 mtime 最新在 `build/*/python` 里选扩展。多构建树并存（如 AMGCL ON/OFF 各自验证）时，更晚构建的树会遮蔽目标树，手工 `python -m unittest discover` 出现**假 skip**（如 AMGCL 用例在 OFF pyd 下静默跳过）。手工复跑绑定测试必带 `FVM_PYFVM_DIR=<目标构建树>/python`，或只采信 ctest 结果。
2. **异步 application ctest 的 python 启动停驻**（2026-11-09 ZCode 环境实测，根因未定，非仓库代码缺陷）：MCP `build_and_test(run=application/all)` 的后台 ctest 中，python 测试进程可能在解释器启动阶段 0-CPU 停驻直至 TIMEOUT；同一命令从 bash 直接后台运行正常。fallback：`ctest --test-dir <构建树> -L application --output-on-failure -C Release` 从 bash 直跑。
3. **满载变慢**：持续满载后 application 条目可能显著变慢、个别 TIMEOUT 余量薄；实测数据见 `docs/python.md` §测试分工。
