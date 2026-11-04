# math 模块说明

`fvm::math` 命名空间，提供稀疏矩阵装配与线性方程组求解能力。模块对上层（numerical/app）隐藏 Eigen 的细节：装配侧只看到三元组 API，求解侧只看到抽象接口 + 工厂函数，后端可替换——现有 Eigen 三个实现，另有经 `FVM_BUILD_AMGCL=ON` 可选启用的 AMGCL 后端（见下文），Hypre 等预留。

## SparseMatrix：三元组装配的稀疏矩阵

### 设计思路

- **两阶段生命周期**：装配阶段通过 `insert(row, col, value)` 追加三元组 `(row, col, value)` 到内部 `std::vector<Eigen::Triplet>`；装配完成后调用 `finalize()`，由 `Eigen::SparseMatrix::setFromTriplets` 一次性构建压缩列存储（CCS）格式。三元组装配允许任意顺序插入且自动合并重复位置，非常适合有限体积法中"逐面累加通量贡献"的装配模式。
- **显式状态保护**：`finalized_` 标志区分两种状态——
  - 未 finalize：`insert` 可用，`native()` 抛异常；
  - 已 finalize：`native()` 可用，`insert` 抛异常。
  
  防止"装配一半就拿去求解"或"求解后又偷偷插入"两类错误。
- **隐藏内部存储**：对外只暴露 `insert/finalize/native` 等少量接口。`native()` 返回底层 Eigen 矩阵的常量引用，仅供求解器实现使用；若未来替换后端，只需修改 `native()` 的返回类型或增加内部访问器，上层装配代码不变。
- `setZero()` 将矩阵重置回未 finalize 的空状态，可重新装配。
- **矩阵缩放**：`scale(factor)` 将全部已存系数乘以 `factor`——未 finalize 时缩放已累加的三元组，已 finalize 时缩放底层矩阵。用于隐式 θ 时间格式：由一次稳态空间算子装配 $A_{spatial}$ 廉价地得到 $\theta A_{spatial}$，无需重新装配。

## LinearSolver：抽象求解器接口

### 设计思路

- **接口与后端分离**：`LinearSolver` 只声明 `virtual Vector solve(const SparseMatrix& A, const Vector& b) = 0`。三个 Eigen 实现类（`EigenBiCGSTABSolver`、`EigenCGSolver`、`EigenSparseLUSolver`）定义在 `.cpp` 的匿名命名空间中，对外不可见；用户只能通过工厂函数创建：

  ```cpp
  auto solver = fvm::math::createEigenBiCGSTAB(cfg);  // std::unique_ptr<LinearSolver>
  ```

  新增后端时只需添加新的子类 + 工厂函数，不触动任何调用点。

- **统一配置**：`SolverConfig { tolerance, maxIterations, verbose, backend, method }`，语义为**相对残差**判据 $\lVert Ax-b \rVert / \lVert b \rVert \le \text{tolerance}$。求解后可通过 `lastIterations()` / `lastResidual()` 查询实际迭代数与残差。
- **`backend` × `method` 正交字段**：`backend`（实现族，默认 `"eigen"` | `"amgcl"`）与 `method`（外层 Krylov，默认 `"bicgstab"` | `"cg"`）独立选择，pyfvm 的 `makeLinearSolver` 按二维组合分派四家工厂（含经此可达的 `(eigen, cg)` 组合）。`amgcl` 族需 `FVM_BUILD_AMGCL=ON` 构建（未知值 → `invalid_argument`；OFF 构建选 amgcl → `runtime_error`，与方法无关）。C++ 驱动不消费这两个字段——C++ 直接调工厂，字段由 pyfvm 单系统入口消费。
- **失败即异常**：所有实现在不收敛或分解失败时抛 `std::runtime_error`，调用方无需检查错误码。

### 实现一览

| 实现 | 算法 | 适用场景 | 备注 |
|------|------|---------|------|
| `createEigenBiCGSTAB` | 双共轭梯度稳定法 + 对角（Jacobi）预条件 | 非对称系统（含对流的输运方程） | 见下方容差换算说明 |
| `createEigenCG` | 共轭梯度法 + 对角预条件 | 对称正定系统（纯扩散） | 非对称矩阵上行为未定义 |
| `createEigenSparseLU` | 稀疏 LU 直接分解 | 小问题、验证用、病态系统 | `lastIterations`=1，残差为显式计算的 $\lVert Ax-b \rVert / \lVert b \rVert$ |
| `createAmgclCG` | AMG（smoothed_aggregation + spai0）预条件 CG | 压力类 SPD 系统 | 需 `FVM_BUILD_AMGCL=ON`，见下节 |
| `createAmgclBiCGSTAB` | AMG（smoothed_aggregation + ilu0）预条件 BiCGSTAB | 动量类非对称系统 | 需 `FVM_BUILD_AMGCL=ON`，见下节 |

### AMGCL 后端（可选）

两个门控工厂在 `#ifdef FVM_HAVE_AMGCL` 下声明（编译开关 `FVM_BUILD_AMGCL`，默认 OFF），实现集中在单一翻译单元 `src/math/src/AmgclSolver.cpp`（整个文件由该宏包裹）——重模板的 amgcl 头文件只在这一个 TU 展开，OFF 构建的编译时间与产物完全不变。内部类结构镜像 `LinearSolver.cpp` 的 Eigen 包装（无状态、每次 solve 现建求解器）。设计要点：

- **固定组合，不加旋钮**：`createAmgclCG` = `amg<builtin, smoothed_aggregation, spai0>` + CG；`createAmgclBiCGSTAB` = `amg<builtin, smoothed_aggregation, ilu0>` + BiCGSTAB（均为 builtin backend）。
- **参数映射**：`rtol = SolverConfig::tolerance`（AMGCL 的 `rtol` 即 $\lVert r \rVert / \lVert b \rVert$，与项目容差定义一致）、`maxiter = maxIterations`；`verbose` 暂不消费（与 Eigen 包装一致）。
- **自检收敛**：`lastIterations()` / `lastResidual()` 取 amgcl solve 返回的 `(iters, error)`；`error > tolerance` 时抛 `std::runtime_error`——与 Eigen BiCGSTAB 包装同约定，不依赖求解器状态标志。
- **矩阵对接**：`SparseMatrix::native()` 是 ColMajor Eigen 矩阵，solve 内先做一次 ColMajor→RowMajor 存储序转换，再经 `amgcl/adapter/eigen.hpp` 适配。
- **TU 级 `AMGCL_NO_BOOST`**：amgcl 1.5.0 的 `util.hpp` 默认拉 `boost/property_tree`（只有 ptree 参数桥接需要）；本项目使用 plain-struct params，不需要 Boost，故在该 TU 的所有 amgcl include 之前定义 `AMGCL_NO_BOOST`。

**依赖接入**：自建 vcpkg overlay port `ports/amgcl/`（锁 1.5.0 + SHA512，仅依赖 eigen3），经 `vcpkg-configuration.json` 的 `overlay-ports` 接入；`FVM_BUILD_AMGCL` 默认 OFF，OFF 构建不执行任何 amgcl 接线。

### 关键实现细节：BiCGSTAB 的容差换算

Eigen 5.x 的 `BiCGSTAB` 内部以**绝对残差**作为停机判据，且其收敛标志存在绝对/相对量纲不一致的问题。因此 `EigenBiCGSTABSolver::solve` 中做了两步处理：

1. **容差缩放**：设右端项范数为 $\lVert b \rVert$，传给 Eigen 的停机容差为

   $$\text{tol}_{\text{eigen}} = \text{tolerance} \cdot \lVert b \rVert \qquad (\lVert b \rVert > 0)$$

   使 Eigen 的绝对判据等价于本项目约定的相对判据 $\lVert Ax-b \rVert / \lVert b \rVert \le \text{tolerance}$。

2. **自行检查收敛**：求解后取 `solver.error()`（相对残差估计）存入 `lastResidual_`，若超过 `config_.tolerance` 则抛出"did not converge"异常——**不依赖 `solver.info()` 判断 BiCGSTAB 是否收敛**（`info()` 仅用于检查分解失败与数值崩溃）。

