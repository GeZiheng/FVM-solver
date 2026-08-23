# math 模块说明

`fvm::math` 命名空间，提供稀疏矩阵装配与线性方程组求解能力。模块对上层（numerical/app）隐藏 Eigen 的细节：装配侧只看到三元组 API，求解侧只看到抽象接口 + 工厂函数，从而为将来替换后端（AMGCL、Hypre 等）预留空间。

## 文件结构

| 文件 | 内容 |
|------|------|
| `include/SparseMatrix.h` + `src/SparseMatrix.cpp` | `SparseMatrix`：三元组装配式稀疏矩阵包装 |
| `include/LinearSolver.h` + `src/LinearSolver.cpp` | `SolverConfig`、`LinearSolver` 抽象接口、三个 Eigen 求解器实现及工厂函数 |

## SparseMatrix：三元组装配的稀疏矩阵

### 设计思路

- **两阶段生命周期**：装配阶段通过 `insert(row, col, value)` 追加三元组 `(row, col, value)` 到内部 `std::vector<Eigen::Triplet>`；装配完成后调用 `finalize()`，由 `Eigen::SparseMatrix::setFromTriplets` 一次性构建压缩列存储（CCS）格式。三元组装配允许任意顺序插入且自动合并重复位置，非常适合有限体积法中"逐面累加通量贡献"的装配模式。
- **显式状态保护**：`finalized_` 标志区分两种状态——
  - 未 finalize：`insert` 可用，`native()` 抛异常；
  - 已 finalize：`native()` 可用，`insert` 抛异常。
  
  防止"装配一半就拿去求解"或"求解后又偷偷插入"两类错误。
- **隐藏内部存储**：对外只暴露 `insert/finalize/native` 等少量接口。`native()` 返回底层 Eigen 矩阵的常量引用，仅供求解器实现使用；若未来替换后端，只需修改 `native()` 的返回类型或增加内部访问器，上层装配代码不变。
- `setZero()` 将矩阵重置回未 finalize 的空状态，可重新装配。

## LinearSolver：抽象求解器接口

### 设计思路

- **接口与后端分离**：`LinearSolver` 只声明 `virtual Vector solve(const SparseMatrix& A, const Vector& b) = 0`。三个 Eigen 实现类（`EigenBiCGSTABSolver`、`EigenCGSolver`、`EigenSparseLUSolver`）定义在 `.cpp` 的匿名命名空间中，对外不可见；用户只能通过工厂函数创建：

  ```cpp
  auto solver = fvm::math::createEigenBiCGSTAB(cfg);  // std::unique_ptr<LinearSolver>
  ```

  新增后端时只需添加新的子类 + 工厂函数，不触动任何调用点。

- **统一配置**：`SolverConfig { tolerance, maxIterations, verbose }`，语义为**相对残差**判据 `|Ax−b|/|b| ≤ tolerance`。求解后可通过 `lastIterations()` / `lastResidual()` 查询实际迭代数与残差。
- **失败即异常**：所有实现在不收敛或分解失败时抛 `std::runtime_error`，调用方无需检查错误码。

### 三个实现

| 实现 | 算法 | 适用场景 | 备注 |
|------|------|---------|------|
| `createEigenBiCGSTAB` | 双共轭梯度稳定法 + 对角（Jacobi）预条件 | 非对称系统（含对流的输运方程） | 见下方容差换算说明 |
| `createEigenCG` | 共轭梯度法 + 对角预条件 | 对称正定系统（纯扩散） | 非对称矩阵上行为未定义 |
| `createEigenSparseLU` | 稀疏 LU 直接分解 | 小问题、验证用、病态系统 | `lastIterations`=1，残差为显式计算的 `‖Ax−b‖/‖b‖` |

### 关键实现细节：BiCGSTAB 的容差换算

Eigen 5.x 的 `BiCGSTAB` 内部以**绝对残差**作为停机判据，且其收敛标志存在绝对/相对量纲不一致的问题。因此 `EigenBiCGSTABSolver::solve` 中做了两步处理：

1. **容差缩放**：设 `rhsNorm = ‖b‖`，传给 Eigen 的停机容差为

   ```
   tol_eigen = tolerance · ‖b‖    （‖b‖ > 0 时）
   ```

   使 Eigen 的绝对判据等价于本项目约定的相对判据 `|Ax−b|/|b| ≤ tolerance`。

2. **自行检查收敛**：求解后取 `solver.error()`（相对残差估计）存入 `lastResidual_`，若超过 `config_.tolerance` 则抛出"did not converge"异常——**不依赖 `solver.info()` 判断 BiCGSTAB 是否收敛**（`info()` 仅用于检查分解失败与数值崩溃）。

## 依赖关系

```
math → core（Scalar/Index/Vector 别名）
     → Eigen（Sparse、IterativeLinearSolvers、SparseLU）
```

math 不知道网格、场、离散化的存在：`solve` 的输入只有矩阵与向量。
