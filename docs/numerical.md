# numerical 模块说明

`fvm::numerical` 命名空间，实现稳态标量输运方程的有限体积离散，将对流、扩散、源项和边界条件装配为线性系统 $A\phi = b$；并在其之上实现稳态不可压缩 Navier-Stokes 方程的 SIMPLE 算法。这是求解器的数值核心。

控制方程：

$$\nabla \cdot (\rho\, \mathbf{u}\, \phi) = \nabla \cdot (\gamma \nabla \phi) + S(\phi), \qquad S(\phi) = S_c + S_p\, \phi \quad \text{（单位体积源项）}$$

对单元 $P$ 做有限体积积分后得到离散形式（$\sum_f$ 遍历 $P$ 的四个面）：

$$\underbrace{\sum_f F_f\, \phi_f}_{\text{对流通量}} = \underbrace{\sum_f D_f\, (\phi_N - \phi_P)}_{\text{扩散通量}} + \underbrace{(S_c + S_p\, \phi_P)\, V_P}_{\text{源项}}$$

## 文件结构

| 文件 | 内容 |
|------|------|
| `include/BoundaryCondition.h` | `BCType`、`BoundaryCondition`、`BoundaryField`（header-only） |
| `include/Diffusion.h` + `src/Diffusion.cpp` | `assembleDiffusion`：扩散项装配 |
| `include/Convection.h` + `src/Convection.cpp` | `ConvectionScheme`、`assembleConvection`：对流项装配 |
| `include/TransportEquation.h` + `src/TransportEquation.cpp` | `EquationSystem`、`assembleTransport`：完整方程装配 |
| `include/Simple.h` + `src/Simple.cpp` | `SimpleConfig`、`assembleMomentum`、`solveSimple`：SIMPLE 算法 |

## BoundaryCondition：边界条件

- `BCType { Dirichlet, Neumann }`：Dirichlet 给定边界面值 $\phi_b$；Neumann 给定外法向导数 $g = \mathrm{d}\phi / \mathrm{d}n$。
- `BoundaryField` 持有矩形域**四条边**的边界条件（`std::array<BoundaryCondition, 4>`），边索引与网格面索引约定一致：`East=0, North=1, West=2, South=3`。默认四边均为零通量 Neumann（$g = 0$）。
- 设计要点：边界条件按"边"而非按"单元"存储——同一物理边上所有边界面共享一个条件，与均匀矩形域的设定匹配；`set/get` 对越界 side 抛异常。

## Diffusion：扩散项装配

装配算子 $-\nabla \cdot (\gamma \nabla \phi)$。

### 内部面：中心差分

面扩散系数取两侧单元值的**算术平均** $\gamma_f = (\gamma_P + \gamma_N) / 2$，梯度用中心差分 $(\phi_N - \phi_P) / d_{PN}$，扩散传导率：

$$D_f = \frac{\gamma_f\, S_f}{d_{PN}}$$

对每个内部面，向矩阵插入对称的四项：

```
A(P,P) += D_f      A(P,N) −= D_f
A(N,N) += D_f      A(N,P) −= D_f
```

由于只取东、北两个方向处理内部面（见下文"装配循环结构"），每对相邻单元恰好贡献一次。该格式为二阶精度 $O(h^2)$，且矩阵对称。

### 边界面

- **Dirichlet**（$\phi = \phi_b$）：用单元中心到面中心的半距 $d_{Pb}$ 构造边界传导率

  $$D_b = \frac{\gamma_P\, S_f}{d_{Pb}}$$

  ```
  A(P,P) += D_b,   b(P) += D_b·φ_b
  ```

  即将通量 $D_b\, (\phi_P - \phi_b)$ 的未知部分留在矩阵、已知部分移入右端项。

- **Neumann**（$\mathrm{d}\phi / \mathrm{d}n = g$，外法向）：通量已知，整体移入右端项，不触碰矩阵：

  $$b_P \mathrel{+}= \gamma_P\, g\, S_f$$

### 矩阵性质

仅有 Dirichlet/Neumann 边界时矩阵对称半正定；存在至少一条 Dirichlet 边时为对称正定（SPD），可用 CG 求解。

## Convection：对流项装配

装配算子 $\nabla \cdot (\rho\, \mathbf{u}\, \phi)$，面质量通量定义为（正号表示沿 $P$ 的外法向流出）：

$$F_f = \rho\, (\mathbf{u}_f \cdot \mathbf{n})\, S_f$$

- **内部面**速度：$\mathbf{u}_f$ 取相邻两单元中心速度的算术平均（分量各自平均后与法向点乘）。
- **边界面**速度：直接取单元中心速度 $\mathbf{u}_P$ 与法向点乘（无邻居可平均）。

### 面值的两种插值格式

`ConvectionScheme` 枚举选择面上面值 $\phi_f$ 的取法：

**Upwind（一阶迎风）**——面值取上游单元的值：

```
F_f > 0（流向 P→N）：φ_f = φ_P  →  A(P,P) += F_f,  A(N,P) −= F_f
F_f < 0（流向 N→P）：φ_f = φ_N  →  A(N,N) −= F_f,  A(P,N) += F_f
```

特点：无条件有界、保持对角占优，精度 $O(h)$。是默认的稳健选择。

**Central（二阶中心）**——面值取算术平均 $\phi_f = (\phi_P + \phi_N)/2$，记 $h = F_f/2$：

```
A(P,P) += h    A(P,N) += h
A(N,P) −= h    A(N,N) −= h
```

特点：精度 $O(h^2)$，但在网格 Peclet 数较大时可能产生非物理振荡（无有界性保证）。

### 边界面处理

边界面通量 $F_b$ 按流向分三种情况：

| 情况 | 处理 | 含义 |
|------|------|------|
| $F_b > 0$（出流） | `A(P,P) += F_b` | 面值由单元值外插（迎风），通量未知留在矩阵 |
| $F_b < 0$（入流）+ Dirichlet | `b(P) −= F_b·φ_b` | 入流携带已知值，通量完全已知移入右端项（注意 $F_b < 0$，故为减量） |
| $F_b < 0$（入流）+ Neumann | `A(P,P) += F_b` | 假设法向零梯度，面值取 $\phi_P$ |

## TransportEquation：完整方程装配

### EquationSystem

```cpp
struct EquationSystem {
    SparseMatrix A;   // 未 finalize
    Vector b;         // 已置零
};
```

返回的矩阵**未 finalize**，允许调用方继续插入（如自定义源项），求解前需调用 `A.finalize()`。

### assembleTransport 的装配流程

1. 构造 `EquationSystem`（`b` 置零，`A` 为空的三元组状态）；
2. 调用 `assembleDiffusion` 累加扩散贡献；
3. 调用 `assembleConvection` 累加对流贡献（同一 `bc` 作用于两个算子）；
4. 源项按 **Patankar 线性化**处理 $S(\phi) = S_c + S_p\, \phi$（单位体积）：
   - 常数部分：$b_c \mathrel{+}= S_c(c)\, V$；
   - 线性部分：$A_{cc} \mathrel{+}= -S_p(c)\, V$，**要求 $S_p \le 0$**（隐式处理增强对角占优；$S_p > 0$ 会破坏对角占优导致迭代求解发散，故逐单元检查并抛 `std::invalid_argument`）。

### 装配的累加语义

`assembleDiffusion` / `assembleConvection` 均**向既有的 (A, b) 累加**而不清零——这是刻意设计：`assembleTransport` 借此组合多个算子，用户也可先装配标准算子再叠加自定义项。若需全新系统，调用方须自行清零（`A.setZero()`、`b.setZero()`）。

### 装配循环结构（Diffusion/Convection 共用）

```
for P in cells:
    for face in {0,1,2,3}:
        N = mesh.neighbor(P, face)
        if N 存在:          # 内部面
            只处理 East/North，避免每个面被两侧单元重复计数
        else:               # 边界面
            立即处理（West/South 边界面不会被任何其他单元访问到，
                       因此不存在重复计数问题）
```

该结构保证每个内部面恰好装配一次、每个边界面恰好装配一次，时间复杂度 $O(n_{\text{cells}})$。

## Simple：稳态不可压缩 Navier-Stokes（SIMPLE 算法）

控制方程（$\rho$、$\mu$ 为常数）：

$$\nabla \cdot (\rho\, \mathbf{u}\, \mathbf{u}) = -\nabla p + \nabla \cdot (\mu \nabla \mathbf{u}), \qquad \nabla \cdot \mathbf{u} = 0$$

采用**同位网格**（速度、压力均存单元中心）+ **Rhie-Chow 插值**抑制压力棋盘格振荡。`solveSimple` 为主入口，`velocity`/`pressure` 以 in-out 方式传入（初值 → 收敛解）。

### 公开 API

| 类型 | 说明 |
|------|------|
| `SimpleConfig` | `maxIterations`、`tolerance`（缩放连续性残差收敛判据）、`relaxationU`/`relaxationP`（动量/压力亚松弛因子）、`scheme`（对流格式）、`solverConfig`（内层线性求解器配置）、`verbose` |
| `SimpleResiduals` | 每次迭代的残差快照：`continuity`（缩放质量不平衡）、`u`/`v`（速度最大相对修正量）、`pressure`（最大 $\lvert \alpha_p p' \rvert$） |
| `SimpleResult` | `converged`、`iterations`、`history`（逐迭代残差） |
| `MomentumAssembly` | 动量装配结果：`system`（含压力源与亚松弛）、`diag`（松弛后对角元 $a_P$）、`rhsNoPressure`（不含压力源的右端项，供 Rhie-Chow 使用） |

### assembleMomentum：动量方程装配

单个动量分量（`component` 0=u / 1=v）的方程复用 `assembleDiffusion`（$\gamma = \mu$）与 `assembleConvection`，再累加两项：

**压力梯度源**——用 Gauss 定理形式（而非纯中心差分），使 Dirichlet 压力边界值参与梯度计算（压力驱动流的必要条件）：

$$b_P \mathrel{-}= \sum_f p_f\, n_{f,d}\, S_f$$

内部面 $p_f$ 取算术平均（退化为中心差分）；边界面 Dirichlet 取给定值、Neumann 取 $p_P$。

**Patankar 亚松弛**（$\alpha$ = `relaxationU`）：

```
A(P,P) /= α
b(P)   += (1-α)/α · a_P⁽⁰⁾ · φ_old(P)
```

其中 $a_P^{(0)}$ 为松弛前对角元。实现上通过"复制未冻结矩阵 → finalize 副本 → 从 `native()` 读对角"获得 $a_P^{(0)}$，再向原矩阵插入对角增量。注意 `velocity` 同时承担对流速度场与 $\phi_{old}$ 两个角色（SIMPLE 的标准用法）。

### Rhie-Chow 面通量

解出动量预测值 $\mathbf{u}^*$ 后，定义剔除压力贡献的速度（`computeUHat`，利用 `SparseMatrix::native()` 做 Eigen 稀疏运算）：

$$\hat{u}_P = \frac{b^{np}_P - \sum_{N \ne P} A_{PN}\, u^*_N}{a_P}, \qquad d_P = \frac{V_P}{a_P}$$

内部面质量通量（x 向面用 u 方程的 $a_P$，y 向面用 v 方程的）：

$$F_f = \rho\, S_f \left[ \overline{\hat{u}}_f \cdot \mathbf{n} - \bar{d}_f\, \frac{p_N - p_P}{\delta_{PN}} \right]$$

边界面通量 $F_b$ 直接由边界速度给出（Dirichlet 取给定值，Neumann 取 $u^*_P$），保证壁面无穿透。

### 压力修正方程

把通量修正 $F_f = F^*_f - C_f (p'_N - p'_P)$（$C_f = \rho \bar{d}_f S_f / \delta$）代入单元连续性 $\sum_f F_f = 0$，整理得

$$A\, p' = -m, \qquad m_P = \sum_f F^*_f \ \text{（预测净流出量）}$$

> **符号约定**：右端项是质量不平衡的**负值**。若误写为 $+m$，压力修正会形成正反馈，速度场迅速发散。

装配规则：

- 内部面：`A(P,P) += C_f, A(N,N) += C_f, A(P,N) -= C_f, A(N,P) -= C_f`（SPD Laplace 型）；
- Dirichlet 压力边（$p'_b = 0$）：`A(P,P) += C_b`，$C_b = \rho\, d_P S_f / d_{Pb}$；
- Neumann 压力边：无矩阵贡献（$F_b$ 仅进入 $m_P$）。

**奇异性处理——参考单元消元**：四条边全为 Neumann 时系数矩阵奇异（零空间为常向量）。此时消去 0 号单元（$p'_0 = 0$，其连续性方程因 $\sum_P m_P = 0$ 而冗余），得到 $n-1$ 阶 SPD 系统；存在 Dirichlet 压力边时系统本已正定，保留全部单元。p' 方程用 CG 求解，动量方程用 BiCGSTAB（对流使矩阵非对称）。

### 修正与收敛判据

$$u_P \leftarrow u^*_P - d_P\, (\nabla p')_{P,x}, \qquad v_P \leftarrow v^*_P - d_P\, (\nabla p')_{P,y}, \qquad p \leftarrow p + \alpha_p\, p'$$

$p'$ 的梯度同样用 Gauss 形式，但 Dirichlet 边界的 $p'_b$ 恒取 0（压力已被固定，修正为零）。

收敛判据（三者同时小于 `SimpleConfig::tolerance`）：

- 连续性：$\max_P |m_P| / F_{ref}$，$F_{ref} = \rho \cdot \max\lVert\mathbf{u}\rVert \cdot (dx+dy)/2$；
- 速度：$\max_P |\Delta u| / \max\lVert\mathbf{u}\rVert$（$v$ 同理）。

## 依赖关系

```
numerical → core（Mesh/Field/Types）
          → math（SparseMatrix, Vector；solveSimple 另用 LinearSolver）
```

transport 部分不依赖线性求解器与 io——它只负责装配，求解与输出由 app 层组织；`solveSimple` 是例外，它内部持有 BiCGSTAB（动量）与 CG（压力修正）求解器以驱动整个迭代循环。
