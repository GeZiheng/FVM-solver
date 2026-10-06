# numerical 模块说明

`fvm::numerical` 命名空间，实现标量输运方程的有限体积离散，将对流、扩散、源项和边界条件装配为线性系统 $A\phi = b$；并在此之上实现不可压缩 Navier-Stokes 方程：稳态用 SIMPLE，瞬态用 θ 时间格式 + PISO。这是求解器的数值核心。

控制方程（标量输运；动量方程见 [Momentum](#momentum动量方程装配与预测子)）：

$$\nabla \cdot (\rho\, \mathbf{u}\, \phi) = \nabla \cdot (\gamma \nabla \phi) + S(\phi), \qquad S(\phi) = S_c + S_p\, \phi \quad \text{（单位体积源项）}$$

对单元 $P$ 做有限体积积分后得到离散形式（$\sum_f$ 遍历 $P$ 的四个面）：

$$\underbrace{\sum_f F_f\, \phi_f}_{\text{对流通量}} = \underbrace{\sum_f D_f\, (\phi_N - \phi_P)}_{\text{扩散通量}} + \underbrace{(S_c + S_p\, \phi_P)\, V_P}_{\text{源项}}$$

## BoundaryCondition：边界条件

- `BCType { Dirichlet, Neumann }`：Dirichlet 给定边界面值 $\phi_b$；Neumann 给定外法向导数 $g = \mathrm{d}\phi / \mathrm{d}n$。
- `BoundaryField` 持有矩形域**四条边**的边界条件（`std::array<BoundaryCondition, 4>`），边索引与网格面索引约定一致：`East=0, North=1, West=2, South=3`。默认四边均为零通量 Neumann（$g = 0$）。
- 设计要点：边界条件按"边"而非按"单元"存储——同一物理边上所有边界面共享一个条件，与均匀矩形域的设定匹配；`set/get` 对越界 side 抛异常。

## GridOperators：公共离散算子

存放跨方程复用的离散算子。目前提供单元中心 Gauss 梯度（`cellGradient`），被动量方程的压力梯度源与压力修正的速度重建共同使用：

$$(\nabla \phi)_{P,d} = \frac{1}{V_P} \sum_f \phi_f\, n_{f,d}\, S_f$$

内部面 $\phi_f$ 取算术平均（均匀网格上退化为中心差分）；边界面 Dirichlet 取给定值、Neumann 取 $\phi_P$（零法向梯度）。

设计意图：把这类"任何 solver 都可能要用的网格算子"集中为公共接口（`include/`），后续如 `divergence` 等直接加入即可被其他 solver 复用。

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

## Convection：通量构造与对流项装配

装配算子 $\nabla \cdot (F\, \phi)$。OpenFOAM 风格的设计：**面质量通量 $F$ 由调用方以 `FaceFluxField`（见 [core.md](core.md)）显式提供**，装配器不再自行插值速度——对流、连续性方程与标量输运由此共用同一套通量。

### 通量构造函数

两个自由函数把单元中心速度场转为面通量场（$F_f = \rho\, (\mathbf{u}_f \cdot \mathbf{n})\, S_f$）：

| 函数 | 内部面 | 边界面 | 适用场景 |
|------|--------|--------|----------|
| `interpolateCellVelocityFlux(mesh, velocity, rho)` | $\mathbf{u}_f$ 算术平均 | 相邻单元中心速度 | "给定速度场"的独立输运问题（无速度 BC 可用） |
| `computeMassFlux(mesh, velocity, rho, bcU, bcV, flux)` | $\mathbf{u}_f$ 算术平均 | 对应分量的 Dirichlet BC 值，无则取单元速度（零梯度） | 有速度边界条件时（`solveSimple`/`solvePiso` 入口即用它初始化通量，对应 OpenFOAM 的 `createPhi.H`）；**原地填充**（`FaceFluxField` 不可赋值） |
| `checkFluxCompatibility(flux, relTol)` | — | 检查边界净流出量是否为零（相对容差 `relTol * Σ\|F_b\|`，默认 1e-10），不满足则抛 `std::runtime_error` | 纯 Neumann 压力（封闭域）问题的相容性检查（对应 OpenFOAM 的 `adjustPhi`）；`solveSimple`/`solvePiso` 入口在初始化通量后调用 |

返回值/填充结果均为正方向为正的存储约定；`assembleConvection` 读边界面时经 `outwardFlux` 还原外法向出流通量 $F_b$。

### 面值的两种插值格式

`ConvectionScheme` 枚举选择面上面值 $\phi_f$ 的取法（$F_f$ 直接读自通量场，东/北面即 $P \to N$ 方向）：

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

## TransportEquation：标量输运装配

### EquationSystem

```cpp
struct EquationSystem {
    SparseMatrix A;   // 未 finalize
    Vector b;         // 已置零
};
```

返回的矩阵**未 finalize**，允许调用方继续插入（如自定义源项），求解前需调用 `A.finalize()`。

### assembleTransport 的装配流程（稳态）

1. 构造 `EquationSystem`（`b` 置零，`A` 为空的三元组状态）；
2. 调用 `assembleDiffusion` 累加扩散贡献；
3. 调用 `assembleConvection` 累加对流贡献（同一 `bc` 作用于两个算子；通量由调用方传入——可以是 `solveSimple` 输出的守恒通量，也可以是 `interpolateCellVelocityFlux` 构造的一次性通量）；
4. 源项按 **Patankar 线性化**处理 $S(\phi) = S_c + S_p\, \phi$（单位体积）：
   - 常数部分：$b_c \mathrel{+}= S_c(c)\, V$；
   - 线性部分：$A_{cc} \mathrel{+}= -S_p(c)\, V$，**要求 $S_p \le 0$**（隐式处理增强对角占优；$S_p > 0$ 会破坏对角占优导致迭代求解发散，故逐单元检查并抛 `std::invalid_argument`）。

### assembleTransientTransport 的装配流程（瞬态，θ 格式）

设稳态空间算子装配为 $A_{sp}$、右端为 $b_{sp}$（即 `assembleTransport` 的结果），瞬态方程在时间步 $n \to n+1$ 上按 θ 格式离散：

$$\frac{V}{\Delta t}\left(\phi^{n+1} - \phi^n\right) + \theta\, L(\phi^{n+1}) + (1-\theta)\, L(\phi^n) = b_{sp}$$

其中 $L(\phi) = A_{sp}\phi - b_{sp}$ 为对流+扩散空间算子，整理为关于 $\phi^{n+1}$ 的线性系统：

$$\left(\theta\, A_{sp} + \frac{V}{\Delta t} I\right)\phi^{n+1} = \frac{V}{\Delta t}\,\phi^n - (1-\theta)\, A_{sp}\,\phi^n + b_{sp}$$

- $\theta = 1$：Backward Euler（一阶）；$\theta = 1/2$：Crank–Nicolson（二阶）。
- 实现：先按稳态装配一次得到未 finalize 的 $A_{sp}$，复制后 finalize 以求显式项 $A_{sp}\phi^n$；再对原矩阵调用 `SparseMatrix::scale(theta)` 并插入 $\frac{V}{\Delta t}I$ 对角，右端按上式组装。
- 约定：对流所用面通量在整步内**冻结**（标量输运的常规做法）；源项 $(S_c, S_p)$ 视为步内常量。

### 装配的累加语义

`assembleDiffusion` / `assembleConvection` 均**向既有的 (A, b) 累加**而不清零——这是刻意设计：`assembleTransport` 借此组合多个算子，用户也可先装配标准算子再叠加自定义项。若需全新系统，调用方须自行清零（`A.setZero()`、`b.setZero()`）。

## TimeScheme：时间离散

`TimeScheme.h` 定义隐式 θ 族时间格式（header-only）：

```cpp
enum class TimeScheme { Euler, CrankNicolson };   // theta = 1.0 / 0.5
Scalar thetaOf(TimeScheme);

struct TimeTerm {           // dt <= 0 表示稳态
    Scalar rho = 1.0;       // 动量方程中守恒量为 rho·u
    Scalar dt  = 0.0;
    TimeScheme scheme = TimeScheme::Euler;
    bool   active() const;  // dt > 0
    Scalar theta()  const;
};
```

- 标量输运直接传 `TimeScheme`（其 ddt 系数为 $V/\Delta t$，无需密度）；
- 动量方程用 `TimeTerm`（ddt 系数为 $\rho V/\Delta t$）——同一个结构被 `assembleMomentum`/`predictMomentum` 共享，`dt <= 0` 时完全退化为稳态（供 SIMPLE 使用）。

## Momentum：动量方程装配与预测子

不可压缩 Navier-Stokes（$\rho$、$\mu$ 为常数）：

$$\nabla \cdot (\rho\, \mathbf{u}\, \mathbf{u}) = -\nabla p + \nabla \cdot (\mu \nabla \mathbf{u}), \qquad \nabla \cdot \mathbf{u} = 0$$

采用**同位网格**（速度、压力均存单元中心）+ **Rhie-Chow 插值**抑制压力棋盘格振荡。本节的算子被 SIMPLE 与 PISO 共用。

### 公开 API

| 类型 | 说明 |
|------|------|
| `MomentumAssembly` | 单分量动量装配结果：`system`（含压力源与松弛）、`diag`（对角 $a_P$）、`rhsNoPressure`（不含压力源的右端项，供 Rhie-Chow 使用） |
| `MomentumPrediction` | 动量预测结果：`momU`/`momV`（两分量装配）、`uStar`/`vStar`（预测速度）、`uHatU`/`uHatV`（$\hat{u} = H/a_P$）、`dU`/`dV`（$V/a_P$） |

### assembleMomentum：单分量动量装配

复用 `assembleDiffusion`（$\gamma = \mu$）与基于通量场的 `assembleConvection`（对流通量直接取持久通量场，与连续性方程共用同一通量），再累加：

**压力梯度源**——用 [GridOperators](#gridoperators公共离散算子) 的 Gauss 形式（而非纯中心差分），使 Dirichlet 压力边界值参与梯度计算（压力驱动流的必要条件）：

$$b_P \mathrel{-}= \sum_f p_f\, n_{f,d}\, S_f$$

内部面 $p_f$ 取算术平均（退化为中心差分）；边界面 Dirichlet 取给定值、Neumann 取 $p_P$。

**稳态分支（`time.active() == false`）——Patankar 亚松弛**（$\alpha$ = `relaxation`）：

```
A(P,P) /= α
b(P)   += (1-α)/α · a_P⁽⁰⁾ · φ_old(P)
```

其中 $a_P^{(0)}$ 为松弛前对角元。实现上通过"复制未冻结矩阵 → finalize 副本 → 从 `native()` 读对角"获得 $a_P^{(0)}$，再向原矩阵插入对角增量。注意入参 `velocity` 在此仅承担 $\phi_{old}$ 一个角色（对流速度来自通量场）。

**瞬态分支（`time.active() == true`）——θ 格式 ddt 项**（$\theta$ = `time.theta()`，$\rho V/\Delta t$ 为 ddt 对角）：

$$\left(\theta\, A_{sp} + \frac{\rho V}{\Delta t} I\right) u^{n+1} = \frac{\rho V}{\Delta t}\,u^n - (1-\theta)\, A_{sp}\,u^n + b_{sp} - (\nabla p^n)\,V$$

- $A_{sp}$ 为对流+扩散空间算子；压力梯度按 $p^n$ **显式**进入右端（$\theta$ 只作用于空间算子），$b_{sp}$ 以全权重进入右端以保证 $\theta = 1$ 极限回到 Backward Euler。
- 关键：ddt 对角 $\rho V/\Delta t$ 计入 $a_P$，于是 Rhie–Chow 的 $d = V/a_P$ 自动是瞬态一致的（对应 OpenFOAM 的 `rAU = 1/UEqn.A()`）。
- **约束**：瞬态分支要求 `relaxation == 1`（欠松弛是稳态 SIMPLE 的手段，不用于 PISO），否则抛 `std::invalid_argument`。

### predictMomentum：预测子

装配并求解 u/v 两个分量，再计算 Rhie–Chow 数据（对应 OpenFOAM `UEqn.H`）。求解出 $\mathbf{u}^*$ 后，定义剔除压力贡献的速度（内部 `computeUHat`，利用 `SparseMatrix::native()` 做 Eigen 稀疏运算）：

$$\hat{u}_P = \frac{b^{np}_P - \sum_{N \ne P} A_{PN}\, u^*_N}{a_P}, \qquad d_P = \frac{V_P}{a_P}$$

其中 $b^{np}$ 即 `rhsNoPressure`。返回 `MomentumPrediction`。

## Pressure：绝对压力方程与修正子

`correctPressure` 执行一次压力修正（对应 OpenFOAM-14 的 `applications/modules/incompressibleFluid/correctPressure.C`，即旧版 `pEqn.H`）：写入**无压力**预测通量 `phiHbyA` → 装配并求解**绝对压力** $p$ → 就地重建面通量、单元中心速度与压力。返回 `CorrectorResult`。

### Rhie–Chow 面通量与预测通量

压力方程使用的面通量（x 向面用 u 方程的 $a_P$，y 向面用 v 方程的；$C_f = \rho \bar{d}_f S_f / \delta_{PN}$）：

$$F_f = \rho\, S_f\, \overline{\hat{u}}_f \cdot \mathbf{n} \; - \; C_f\,(p_N - p_P)$$

边界面与内部面**同源**（$d_{Pb}$ 为单元中心到面的距离，$\mathbf{n}$ 为外法向，$C_b = \rho\, d_P S_f / d_{Pb}$）：

$$F_b = \rho\, S_f\, (\hat{u}\cdot\mathbf{n})_b - C_b\,(p_b - p_P) \ \text{（Dirichlet 压力边）}, \qquad F_b = \rho\, S_f\, (\hat{u}\cdot\mathbf{n})_b \ \text{（Neumann 压力边）}$$

Neumann 压力边法向梯度为零，故不加压力项。速度分量 $(\hat{u}\cdot\mathbf{n})_b$ 在 Dirichlet 速度边取给定值，否则取 $\hat{u}_P$（零梯度），保证壁面无穿透。

装配时先把**无压力部分** $F^{\hat u}_f \equiv \rho S_f\, \overline{\hat{u}}_f\cdot\mathbf{n}$（OpenFOAM 的 `phiHbyA = fvc::flux(HbyA)`）逐面写入持久通量场，压力项整体由方程未知量 $p$ 承担。这份 $F^{\hat u}$ 同时是右端来源、封闭域 `adjustPhi` 的操作对象、以及求解后通量重建的基准——OpenFOAM 里它是局部量 `phiHbyA`，本项目因 `FaceFluxField` 不可赋值而复用持久通量场作缓冲，最后就地修正成守恒通量。

> **不能**用 $u^*_P$ 代替 $\hat{u}_P$：$u^*$ 含旧压力的*单元中心*梯度，而这里要的是**面法向** RC 通量；边界通量必须与内部面同源（Dirichlet 速度边取给定值，否则取 $\hat u_P$）。旧版曾因边界面误用 $u^*$ 而与右端不自洽，每次修正重复计入整段边界压降，修正子迭代因此发散。

### 压力方程（绝对压力形式）

把 $F_f = F^{\hat u}_f - C_f (p_N - p_P)$ 代入单元连续性 $\sum_f F_f = 0$，得到关于**新压力**（绝对压力）的方程

$$A\, p = b, \qquad b_P = -\sum_f \text{outward}\big(F^{\hat u}_f\big) \; + \sum_{\text{Dirichlet 压力边}} C_b\, p_b$$

> **符号约定**：右端第一项是预测通量净流出的**负值**。若符号写反，压力方程会形成正反馈，速度场迅速发散。

装配规则：

- 内部面：`A(P,P) += C_f, A(N,N) += C_f, A(P,N) -= C_f, A(N,P) -= C_f`（SPD Laplace 型）；
- Dirichlet 压力边：`A(P,P) += C_b`，边界值 $p_b$ **直接进右端**（`b(P) += C_b·p_b`）——不再需要 $p'$ 的镜像边界条件；
- Neumann 压力边：无矩阵贡献，只有 $F^{\hat u}_b$ 进入右端。

**与增量形式（$p'$）的等价性**：把 $p$ 写作 $p_{\text{old}} + p'$，则 $A p' = b - A p_{\text{old}} = -m$，其中 $m_P$ 正是旧形式的预测净流出量（$-m + A p_{\text{old}} \equiv b$ 是恒等式）。两种写法组装同一个矩阵、解出同一个修正量；绝对形式的收益是右端直接就是 `div(phiHbyA)`，不必形成"旧压力梯度 − 矩阵系数"的相消差，右端的量级（因此 CG 相对残差容差的含义）也由此明确。

**奇异性处理——参考单元消元**：四条边全为 Neumann 时矩阵奇异（零空间为常向量）。此时消去 0 号单元并把它钉在**旧值** $p_0 = p_{\text{old},0}$（等价于旧形式的 $p'_0 = 0$，压力水平不漂移），被消掉的第 0 列对邻居行的贡献移到右端（2D 最多两个邻居），得到 $n-1$ 阶 SPD 系统；存在 Dirichlet 压力边时系统本已正定，保留全部单元。压力方程用 CG 求解，动量方程用 BiCGSTAB（对流使矩阵非对称）。

**封闭域相容性（OpenFOAM adjustPhi）**：纯 Neumann 压力时压力项改变不了边界净流出量——内部面两两抵消，Neumann 边界又无压力项——因此 $F^{\hat u}$ 的边界净通量必须为零。不满足时方程不相容，参考单元消元会静默违反被消元单元（0 号）的连续性并累积误差。装配时先把残余净通量按"法向速度非 Dirichlet"的边界面积均摊回这些边界面，**同一调整同步进入右端**；存在 Dirichlet 压力边（开放域）时不做调整。入口处的 `checkFluxCompatibility` 保留为防御性检查。

### 修正与收敛判据

**通量重建**（对应 OpenFOAM `phi = phiHbyA - pEqn.flux()`）：在已写入的 $F^{\hat u}$ 上就地减去压力项

$$F_f \leftarrow F^{\hat u}_f - C_f\,(p_N - p_P) \quad \text{（内部面）}, \qquad F_b \leftarrow F^{\hat u}_b - C_b\,(p_b - p_P) \quad \text{（Dirichlet 压力边，外法向）}$$

Neumann 压力边的边界通量保持 $F^{\hat u}_b$。由于右端与重建用的是**同一份**（封闭域下还经过 `adjustPhi` 调整的）$F^{\hat u}$，$A p = b$ 精确等价于"修正后每单元净流出量为零"，该步之后通量场在**压力求解器精度内严格离散守恒**；下一轮/下一步动量装配直接使用该通量。

**压力与速度重建**：压力场按松弛更新 $p \leftarrow p + \alpha_p\,(p_{\text{sol}} - p_{\text{old}})$，而速度和面通量用**未松弛**的解 $p_{\text{sol}}$ 重建：

$$\mathbf{u}_P \leftarrow \hat{\mathbf{u}}_P - d_P\, (\nabla p_{\text{sol}})_P$$

梯度用 Gauss 形式，Dirichlet 压力边界直接取 $p_b$。稳态 SIMPLE 下这恒等于旧的增量写法 $u^*_P - d_P(\nabla p')_P$：因为 $\hat u = u^* + d\nabla p_{\text{old}}$，而 $\nabla p_{\text{sol}} - \nabla p_{\text{old}}$ 在 Dirichlet 边上取零、在 Neumann 边上取零梯度，正是原来 $p'$ 的镜像边界。PISO 下 $\alpha_p = 1$，$p_{\text{sol}}$ 就是存储压力，因此多个修正子自然累积。这条重建要求**每个附加修正子前刷新 $\hat u$**（$\hat u = H(u)/a_P$）：若冻结 $\hat u$，第二个修正子的右端会退化成上一次修正留下的求解器残差，修正子循环在第一次之后就到达不动点。为什么第一个修正子与 SIMPLE 的增量写法等价，见"Piso：瞬态驱动"的"PISO 里的三个速度"。

> **松弛次序（与 OpenFOAM 的差异）**：本项目速度/通量用未松弛解、只有存储压力吃 $\alpha_p$（Patankar 写法）；OpenFOAM-14 的 `correctPressure()` 先 `p.relax()` 再算 `U = HbyA - rAU·grad(p)`，速度修正也被 $\alpha_p$ 阻尼（通量仍在 relax 之前取，故同样守恒）。两者都是合法 SIMPLE 变体，本项目保留前者以维持既有测试标定；PISO（$\alpha_p = 1$）下二者无差别。

> **顺序要求已消解**：早期版本要求"压力先整体更新、再重建速度"，因为那时的速度重建读的是就地更新的压力场梯度 $\nabla p$——若与压力更新写在同一循环里，重建只能看到"前面单元已更新、后面仍是旧值"的半成品场，梯度会错到量级失真（实测某单元 $(\nabla p)_x = -58.4$，完整更新后为 $+2.42$）。绝对压力形式下速度重建读的是只读的 $p_{\text{sol}}$ 场，该顺序约束在结构上不再存在。修正后与 OpenFOAM-14 单步结果逐格对比：$\max|\Delta u| \approx 5\times10^{-11}$、$\max|\Delta v| \approx 1\times10^{-11}$、$\max|\Delta p| \approx 8\times10^{-9}$（均为求解器容差量级）。

## Simple：稳态 SIMPLE 驱动

`solveSimple` 为主入口：`velocity`/`pressure` 以 in-out 方式传入（初值 → 收敛解），`flux`（`FaceFluxField&`）为出参。入口用 `computeMassFlux` 初始化通量；纯 Neumann 压力时随即 `checkFluxCompatibility`。迭代体只做三件事：

1. `predictMomentum(..., TimeTerm{}, momSolver)`——稳态动量预测（`dt = 0`，走 Patankar 松弛分支）；
2. `correctPressure(..., relaxationP, ...)`——单次压力修正（解绝对压力；压力场松弛 $\alpha_p$，速度/通量用未松弛解重建）；
3. 残差与收敛判断。

收敛判据（三者同时小于 `SimpleConfig::tolerance`）：

- 连续性：$\max_P |b_P - (A p_{\text{old}})_P| / F_{ref}$（压力方程在旧压力处的初始残差，即该次修正要消除的质量不平衡；与旧写法的 $\max_P|m_P|$ 等价。纯 Neumann 时被消元的 0 号行按冗余关系 $-\sum$ 还原后一起参与取最大），$F_{ref} = \rho \cdot \max\lVert\mathbf{u}\rVert \cdot (dx+dy)/2$；
- 速度：$\max_P |\Delta u| / \max\lVert\mathbf{u}\rVert$（$v$ 同理）。

| 类型 | 说明 |
|------|------|
| `SimpleConfig` | `maxIterations`、`tolerance`、`relaxationU`/`relaxationP`、`scheme`、`solverConfig`、`verbose` |
| `SimpleResiduals` | 每次迭代的残差快照：`continuity`、`u`/`v`、`pressure` |
| `SimpleResult` | `converged`、`iterations`、`history` |

## Piso：瞬态驱动

`solvePiso` 复用共享的 `predictMomentum` / `correctPressure`，每个时间步执行**一次动量预测 + `nCorrectors` 次压力修正**，不使用欠松弛（对应 OpenFOAM `PISO`）：

1. 入口 `computeMassFlux` 初始化持久通量（纯 Neumann 压力时 `checkFluxCompatibility`）；
2. 每个时间步：
   - `predictMomentum(..., 1.0, time, momSolver)`，`time = TimeTerm{rho, dt, timeScheme}`（θ 格式 ddt）；它同时给出 $u^{*}$ 与 $\hat u_1 = H(u^{*})/a_P$；
   - 第一个修正子：`correctPressure(..., 1.0, ...)`——解绝对压力；$\alpha_p = 1$ 时存储压力就是解；
   - 之后每个修正子：**先用当前速度重算 $\hat u = H(u)/a_P$**（`refreshUHat`，等价于 OpenFOAM 每个修正子开头的 `HbyA = rAU*UEqn.H()`），再调用 `correctPressure`；
3. 持久通量跨时间步复用，保持守恒。

| 类型 | 说明 |
|------|------|
| `PisoConfig` | `dt`、`nSteps`、`nCorrectors`、`timeScheme`、`scheme`、`solverConfig`、`verbose` |
| `PisoStepInfo` | 每步诊断：`time`、`continuity`（修正后真实连续性误差）、`maxSpeed` |
| `PisoResult` | `steps`、`history` |

### PISO 里的三个速度

多修正子的正确性依赖三者严格区分（OpenFOAM 里 $u^{*}$ 与 $u$ 是同一个 `U` 场被就地覆盖，本实现拆成独立变量，因此刷新步骤必须显式写出）：

| 量 | 代码 | 定义 | 更新时机 | 作用 |
|---|---|---|---|---|
| 预测速度 $u^{*}$ | `pred.uStar` | 解动量方程 $A u = b_{np} - V\nabla p^{\text{old}}$ | 每个时间步一次 | 给 $\hat u_1$ 提供求值点（$\hat u_1 = u^{*} + d\nabla p^{\text{old}}$）；修正子里不再直接使用 |
| 无压力速度 $\hat u$（HbyA） | `pred.uHatU/uHatV` | $\hat u = H(u)/a_P$ | 步初一次（用 $u^{*}$），之后每个附加修正子前用当前 $u$ 重算 | 组装无压力预测通量 $F^{\hat u} = \rho S\, \hat u_f$，即压力方程右端 |
| 修正速度 $u$ | `velocity` | $u = \hat u - d\nabla p_{\text{sol}}$（未松弛解） | 每个修正子末尾 | 本步输出、下一步初值、重算 $\hat u$ 的输入 |

索引 $k$ 为修正子编号（$k \ge 1$）：

$$\hat u_1=\frac{H(u^{*})}{a_P},\quad \hat u_{k+1}=\frac{H(u^{(k)})}{a_P},\qquad
F^{\hat u}_k=\rho S\, \hat u_{k,f}\ \Rightarrow\ b_k\ \Rightarrow\ p^{(k)}$$

$$u^{(k)}=\hat u_k-d\,\nabla p^{(k)} \qquad (\alpha_p = 1)$$

绝对压力形式下，第 $k$ 个修正子的右端**不显含** $p^{(k-1)}$（纯 Neumann 时只通过参考钉值 $p_0 = p^{(k-1)}_0$ 进入），每个修正子都是对当前 $\hat u_k$ 重新解一次绝对压力；这与"每次都要用上一步的压力重算 $m$"的增量写法在代数上等价。

**为什么第一个修正子与 SIMPLE 的增量写法等价**：动量方程按"对角 + 非对角"拆开并移项

$$a_P u_P + \sum_N A_{PN}u_N = b_{np,P} - V(\nabla p)_P
\quad\Longleftrightarrow\quad
u_P = \underbrace{\frac{b_{np,P}-\sum_N A_{PN}u_N}{a_P}}_{\hat u_P} - d_P\,(\nabla p)_P,$$

也就是说 $u=\hat u-d\nabla p$ 只是动量方程的**移项**（代码里 $b_{np}$ 即 `rhsNoPressure`：含 ddt 与松弛源，不含压力源），而它成立的**前提是 $u$ 恰为该 $\nabla p$ 下的动量解**。$u^{*}$ 正是 $p^{(0)}$ 的动量解，所以

$$\hat u_1 = u^{*} + d\,\nabla p^{(0)}
\;\Longrightarrow\;
u^{(1)} = \hat u_1 - d\nabla p^{(1)} = u^{*} - d\,\nabla p'^{(1)} .$$

对 $k\ge2$ 这个前提不再成立：$u^{(k-1)}$ 不是 $p^{(k-1)}$ 的动量解，两者相差残差 $R=-\sum_N a_N\big(u^{(k-1)}_N-u^{(k-2)}_N\big)$（即被冻结的非对角耦合），所以必须用当前的 $\hat u_k$ 重新解绝对压力并以 $u^{(k)}=\hat u_k-d\nabla p^{(k)}$ 重建速度——这正是"每个附加修正子前刷新 $\hat u$"的原因，也是 $k\ge2$ 的路径与"只做一次增量修正再叠加"不同的根源。

> **状态（2026-10，已修复并通过测试）**：PISO 已实现并验收（`tests/test_piso.cpp` 已启用在 CMake 中；全量 50 用例 / 3271 条断言通过）。排查共修复**四处**缺陷——`correctPressure` 内三处（边界面预测通量未与内部面同源、封闭域边界净通量不平衡、压力更新与速度重建的顺序）加 `solvePiso` 驱动循环一处（**未在每个附加修正子前用当前速度重算 $\hat u$**）。标量瞬态路径（`assembleTransientTransport`，`tests/test_transient.cpp`）已验证时间阶数正确。四条不变量与 OpenFOAM-14 的定量对照见下文"PISO 与 OpenFOAM 的对照结论"。
>
> **2026-10-07 重构**：压力方程由 $p'$ 增量形式改为 OpenFOAM 式的**绝对压力形式**（右端 `div(phiHbyA)`、Dirichlet 压力值直接进右端；`bcPrime` 与 `cumulativeVelocityCorrection` 已删除）。两者代数等价，重构后全量测试仍为 50 用例 / 3271 断言全绿，单步指纹与 OpenFOAM 一致（$p(0,0)$ 差 $\sim2\times10^{-11}$）。

## 与 OpenFOAM 实现的对比

### PISO 与 OpenFOAM 的对照结论（2026-10）

本项目的 PISO 与 OpenFOAM-14 做过逐单元/逐面、单步与多步的对照（参照算例 `~/OpenFOAM/gzh1057-14/run/pisoPoiseuille`：16×16、$\nu=1$、$\Delta t=0.02$、进出口 `fixedValue p`、进出口 U 零梯度、上下壁 no-slip、迎风、Euler、纯 PISO）。**完整排查过程、已排除假设清单与踩坑**见 `.agents/skills/openfoam-crosscheck/`。

**必须维持的四条不变量**（对应上文的"Rhie–Chow 面通量与预测通量""封闭域相容性""修正与收敛判据"与下文 Key Design Decisions）：

1. 边界面预测通量 $F^{\hat u}_b$ 与内部面**同源**（都取 $\hat u$ 的面值：Dirichlet 速度边取给定值，否则零梯度），不能用 $u^{*}$ 的单元中心梯度；
2. 纯 Neumann 压力域的 $F^{\hat u}$ 边界净通量必须**再平衡**（adjustPhi 式），且同一调整要同步进入方程右端，否则系统不相容；
3. 压力方程右端与通量重建必须用**同一份**（调整后的）$F^{\hat u}$；速度与通量用未松弛解重建，只有存储压力吃松弛；
4. 每个附加修正子前**用当前速度重算 $\hat u = H(u)/a_P$**（OpenFOAM 的 `HbyA = rAU*UEqn.H()`）；冻结它等价于 `nCorrectors = 1`。

**与 OpenFOAM-14 的定量一致性**（同网格/边界/Δt、单时间步、冲动启动）：

| 单步 Poiseuille | 本项目 | OpenFOAM-14 |
|---|---|---|
| `nCorrectors = 1` | $p(0,0)=2.20767079142$；进口 $\varphi/S=-0.0155562368675$ | $p(0,0)=2.20767079142$；$\varphi/S=-0.0155562368832$ |
| `nCorrectors = 2` | $p(0,0)=1.11519142929$；$\varphi/S=-0.00568435008099$ | $p(0,0)=1.11519143114$；$\varphi/S=-0.00568435009419$ |

一次修正后一致到 $\sim10^{-11}$，两次后 $\sim10^{-9}$（即 OpenFOAM 的 p 求解容差量级）。**单修正子在两边同样发散**：OpenFOAM 纯 PISO `nCorrectors = 1` 在 $t=0.5$ 时 $\max u_x=-2.3\times10^{10}$，`nCorrectors = 2` 为 0.12352 —— 所以"$n_{\text{correctors}}\ge2$"是算法本身的要求，不是本实现的缺陷。

**瞬态验收**：突启 Couette $\max|u-u_{\text{exact}}|=2.0\times10^{-3}$（限值 $2\times10^{-2}$）、瞬态 Poiseuille 相对 $L_2$ 误差 $5.3\times10^{-3}$（限值 $8\times10^{-2}$）、$Q=0.08398\approx1/12$、连续性 $2.1\times10^{-16}$。

以 OpenFOAM-14 的 `incompressibleFluid` 模块（`correctPressure.C` / `momentumPredictor.C`）为参照。两者数学上是同一算法，差异集中在公式写法、数据结构与工程化程度上。

### 公式形式：绝对压力（本项目已采用）

| | 本项目 | OpenFOAM |
|---|--------|----------|
| 压力方程 | `A p = b`，$b = -\sum_f \text{outward}(F^{\hat u}_f) + \sum C_b p_b$，直接解**新压力** | `fvm::laplacian(rAtU, p) == fvc::div(phiHbyA)`，同样直接解**新压力** |
| 边界条件 | 直接复用 p 的物理边界（Dirichlet 值进右端、Neumann 零梯度） | 直接复用 p 的 patch 边界条件 |
| 通量重建 | `F_f = F^{\hat u}_f - C_f(p_N - p_P)`，就地修正持久通量场 | `phi = phiHbyA - pEqn.flux()` |
| 参考单元 | 消元（钉住 0 号单元旧压力，矩阵缩一维） | `setReference` 软钉（`b_ref += A_ref,ref·p_ref; A_ref,ref *= 2`，矩阵尺寸不变） |

**为什么改**：绝对形式与 $p'$ 增量形式**代数等价**（$p' = p - p_{\text{old}}$ 解同一个系统，单步指纹一致到 $\sim10^{-11}$ / $\sim10^{-9}$），但右端可以直接由 `div(phiHbyA)` 装配，不必形成"旧压力梯度 − 矩阵系数"的相消差；同时不再需要 $p'$ 的镜像边界对象，Dirichlet 压力边界、速度重建与通量重建都只依赖物理边界条件。重构前本项目用的是 $p'$ 形式；当时排查一度怀疑两种形式会导致多修正子路径不同，**最终确认与公式形式无关**——差别来自驱动循环是否按修正子刷新 $\hat u$（OpenFOAM 因为 `H()` 读矩阵当前 `psi_` 而自动刷新），见上文"PISO 与 OpenFOAM 的对照结论"。

**一处有意的差异**：稳态 SIMPLE 中本项目用未松弛解重建速度与通量、只有压力场松弛（Patankar 写法）；OpenFOAM 在 `p.relax()` **之后**才 `U = HbyA - rAU·grad(p)`（通量仍在 relax 之前取，故两边的通量都守恒）。PISO（$\alpha_p = 1$）下无差别。

### 数据结构与核心环节

| 环节 | 本项目 | OpenFOAM |
|------|--------|----------|
| 动量对角 | `MomentumAssembly.diag` 显式导出 | `rAU = 1.0/UEqn.A()` |
| 非压力速度 | `computeUHat`：$(b^{np}_P - \sum_{N\ne P} A_{PN} u_N)/a_P$；步初用 $u^{*}$ 求值，之后每个附加修正子前用当前 $u$ 重算（`refreshUHat`）；装配右端的 `phiHbyA` 取 $\rho S \hat u_f$ | `HbyA = rAU * UEqn.H()`；`phiHbyA = fvc::flux(HbyA) + rAUf·ddtCorr(...)`；`H()` 用矩阵当前持有的 `psi_`，因此每个修正子自动重算 |
| 瞬态算法 | PISO：一次预测 + `nCorrectors` 次修正，无欠松弛；每个附加修正子前重算 $\hat u$ | PISO / PIMPLE，`nCorrectors`/`nOuterCorrectors` |
| 奇异性 | 参考单元**消元**（矩阵缩一维，严格 SPD） | `pEqn.setReference(refCell, refValue)`（矩阵尺寸不变） |
| 封闭域相容性 | `checkFluxCompatibility`：入口检查边界净通量，不平衡则抛异常 | `adjustPhi` 强制边界净通量为零（自动修正而非报错） |
| 松弛 | 稳态 Patankar 动量松弛；压力场 `p += α_p(p_sol − p_old)`，速度/通量用未松弛解；瞬态无松弛 | `UEqn.relax()`；`p.relax()` 在 `U = HbyA − rAU·grad(p)` 之前；另有 SIMPLEC 选项 |

### 值得借鉴与不宜照搬

**值得借鉴**（尚未实现，按对本项目的价值排序）：

1. **SIMPLEC 选项**——仅需改对角系数 $d = 1/(a_P - \sum_N a_N)$，教学上可直接对比迭代数差异。
2. **PIMPLE 外层迭代**（`nOuterCorrectors > 1`，每个外层重新解动量方程）——大 Courant 数时的标准做法，是修正子数不够用时的正解。（`ddtCorr` 已排除，见 skill 的 facts。）
3. 次要项：动量预测开关、按场独立的收敛阈值。

**不宜照搬**：`setReference`（消元法更干净、矩阵更小）；patch/fvMatrix 重型抽象层（为非结构网格通用性付的代价，教学项目会淹没算法主线）；非正交修正循环（仅当引入斜交网格时才有意义）。

