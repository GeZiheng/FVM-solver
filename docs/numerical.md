# numerical 模块说明

`fvm::numerical` 命名空间，实现稳态标量输运方程的有限体积离散，将对流、扩散、源项和边界条件装配为线性系统 $A\phi = b$。这是求解器的数值核心。

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

## 依赖关系

```
numerical → core（Mesh/Field/Types）
          → math（SparseMatrix, Vector）
```

numerical 不依赖线性求解器与 io——它只负责装配，求解与输出由 app 层组织。
