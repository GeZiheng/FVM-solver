# app 模块说明

`src/app/main.cpp` 是可执行目标 `fvm_solver` 的入口，包含两个演示算例：稳态对流-扩散（标量输运）与顶盖驱动方腔流（SIMPLE），串联 core → numerical → math → io 全部模块，展示完整求解流程。

## 算例 1：稳态对流-扩散

**物理问题**：单位正方形 $[0,1] \times [0,1]$ 内的稳态温度输运——

$$\nabla \cdot (\rho\, \mathbf{u}\, T) = \nabla \cdot (\gamma \nabla T)$$

- **网格**：$64 \times 64$ 均匀笛卡尔网格。
- **速度场**：由流函数 $\psi = \sin(\pi x)\, \sin(\pi y)$ 派生的无散度回流场

  $$u = \frac{\partial \psi}{\partial y} = \pi \sin(\pi x) \cos(\pi y), \qquad v = -\frac{\partial \psi}{\partial x} = -\pi \cos(\pi x) \sin(\pi y)$$

  无散度（$\partial u / \partial x + \partial v / \partial y = 0$）保证对流算子装配的守恒性测试意义；流线为域内闭合回卷，热量主要靠对流在环内搬运。
- **参数**：$\rho = 1$，$\gamma = 0.01$，特征 Peclet 数 $\mathrm{Pe} \sim \rho\, \lVert \mathbf{u} \rVert L / \gamma \sim 100$（对流占优）。
- **边界条件**：西墙 Dirichlet $T = 1$（热壁），东墙 Dirichlet $T = 0$（冷壁），北/南墙取 `BoundaryField` 默认的零通量 Neumann（绝热）。

## 求解流程

```
1. 建网格          CartesianMesh mesh(64, 64, 0, 0, 1, 1)
2. 填场量          gamma.setConstant(0.01); 逐单元按解析式填 velocity
3. 设边界          bc.set(West, Dirichlet, 1.0); bc.set(East, Dirichlet, 0.0)
4. 装配            assembleTransport(..., ConvectionScheme::Upwind, bc)
5. 冻结矩阵        sys.A.finalize()           ← 求解前必须的一步
6. 求解            createEigenBiCGSTAB({tol=1e-10, maxIter=2000})->solve(A, b)
7. 回填场          temperature.data() = sol  ← 场数据即 Eigen 向量，直接整体赋值
8. 输出            VtkWriter::write("convection_diffusion.vti", mesh,
                                     {{"temperature", &T}}, {{"velocity", &u}})
```

### 流程体现的设计决策

- **求解器选择 BiCGSTAB 而非 CG**：含一阶迎风对流的矩阵非对称，不满足 CG 的对称正定性要求。
- **`finalize()` 显式分离装配与求解**：`assembleTransport` 返回未冻结矩阵（允许调用方追加自定义项），本例无追加项，直接冻结后求解。
- **场与向量的零拷贝对接**：`temperature.data() = sol` 成立是因为 `ScalarField` 底层就是按网格线性索引排列的 `Eigen::VectorXd`（见 [core.md](core.md)）。

## 输出

生成 `convection_diffusion.vti`（运行目录下），包含标量场 `temperature` 与向量场 `velocity`，可用 ParaView 打开查看温度分布与流场矢量。

## 算例 2：顶盖驱动方腔流（SIMPLE）

**物理问题**：单位正方形内的稳态不可压缩流动——顶盖（北墙）以 $U = 1$ 水平拖动，其余三壁面无滑移，$\rho = 1$、$\mu = 0.01$，即 $\mathrm{Re} = \rho U L / \mu = 100$。

- **网格**：$64 \times 64$ 均匀笛卡尔网格。
- **边界条件**：`bcU` 北墙 Dirichlet 1、其余三墙 Dirichlet 0；`bcV` 四墙均 Dirichlet 0（无穿透）；`bcP` 四墙均为默认零梯度 Neumann（纯 Neumann 情形，`solveSimple` 内部消去参考单元，见 [numerical.md](numerical.md)）。
- **算法配置**：`tolerance = 1e-6`，`maxIterations = 3000`，亚松弛 `relaxationU = 0.7` / `relaxationP = 0.3`，内层线性求解器容差 1e-9。

```
1. 建网格与场      CartesianMesh(64,64,0,0,1,1); velocity/pressure 置零
2. 设边界          bcU 北墙 = 1，其余速度墙 = 0；bcP 默认零梯度
3. SIMPLE 迭代     solveSimple(mesh, rho, mu, bcU, bcV, bcP, cfg, u, p)
4. 输出            VtkWriter::write("cavity.vti", mesh,
                                   {{"pressure", &p}}, {{"velocity", &u}})
```

当前配置下约 1000 次外迭代收敛（连续性残差降至 ~1e-8）。输出 `cavity.vti` 包含标量场 `pressure` 与向量场 `velocity`，可在 ParaView 中观察主涡与角隅二次涡。

## 扩展指引

对流-扩散算例可：换 `ConvectionScheme::Central` 观察二阶格式的振荡、调整 `gammaVal` 改变 Pe 数、给 `assembleTransport` 传 `Sc/Sp` 增加源项、或换用 `createEigenSparseLU` 做直接法对照。方腔算例可：调整 `mu` 改变 Re（提高 Re 需加密网格并加强亚松弛）、改 `bcU` 的 North 值改变盖速、或对 `bcP` 设置 Dirichlet 边模拟压力出口。
