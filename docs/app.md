# app 模块说明

`src/app/main.cpp` 是可执行目标 `fvm_solver` 的入口：一个稳态对流-扩散演示算例，串联 core → numerical → math → io 全部模块，展示完整求解流程。

## 算例设置

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

## 扩展指引

以此算例为模板可快速变化：换 `ConvectionScheme::Central` 观察二阶格式的振荡、调整 `gammaVal` 改变 Pe 数、给 `assembleTransport` 传 `Sc/Sp` 增加源项、或换用 `createEigenSparseLU` 做直接法对照。
