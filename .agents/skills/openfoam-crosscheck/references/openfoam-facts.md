# 已确认的 OpenFOAM-14 事实索引

先查这里，避免重复读源码。每条都标了**来源**：`实测` = 本项目跑算例验证过；`源码` = 读过对应文件；`记录` = 结论已写入 `docs/numerical.md`（含判据）。

路径以 `$OF14 = ~/OpenFOAM/OpenFOAM-14` 记。

## 求解器与算例

- 入口：`foamRun -solver incompressibleFluid`（分派器在 `$OF14/applications/solvers/foamRun/foamRun.C`），求解器模块在 `$OF14/applications/modules/incompressibleFluid/`。
- **OF-14 里没有 `pEqn.H`/`UEqn.H`**：压力修正逻辑在 `applications/modules/incompressibleFluid/correctPressure.C`，动量预测在 `momentumPredictor.C`，主循环在 `incompressibleFluid.C`。找 `pEqn.H` 只会命中 legacy 求解器（如 `porousSimpleFoam`），别被误导。`实测`
- 纯 PISO 用 PIMPLE 字典表达：`PIMPLE { momentumPredictor on; nOuterCorrectors 1; nCorrectors N; nNonOrthogonalCorrectors 0; }`，启动日志会打印 `PIMPLE: Operating solver in transient mode with 1 outer corrector` + `PIMPLE: Operating solver in PISO mode`。`实测`
- 压力是**运动学压力**（`dimensions [kinematicPressure]`，m²/s²），物性用 `constant/physicalProperties` 的 `nu`；对照本项目的 `rho`/`mu` 时按 `nu = mu/rho` 换算。`实测`

## Rhie–Chow 与压力修正

- **`fvMatrix::H()` 用矩阵当前持有的 `psi_`**（`$OF14/src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C`）：

  ```cpp
  Hphi.primitiveFieldRef() += lduMatrix::H(psi_.primitiveField()) + source_;
  ```

  因此 OpenFOAM 在每个压力修正子里重算 `HbyA = rAU*UEqn.H()` 时，用的是**上一次修正后的 U**。冻结 $\hat u$ 会让修正子循环在第一次之后到达自身不动点（等价 `nCorrectors = 1`）。`源码`+`实测`
- `rAU = 1.0/UEqn.A()`；`A()` 的对角含边界对角贡献（`addBoundaryDiag`）并除以体积。`源码`
- 压力修正序列（`correctPressure.C`，对照本项目 `predictMomentum`/`correctPressure`）：
  `HbyA = constrainHbyA(rAU*UEqn.H(), U, p)` → `phiHbyA = fvc::flux(HbyA) + interpolate(rAU)*ddtCorr(U, phi, Uf)` → `adjustPhi` → `fvm::laplacian(rAU, p) == fvc::div(phiHbyA)`（**绝对压力形式**）→ `setReference` → `phi = phiHbyA - pEqn.flux()` → `U = HbyA - rAU*fvc::grad(p)` → `U.correctBoundaryConditions()`。`记录`
- **`p.relax()` 在 `U` 重建之前**：`phi = phiHbyA - pEqn.flux()` 在 `pEqn.solve()` 之后、`p.relax()` 之前取走通量（所以 `phi` 用**未松弛**解，离散守恒），而 `p.relax()` 之后才 `U = HbyA - rAU*fvc::grad(p)`（所以稳态速度吃到的是**松弛后**的压力）。本项目速度/通量都用未松弛解、只松弛压力场——这是有意的 SIMPLE 变体差异（PISO 时 α_p=1 无差别）。`源码`（`correctPressure.C`）+`记录`
- `fvMatrix::setReference` 的实现是软钉：`source()[ref] += diag()[ref]*value; diag()[ref] *= 2.0;`（矩阵尺寸不变、仍对称正定）。本项目用参考单元消元（矩阵缩一维、钉住旧压力），两者压力水平等价。`源码`（`src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C`）+`记录`
- **RC 的落点**：面通量用**面法向差商**（`snGrad`/`laplacian`），而动量源与速度重构用**单元中心 Gauss 梯度**（`fvc::grad`）。两者之差就是 RC 修正项；棋盘格在单元梯度下恒为零、在面差商下被强惩罚。`记录`
- `createPhi`（`$OF14/src/finiteVolume/cfdTools/incompressible/createPhi.H`）用 `linearInterpolate(U) & mesh.Sf()` —— **初始通量不是 RC**，与本项目 `computeMassFlux`/`interpolateCellVelocityFlux` 对应。`记录`
- `adjustPhi`（`$OF14/src/finiteVolume/cfdTools/general/adjustPhi/`）在封闭域强制边界净通量为零（自动修正）；本项目对应 `checkFluxCompatibility`（报错）+ `correctPressure` 内的纯 Neumann 再平衡。`记录`
- `constrainHbyA`（`$OF14/src/finiteVolume/cfdTools/general/constrainHbyA/`）只对**固定值速度边界**（壁面）生效，零梯度进出口不触发。`记录`

## 已排除的假设（不要重复试错）

以下都针对本项目的瞬态通道/库埃特算例做过对照，结论都记在 `docs/numerical.md` 的"PISO 与 OpenFOAM 的对照结论"里：

- `ddtCorr`（`EulerDdtScheme::fvcDdtPhiCorr`，`$OF14/src/finiteVolume/finiteVolume/ddtSchemes/EulerDdtScheme/EulerDdtScheme.C`）：逐字实现后无改善；限幅器在"通量—速度失配与通量同量级"处把该项关闭。`记录`
- `pimple.consistent()`：显式设 `consistent no` 后对照结果**逐位相同**。`记录`
- 壁面压力 BC（`fixedFluxPressure` vs `zeroGradient`）：两种都稳定，非差异点。`记录`
- 动量方程去掉压力梯度源、加 PIMPLE 外层：都会改变不动点或不属于 PISO 定义，不是差异原因。`记录`

## 日志可读量

- `smoothSolver: Solving for Ux/... Initial residual = ..., No Iterations ...`
- `GAMG: Solving for p, Initial residual = ...` —— 每个修正子一条，**第二条**（PISO 2 个修正子时）是判断"第二次求解是否仍在推进"的关键指纹（本项目对照时是 `0.0886`）。
- `time step continuity errors : sum local/global/cumulative`
- `Courant Number mean: ... max: ...` —— 打印在 `Time = ...` **之前**，对应进入该步的状态；`max|U| = Co_max*Δx/Δt`（均匀方形网格）是免额外输出的每步场量代理。
