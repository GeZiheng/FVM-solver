# ROM（降阶模型）

Phase 6 的设计、算法与验证证据。脚本用法、CLI 参数与 pyfvm 机制见 [`docs/python.md`](python.md)；批次划分与执行状态见 [`README.md`](../README.md) §路线图 → Phase 6。

## 定位与数据流

Phase 6 的层级关系（POD/DMD/Galerkin 各是什么、怎么配合）：

- **POD** 是"选基"技术：快照矩阵的 SVD，给出能量最优低秩子空间与重构误差曲线——本身不含动力学；
- **DMD** 是最简单的完整（线性）预测模型：从快照序列拟合最佳线性演化算子，模态自带频率/增长率；
- **侵入式 Galerkin**（6.3 已做，`rom.galerkin`）把 FOM 算子投影到 POD 基上，保留 Navier–Stokes 二次非线性，严格包含 DMD 的能力。

数据流（6.1 交付）：

```text
FOM: pyfvm PISO (n_steps=1 逐步循环)
  → rom.SnapshotCollector（append 时立即 float64 拷贝）
  → npz 快照文件（契约 v1，本仓库与外部 ROM 项目的数据边界）
  → 离线分析: rom.pod_decompose / rom.dmd（纯 numpy，无需 pyfvm）
```

## 快照契约（format_version = 1）

| key | 形状/类型 | 含义 |
| --- | --- | --- |
| `format_version` | 标量 int64 == 1 | 契约版本；只允许向后兼容地新增 key |
| `case` | 标量 str | 算例名 |
| `params` | 标量 str（JSON） | 算例参数（网格/dt/nu/rho/窗口等） |
| `t` | (S,) float64 | 快照时刻，含 t=0 初值，严格递增、等间距（DMD 前提） |
| `u` / `v` / `p` | (S, N) float64 | **行=快照**；单元顺序=pyfvm cell 顺序（结构网格 `j*nx+i`，SU2=reader 顺序） |
| `cx` / `cy` | (N,) float64 | 单元中心坐标（可选；外部项目重建几何用） |

校验与错误路径：`SnapshotCollector.save` 拒绝 S<2、非严格递增 t、三场形状不一致；`load_snapshots` 拒绝缺失 key、`format_version != 1`、违反形状契约。`to_numpy()` 是存活视图，**禁止把视图存进快照栈**——append 时立即拷贝。6.2 若需通量快照，走版本化新增 key（如 `face_phi`），不破坏 v1 读取。

## POD（`rom.pod_decompose`）

对 $X \in \mathbb{R}^{S\times N}$（行=快照）默认**均值中心化**（fluctuation POD）后做经济 SVD：$X_c = U\Sigma V^{\top}$。

- 模态为**行向量** `modes (r_full, N)`（随 SVD 惯例）；符号不固定（SVD 固有），断言只用重构不变量；
- 能量 $e_k = \sigma_k^2 \big/ \sum_k \sigma_k^2$；重构误差 $\mathrm{err}(r) = \sqrt{\sum_{k>r} \sigma_k^2 \big/ \sum_k \sigma_k^2}$ 由谱直接计算，不重建矩阵；
- 重构：`X̂ = A @ modes + mean`，`A = Xc @ modes.T` 为时间系数（numpy 语义）；
- `center=False` 保留均值（mean 返回全零），用于需要含均值基的场景。

**解读要点**（实测证据见下节）：瞬态会显著抬高所需秩——极限环流场应先做瞬态丢弃窗口再分解；模态**不是物理速度场**（无量纲基向量），mean 是有量纲时均场。

`PodResult.truncated(r)` 返回仅保留前 r 阶模态的拷贝（r 封顶于满秩；谱属性沿切片保留，`reconstruction_error` 仍反映分解的完整谱）——Galerkin ROM 按秩截断基的标准入口。

**已知 caveat**：collocated Rhie–Chow 排布下 POD 速度模态**非逐点散度自由**，侵入式路径的重构场缺 $\nabla p$ 约束——6.3 已在 TG 上定量（见 Galerkin 节），6.4 以预测性压力投影补上缺失约束（TG 上泄漏降 ~8×，见压力投影节）。

## DMD（`rom.dmd`，exact DMD，Tu et al. 2014）

输入 X (S, N) + 均匀 dt（快照间距 = `snap_every × dt`），内部转置为文献布局（列=快照）：

1. $X_1 = X_{[:-1]}^{\top}$，$X_2 = X_{[1:]}^{\top}$；$X_1$ 经济 SVD；
2. 秩：显式 `r` 或相对截断 $\sigma > \sigma_{\max}\cdot\mathrm{rtol}$（默认 1e-12）；
3. $\tilde{A} = U_r^{\top} X_2 V_r \Sigma_r^{-1}$，特征分解 $(\lambda, W)$；
4. exact DMD 模态 $\Phi = X_2 V_r \Sigma_r^{-1} W$（**(N, r) 列向量**，与 POD 的行约定不同，docstring 已互警）；
5. 连续特征值 $\omega = \ln(\lambda)/\mathrm{dt}$；幅值 $b = \operatorname{lstsq}(\Phi, x_1)$。

**主导频率规则**（确定性，入 API 不入 CI 断言）：$|\operatorname{Im}\omega| > 0$ 的模态中取 $|b|$ 最大者，返回 $|\operatorname{Im}\omega|/(2\pi)$。涡街 $\mathrm{St} = f\,D/U_\infty$（$D=1$、$U_\infty=1$，取自 cylinder.py 参数约定）。负实特征值映到 Nyquist 频率是 DMD 固有现象，规则不受其影响。

## Galerkin ROM（`rom.galerkin`，Phase 6.3）

`GalerkinRom` 是纯速度侵入式 POD-Galerkin ROM 的在线积分器（显式 RK4）。与 `rom` 包其余部分不同，它通过 pyfvm 扩展驱动 FOM——`pyfvm` 在 `GalerkinRom` 构造内**惰性 import**，单独 `import rom.galerkin` 无需扩展。

**降阶模型**：u、v 各自独立 POD 基（中心化，行正交），$u = \bar{u} + a_u \Phi_u$；半离散 FOM 为 $M\,\mathrm{d}u/\mathrm{d}t = -A_{\mathrm{sp}}(t, u)\,u + b_{\mathrm{sp}}(t)$，其中 $M = \rho\,\mathrm{diag}(V)$（6.2 theta 分离恒等式）、`A_sp`/`b_sp` 来自 `assemble_momentum(dt=0, 零压场, Neumann-p)`。投影得降阶 ODE（逐分量，r×r 稠密 SPD）：

$$M_r\,\dot{a} = \Phi\,(b - A u)^{\top}, \qquad M_r = \Phi\,\mathrm{diag}(\rho V)\,\Phi^{\top}$$

**$M_r$ 不是单位阵**：POD 模态非质量正交，$M_r$ 构造后缓存一次，每次残差评估用 `numpy.linalg.solve` 求解。初值投影 $a_0 = (u_0 - \bar{u})\,\Phi^{\top}$（行正交基，全秩时往返精确）。

**每 RK4 阶段的残差评估**：重构 u/v 写入内部 VectorField → `bc_refresh(t)`（时间相关 BC 在阶段时刻刷新；静态 BC 传 None）→ `compute_mass_flux` → `assemble_momentum(dt=0)` 两个分量 → $\dot{a} = M_r^{-1}\,\Phi\,(b - A\,u)^{\top}$。ROM 的 dt 独立于 FOM dt（显式积分器继承投影算子刚性）。重构场不精确满足 BC（子空间不含边界修正），边界效应经装配 RHS 进入——Galerkin-ROM 标准行为。

**pressure 注入口**：构造器接受可选 `pressure` ScalarField，被每次残差评估**活读**（`bc_refresh` 可在阶段时刻更新它）。零压 = 纯速度 ROM，承受结构性压力泄漏；供入精确压力 = oracle 隔离实验，把泄漏从机构误差中分离。预测性压力投影（6.4）即经注入口逐 stage 写入投影压力（见压力投影节）。

**Taylor–Green 实测**（16² 全秩，2026-11-13）：

| 变体 | 误差 | 结论 |
| --- | --- | --- |
| 单步一致性（ROM RK4 vs FOM PISO，同 dt） | 相对差 0.0067 | 机构正确（不同积分器的预期量级） |
| 零压，T=0.3 / T=1.0 | 0.182 / 43% | 误差线性漂移；dt_rom/4 与 32² 网格下不变 ⇒ 排除积分器刚度与空间分辨率，**结构性压力泄漏**（collocated 模态非散度自由 + ∇p 缺失） |
| oracle（阶段时刻注入精确压力） | 0.0091 | 稳定在 FOM 离散地板（oracle|FOM = 0.0065）⇒ Galerkin 机构正确，漂移全归因压力项 |

**涡街诊断**：`vortex_shedding_galerkin.py` 做窗口 POD（r=16）→ ROM 积分 → a₁/a₂ 系数谱取 ROM St——**只诊断不断言**（纯速度 ROM 无压力约束，极限环保持无保证）。注意脚本的 FFT 输出受窗长限制：150 时间单位的频率分辨率为 ±0.0033，"St=0.1533" 只是 bin 中心。**逐项测量**（2026-11-14，同一 npz 数据集，过零点法测频 + 最优时移误差分解）：

- **频率**：ROM St ≈ 0.152–0.154（过零点法，全窗 0.1522、后 2/3 窗 0.1536），比 FOM（0.1593）低约 4%，三分窗内稳定——频率抓得住；
- **幅值**：主导系数 a₁/a₂ 的持续振荡 std ≈ 0.33，而 FOM 投影系数 std ≈ 2.68/2.47——**ROM 的持续振幅塌缩到 ~1/8**（此前报告的"包络 3.67/5.71"是初值瞬态的峰值，不是持续振幅——该解读已更正）；同时 a₃ 出现 +0.38 的均值偏移（时均尾迹被扭曲）；
- **误差形态**：全程相对误差从 0.001 快速升至 ~0.14 后平台缓降（t=250 时 0.115）；**最优时移只把中位误差从 0.118 降到 0.114**——差距主体是幅值/模态构成与均值的结构性偏移，不是相位滑动（此前"主因相位错位"的说法有误，已在此更正）；
- **时间外推**（POD 只用 $t \in [100, 200]$、从 t=200 预报 50 单位）：误差有界（中位 13.8%）、振荡持续，但外推窗 50 单位的 FFT 分辨率仅 ±0.01，此前"预报 St 降级至 0.1400（−12%）"在分辨率上不成立，外推窗内的真实频率未单独测定；
- **每步成本**：实测 ROM 48.9 ms/步 vs FOM 46.8 ms/步（比值 1.05）——本实现的逐降阶步重装配设计**没有任何步级加速**，其价值在 r 维状态与机构验证；经典 Galerkin 的加速来自预计算降阶算子张量（未实现）。

结论：纯速度 ROM 在涡街上抓住了振荡频率与有界性，但持续幅值塌缩 ~8×、时均场偏移——这些结构性偏差与 TG 上的压力泄漏同源，是 6.4 压力投影的最强动机。

## 压力投影（Phase 6.4）

6.3 量化的结构性压力泄漏的修复：**预测性压力投影**——逐 RK4 stage 在冻结重构场上用 FOM 的 `correctPressure` 机器求解一次离散压力方程（= PISO 单 corrector 的压力半步，不动动量解），结果写入 `GalerkinRom` 既有的 pressure 注入口。落点 = `pyfvm.pressure_projection` 绑定（签名与语义见 [`docs/python.md`](python.md) §API 一览）+ `GalerkinRom(pressure_projection=True)`；C++ numerical 模块零改动（延续 6.1–6.3 惯例）。

**投影链**（绑定内一次完成）：稳态装配两分量（`assembleMomentum`，relaxation=1、`TimeTerm{}`，与 ROM 残差同算子）→ `finalize` → `refreshUHat` → `d = V/a_P` → `correctPressure(relaxationP=1)`（线性求解器由 `solver_config` 内部创建，不暴露 LinearSolver 类）。传给装配的 pressure 内容无关紧要（投影只消费 `diag` 与 `rhsNoPressure`）；`relaxationP=1` ⇒ 存储 pressure = p_solved。`velocity` 按契约被**原位**写为修正速度 uHat − d∇p（丢弃与否由调用方自备 scratch 决定）；`flux` 被覆写为修正通量（phiHbyA − C∇p）。

**galerkin.py 侧时序**（逐 stage，镜像 oracle 时刻语义）：`compute_mass_flux` 之后、装配之前——scratch ← 复制重构场 → 投影（修正速度写入 scratch 后丢弃，ROM 状态必须保持系数 a）→ 重算 `compute_mass_flux` 把 flux 恢复为重构场质量通量。**与 oracle 通路的唯一差异 = pressure 场内容**（干净 A/B 对照）。纯 Neumann-p（TG）走 correctPressure 既有参考单元消元 + 闭域平衡路径。每 stage 成本 = 2 次装配 + 1 次压力求解（`last_projection` 存最近一次诊断四元组）。

**稳态系数的理由**：ROM 残差算子就是稳态装配（dt=0），投影出的 ∇p 与被投影算子一致；TG（μ=0.1，Pe O(1–2)）上验证成立。

**瞬态对角变体**（涡街裁决）：稳态投影在涡街 coarse 网格不可用——Central、Re=100、Pe≫2 下稳态算子对角不定（60/4224 单元 a_P ≤ 0，min −0.0303 ⇒ d min −671 ⇒ 压力系统非 SPD，CG/BiCGSTAB 均不收敛）；FOM 因瞬态 ρV/dt 对角（min +0.1997）从不触发，零压 ROM 双胞胎也不受影响（只做矩阵-向量乘、从不求逆对角）。`dt>0` ⇒ 装配带 `TimeTerm{rho, dt, Euler}`，ddt 源以冻结状态为 u_old ⇒ uHat = u_stage + r_s/(a_P+ρV/dt)（阻尼残差步，a_P^tr > 0）；dt=0 保持稳态算子 bit-identical。TG 默认稳态（μ=0.1、Pe O(1–2) 不受影响），涡街 `--projection` 传瞬态 dt。

**实测证据**（2026-11-14，TG 16² 全秩 / 涡街 coarse rank 8）：

| 场景 | 变体 | 结果 | 对照 |
| --- | --- | --- | --- |
| TG，T=0.2（20 步） | 稳态投影 err 0.0203 | 泄漏降 ~8× | 零压 0.124、oracle 地板 0.0091 |
| TG，T=0.3 | 稳态投影 err 0.0229 | 同上；驱动 `--test`/`test_rom` 锁定 ≤0.05 | 零压 0.182、oracle 0.0091 |
| TG，T=0.2 / T=0.3 | 瞬态投影（dt=0.01）err 0.0101 / 0.0110 | 优于稳态，距 oracle 地板 ~10–20% | `test_projection_transient_horizon` 锁定 ≤0.05 |
| 涡街，T=20 短窗 `--test --projection` | 瞬态投影 | err\|FOM 尾段均值 ~1e-4（零压 0.1709）、std(a1) 3.907→1.333 | 跑通且诊断有限；全视界运行留作用户后续 |

涡街 St(a1/a2) 在 T=20 短窗内与零压基线同值（0.0833）——窗口内极限环尚未建立（FOM 同因），只诊断不断言。契约测试：`TestPressureProjection`（Poiseuille 精确性、散度修复、原位语义、`relaxation_p` 校验）+ `TestGalerkinRomProjection`（单步一致性、泄漏视界、瞬态视界、投影/注入口互斥），见 `docs/python.md` §测试分工。

## 已验证结论

**Taylor–Green**（64²、dt=0.0025、T=2.0、201 帧，2026-11-13）：单模衰减解 → 第 1 阶 POD 模态占 99.99997% 能量，err(1)=5.0e-4、err(8)=9.2e-7——管线正确性的机器级验证（多余模态即时间离散误差的痕迹）。

**涡街 Re=100**（coarse、T=250、5001 帧，窗口 t≥100 共 3001 帧，2026-11-13）：

| r | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| 窗口累计能量 | 51.9% | 95.9% | 99.6% | 99.97% | — |
| 窗口 err(r) | 0.69 | 0.20 | 0.061 | 0.016 | 2.2e-3 |
| 全程 err(r) | — | — | — | — | 7.6e-2 |

- 模态 1+2 为共轭行波涡对（空间相位错 ~90°）——窗口 vs 全程 err(16) 差 30 倍，即瞬态污染秩的直接证据；
- **DMD St = 0.1593 = FOM St = 0.1593**（同 run 的 Cl 峰谷法，24 峰），主导共轭对 $|\lambda| = 1.0000$（极限环的数学签名），谱中另见二次谐波 $0.3185 = 2f$；
- 与文献带 0.164–0.167 的差距是 coarse 网格的已知偏移（CI 校准记录同值 0.159），DMD 忠实还原其 FOM。

## 使用

脚本：`taylor_green_rom.py` / `vortex_shedding_rom.py`（CLI 与落盘位置见 `docs/python.md` §示例）；`--modes N` 写前 N 阶 u 分量模态 + 均值场 VTK（模态无量纲、mean 有量纲）。离线分析无需求解器：

```python
import sys; sys.path.insert(0, "python")
from rom import load_snapshots, pod_decompose, dmd
data = load_snapshots(".../vortex_re100_coarse.npz")
pod_decompose(data.u)            # 能量谱 / err(r)
dmd(data.u[2000:], 0.05)         # 窗口 DMD → dominant_frequency() → St
```

测试：`python/tests/test_rom.py` 15 用例（POD 代数 ×2、DMD 双正弦合成信号 ×1、npz 契约往返+错误路径 ×3、TG 32² 端到端冒烟 ×1、Galerkin TG ×4、Galerkin 投影 ×4），随 `pyfvm_tests` 自动收编；`rom` 包核心不 import pyfvm（`rom.galerkin` 惰性 import），代数/契约用例无需扩展构建（pyfvm 缺失时仅 TG 冒烟与 Galerkin/投影用例 skip）。

## 后续批次与已知问题

- ✅ **6.2 pyfvm 装配矩阵绑定（2026-11-13 完成）**——`fvm.assemble_momentum`（用法与契约见 `docs/python.md` §API 一览）。原未决问题全部落地：暴露**稳态单次装配**（`dt=0` 给 `A_sp`/`b_sp`，质量阵 `M = ρ·diag(V)` 无需装配）；finalize 语义 = 返回 finalized COO（压缩后无重复项）；装配状态无所有权耦合（每次调用按传入状态独立装配，干净但每次重算——降阶步粒度下开销可忽略）。分离恒等式 $A_{\mathrm{tr}} = \theta\,A_{\mathrm{sp}} + \mathrm{diag}(\rho V/\mathrm{dt})$、$b_{\mathrm{tr}} = b_{\mathrm{sp}} + (\rho V/\mathrm{dt})\,u^{n} - (1-\theta)\,(A_{\mathrm{sp}}\,u^{n})$ 由 `test_pyfvm.py::TestAssembleMomentum` 契约锁定；压力项语义：逐单元 $b_P = b^{\mathrm{np}}_P - (\nabla p)_P\,V_P$（`rhs_no_pressure` 即 $b^{\mathrm{np}}$），velocity-only ROM 传零压场 + Neumann-p 即剥离压力源。
- ✅ **6.3 纯速度 POD-Galerkin ROM（2026-11-13 完成）**——`rom/galerkin.py`（`GalerkinRom`、稠密 SPD `M_r`、RK4、pressure 注入口）+ `taylor_green_galerkin.py`（双变体：零压 vs oracle）+ `vortex_shedding_galerkin.py`（诊断）+ `test_rom.py` 4 用例。证据一句话：TG 16² 上单步一致性 0.0067、oracle 变体稳定在 FOM 离散地板（0.0091 vs oracle|FOM=0.0065），零压变体线性漂移（T=0.3→0.182、T=1.0→43%，对 dt_rom/4 与 32² 不变）确认为结构性压力泄漏——机构正确，漂移全归因压力项（实测表见 Galerkin 节）。
- ✅ **6.4 预测性压力投影（2026-11-14 完成）**——`pyfvm.pressure_projection` 绑定 + `GalerkinRom` 投影模式：逐 stage 稳态压力修正（PISO 单 corrector 压力半步），瞬态对角变体（`projection_dt`）覆盖 Pe≫2 场合。TG 16² 泄漏降 ~8×（零压 0.182 → 投影 0.0229，oracle 地板 0.0091），瞬态变体 0.0110 更优；涡街 `--projection` 瞬态模式跑通（err|FOM 尾段均值 ~1e-4、std(a1) 3.907→1.333，T=20 短窗诊断）。**Re 参数化 ROM 仍为 6.4 未实施项**（无量化依据，Phase 6 后续候选）；设计、机制与证据见压力投影节，绑定面见 `docs/python.md` §API 一览；
- 条件触发候选：ROM 分析 MCP（README §路线图 Phase 6 登记）；
- 已知问题：collocated 模态非散度自由（见 POD 节）——其后果已从预测变为**已定量**：纯速度 Galerkin ROM 在 TG 上呈线性误差漂移（T=1.0 达 43%），oracle 压力注入把误差压回 FOM 离散地板（见 Galerkin 节实测表）；该泄漏已量化并可由 6.4 投影模式消除（TG 上 ~8× 泄漏削减，见压力投影节）。Galerkin ROM 继承 FOM 离散误差，TG 的 $\mathrm{dt}^2/h$ 混合项意味着粗 dt 下 FOM 轨迹有精度地板（`docs/numerical.md` Taylor–Green 节）。
