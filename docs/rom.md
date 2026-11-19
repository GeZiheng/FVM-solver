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

**已知 caveat**：collocated Rhie–Chow 排布下 POD 速度模态**非逐点散度自由**，侵入式路径的重构场缺 $\nabla p$ 约束——6.3 已在 TG 上定量（见 Galerkin 节），6.4 以预测性压力投影补上缺失约束（TG 上泄漏降 ~8×，见压力响应的演进节）。

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

**降阶模型与推导**：u、v 各自独立 POD 基（中心化，行正交），$u = \bar{u} + a_u \Phi_u$（$\Phi$ 为 $r\times N$ 行模态，$a$ 为系数向量）。半离散 FOM 是**逐单元的动量积分平衡**（有限体积把 PDE 乘上单元体积）：

$$\underbrace{\rho\,\mathrm{diag}(V)}_{M}\,\frac{\mathrm{d}u}{\mathrm{d}t} = b_{\mathrm{sp}}(t) - A_{\mathrm{sp}}\,u$$

$M = \rho\,\mathrm{diag}(V)$ 是 FOM 的**集中质量算子**：第 P 个方程的物理意义是"单元 P 的动量 $\rho V_P u_P$ 的变化率 = 该单元所受合力"；`A_sp`/`b_sp` 来自 `assemble_momentum(dt=0, 零压场, Neumann-p)`（6.2 theta 分离恒等式）。代入 $u = \bar{u} + a\Phi$、以模态为测试函数对单元求和（Galerkin：trial = test = POD 基，本质是**加权残差法**——要求 FOM 残差与模态子空间正交，方程只在投影/弱意义下满足，与有限元对连续弱形式做 Galerkin 是同一原理；差别仅在 FV 半离散方程已是逐单元的积分平衡，无需再分部积分）：

$$\big\langle \varphi_i,\ M \varphi_j \big\rangle\, \dot{a}_j = \big\langle \varphi_i,\ b - A u \big\rangle \;\;\Longrightarrow\;\; M_r\,\dot{a} = \Phi\,(b - A u)^{\top}$$

$$M_r \;=\; \Phi\,M\,\Phi^{\top} \;=\; \Phi\,\mathrm{diag}(\rho V)\,\Phi^{\top}$$

（逐分量各一组，$r\times r$ 稠密 SPD；内积为对单元的欧氏求和。）

**$M_r$ 的含义**：

- **模态坐标下的质量阵（广义惯性）**——FOM 质量算子到系数空间的 Galerkin 投影。$M_r \dot{a}$ 是各模态广义动量的变化率，右端是对应的广义力：牛顿第二定律的模态形式。
- **动能度量**——涨落场动能 $\tfrac{1}{2}\sum_P \rho V_P\,|u_P - \bar{u}_P|^2 = \tfrac{1}{2}\,a^{\top} M_r\,a$，即 $M_r$ 是系数空间上的动能（黎曼）度量；权重 $\rho V$ 正是动量方程积分形式带来的。
- **为何不是单位阵**——POD 用欧氏 SVD，模态满足 $\Phi\Phi^{\top} = I$（欧氏正交），但 FOM 质量是 $\rho V$ 加权的：均匀笛卡尔网格上 $V$ 为常数，$M_r$ 退化为纯标量 $\rho V \cdot I$；非均匀/非结构网格（涡街 O-4）上 $V$ 逐单元不同，模态在质量加权内积下不再正交，$M_r$ 出现真实非对角结构。若改用质量加权 POD（能量范数正交模态）则恒有 $M_r = I$——本仓库取欧氏 SVD（numpy 直截），代价是每次残差评估 `numpy.linalg.solve(M_r, ·)`（$O(r^3)$，$r \le 24$ 可忽略；满秩 SPD：$x^{\top} M_r x = \sum_P \rho V_P\,(x^{\top}\Phi)_P^2 > 0$，构造后缓存一次）。

初值投影 $a_0 = (u_0 - \bar{u})\,\Phi^{\top}$（行正交基，全秩时往返精确）。

**每 RK4 阶段的残差评估**：重构 u/v 写入内部 VectorField → `bc_refresh(t)`（时间相关 BC 在阶段时刻刷新；静态 BC 传 None）→ `compute_mass_flux` → `assemble_momentum(dt=0)` 两个分量 → $\dot{a} = M_r^{-1}\,\Phi\,(b - A\,u)^{\top}$。ROM 的 dt 独立于 FOM dt（显式积分器继承投影算子刚性）。重构场不精确满足 BC（子空间不含边界修正），边界效应经装配 RHS 进入——Galerkin-ROM 标准行为。

**pressure 注入口**：构造器接受可选 `pressure` ScalarField，被每次残差评估**活读**（`bc_refresh` 可在阶段时刻更新它）。零压 = 纯速度 ROM，承受结构性压力泄漏；供入精确压力 = oracle 隔离实验，把泄漏从机构误差中分离。预测性压力投影（6.4）即经注入口逐 stage 写入投影压力（见压力响应的演进节）。

**为什么压力项不可省（同位网格的根源链）**：本仓库 FOM 是 collocated 排布，连续性约束施加在**面通量**上（修正通量 `phi =
phiHbyA − C·∇p` 守恒），而面速度不是单元心速度的普通插值——差一个 Rhie–Chow
压力修正项，这是同位格式压住压力棋盘解耦的**设计代价**。因此**面通量守恒 ≠ 单元心速度无散**：快照存的是单元心
u/v，每个快照都带残差散度；散度算子线性，POD 模态与均值场作为快照的线性组合**原样继承**该残差（截断不引入新散度——重构仍在子空间内）。于是 ∇p
在模态上的投影非零，是降阶方程的真实项：删掉它误差线性漂移（下表 TG 实测；对 dt_rom/4 与网格加密不变 ⇒
结构性），供压力即回离散地板。对照：交错网格（速度即通量）或散度自由基的谱/FE 方法，快照可精确无散，其无压力 Galerkin
才严格合法——"教科书无压力标准形式"的隐藏前提正在于此。

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
- **每步成本**：实测 ROM 48.9 ms/步 vs FOM 46.8 ms/步（比值 1.05）——本实现的逐降阶步重装配设计**没有任何步级加速**，其价值在 r 维状态与机构验证；经典 Galerkin 的加速来自预计算降阶算子张量（6.5b 起交付，见压力响应的演进节）。该数为零压路径的历史记录（2026-11-14，去重前后不变——零压路径无重复装配）；投影变体与 6.5a 装配去重后的最新成本见压力响应的演进节。

结论：纯速度 ROM 在涡街上抓住了振荡频率与有界性，但持续幅值塌缩 ~8×、时均场偏移——这些结构性偏差与 TG 上的压力泄漏同源，是 6.4 压力投影的最强动机。

## 压力响应的演进（6.4 投影 → 6.5c M2 约化解）

6.3 的结论（见 Galerkin 节）：纯速度 POD-Galerkin ROM 的误差漂移全部归因**结构性压力
泄漏**——collocated 模态非逐点散度自由、降阶方程缺 ∇p 约束（TG 上 T=1.0 达 43%，oracle
注入精确压力即回到 FOM 离散地板 0.0091）。此后各批次回答同一问题——**ROM 里压力的
状态相关椭圆响应怎么供**——差别只在"解多少、在哪解"：

| 变体 | 逐 stage 压力处置 | 每 stage FOM 侧工作 | 涡街 T=250 err\|FOM | 状态 |
| --- | --- | --- | --- | --- |
| 零压（6.3 基线） | 无 | 2 装配（残差） | 0.1173 | 泄漏基线 |
| oracle（6.3 隔离实验） | 注入精确 p | 2 装配 | —（TG 0.0091） | 机制归因 |
| 投影（6.4） | 全量解（冻结场装配） | 2 装配 + 1 解 | 0.0021 | 修复机制 |
| └ 装配去重（6.5a） | 同上（装配共享） | 1 装配 + 1 解 | 0.0021 | 成本优化，轨迹逐位一致 |
| 闭包（6.5b） | 数据拟合 c(a)=Ka+c₀ | 无（纯 numpy 0.2 ms/步） | 发散 @t≈2.25 | **熔断**，UNSTABLE 保留 |
| 压力入基（6.5c M1） | 全量解（离线场重构驱动） | 1 解 | 0.0087 / 0.0019（r_p=24） | 交付 |
| 约化解（6.5c M2） | Ψ 投影 r_p×r_p 稠密解 | 1 导出 + numpy 投影 | 0.0040 / 0.0036（r_p=24） | 交付 |

### 6.4 预测性压力投影

逐 RK4 stage 在冻结重构场上用 FOM 的 `correctPressure` 解一次离散压力方程（= PISO 单
corrector 的压力半步，不动动量解），结果写入 `GalerkinRom` 既有的 pressure 注入口。落点
= `pyfvm.pressure_projection` 绑定 + `GalerkinRom(pressure_projection=True)`，C++
numerical 零改动；签名与原位语义（velocity 原位修正、调用方自备 scratch、flux 于内部
工作副本）见 `docs/python.md` §API 一览。投影链（6.5a 起为融合入口，一次完成）：稳态
装配两分量（relaxation=1、与 ROM 残差同算子）→ `refreshUHat` → d = V/a_P →
`correctPressure(relaxationP=1)`。

**瞬态对角变体**（涡街必需）：稳态投影在涡街 coarse 不可用——Central、Re=100、Pe≫2 下
稳态算子对角不定（60/4224 单元 a_P ≤ 0，min −0.0303 ⇒ CG/BiCGSTAB 均不收敛；FOM 因瞬态
ρV/dt 对角 min +0.1997 从不触发）。`dt>0` ⇒ 装配带 `TimeTerm{rho, dt, Euler}`，uHat =
u_stage + r_s/(a_P+ρV/dt)（阻尼残差步，对角恒正）；TG（Pe O(1–2)）默认稳态。

**证据**：TG 16² 泄漏 0.182 → 稳态投影 0.0229 / 瞬态 0.0110（oracle 地板 0.0091，~8×
削减，`test_rom` 锁定 ≤0.05）；涡街 T=250 全视界 err|FOM 尾 20% 均值 0.117 → **0.0021**、
std(a₁) 0.604 → **2.668**（FOM 投影幅值 2.68，幅值塌缩修复）、St **0.1600** vs FOM
0.1593（0.4%）——零压变体的频率偏低与幅值塌缩两个结构性偏差均被修复。契约：
`TestPressureProjection` + `TestGalerkinRomProjection`（单步一致性/泄漏视界/瞬态视界/
注入口互斥）。

### 6.5a 装配去重

投影变体每 stage 原本做两遍同一冻结状态的装配（投影内部 + 残差侧）。融合入口
`pressure_projection(..., return_systems=True)` 一次稳态装配同时驱动投影与残差算子导出
（`dt>0` 按 theta=1 恒等式派生瞬态形态，逐位一致、仅 Euler），`GalerkinRom` 投影分支直接
消费返回系统，每 stage O(N) 全场操作 6→4；**轨迹与去重前逐位一致**
（`test_projection_dedup_trajectory_identity` 手工重演去重前流程锁定）。实测投影变体
−16~30%/步（TG 相对 FOM 6.62×→4.81×）；**dt_rom 放大 4× 无精度损失**（投影 4×dt_rom 后
≈1.2× FOM；8× 触及 RK4 显式稳定边界）。

### 6.5b 闭包熔断（离线预计算的交付与失败）

`python/rom/precomputed.py` 把零压降阶右端离线成纯 numpy 二次算子：Central 下 f_np 严格
二次，分级扰动采样（0.1/0.2/0.3σ——吸引子低维，仅用吸引子状态会使联合二次设计欠定），
留出验证 **1.4e-13**；`PrecomputedGalerkinRom` 在线 **0.2 ms/步**；逐模态压力源走 b-差
技巧 g_k = rhs(p=ψ_k) − rhs(p=0)（与 a_ref 无关），g₀ 同法取自 p̄。**仿射压力闭包**
c(a) = Ka + c₀（监督拟合于 FOM 快照对，流形残差 3.7e-4）在涡街极限环长视界**失稳**：
T=250 于 t≈2.25 发散；二分定位——dt 减半失稳点不动（非显式稳定性）、闭包关闭即稳定、
岭收缩/主子空间投影仅延后（223→822→~4400 步）；机制 = 常数雅可比闭包在流形外任意
外推形成正反馈。**教训：监督式数据驱动闭包替代不了状态相关的椭圆压力响应**——这是
M1/M2 保留"解方程"路线的直接依据。闭包保留为 UNSTABLE 基础设施（诊断/短视界用）；
完整失败记录：`.agents/plans/phase6-5b-offline-reduced-operators.md` 变更记录。

### 6.5c M1 压力入基（全量压力解 + 离线算子）

保留 6.5b 全部离线设施，把闭包换回**逐 stage 状态相关的全量压力解**（6.4 的同一物理
机制，动量侧零装配）。`ReducedPressureRom` 每 stage 三步：

1. **Rhie–Chow 场重构**（`fit_rhie_chow_fields`，单元空间精确拟合）：diag(a) = a_P(a) +
   ρV/dt_p（a_P 严格仿射），uHat(a) = u(a) + r_np(a)/diag(a)，r_np = b^np − A·u 严格二次
   （瞬态对角下 ddt 源与对角项在 Euler theta=1 形式中代数相消 ⇒ 只需稳态装配公开输出），
   留出 **1.1e-12**；
2. **全量压力解**（`pressure_projection_from_fields`）：外部 (d, uHat) 场直调
   `correctPressure`（消元/adjustPhi/crossP 环全部沿用）；契约**同输入同方程**——与
   6.5a 融合入口同态解出的 p 相对差 Neumann **3.6e-12** / Dirichlet-p **8.4e-14**；
3. **压力源作用**：c = Ψᵀ(p − p̄)，f = quad(a) + g₀ + G·c；rhs vs GalerkinRom 投影模式
   快照态 **2.8e-9** / 扰动态 4.2e-5。

**两条执行期教训（D3 修订）**：压力基 Ψ 必须取**投影解**压力快照（FOM 收敛压力在满秩
下仍留 0.42 子空间外残差——单 corrector 投影解与收敛解不同子空间）；g₀ 必须作用（∇
线性，漏掉 g₀ 使 rhs 相对差 1.3）。另 `QuadraticFit` 求值改展平上三角 GEMM（einsum
等价 8.9e-16）。成本 **67–106 ms/步 = 1.5–2.3× vs 投影变体（154–175）**：动量侧已全部
离线化，残差成本就是全量压力解本身——M2 的直接目标。驱动 `--pressure-basis`；测试
`TestReducedPressureRom` 4 用例（含 6000 步稳定性、dt_p>0 校验）。

### 6.5c M2 约化压力解（2026-11-18）

把全量压力解替换为**压力方程自身的 Galerkin 投影**：`correctPressure` 增可选
`PressureSystemExport*` 尾参（唯一触 `Pressure.cpp` 的改动），绑定
`pressure_projection_from_fields(..., return_systems=True)` 导出
`(ProjectionResult, PressureSystem{rows/cols/vals/n/b})`，numpy 侧 `A_r = Ψ A_p Ψᵀ`
（bincount O(nnz·r_p) + GEMM O(N·r_p²)）、`b_r = Ψ (b_p − A_p p̄)`（绝对压力方程，
p̄ 分解必须进 RHS）、r_p×r_p 稠密解；`ReducedPressureRom(reduced_solve=True)` 切换，
COO 模式跨 stage 固定 → idx/Ψᵀ 列/p̄ 列缓存（轨迹与未缓存逐位一致）。

**两处关键设计**：导出恒为**未消元 N×N 系统**——均值中心化模态 Ψ·1 = 0 使 ΨA_pΨᵀ 自行
消去纯 Neumann 零空间，常数偏移对 c 不可见，一条公式统一两族 BC（adjustPhi 仍运行）；
**crossP 裁决 = 第一遍系统 + 单遍塌缩**——导出 RHS 的显式交叉项取传入压力状态，与
correctors=0（驱动/夹具口径）下 M1 首解装配同一方程。

**实测证据**（涡街 coarse T=250，r=16、窗口 0.6，与前批次同口径）：

| 指标 | 零压 | 投影（6.4） | M1（r_p=16/24） | **M2（r_p=16/24）** |
| --- | --- | --- | --- | --- |
| err\|FOM 尾 20% 均值 | 0.1173 | 0.0021 | 0.0087 / 0.0019 | **0.0040 / 0.0036** |
| St(a₁) | — | 0.1600 | 0.1600 / 0.1600 | **0.1600 / 0.1600**（FOM 0.1593，0.44%） |
| std(a₁) | 0.604 | 2.668 | 2.646 / 2.660 | **2.654 / 2.656** |
| 长视界 | 稳定 | 稳定 | 15000 步稳定 | **15000 步稳定** |
| 成本 ms/步 | 46.1 | — | 60.3 / 61.1 | **29.5 / 33.5** |

**门槛裁定（如实记录，两项未达）**：**轨迹门 ✗**（vs M1 ≤1e-3 → 实测 7.3e-2，r_p=24
上探 2.3e-2——能量残差缩 9× 偏差仅缩 3.1×，T=20 满能量基仍有 9e-3，地板是 crossOut
状态依赖与 Galerkin-vs-L² 系数差的逐步累积而非截断单项；依据：两轨迹在同一极限环上，
St 四位一致、std 相对差 0.3%，且 r_p=16 下 M2 的 err|FOM 反而优于 M1 2.2×——偏差是
轨迹/相位级而非质量级）；**成本门 ✗**（<10 ms/步 → 实测 29.5 = M1 的 2.0×——每 stage
场重构 ~2.5 + C++ 导出 ~1.4 + numpy 投影 ~2.5 ms ×4，压力解本身 12.3→~2 ms/stage（6×）
但 M1 机制的共享底板使其无法兑现到步级 <10）。CI 契约：`TestReducedPressureSolve` 4
用例（rhs vs M1 快照态 1.1e-6/扰动态 3.7e-4、20 步轨迹 1.9e-4、2000 步稳定、空基拒绝）
+ 绑定契约 ×2（导出与 M1 解满足同一方程 ≤1e-8 双 BC 族、导出后 pressure 不动）。驱动
`--reduced-pressure`（依赖 `--pressure-basis`）。后续候选（未立项）：步级成本再下探需动
M1 场重构/导出装配的共享底板（C++ 侧几何缓存或装配融合），超出 M2 边界。

## 验证算例扩容（Stokes 振荡板 + 后向台阶，2026-11-20）

Phase 6 的 ROM 验证原本只有 TG（封闭域、纯 Neumann、线性、衰减瞬态）与涡街（开域、
Dirichlet-p、强非线性、极限环）两个极端；本批补中间形态，并顺势落地 6.5b 计划登记的
**D1 边界参数机制**（`.agents/plans/phase6-rom-validation-cases.md`）：

| 算例 | 驱动 | 填补的缺口 | 核心证据（2026-11-19/20） |
| --- | --- | --- | --- |
| Stokes 振荡板 | `stokes_galerkin.py` | 纯 Neumann 的**真实物理算例**（此前 M1/M2 只有 CI 夹具）、时变 BC、近满降阶（线性问题 POD 秩 2） | err\|FOM 尾段 零压 0.0298 / 投影 0.0334 / M1 0.0290 / **M2 0.0310**（同档）；相位 0 剖面 L∞ **M2 2.38e-3 全场最佳**（FOM 自身 2.82e-3）；10 额外周期全稳定；成本 M2 **3.8 ms/步**（M1 10.7 = 2.8×） |
| 后向台阶启动瞬态 | `bfs_galerkin.py` | **定态吸引子**（第三种吸引子类型）、分离流 / L 形 SU2 几何、Dirichlet-p 出口 | err\|FOM 零压 **1.91 完全失锁**（启动段压力动力学强烈）/ 投影 0.0242 / M1 0.0248 / M2 0.0241；x_r/h 偏差 ≤0.44%（FOM 6.685）；M2 **24.4 ms/步**全场最低（低于逐装配的零压变体） |

两算例合起来给出机制排序的完整谱：**压力供给的收益随压力动力学强度单调**——Stokes
（连续 ∇p=0，只剩离散残差散度泄漏）四变体同档，BFS（强压力动力学）零压崩溃、供压
三变体全部恢复。这也是 6.3"结构性泄漏"结论在两种新吸引子上的外推验证。

**D1 边界参数机制（γ 参数化）**：Stokes 北壁 u = cos(ωt) 是时变 Dirichlet，与离线算子
的固定 BC 假设冲突。解法 = 把 γ(t) 作为**已知额外坐标**并入二次拟合：
离线 `make_fit_samples(parameter=...)` 产出增广样本（分级扰动覆盖 γ 列）、
`fit_reduced_operators`/`fit_rhie_chow_fields`/`projected_pressure_snapshots` 经
`parameter_samples`+`parameter_bc` 逐样本刷新活 BC 后装配（参数必须贯穿**采样侧**，
仅改系数列不充分）；在线 `PrecomputedGalerkinRom(parameter_fn=t→γ(t))` 逐 RK4 stage
求值增广进 rhs，`advance(bc_refresh=...)` 同步活 BC 供 M1/M2 压力解读取。默认路径
（parameter_fn=None）逐位不变。这是 Re 参数化 ROM 登记项的最小实例（参数不进状态、
时钟已知）；a–γ 双线性交叉项由联合二次拟合自动覆盖。契约 `TestParameterizedRom` ×2
（参数化 rhs vs 投影 twin：快照态 1.5e-7 / 扰动态 3.3e-4@r_p=12——γ 族使压力流形
变肥，r_p=8 残留 3e-2，12 秩饱和）。

**执行期发现**（细节与失败形态入计划变更记录）：

- **投影变体的 RK4 显式稳定边界**（Stokes）：逐 stage 压力投影使降阶 rhs 变刚，
  边界在 2.5–5× FOM dt 之间（0.025 爆、0.0125 稳）——两驱动的 dt_rom 默认 FOM dt
  （TG 惯例，4× 余量），快照节拍不再是安全的 ROM 步长；
- **holdout 度量细化**：逐行/逐单元相对度量在消失分量上地板化（Stokes 的 v 是
  噪声场，|f_v|~1e-9 ⇒ 假性失配）——改为逐输出/逐分量参考尺度（留出集峰值），
  健康算例仍机器精度；(a_u, γ) 设计手工验证 1e-14 机器精确；v 分量的拟合地板
  （rhs ~1e-3、场 ~1e-2，Stokes 驱动按算例校准容差 1e-2/1e-1）机制未解析，入档
  open；
- u/v 等宽假设放宽为行对齐 + 各自宽度（Stokes 2+1 不等秩）。

两驱动均带 `--frames N`（各变体重构帧 + 同时刻 FOM 快照；Stokes 写 `.vti`、BFS 写
`.vtu`，镜像 TG/涡街驱动惯例）与 `--test` 机检；证据口径同涡街（驱动证据运行 +
本节记录，不进 application 层）。

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

测试：`python/tests/test_rom.py` 30 用例（POD 代数 ×2、DMD 双正弦合成信号 ×1、npz 契约往返+错误路径 ×3、TG 32² 端到端冒烟 ×1、Galerkin TG ×4、Galerkin 投影 ×5（含 6.5a 去重轨迹恒等）、离线预计算 ×4（`TestPrecomputedRom`，6.5b）、压力入基 ×4（`TestReducedPressureRom`，6.5c，含 6000 步稳定性与 dt_p>0 校验）、约化压力解 ×4（`TestReducedPressureSolve`，6.5c M2，含 rhs/轨迹/2000 步稳定性/空基校验）、边界参数化 ×2（`TestParameterizedRom`，Phase 6 验证算例，含参数化 rhs 对照与轨迹）），随 `pyfvm_tests` 自动收编（pyfvm_tests 79 = test_pyfvm 49 + test_rom 30）；`rom` 包核心不 import pyfvm（`rom.galerkin`/`rom.precomputed` 惰性 import），代数/契约用例无需扩展构建（pyfvm 缺失时 TG 冒烟与 Galerkin/投影/预计算/压力入基用例 skip）。

## 后续批次与已知问题

- ✅ **6.2 pyfvm 装配矩阵绑定（2026-11-13 完成）**——`fvm.assemble_momentum`（用法与契约见 `docs/python.md` §API 一览）。原未决问题全部落地：暴露**稳态单次装配**（`dt=0` 给 `A_sp`/`b_sp`，质量阵 `M = ρ·diag(V)` 无需装配）；finalize 语义 = 返回 finalized COO（压缩后无重复项）；装配状态无所有权耦合（每次调用按传入状态独立装配，干净但每次重算——降阶步粒度下开销可忽略）。分离恒等式 $A_{\mathrm{tr}} = \theta\,A_{\mathrm{sp}} + \mathrm{diag}(\rho V/\mathrm{dt})$、$b_{\mathrm{tr}} = b_{\mathrm{sp}} + (\rho V/\mathrm{dt})\,u^{n} - (1-\theta)\,(A_{\mathrm{sp}}\,u^{n})$ 由 `test_pyfvm.py::TestAssembleMomentum` 契约锁定；压力项语义：逐单元 $b_P = b^{\mathrm{np}}_P - (\nabla p)_P\,V_P$（`rhs_no_pressure` 即 $b^{\mathrm{np}}$），velocity-only ROM 传零压场 + Neumann-p 即剥离压力源。
- ✅ **6.3 纯速度 POD-Galerkin ROM（2026-11-13 完成）**——`rom/galerkin.py`（`GalerkinRom`、稠密 SPD `M_r`、RK4、pressure 注入口）+ `taylor_green_galerkin.py`（双变体：零压 vs oracle）+ `vortex_shedding_galerkin.py`（诊断）+ `test_rom.py` 4 用例。证据一句话：TG 16² 上单步一致性 0.0067、oracle 变体稳定在 FOM 离散地板（0.0091 vs oracle|FOM=0.0065），零压变体线性漂移（T=0.3→0.182、T=1.0→43%，对 dt_rom/4 与 32² 不变）确认为结构性压力泄漏——机构正确，漂移全归因压力项（实测表见 Galerkin 节）。
- ✅ **6.4 预测性压力投影（2026-11-14 完成）**——逐 stage 冻结场压力修正修复结构性泄漏（TG ~8×；涡街 err|FOM 0.117→0.0021、幅值塌缩修复）；瞬态对角变体覆盖 Pe≫2。机制与证据见压力响应的演进节 §6.4，绑定面见 `docs/python.md` §API 一览。**Re 参数化 ROM 仍为未实施项**（Phase 6 后续候选）；
- ✅ **6.5a 装配去重（2026-11-15 完成）**——融合出口一次装配共享投影与残差导出，轨迹逐位一致；投影变体 −16~30%/步，dt_rom 放大 4× 无损。见压力响应的演进节 §6.5a。
- **6.5b 离线预计算（2026-11-16 部分交付，验收熔断）**——二次算子机器精度（留出 1.4e-13）+ `PrecomputedGalerkinRom` 0.2 ms/步交付；**仿射压力闭包在涡街长视界失稳**（t≈2.25，机制=常数雅可比离流形外推正反馈），保留为 UNSTABLE 基础设施。机制二分与教训见压力响应的演进节 §6.5b；完整记录：`.agents/plans/phase6-5b-offline-reduced-operators.md`。
- ✅ **6.5c M1 压力入基 ROM（2026-11-17 完成）**——离线场重构驱动的逐 stage 全量压力解：涡街 err|FOM 0.0087、St 0.1600、15000 步稳定、67–106 ms/步。机制与 D3 教训见压力响应的演进节 §6.5c M1；执行记录：`.agents/plans/phase6-5c-pressure-basis-rom.md`。
- ✅ **6.5c M2 约化压力解（2026-11-18 完成）**——压力方程 Galerkin 投影 r_p×r_p 稠密解：涡街 err|FOM 0.0040、St 0.1600、15000 步稳定、29.5 ms/步 = M1 的 2.0×；轨迹门与成本门如实未达并文档化（见压力响应的演进节 §6.5c M2）；执行记录：`.agents/plans/phase6-5c-m2-reduced-pressure-solve.md`。
- 条件触发候选：ROM 分析 MCP（README §路线图 Phase 6 登记）；
- 已知问题：collocated 模态非散度自由（见 POD 节）——其后果已从预测变为**已定量**：纯速度 Galerkin ROM 在 TG 上呈线性误差漂移（T=1.0 达 43%），oracle 压力注入把误差压回 FOM 离散地板（见 Galerkin 节实测表）；该泄漏已量化并可由 6.4 投影模式消除（TG 上 ~8× 泄漏削减，见压力响应的演进节）。Galerkin ROM 继承 FOM 离散误差，TG 的 $\mathrm{dt}^2/h$ 混合项意味着粗 dt 下 FOM 轨迹有精度地板（`docs/numerical.md` Taylor–Green 节）。
