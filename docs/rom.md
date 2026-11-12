# ROM（降阶模型）

Phase 6 的设计、算法与验证证据。脚本用法、CLI 参数与 pyfvm 机制见 [`docs/python.md`](python.md)；批次划分与执行状态见 [`README.md`](../README.md) §路线图 → Phase 6。

## 定位与数据流

Phase 6 的层级关系（POD/DMD/Galerkin 各是什么、怎么配合）：

- **POD** 是"选基"技术：快照矩阵的 SVD，给出能量最优低秩子空间与重构误差曲线——本身不含动力学；
- **DMD** 是最简单的完整（线性）预测模型：从快照序列拟合最佳线性演化算子，模态自带频率/增长率；
- **侵入式 Galerkin**（6.2/6.3 未做）把 FOM 算子投影到 POD 基上，保留 Navier–Stokes 二次非线性，严格包含 DMD 的能力。

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

对 X∈R^{S×N}（行=快照）默认**均值中心化**（fluctuation POD）后做经济 SVD：Xc = UΣVᵀ。

- 模态为**行向量** `modes (r_full, N)`（随 SVD 惯例）；符号不固定（SVD 固有），断言只用重构不变量；
- 能量 `e_k = σ_k²/Σσ_k²`；重构误差 `err(r) = sqrt(Σ_{k>r} σ_k²/Σ σ_k²)` 由谱直接计算，不重建矩阵；
- 重构：`X̂ = A @ modes + mean`，`A = Xc @ modes.T` 为时间系数；
- `center=False` 保留均值（mean 返回全零），用于需要含均值基的场景。

**解读要点**（实测证据见下节）：瞬态会显著抬高所需秩——极限环流场应先做瞬态丢弃窗口再分解；模态**不是物理速度场**（无量纲基向量），mean 是有量纲时均场。

**已知 caveat**：collocated Rhie–Chow 排布下 POD 速度模态**非逐点散度自由**，6.2/6.3 侵入式路径重构后可能需要压力投影步压 checkerboard。

## DMD（`rom.dmd`，exact DMD，Tu et al. 2014）

输入 X (S, N) + 均匀 dt（快照间距 = `snap_every × dt`），内部转置为文献布局（列=快照）：

1. X₁ = X[:-1]ᵀ，X₂ = X[1:]ᵀ；X₁ 经济 SVD；
2. 秩：显式 `r` 或相对截断 `σ > σ_max·rtol`（默认 1e-12）；
3. Ã = Uᵣᵀ X₂ Vᵣ Σᵣ⁻¹，特征分解 (λ, W)；
4. exact DMD 模态 Φ = X₂ Vᵣ Σᵣ⁻¹ W（**(N, r) 列向量**，与 POD 的行约定不同，docstring 已互警）；
5. 连续特征值 ω = ln(λ)/dt；幅值 b = lstsq(Φ, x₁)。

**主导频率规则**（确定性，入 API 不入 CI 断言）：|Im ω| > 0 的模态中取 |b| 最大者，返回 |Im ω|/(2π)。涡街 St = f·D/U∞（D=1、U∞=1，取自 cylinder.py 参数约定）。负实特征值映到 Nyquist 频率是 DMD 固有现象，规则不受其影响。

## 已验证结论

**Taylor–Green**（64²、dt=0.0025、T=2.0、201 帧，2026-11-13）：单模衰减解 → 第 1 阶 POD 模态占 99.99997% 能量，err(1)=5.0e-4、err(8)=9.2e-7——管线正确性的机器级验证（多余模态即时间离散误差的痕迹）。

**涡街 Re=100**（coarse、T=250、5001 帧，窗口 t≥100 共 3001 帧，2026-11-13）：

| r | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| 窗口累计能量 | 51.9% | 95.9% | 99.6% | 99.97% | — |
| 窗口 err(r) | 0.69 | 0.20 | 0.061 | 0.016 | 2.2e-3 |
| 全程 err(r) | — | — | — | — | 7.6e-2 |

- 模态 1+2 为共轭行波涡对（空间相位错 ~90°）——窗口 vs 全程 err(16) 差 30 倍，即瞬态污染秩的直接证据；
- **DMD St = 0.1593 = FOM St = 0.1593**（同 run 的 Cl 峰谷法，24 峰），主导共轭对 |λ| = 1.0000（极限环的数学签名），谱中另见二次谐波 0.3185 = 2f；
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

测试：`python/tests/test_rom.py` 7 用例（POD 代数 ×2、DMD 双正弦合成信号 ×1、npz 契约往返+错误路径 ×3、TG 32² 端到端冒烟 ×1），随 `pyfvm_tests` 自动收编；`rom` 包不 import pyfvm，代数/契约用例无需扩展构建（pyfvm 缺失时仅 TG 冒烟 skip）。

## 后续批次与已知问题

- **6.2** pyfvm 绑定装配矩阵 A/b（COO 数组 + shape，避免 scipy 硬依赖）；未决：暴露哪些装配态（稳态/含 ddt 分离）、finalize 语义、装配状态所有权；
- **6.3** Galerkin 投影 ΦᵀAΦ + RK4 在线积分，对照 FOM 轨迹与 TG 精确解验证；
- **6.4（可选）** 压力重构、Re 参数化；
- 条件触发候选：ROM 分析 MCP（README §路线图 Phase 6 登记）；
- 已知问题：collocated 模态非散度自由（见 POD 节）；Galerkin ROM 继承 FOM 离散误差，TG 的 dt²/h 混合项意味着粗 dt 下 FOM 轨迹有精度地板（`docs/numerical.md` Taylor–Green 节）。
