# 对照算例：搭建、运行、比对

配套脚本：`../scripts/make_of_case.sh`（建算例+跑+汇总）、`../scripts/of_log_summary.py`（日志表格）。源码/字典语义见 `openfoam-facts.md`。

## 1. 对照契约（先写下来，再动手）

任何一项不同，对照就无效。逐项确认并在报告里列出：

| 项目 | 本项目 | OpenFOAM | 必须一致？ |
|---|---|---|---|
| 网格 | `CartesianMesh(nx,ny,Lx,Ly,...)` | `blockMeshDict` | 是（含单元顺序约定） |
| 物性 | `rho`、`mu` | `nu`（`kinematicPressure`） | 是（`nu=mu/rho`） |
| 时间格式 | `PisoConfig::timeScheme` | `fvSchemes/ddtSchemes` | 是 |
| 对流格式 | `ConvectionScheme` | `divSchemes` | 是 |
| 梯度/插值 | `cellGradient`（Gauss） | `gradSchemes/Gauss linear`、`interpolationSchemes/linear` | 是 |
| 边界条件 | `bcU/bcV/bcP` | `0/U`、`0/p` | 是（含"零梯度"的写法） |
| 时间步/时长 | `dt`、`nSteps` | `deltaT`、`endTime` | 是 |
| 线性求解器容差 | `SolverConfig::tolerance` | `fvSolution/solvers` 的 `tolerance` | 是（量级一致即可） |
| **算法配置** | `nCorrectors`、松弛 | `PIMPLE{nOuterCorrectors,nCorrectors}`、`relaxationFactors` | **是**（最易漏） |

本项目参照算例（压力驱动通道，已与 OpenFOAM 对齐过）：`~/OpenFOAM/gzh1057-14/run/pisoPoiseuille` —— 16×16 单位方腔、`nu=1`、`Delta t=0.02`、进出口 `fixedValue p`（1/0）、进出口 U 零梯度、上下壁 no-slip、Euler + 迎风、纯 PISO、`deltaT` 无自适应的固定步长。

## 2. 运行

```bash
# 在 WSL 中执行（从 Windows 侧调用时用 wsl.exe -e bash -lc "..."，需要提权）
cd /mnt/d/Projects/FVM-solver
bash .agents/skills/openfoam-crosscheck/scripts/make_of_case.sh \
    --template ~/OpenFOAM/gzh1057-14/run/pisoPoiseuille \
    --root     ~/OpenFOAM/gzh1057-14/run \
    --name     chan --nc 2 --dt 0.02 --end 0.5 --dx 0.0625
```

脚本会把配置写进目录名（`chan_nc2_dt0.02_t0.5_<stamp>`）并打印每步摘要。**永远不要复用同一个目录改配置再跑**——本项目曾因混用不同 `nCorrectors` 的输出得出"边界通量差 2.7 倍"的错误结论。

只看单步指纹时用 `--end` = `--dt`（只跑 1 步）。

## 3. 提取场与映射到本项目布局

- **单元场**（`U`、`p`）：时间目录里的 ascii `List<vector>`/`List<scalar>`，顺序即 OpenFOAM 单元序；与我们的 `cellIndex(i,j)` 顺序要对齐**并实测验证**（对照时验证过 `p(0,0)` 数值一致）。
- **面通量**（`phi`）：内部面按 owner→neighbour 为正（对内部面就是 +x/+y，与我们 `FaceFluxField` 的约定一致），随后是各 boundary patch 的面。映射到我们的 $(type,i,j)$ 需要 owner/neighbour → $(\text{方向},i,j)$ 的换算；**注意西/南边界**：我们的存储方向是 +x/+y，向外的西/南面通量在数值上要取相反号。
- 需要更细的场（$\hat u$、$d$、每修正子增量）时，本项目侧写一个小驱动：链接 `build/Release-agent/fvm_core.lib`，复刻 `solvePiso` 的循环（先在 `predictMomentum` 后、再在每个修正子后 dump）。先前的做法见 `%TEMP%\piso_repro\driver.cpp`（含一个提供全局量的 `stub.cpp`）。

## 4. 单步指纹（先做这个）

1. 两边都只跑**一个时间步**，同一初始场（冲动启动：$u=0$、$p$ 按边界线性）。
2. 比 $p$ 与进口通量 $\varphi/S$、$\max|\Delta u|$、$\max|\Delta p|$。一致到求解器容差量级（~1e-11）才算"第一步没问题"。
3. **必须同时跑 `nCorrectors = 1` 和 `= 2` 两组**：这能立刻区分"单次修正就不一致"和"多修正子路径不同"。
4. OpenFOAM 侧可直接看日志里**第二次 p 求解的 Initial residual**（本项目对照时是 `0.0886`），它一非零就说明第二个修正子确实在推进方程。

## 5. 多步 / 多配置扫描

- 跑到发散或到关注时刻，用脚本批量建算例（每次只改一个变量）。
- **先写下可证伪的预测再跑**：例如"Δt 变小应该能救单修正子"→ 实测 Δt=0.02/0.005/0.002 全发散 → 假设被推翻，如实记录并据此修正机制解释。
- 用 `max|U|`（由 `Co_max*Δx/Δt` 换算）和每步放大因子判断"是慢速失稳还是立刻爆"。

## 6. 坑清单（都踩过）

- **命令引号**：PowerShell → `wsl.exe -e bash -lc` 会破坏含 `()`、`<>`、`$()`、双引号的长内联命令。把命令写成仓库里的 `.tmp_*.sh`，用 `bash /mnt/d/Projects/FVM-solver/<脚本>` 执行。
- `source .../etc/bashrc` 之前必须 `set +u`（否则 `ZSH_NAME: unbound variable` 直接退出）。
- **`sed` 替换要容忍缩进**：`fvSolution` 里的键有前导空格（`^nCorrectors` 匹配不到），`controlDict` 的键通常顶格。用 `^([[:space:]]*key[[:space:]]+)[^;]*;` 并**替换后 grep 验证**。
- **不要手写日志采样循环**：`grep | awk` 里用文件行号当步号会得到"步数超过总步数"的假数据。先用 `grep -c` 对齐计数，或直接用 `of_log_summary.py`。
- `Courant Number` 行在 `Time =` **之前**打印，属于进入该步的状态。
- WSL 访问需要提权；`rm -rf` 之类先确认路径，优先用带时间戳的新目录名避免覆盖。
