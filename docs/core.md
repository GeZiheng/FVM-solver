# core 模块说明

`fvm::core` 命名空间，提供整个求解器的基础数据结构与几何描述：标量/索引类型别名、二维均匀笛卡尔网格、单元中心场。本模块不依赖其他 fvm 模块，是依赖关系的最底层。

## 文件结构

| 文件 | 内容 |
|------|------|
| `include/Types.h` | `Scalar`、`Index`、`Vector` 类型别名 |
| `include/Mesh.h` + `src/Mesh.cpp` | `CartesianMesh` 类 |
| `include/Field.h` | `ScalarField`、`VectorField` 类（header-only，`src/Field.cpp` 仅为保持构建结构一致而存在的空文件） |

## Types.h：类型别名

```cpp
using Scalar = double;
using Index  = std::size_t;
using Vector = Eigen::VectorXd;
```

全项目统一使用 `fvm::core::Scalar` 表示浮点数、`Index` 表示网格索引、`Vector` 表示稠密向量（底层为 Eigen）。其他模块通过 `using` 声明引入这些别名，保证精度与索引类型可在一处全局切换。

## CartesianMesh：二维均匀笛卡尔网格

### 设计思路

- **均匀网格**：`dx = (xMax-xMin)/nx`，`dy = (yMax-yMin)/ny` 全局恒定，因此单元体积、面面积、中心距等几何量无需逐单元存储，全部由 `dx_`、`dy_` 即时计算，内存开销为 O(1)。
- **单元中心（cell-centered）**：未知量存储在单元中心，边界面上无网格点。这是有限体积法的标准布局，天然保证守恒性。
- **行优先线性索引**：`cellIndex = j * nx + i`（i 沿 x 方向，j 沿 y 方向）。线性索引是场数据存储和稀疏矩阵行号的基础。
- **面索引约定**（全项目统一，边界条件、通量装配均依赖此约定）：

  | face | 方向 | 外法向 |
  |------|------|--------|
  | 0 | east 东 | (+1, 0) |
  | 1 | north 北 | (0, +1) |
  | 2 | west 西 | (−1, 0) |
  | 3 | south 南 | (0, −1) |

- **边界哨兵值**：`neighbor(cell, face)` 在面位于计算域边界时返回 `cellCount()`（越界索引）作为哨兵，`isBoundaryFace` 即据此判断。装配循环据此区分内部面与边界面。

### 关键函数

| 函数 | 说明 / 公式 |
|------|------------|
| `cellCenter(i, j)` | 单元中心坐标：`x = xMin + (i + 0.5)·dx`，`y = yMin + (j + 0.5)·dy` |
| `cellIndex(i, j)` / `cellIJ(cell)` | 线性索引与 (i, j) 互转：`cell = j·nx + i`；`i = cell % nx`，`j = cell / nx` |
| `neighbor(cell, face)` | 越过指定面的邻居单元索引；边界面返回 `cellCount()`；非法 face 抛 `std::invalid_argument` |
| `faceArea(face)` | 2D 中面面积即边长：东/西面为 `dy`，北/南面为 `dx` |
| `cellToCellDistance(cell, face)` | 相邻单元中心距：东西向 `dx`，南北向 `dy` |
| `cellToFaceDistance(face)` | 单元中心到面中心距离：`dx/2` 或 `dy/2` |
| `cellVolume(cell)` | 单元体积：`dx·dy`（构造时预计算为 `volume_`） |

构造函数会校验 `nx`、`ny` 非零，否则抛 `std::invalid_argument`。

## ScalarField / VectorField：单元中心场

### 设计思路

- **场绑定网格**：`ScalarField` 持有 `const CartesianMesh&`，构造时按 `mesh.cellCount()` 分配数据并置零。场的生命周期必须覆盖网格的生命周期（注意悬挂引用）。
- **存储即一个 `Vector`**：数据是长度为 `cellCount` 的 `Eigen::VectorXd`，与网格线性索引一一对应。这使场可以直接参与 Eigen 向量运算（如 `field.data() = solver.solve(...)`）。
- **双索引访问**：`operator()` 同时支持线性索引 `(cell)` 和二维索引 `(i, j)`，后者内部经 `cellIndex` 转换，便于写逐点初始化循环。
- **VectorField = 两个 ScalarField**：2D 向量场由 `u_`（x 分量）和 `v_`（y 分量）两个标量场组成，名称自动加 `_u` / `_v` 后缀（供 VTK 输出区分）。分量场可独立访问，便于分别施加边界条件或输出。

### 关键接口

- `field(cell)` / `field(i, j)`：可读写引用。
- `field.data()`：返回底层 `Vector&`，用于整体赋值或与线性求解器对接。
- `setZero()` / `setConstant(value)`：整体初始化。
- `VectorField::u()` / `v()`：返回分量标量场引用。
- `VectorField::operator()(cell)`：返回 `{u, v}` 值对（只读）。

## 依赖关系

```
Types.h  ←（被所有文件包含）
Mesh.h   → Types.h
Field.h  → Mesh.h, Types.h
```

core 不依赖 math / io / numerical；上层模块通过 `using fvm::core::Scalar` 等声明引用本模块类型。
