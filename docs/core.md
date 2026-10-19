# core 模块说明

`fvm::core` 命名空间，提供求解器的基础数据结构与网格描述：抽象二维有限体积网格接口、结构化/非结构化两套实现、单元中心场与面通量场。本模块不依赖其他 fvm 模块，是依赖关系的最底层。

## Types.h：基础类型

```cpp
using Scalar = double;
using Index  = std::size_t;
using Vector = Eigen::VectorXd;

struct Point2 { Scalar x; Scalar y; };       // 二维点/向量
struct IndexSpan;                            // C++17 下的非拥有索引区间
```

全项目统一使用 `fvm::core::Scalar` 表示浮点数、`Index` 表示网格索引、`Vector` 表示稠密向量。`IndexSpan` 只暴露 `begin/end/size/empty/operator[]`，用于 `cellFaces`、`patchFaces` 这类连续存储的只读视图。

## Mesh：抽象网格接口

`Mesh`（`include/Mesh.h`）是所有数值代码可见的唯一网格接口，描述面优先的二维 FVM 拓扑：

| 接口 | 说明 |
|------|------|
| `cellCount()` / `faceCount()` | 单元数；内部面与边界面使用同一套全局面编号 |
| `faceOwner(face)` / `faceNeighbour(face)` | 面的 owner/neighbour；边界面返回 `Mesh::kInvalidCell` |
| `isBoundaryFace(face)` | 边界面判定 |
| `cellFaces(cell)` | 单元相邻面列表（`IndexSpan`） |
| `cellVolume(cell)` / `cellCentroid(cell)` | 单元体积与面积加权中心 |
| `faceCentroid(face)` / `faceAreaVector(face)` | 面中心与面积向量 $\mathbf S_f$ |
| `patchCount()` / `patchName(patch)` / `patchFaces(patch)` | 边界 patch 拓扑 |
| `facePatch(face)` | 边界面所属 patch（仅边界面有效） |

### 全局符号约定

这些约定对所有实现一致，数值装配不得自行假设坐标方向：

- $\mathbf S_f$ 相对 owner 向外；内部面因此从 owner 指向 neighbour；
- `FaceFluxField` 的内部面通量 $\phi_f > 0$ 表示 owner → neighbour；
- 边界面通量为 owner 的外法向通量，出流为正。

## CartesianMesh：二维均匀笛卡尔网格

### 设计思路

- **继承 `Mesh`，保留结构化便捷层**：`nx/ny/dx/dy/bounds/cellIndex/cellIJ/cellCenter` 服务于测试与 Python 绑定；数值代码仍只应使用 `const Mesh&`。
- **单元中心（cell-centered）**：未知量存储在单元中心，线性索引仍为 $\text{cell}=j\,n_x+i$。
- **全局面编号**：

  $$
  x_f(i,j)=j\,(n_x+1)+i,\quad i\in[0,n_x],\ j\in[0,n_y);
  $$
  $$
  y_f(i,j)=(n_x+1)\,n_y+j\,n_x+i,\quad i\in[0,n_x),\ j\in[0,n_y].
  $$

  总面数 $(n_x+1)n_y+n_x(n_y+1)$。内部面 owner 取西/南单元，owner → neighbour 为 $+x/+y$；边界面 owner 是唯一相邻单元。
- **单元面顺序**：`cellFaces(cell)` 按 `[east, north, west, south]` 返回；旧局部罗盘查询接口（`neighbor`、`faceArea(int)`、`faceNormal`、`cellToCellDistance`、`cellToFaceDistance`、`isBoundaryFace(cell,int)`）已移除。
- **patch 顺序**：名称与索引固定为 `east/north/west/south`，对应 $0/1/2/3$。

构造函数校验 `nx`、`ny` 非零，否则抛 `std::invalid_argument`。

### 结构化便捷接口

| 函数 | 说明 / 公式 |
|------|------------|
| `cellCenter(i, j)` | $x=x_{\min}+(i+\tfrac12)\Delta x$，$y=y_{\min}+(j+\tfrac12)\Delta y$ |
| `cellIndex(i, j)` / `cellIJ(cell)` | $\text{cell}=j\,n_x+i$ 及其逆映射 |
| `xFace(i, j)` / `yFace(i, j)` | 上述全局 x/y 面编号 |
| `cellVolume(cell)` | $\Delta x\,\Delta y$ |
| `xMin/yMin/xMax/yMax` | 计算域边界 |

## UnstructuredMesh：多边形非结构网格

`UnstructuredMesh` 采用面优先存储，由 `MeshInput` 构造：

```cpp
struct MeshInput {
    std::vector<Point2> points;
    std::vector<std::vector<Index>> cells;   // 每个单元为顶点环
    std::vector<PatchInput> patches;          // patch 由顶点对表示边
};
```

构造过程：

1. 顶点环按逆时针规范化；顺时针输入也被接受并自动反转；
2. 按排序后的顶点对匹配边并建立面；owner 为首次遍历该边的单元，owner 的逆时针走向自动给出 owner 外向 $\mathbf S_f$；
3. 预计算单元体积（shoelace）、面积加权单元中心、边中点面中心与 $\mathbf S_f=(\Delta y,-\Delta x)$；
4. 建立 `cellFaces` / `patchFaces` 的 CSR 存储；
5. 校验顶点范围、每单元至少 3 个顶点、正面积、流形且方向一致的边连接、patch 恰好覆盖全部边界边，以及每单元 $\sum_f\mathbf S_f=0$（相对容差 `1e-9`）。不满足则抛异常。

`makeStructuredMesh(nx, ny, xMin, yMin, xMax, yMax)` 用非结构路径生成与 `CartesianMesh` 相同的结构网格：单元按 j-major 创建，因此单元编号同为 `cellIndex(i,j)`，patch 也保持 east/north/west/south 顺序。它是双实现回归测试的基准，不用于替代日常结构网格。

`UnstructuredMesh` 保留 `MeshInput` 输入点，并暴露 `pointCount()` / `point(i)` / `cellNodeCount(c)` / `cellNode(c, local)`。单元顶点访问按构造时逆时针规范化后的环序返回；该 CCW 约定与 owner 外向面积向量的推导一致。

## MeshGeomCache：派生几何系数缓存

`MeshGeomCache(const Mesh&)` 预计算所有面相关的几何系数，避免装配循环重复求距离和点积：

| 接口 | 定义 | 说明 |
|------|------|------|
| `interpWeight(face)` | $g_f=\lVert x_f-x_N\rVert/\lVert x_P-x_N\rVert$ | owner 到面/邻点的线性权重；边界面返回 1 |
| `innerCoef(face)` | $\alpha_f=(\mathbf S_f\cdot\mathbf d)/(\mathbf d\cdot\mathbf d)$，$\mathbf d=x_N-x_P$ | 内部扩散几何系数；边界面调用抛异常 |
| `boundaryCoef(face)` | $\beta_f=(\mathbf S_f\cdot\mathbf S_f)/(\mathbf S_f\cdot\boldsymbol\delta)$，$\boldsymbol\delta=x_f-x_P$ | 边界扩散几何系数；内部面调用抛异常 |
| `nonOrthCorrection(face)` | $\mathbf k_f=\mathbf S_f-\alpha_f\mathbf d_f$ | 内部面非正交部分；正交网格为零，边界面调用抛异常 |
| `maxNonOrthogonality()` | $\max_f\angle(\mathbf S_f,\mathbf d_f)$ | 以度为单位，仅统计内部面 |

## ScalarField / VectorField：单元中心场

- 场持有 `const Mesh&`，按 `mesh.cellCount()` 分配并置零；生命周期必须覆盖网格。
- 数据是一个 `Vector`，与全局单元编号一一对应，可直接交给 Eigen 求解。
- `ScalarField` 只支持单元编号访问 `field(cell)`；二维 $(i,j)$ 重载已移除，结构网格需先经 `CartesianMesh::cellIndex(i,j)` 转换。
- `VectorField` 由 `u()` / `v()` 两个 `ScalarField` 组成；`operator()(cell)` 返回只读 `{u,v}`，`setConstant(value)` 同时设置两个分量。

## FaceFluxField：面通量场

OpenFOAM `surfaceScalarField` 的对应物：一个长度为 `mesh.faceCount()` 的 `Vector` 同时覆盖内部面与边界面。

- `operator[](face)` / `data()`：按全局面编号读写；
- 旧的 `x()/y()` 与 `xData()/yData()` 分量存储接口已移除；
- 内部面 $\phi_f>0$ 表示 owner → neighbour；边界面存储 owner 外向通量；
- `outwardFlux(cell, face)`：若该单元是面 owner 则返回 `+phi[face]`，否则返回 `-phi[face]`；
- `cellImbalance(cell)`：遍历 `cellFaces(cell)` 求净流出量，守恒场在求解精度内为零；
- `setZero()` 整体清零。

类持有 `const Mesh&`，不可拷贝赋值；需要覆盖写时使用 numerical 模块的原地填充接口（如 `computeMassFlux`）。

## VTK 输出

`writeVti` 保持结构网格 `.vti` ImageData 路径，仅接受 `CartesianMesh`；`writeVtu` 输出 `UnstructuredMesh` 的 `.vtu` UnstructuredGrid。文件结构与格式细节见 `docs/io.md`。
