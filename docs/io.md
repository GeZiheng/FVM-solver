# io 模块说明

`fvm::io` 命名空间，负责网格输入与结果输出：`readSu2` 读取 2D SU2 ASCII 网格（`.su2`），`writeVti` 为 `CartesianMesh` 写出 VTK ImageData（`.vti`），`writeVtu` 为 `UnstructuredMesh` 写出 VTK UnstructuredGrid（`.vtu`）。

接口为命名空间下的**自由函数**，不设类：每个入口都是无状态的一次性转换，函数即完整接口。写出函数按目标格式命名（`writeVti` / `writeVtu`）而非按网格类型重载，调用点直接可见输出格式，并与 `pyfvm` 的 `write_vti` / `write_vtu` / `read_su2` 一一对应。

## readSu2：.su2 输入

### 设计思路

- **只做映射，不做几何**：reader 把文件内容映射到 `MeshInput`（点、多边形单元、patch 边）后交给 `UnstructuredMesh` 构造器；拓扑校验（流形性、边界全覆盖、非零面积）全部复用构造器，reader 只负责文法与 SU2 特有的约定。
- **无状态自由函数**：`readSu2(filename)` 一次调用完成读文件、映射、构造，返回校验过的 `UnstructuredMesh`。
- **严格拒绝**：契约之外的任何变体（未知关键字、节缺失/重复/乱序、计数不符、未知单元类型）都抛异常而不是静默容忍。唯一容忍点是 `NELEM` 行尾的可选单元号——各家导出器实践不一，存在即忽略。

### 文法契约

- **范围**：2D SU2 ASCII，首节 `NDIME= 2`（其余拒绝）。按空白分词（`NDIME=2` 与 `NDIME= 2` 等价）；关键字大小写敏感（仅大写）；节顺序固定 `NDIME → NPOIN → NELEM → NMARK`。
- **节点**（`NPOIN= n` + n 行 `id x y`）：id 为 1-based 正整数、允许稀疏，映射为紧凑 0-based 点索引；重复 id、行数与 n 不符、后续引用未知 id 均拒绝；点按文件顺序全量保留。
- **单元**（`NELEM= n` + n 行 `type v... [elemId]`）：仅 VTK 类型 **5（triangle）/ 9（quad）** 为 cell，按文件顺序；环向任意（构造器归一为 CCW）；行尾单元号可选、忽略；其他类型（含此处出现的 line、高阶单元）拒绝。
- **marker → patch**（`NMARK= m`）：每个 marker 依次为 `MARKER_TAG= name`（非空、不得重名）+ `MARKER_ELEMS= k` + k 行 `3 v1 v2`（仅 VTK line）。**marker 在文件中的出现顺序即 patch 索引顺序**（格式自带的确定性，不依赖单元顺序）。marker 边必须落在边界上——由 cell 邻接计数判定：出现 0 次 = 非单元边、2 次 = 内部边、≥3 次 = 非流形，全部拒绝；每条边界边恰好属于一个 marker 由 `UnstructuredMesh` 构造器保证。

### 错误处理

文件打开失败抛 `std::runtime_error`；其余格式/拓扑错误抛 `std::invalid_argument`，消息形如 `Su2Reader: '<file>': line N: <原因>`，尽量带上 marker 名/类型码等上下文。`UnstructuredMesh` 构造器抛出的异常会被捕获并加 `Su2Reader: '<file>': ` 前缀重抛。

## writeVti / writeVtu：.vti/.vtu 输出

### 设计思路

- **选择 ImageData 的原因**：`.vti`（vtkImageData）描述"原点 + 间距 + 维度"的规则网格，与 `CartesianMesh` 的均匀笛卡尔结构一一对应——不需要输出任何点坐标或单元连接表，文件头三行即完整描述几何。文件可直接用 ParaView 打开。
- **无状态自由函数**：`writeVti` / `writeVtu` 各一次调用完成开文件、写 XML、关文件；`writeVti` 接受 `CartesianMesh`，`writeVtu` 接受 `UnstructuredMesh`。场以 `(名称, 指针)` 对的列表传入，一次调用可写任意多个标量场和向量场；空指针被跳过。
- **CellData 而非 PointData**：场存储在单元中心，因此写入 `<CellData>` 段。VTK ImageData 的拓扑描述是基于节点的，`WholeExtent` 写为 `0 nx 0 ny 0 0`（即节点数为 $(n_x+1) \times (n_y+1)$，z 方向退化为 1 层），单元数恰为 $n_x \times n_y$，与场数据长度一致。
- **ASCII 格式**：`format="ascii"`，牺牲文件体积换取可读性与调试便利（可直接文本查看数值）。数据按 j 行 i 列的顺序写出，与网格的行优先线性索引一致。
- **XML 属性转义**：`.vti` 与 `.vtu` 的 `<DataArray Name=...>` 都会对场名中的 `& < > " '` 做 XML 转义；这也是对既有 `.vti` 路径中一个潜在转义缺陷的修复。

### 输出文件结构

```xml
<VTKFile type="ImageData" ...>
  <ImageData WholeExtent="0 nx 0 ny 0 0"
             Origin="xMin yMin 0.0"
             Spacing="dx dy 1.0">
    <Piece Extent="...">
      <CellData>
        <DataArray type="Float64" Name="T" format="ascii"> ... </DataArray>
        <DataArray ... NumberOfComponents="3" ...> u v 0.0 ... </DataArray>
      </CellData>
    </Piece>
  </ImageData>
</VTKFile>
```

要点：

- 标量场：每个单元一个值，按 `field(i, j)` 逐点写出。
- 向量场：`NumberOfComponents="3"`，2D 向量 `(u, v)` 补零成三维 `(u, v, 0)`，以满足 VTK 对向量三分量的要求（ParaView 的 Glyph 等滤镜需要）。
- `Origin` 取 `(xMin, yMin)`，`Spacing` 取 `(dx, dy)`——几何信息直接来自网格对象，保证可视化坐标与离散坐标严格一致。

### 错误处理

文件打开失败时抛 `std::runtime_error`（含路径信息）。

### `.vtu` 输出

`writeVtu(filename, const UnstructuredMesh&, scalars, vectors)` 写 ascii VTK XML UnstructuredGrid，用于任意多边形非结构网格。几何写入 `<Points>` 的 `Float64` 三分量坐标（2D 时 z 固定为 `0.0`）；拓扑写入 `<Cells>`，其中 `connectivity` / `offsets` 为 `Int64`，`types` 为 `UInt8`。单元类型按节点数映射：3 → `VTK_TRIANGLE`（5），4 → `VTK_QUAD`（9），其余 → `VTK_POLYGON`（7）。

`<CellData>` 保持在单元中心：标量场每单元一个值，向量场每单元写 `u v 0.0`，两者都按 `UnstructuredMesh` 的单元顺序写出。单元连接表使用构造后的 CCW 顶点环——它与“面积向量相对 owner 向外”的网格符号约定同源，写出时不再隐式调整方向。文件打开失败同样抛 `std::runtime_error`。

