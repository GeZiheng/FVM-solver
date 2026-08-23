# io 模块说明

`fvm::io` 命名空间，负责将计算结果写出为可视化文件。目前只有一个类 `VtkWriter`，输出 VTK ImageData（`.vti`）格式。

## 文件结构

| 文件 | 内容 |
|------|------|
| `include/VtkWriter.h` + `src/VtkWriter.cpp` | `VtkWriter` 类（仅静态方法，无状态） |

## VtkWriter：.vti 输出

### 设计思路

- **选择 ImageData 的原因**：`.vti`（vtkImageData）描述"原点 + 间距 + 维度"的规则网格，与 `CartesianMesh` 的均匀笛卡尔结构一一对应——不需要输出任何点坐标或单元连接表，文件头三行即完整描述几何。文件可直接用 ParaView 打开。
- **无状态静态接口**：`VtkWriter::write` 是纯静态方法，一次调用完成开文件、写 XML、关文件。场以 `(名称, 指针)` 对的列表传入，一次调用可写任意多个标量场和向量场；空指针被跳过。
- **CellData 而非 PointData**：场存储在单元中心，因此写入 `<CellData>` 段。VTK ImageData 的拓扑描述是基于节点的，`WholeExtent` 写为 `0 nx 0 ny 0 0`（即节点数为 `(nx+1)×(ny+1)`，z 方向退化为 1 层），单元数恰为 `nx×ny`，与场数据长度一致。
- **ASCII 格式**：`format="ascii"`，牺牲文件体积换取可读性与调试便利（可直接文本查看数值）。数据按 j 行 i 列的顺序写出，与网格的行优先线性索引一致。

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

## 依赖关系

```
io → core（CartesianMesh, ScalarField, VectorField）
```

io 不依赖 math / numerical，可在任何持有网格与场的上下文中使用。
