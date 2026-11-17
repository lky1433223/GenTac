# GenTac

基于 Genesis 1.4.6 的视触觉仿真。固定台架以 GelSight Mini 的公开仿真配置为尺寸参考，通过原生 FEM + SAP 接触求解球形压头和凝胶的相互作用。输出凝胶表面几何和表面标记的三维运动。

## 目录结构

- `src/gentac/` 存放可导入的仿真、传感器定义和读数代码。
- `scripts/` 存放日常运行的 CLI 入口。
- `examples/` 存放固定示例及其运行命令。
- `tests/` 预留给自动化测试。

## 固定台架

需要 Python 3.12–3.14 和 uv，固定台架使用 CPU。

```bash
uv sync --frozen --python 3.12
uv run --frozen python scripts/run_indentation.py
```

`--depth-mm` 控制压头的目标压入量，默认 0.5 mm。`--radius-mm` 和 `--hydroelastic-modulus-pa` 分别设置球半径和 SAP 凝胶接触模量，默认 3 mm 和 2 MPa。`--steps` 和 `--save-every` 控制仿真与采样。每次运行写入独立的 `output/indent-*/`，可用 `--out` 指定目录，已有目录须显式加 `--overwrite` 才能复用。

- `state.npz` 包含时序 `surface_xyz_m`、共享三角面索引 `surface_faces`、`marker_id`、`marker_xyz_m` 和 `marker_disp_m`。坐标均为传感器局部坐标，单位为米。标记用同一组重心权重跟踪表面材料点。
- `run.json` 记录材料、挂载、目标深度和诊断结果。`surface_overlap_m` 根据球心到变形后三角表面的最近距离计算，记录各采样时刻凝胶与理想球面的交叠深度。

台架采用平板凝胶、底面粘接和暂定材料参数。GelSight Mini 的尺寸与标记行列数参考 UniVTAC 的 TacEx 配置，其余物理参数和边界条件仍待标定。

## 早期位移约束原型

原型入口继续提供 UV 点阵渲染和四种形状的演示。它用解析形状指定受压点的位移，用于查看早期渲染和同步输出。

```bash
uv run --frozen python examples/displacement_indentation.py --shape all
uv run --frozen python examples/displacement_indentation.py --shape sphere --depth-mm 0.8 --vis --show-indenter
```

`--depth-mm` 是从凝胶静止表面量起的压入量。相机默认只拍凝胶，`--show-indenter` 会在画面中显示压头。

每种压头的输出位于 `output/<shape>/`，也可以通过 `--out` 指定目录。

- `rgb_*.png` 是按仿真步编号的相机帧。
- `summary.png` 并排显示压入前和压入后的 RGB。
- `state.npz` 保存与 RGB 对齐的 `step`、`time_s`、`indenter_tip_z`、相机原始 `depth` 和顶面位移 `top_disp`，以及网格静止位置 `rest`、末帧位置 `pos` 和顶面索引 `top_rest_idx`。长度单位为米。

## 查看固定台架结果

把运行结束时打印的目录传给绘图脚本。绘图直接读取 `state.npz`，也支持已有的固定台架结果。

```bash
uv run --frozen python scripts/plot_indentation.py output/<run-id>
uv run --frozen python scripts/plot_indentation.py output/<run-id> --frame 10 --texture ./pattern.png
```

将命令中的 `<run-id>` 替换为实际目录名。默认使用最后一个采样帧。`--frame` 是从零开始的采样索引，支持负数。每帧生成独立的 `plots-step-*/` 目录，已有同名目录时报错。

- `diagnostics.png` 显示压陷、标记位移、球面交叠的时序及选中帧的位移场。
- `simulation.png` 对比静止和变形后的 FEM 三维表面及标记位置。
- `texture.png` 在传感器 XY 平面俯视比较静止和变形贴图。默认使用棋盘格。指定图片时，图像上边对应传感器的 +Y 方向。每个表面三角形携带从静止位置计算的 UV，绘图时用变形后的顶点插值。该图显示面内形变，三维高度变化可在仿真网格图和压陷图中查看。
