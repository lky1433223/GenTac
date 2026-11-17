# GenTac

基于 Genesis 1.4.6 的视触觉仿真。固定台架以 GelSight Mini 的公开仿真配置为尺寸参考，通过原生 FEM + SAP 接触求解球形压头和凝胶的相互作用。输出凝胶表面几何和表面标记的三维运动。

## 固定台架

需要 Python 3.12–3.14 和 uv，固定台架使用 CPU。

```bash
uv sync --frozen --python 3.12
uv run --frozen python run_indentation.py
```

`--depth-mm` 控制压头的目标压入量，默认 0.5 mm。`--radius-mm` 和 `--hydroelastic-modulus-pa` 分别设置球半径和 SAP 凝胶接触模量，默认 3 mm 和 2 MPa。`--steps` 和 `--save-every` 控制仿真与采样。每次运行写入独立的 `output/indent-*/`，可用 `--out` 指定目录，已有目录须显式加 `--overwrite` 才能复用。

- `state.npz` 包含时序 `surface_xyz_m`、共享三角面索引 `surface_faces`、`marker_id`、`marker_xyz_m` 和 `marker_disp_m`。坐标均为传感器局部坐标，单位为米。标记用同一组重心权重跟踪表面材料点。
- `run.json` 记录材料、挂载、目标深度和诊断结果。`surface_overlap_m` 根据球心到变形后三角表面的最近距离计算，记录各采样时刻凝胶与理想球面的交叠深度。

台架采用平板凝胶、底面粘接和暂定材料参数。GelSight Mini 的尺寸与标记行列数参考 UniVTAC 的 TacEx 配置，其余物理参数和边界条件仍待标定。

## 早期位移约束原型

原型入口继续提供 UV 点阵渲染和四种形状的演示。它用解析形状指定受压点的位移，用于查看早期渲染和同步输出。

```bash
uv run --frozen python gentac.py --shape all
uv run --frozen python gentac.py --shape sphere --depth-mm 0.8 --vis --show-indenter
```

`--depth-mm` 是从凝胶静止表面量起的压入量。相机默认只拍凝胶，`--show-indenter` 会在画面中显示压头。

每种压头的输出位于 `output/<shape>/`，也可以通过 `--out` 指定目录。

- `rgb_*.png` 是按仿真步编号的相机帧。
- `summary.png` 并排显示压入前和压入后的 RGB。
- `state.npz` 保存与 RGB 对齐的 `step`、`time_s`、`indenter_tip_z`、相机原始 `depth` 和顶面位移 `top_disp`，以及网格静止位置 `rest`、末帧位置 `pos` 和顶面索引 `top_rest_idx`。长度单位为米。
