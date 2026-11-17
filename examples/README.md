# 固定示例

从仓库根目录执行。首次使用先按根目录 README 同步依赖。

## 固定台架球形压入

使用默认 Mini 参考预设、3 mm 球半径和 0.5 mm 目标压入量，输出表面与标记轨迹。

```bash
uv run --frozen python scripts/run_indentation.py
```

## 位移约束原型

依次运行球、立方体、圆柱和楔形压头的 RGB 演示。用 `--out` 指定本次演示的结果目录。

```bash
uv run --frozen python examples/displacement_indentation.py --shape all --out output/displacement-demo
```
