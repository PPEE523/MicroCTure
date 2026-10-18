# 保存结果与可视化

这些文件是第一版计算输出的字节级快照，[manifest](results/manifest.json)保留来源和SHA。本轮没有重训任务二、五、六或重绘全部轨道。当前任务一的主结果在[新版比较图](../submission/figures/task1_formal_comparison.png)和[新OOF](../submission/task1/formal/oof/)，这里旧任务一表仅作失败历史。

| 内容 | 图/交互 | 数值 |
| --- | --- | --- |
| 新版分类 | [四模型图](../submission/figures/task1_formal_comparison.png) | [新比较表](../submission/task1/formal/comparison.csv) |
| 候选发现 | [重复对照](results/task2_v2/replication_comparison.png) | [候选](results/task2_v2/unannotated_supported_disjoint.csv) |
| 多条件 | [代表轨道](results/task3/tracks/0050000_0060000.png)、[差异图](results/task4/effects.png) | [汇总](results/task4/summary.csv) |
| 超分辨 | [结构剖面](results/task5/structure_profiles.png)、[结果页](results/task5/index.html) | [指标](results/task5/summary.csv) |
| 三维 | [交互视图](results/task6/index.html)、[接触图](results/task6/contact_maps.png) | [指标](results/task6/metrics.csv) |
| 第一版分类历史 | [旧图](results/task1/comparison.png) | [旧表](results/task1/comparison.csv) |

校验运行 `python -B scripts/export_showcase.py --check`；导出覆盖功能已停用，避免改写历史快照。固定超分辨示例数值仍保留在 `site/assets/matrix_preview.json`，来源为首个CHID测试位置，未按质量挑图。
