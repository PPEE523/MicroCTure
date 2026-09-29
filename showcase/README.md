# 精选成果

先看 [交互作品页](../site/index.html)：超分辨拖动对比、七模型三维切换，以及分类、发现和多条件分析入口。

| 内容 | 可视化 | 数值 |
|---|---|---|
| 超分辨 | [结构剖面](results/task5/structure_profiles.png) | [对照指标](results/task5/summary.csv) |
| 三维嵌入 | [交互视图](results/task6/index.html)、[接触图](results/task6/contact_maps.png) | [评价表](results/task6/metrics.csv) |
| 分类 | [模型与消融](results/task1/comparison.png) | [比较表](results/task1/comparison.csv) |
| 候选发现 | [跨重复对照](results/task2_v2/replication_comparison.png) | [方法汇总](results/task2_v2/method_summary.csv) |
| 多条件 | [基因组轨道](results/task3/tracks/0050000_0060000.png) | [变化汇总](results/task4/summary.csv) |

23 个精选文件从本地原始结果按字节复制，来源与 SHA256 见 [manifest](results/manifest.json)。权重和完整研究输出留在本地。更新精选运行 `python3 scripts/export_showcase.py`，校验运行 `python3 scripts/export_showcase.py --check`。

展示页使用的热图数值为固定首个 CHID 测试示例，来源签名记录在 `site/assets/matrix_preview.json`；封面也由实际矩阵和保存坐标生成，没有用合成图片代替实验结果。
