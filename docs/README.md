# 六任务文档目录

[返回项目概览](../README.md)

每个任务只维护下面一份Markdown。问题定义、方法、结果、可视化分析、运行与验收、剩余目标都在对应文件中；环境安装、共用数据缓存、项目全貌与图表索引命令统一维护在项目README。

| 唯一维护文档 | 内容 |
| --- | --- |
| [TASK1.md：已知结构分类](TASK1.md) | 四模型、I3/H3、空间评价、消融/受控改进、推理与可靠性边界 |
| [TASK2.md：新结构候选](TASK2.md) | 固定预算检索、AE/DAE、重复与新簇判据、形态/居中核查及NA |
| [TASK3.md：全基因组轨道](TASK3.md) | 三级图集、完整465 PNG／HTML、RefSeq坐标核查、作者注释来源与重复一致性 |
| [TASK4.md：条件差异](TASK4.md) | 344区间、688比较、描述性规则与O/E敏感性 |
| [TASK5.md：合成超分辨率](TASK5.md) | 插值/CNN对照、像素和边界取舍、保存结果复核 |
| [TASK6.md：相对三维嵌入](TASK6.md) | MDS/Geometry/Graph、多指标取舍、留出边与构象限制 |

六文档共保留20幅结果图，包含任务二固定21候选证据与任务五新增逐窗口、距离分层及三面板结论图。更新实验时直接修改对应任务，并同步README中的概览；不再新增“阶段总结”“新版结果”或综合说明与它们平行维护。只改文字后执行`run.py report-results refresh`，再`run.py check`。

## 历史材料与冻结证据

历史阶段文档和作业报告仅作留存，不再平行维护。被冻结的旧源码、协议和配置集中在`microcture/history/task1/`、`scripts/task1/history/`和`configs/task1/history/`；`configs/layout.json`保留旧逻辑路径到新任务/实验目录的映射，不改写科学文件的原签名。WORK_LOG已停止维护，不再恢复或新增。

旧索引通过`run.py verify-evidence --all`读取原路径、解析当前位置并核对完整文件集合与哈希。当前运行只使用`microcture/tasks/task1/`中的合并实现；具体结构及数据区别见项目README。
