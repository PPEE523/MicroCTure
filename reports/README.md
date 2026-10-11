# 数据与历史实验审计记录

本目录保留旧记录作为报告出处。数据审计与数值复算通过，仅说明对应范围内的检查完成，不说明三类识别性能达标。

- [新四模型代码检查](task1_comparison_code_check.json)：23 项合成/前向/模拟流程检查通过；没有进行真实数据准备或模型训练。运行方式见 [本地对比说明](../docs/TASK1_COMPARISON.md)。

- [原始数据审计](data_audit.json)、[条件样本审计](conditions_audit.json)：原始文件、样本对应与来源信息。
- [旧分类数据集](classification_dataset.json)：旧划分、类别数、归一化和坐标尚未核实的状态。
- [旧任务一结果](task1_results.json)：M2 未达标的历史结果；见 [失败复盘](../report_materials/task1_failure_review.md)。
- 其他 JSON：局部矩阵、候选发现、条件分析、超分辨和三维实验的历史数值与检查范围。

原始数据、历史数值文件和写作材料的位置见 [项目说明](../README.md)。旧权重和执行流水线已清理，本目录的核验记录保留其当时的出处。
