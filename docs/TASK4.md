# 任务四：已知结构区间的条件差异

[返回项目概览](../README.md) · [任务三](TASK3.md) · [任务五](TASK5.md)

**当前结论：344个作者区间、六样本信号及688次比较已完成并逐条复算；327个质量合格区间中，ΔstpA有10个、双缺失有94个满足固定描述性变化规则。** 这不是统计显著性或结构类型转化的证明。

本文件是任务四唯一持续维护的任务说明。共享环境见[README](../README.md)，数据与轨道方法见[任务三](TASK3.md)。本任务采用作者已知区间，不使用任务一分类器的预测，也不依赖是否发现新结构簇。结果以2026-10-15已保存数值复核为准，没有在该次复核重新训练或扫描原始矩阵。

## 问题定义与实现

比较同一个已知区间在WT、ΔstpA和ΔhnsΔstpA中的相对接触信号。区间边界和类别固定为344个作者标注：OPCID68、CHIN250、CHID26。任务三提供六样本共同有效坐标与接触摘要；任务四进一步在每个区间使用相同接触对掩码，控制缺失位置和样本深度。

| 环节 | 固定规则 |
| --- | --- |
| 样本 | 每条件两个生物学重复，三个条件共六样本 |
| 接触范围 | 100 bp bin，区间内索引距离2–100；只使用上三角有效接触对 |
| 主信号 | 有效接触对平均计数按样本总接触数做每百万相对深度归一化 |
| 质量 | 至少20个有效接触对且有效比例≥0.8；同一区间的六样本共享掩码 |
| 描述性变化 | 四种处理重复/WT重复交叉比值必须全部>1.5或全部<1/1.5 |
| 敏感性 | 用距离期望校正的O/E重复比较，区分效应符号一致与完整阈值通过 |
| 数据单位 | 唯一区间、区间—样本行、区间—条件比较行分别计数；不可混用 |

配置为[conditions.json](../configs/shared/conditions/conditions.json)，实现为[analyze_conditions.py](../scripts/task3/analyze_conditions.py)，与任务三共用[run_conditions_local.py](../scripts/task3/run_conditions_local.py)的`check/review/analyze`入口。计数工具[contact_band.py](../scripts/shared/contact_band.py)不加载旧分类器、AE或PyTorch。

## 比较定义与质量规则

在固定344个作者区间上，提取6样本信号，共2,064条区间—样本数值，比较ΔstpA/WT和双缺失/WT，共688次。要求有效接触对至少20、有效比例至少0.8；327个区间合格、17个排除，实际合格比较行654条。

主效应是`log2((处理组两重复平均信号+0.01)/(WT两重复平均信号+0.01))`。变化还必须满足四种处理重复/WT重复交叉比值**全部>1.5或全部<1/1.5**。四个交叉比值来自每条件两个重复，不是四个独立生物学重复。

## 实际结果

| 条件 | 类别 | 质量合格/总数 | 增强 | 减弱 | 合格区间log2FC中位数 |
| --- | --- | ---: | ---: | ---: | ---: |
| ΔstpA | OPCID | 68/68 | 3 | 6 | 0.0416 |
| ΔstpA | CHIN | 237/250 | 1 | 0 | 0.2344 |
| ΔstpA | CHID | 22/26 | 0 | 0 | 0.2335 |
| ΔhnsΔstpA | OPCID | 68/68 | 0 | 6 | −0.1271 |
| ΔhnsΔstpA | CHIN | 237/250 | 77 | 0 | 0.4409 |
| ΔhnsΔstpA | CHID | 22/26 | 11 | 0 | 0.6441 |

![任务四：条件效应分布、区间质量和变化数量](../submission/figures/task34_delivery_effects.png)

**图1分析。** ΔstpA有10/327个区间满足变化规则，约3.06%；双缺失有94/327个，约28.75%。双缺失的CHIN、CHID分布偏正，分别有77、11个区间相对接触增强；OPCID中位数偏负，6个减弱。这些结果说明已知区间中的相对接触信号具有不同的条件响应，不能改写为CHIN/CHID数量增加或三维结构类型发生转化。

距离背景敏感性核查显示，104条主规则变化记录的O/E效应符号全部同向，其中103条也通过O/E的完整四比值规则。唯一例外是ΔstpA的OPCID_55：主log2FC为0.8497，O/E为0.5333，后者没有通过完整四比值规则。同向与通过同一阈值是两个不同端点，不能混写成104条全部稳健。

2026-10-15结果复核从2,064条样本信号逐条复算688次主效应、O/E方向和质量规则，主效应最大误差为0。每条件只有2个生物学重复，未建立显著性检验、FDR或因果证据。

**任务四交付判定：描述性差异分析及敏感性核查已完成；统计显著性、结构形态变化与机制解释尚未验证。**

证据：[逐条件类别汇总及例外](../submission/task34_delivery_summary.json)、[保留汇总表](../showcase/results/task4/summary.csv)。完整表位于`results/task4/condition_comparison/v1/structure_signals.csv`与`differential_structures.csv`。实现：[条件分析](../scripts/task3/analyze_conditions.py)、[结果复算绘图](../scripts/task3/build_task34_delivery.py)。

## 结果解释与完成状态

质量不合格的17个区间为13个CHIN、4个CHID；按两条件比较则有34行不合格、654行合格。104条变化记录是两个处理合计的区间—条件行，不应写成104个不同区间，更不能变成新增结构数。

| 可支持的结论 | 不支持的扩展 |
| --- | --- |
| 在固定区间和共同掩码下，双缺失更多CHIN/CHID区间的相对接触信号满足增强规则 | CHIN/CHID数量增加、类别转换或细胞中出现新类型 |
| 104条变化的O/E效应方向一致，其中103条也通过O/E完整四比值规则 | 所有尺度完全一致或104条全部通过敏感性判据 |
| 六样本保存信号、688个主效应与质量决策可复算，主效应最大误差0 | 保存表复核已完成原始数据全部字节的独立重验证 |
| 两生物学重复下得到稳健方向的描述性筛选结果 | p值、FDR显著性、绝对接触总量变化或蛋白因果机制 |

共同有效掩码可能排除条件特异性缺失，深度归一化信号不能解释为绝对接触数。注释坐标仍标记`provisional`；在坐标约定、专家形态与独立实验没有确认前，不升级为形态或机制结论。

## 运行、资源与验收

在克隆后的项目根目录、Conda `microcture`中运行。需要真实六样本、作者区间、来源审计和六份100 bp带缓存；取得与缓存构建方式见[README](../README.md)和[数据说明](../data/README.md)。任务三四共享一次运行，不必分别训练或重复计算。

**检查六样本，并复算保留的比较：**

```bash
conda run --no-capture-output -n microcture python -B run.py task34 check \
  --data-root . --run-dir results/task3/verification/check_next_v1
conda run --no-capture-output -n microcture python -B run.py task34 review \
  --data-root . --run-dir results/task3/verification/review_next_v1
```

`review`只读原保存表，输出新目录的`preserved_review.json`与`manifest.json`。成功判据是2,064条唯一结构—样本数值、688条唯一结构—条件比较、344个作者区间身份及坐标一致、逐条主效应/OE效应和质量决策一致、主log2FC最大误差0。仅有汇总数相同不能替代逐条比较。

**从既有缓存完整重算并出图（耗时分析，另选新目录）：**

```bash
conda run --no-capture-output -n microcture python -B run.py task34 analyze \
  --data-root . --run-dir results/task3/condition_tracks/next_v1
```

若六样本缓存另建，在上述命令增加`--band-cache-dir data/shared/cache/conditions_next_v1`。已有输出拒绝覆盖，请使用独立版本名。缓存缺失时不会自动扫描raw矩阵；该命令联动输出任务三的465页轨道，不能将其资源开销看成仅688行CSV计算。

| 操作 | 资源与耗时依据 | 输出/验收 |
| --- | --- | --- |
| 保存表`review` | CPU轻量数值复核；读取CSV、轨道NPZ及缓存哈希，耗时依磁盘而变 | `preserved_review.json`的`comparisons_checked=688`、`structure_sample_values=2064`、`max_log2_fc_error=0` |
| `analyze` | 建议至少2 GB可用内存和完整图片磁盘空间；联动465轨道及20张条件热图，可能数分钟或更久，整合入口完整绘图未实测 | `task4/structure_signals.csv`、`differential_structures.csv`、`summary.csv`、20张热图、效果图及manifest；同时通过任务三覆盖验收 |
| 结果图表重绘 | `scripts/task3/build_task34_delivery.py`只读保留NPZ/CSV，当前为数秒级CPU，建议约1 GB内存 | 任务四效应图、任务三轨道图和`submission/task34_delivery_summary.json`；不改变原始比较 |

完整运行后回传`manifest.json`、`task4/summary.csv`、`task4/differential_structures.csv`、`task4/structure_signals.csv`与stdout/stderr。验收需保留OPCID_55这个O/E阈值例外；不能只筛选效果更强、方向一致的案例。

## 后续工作与验收边界

现阶段描述性条件差异任务可以交付。下一阶段如果要增加“显著性”或“机制”结论，需要更多独立生物学重复、预登记统计模型与多重检验控制，以及针对区间形态/功能的外部验证；不能仅调整1.5倍阈值或将四个交叉比值当四个独立样本来制造显著性。

后续规则、结果图、例外及运行验收继续更新本文件，不另建多份任务四阶段总结。
