# 数据来源与本地输入

原始数据不随 Git 提交。公开仓库只保留小型元信息、注释派生表及划分表，便于审阅来源和方法。

## 来源

- Micro-C：[GSE272159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272159)。原始分辨率 10 bp，参考染色体 `NC_000913.3`，长度 4,641,652 bp。
- 相关论文：[Elementary 3D organization of active and silenced E. coli genome](https://doi.org/10.1038/s41586-025-09396-y)。本项目为独立研究再分析。
- [保存的 GEO SOFT 元数据](metadata/GSE272159_family.soft) 包含样本及补充文件信息；公开记录见 [原始数据审计](../reports/data_audit.json)，完整六样本审计在运行后生成于 `reports/conditions_audit.json`。
- 注释输入为本课程提供的 `标注数据.xlsx`：Table 1 为基因，Table 4/5/6 为 OPCID/CHIN/CHID。文件未重新分发；从课程材料或原研究补充材料取得后，核对表名、列名及内容。它不等同于已经核实完整性的参考注释集。

## 需要放置的文件

```text
micro-c数据/
├── 标注数据.xlsx
├── GSE272159_37C_rep1.mapq_30.10.cool
├── GSE272159_37C_rep2.mapq_30.10.cool
├── GSM8950761_DstpA_rep1.MG1655.mapq_30.10.cool
├── GSM8950762_DstpA_rep2.MG1655.mapq_30.10.cool
├── GSM8950763_DhnsDstpA_rep1.MG1655.mapq_30.10.cool
└── GSM8950764_DhnsDstpA_rep2.MG1655.mapq_30.10.cool
```

前两个 WT 文件与注释用于任务一、二；后四个条件文件用于任务三、四。后者的下载地址明确列在 [conditions.json](../configs/conditions.json)，可用 `scripts/download_conditions.py` 断点续传，随后由 `scripts/prepare_conditions.py` 解压并审计。

WT 下载地址取自保存的 SOFT 元数据：

- [WT 37°C rep1 压缩 Cooler](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE272nnn/GSE272159/suppl/GSE272159_37C_rep1.mapq_30.10.cool.gz)
- [WT 37°C rep2 压缩 Cooler](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE272nnn/GSE272159/suppl/GSE272159_37C_rep2.mapq_30.10.cool.gz)

下载后解压到上述目录，保留原文件名。不要仅改变配置中的样本名称来替代真实条件。完整运行的原始数据与压缩归档占数 GB，请预留数据、缓存和环境空间；不需要为浏览精选结果下载这些文件。

## 目录约定

- `metadata/`：公开来源元信息。
- `processed/`：小型注释派生表、样本表与训练划分，可追踪。
- `cache/`：运行生成的矩阵和预处理缓存，忽略。

`processed/` 中的表用于审阅和追踪，不能替代原始矩阵或 Excel 来执行完整审计。再生成后应比较条目数、坐标和来源，不能只比较文件是否存在。

## 坐标与许可

来源注释未说明坐标约定，目前保留原数值，暂按 0-based 左闭右开解释。所有相关输出保留 `provisional` 标记。要求已核实坐标的流程可调用：

```bash
.venv/bin/python scripts/prepare_data.py --require-verified-coordinates
```

在来源约定未确认前，该命令会拒绝执行。±1 bp 敏感性未改变本批候选的注释重叠状态，但不等于已经核实来源约定。

代码的 MIT 许可不覆盖原始数据、第三方注释或论文。使用这些材料请遵守来源条款并引用原研究。课程任务 PDF 留在本地，不随代码仓库发布。
