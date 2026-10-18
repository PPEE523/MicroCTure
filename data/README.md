# 数据目录

原`micro-c数据/`已经迁入`data/raw/micro_c/`：这是原始输入，不是模型输出。`data/shared/`放共享注释、来源元数据和带状计数缓存；`data/task1/`、`task5/`、`task6/`放任务专用缓存、划分和预处理。旧的实验预测、指标、重建和坐标已从`data/legacy_experiments/results/`迁到项目`results/taskN/<实验>/<版本>/`。

原始Cooler、标注Excel、压缩源文件及所有保留数值未删除。旧科学清单中的路径是历史来源名称，由`configs/layout.json`定位当前文件；不修改旧清单伪造新签名。缓存是从原始计数推导的加速文件，不能当成新增独立样本。

## 来源

- Micro-C：[GSE272159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272159)。原始分辨率 10 bp，参考染色体 `NC_000913.3`，长度 4,641,652 bp。
- 相关论文：[Elementary 3D organization of active and silenced E. coli genome](https://doi.org/10.1038/s41586-025-09396-y)。本项目为独立研究再分析。
- [保存的 GEO SOFT 元数据](shared/metadata/GSE272159_family.soft) 包含样本及补充文件信息；公开记录见[原始数据审计](../reports/shared/data_audit.json)与[六样本保存结果核验](../reports/shared/conditions/task34_verification.json)。
- 注释输入为原研究的官方补充表：本地名为 `标注数据.xlsx`，Table 1 为基因，Table 4/5/6 为 OPCID/CHIN/CHID。可从[出版商补充材料](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-09396-y/MediaObjects/41586_2025_9396_MOESM3_ESM.xlsx)取得；本项目已核对其与本地工作簿字节一致，SHA-256为`a1fa9c0d359509a9eb319442c19e300cb5da85546eeb567368a390d22205167a`。原工作簿不重新分发。

## 需要放置的文件

```text
data/raw/micro_c/
├── 标注数据.xlsx
├── GSE272159_37C_rep1.mapq_30.10.cool
├── GSE272159_37C_rep2.mapq_30.10.cool
├── GSM8950761_DstpA_rep1.MG1655.mapq_30.10.cool
├── GSM8950762_DstpA_rep2.MG1655.mapq_30.10.cool
├── GSM8950763_DhnsDstpA_rep1.MG1655.mapq_30.10.cool
└── GSM8950764_DhnsDstpA_rep2.MG1655.mapq_30.10.cool
```

前两个 WT 文件与注释用于任务一、二；后四个条件文件用于任务三、四。后者的下载地址明确列在 [conditions.json](../configs/shared/conditions/conditions.json)，可用 `scripts/shared/data/download_conditions.py` 断点续传。解压与审计必须指定新的报告目录：

```bash
conda run --no-capture-output -n microcture python -B scripts/shared/data/prepare_conditions.py \
  --report-dir reports/shared/conditions/audit_next_v1
```

它保存新的`condition_samples.csv`与`audit.json`，不会替换已保留的共享样本表和历史审计；已有原始矩阵不会重新解压覆盖。

WT 下载地址取自保存的 SOFT 元数据：

- [WT 37°C rep1 压缩 Cooler](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE272nnn/GSE272159/suppl/GSE272159_37C_rep1.mapq_30.10.cool.gz)
- [WT 37°C rep2 压缩 Cooler](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE272nnn/GSE272159/suppl/GSE272159_37C_rep2.mapq_30.10.cool.gz)

下载后解压到上述目录，保留原文件名。不要仅改变配置中的样本名称来替代真实条件。完整运行的原始数据与压缩归档占数 GB，请预留数据、缓存和环境空间；不需要为浏览精选结果下载这些文件。

## 目录约定

- `shared/metadata/`：公开来源元信息。
- `shared/annotations/`：小型注释派生表；`shared/metadata/`：来源和样本表，可追踪。
- `shared/cache/`、`task1/cache/`、`task6/cache/`：运行生成的矩阵缓存，不随Git发布。
- `task1/ablation/`：原生分辨率等消融专用预处理；`task5/splits/`：固定训练划分。

派生表用于审阅和追踪，不能替代原始矩阵或 Excel 来执行完整审计。再生成后应比较条目数、坐标和来源，不能只比较文件是否存在。

## 坐标与许可

作者结构表的端点约定尚未独立确认，保留原数值与 `provisional` 标记。Annotation QC已完成：结构回查344/344原行；作者基因表4,516行逐项与RefSeq核查，默认新图层采用明确转换后的独立RefSeq注释。作者结构的端点约定与RefSeq基因转换是两个问题，不能相互认证。完整方法见[任务三](../docs/TASK3.md)。要求作者结构端点已经核实的流程可调用：

```bash
conda run --no-capture-output -n microcture python -B scripts/shared/data/prepare_data.py --require-verified-coordinates
```

在来源约定未确认前，该命令会拒绝执行。±1 bp 敏感性未改变本批候选的注释重叠状态，但不等于已经核实来源约定。

代码的 MIT 许可不覆盖原始数据、第三方注释或论文。原论文标明CC BY 4.0；派生作者注释和本项目可视化应注明原研究、来源及转换，单独标有其他权利声明的材料仍按其声明处理。固定RefSeq参考来自NCBI；[NCBI数据使用政策](https://www.ncbi.nlm.nih.gov/home/about/policies/)说明其分子数据的使用范围及提交者权利边界。GEO SOFT中的研究联系人邮箱属于公开数据来源元信息，不是项目作者联系方式。课程PDF、原论文PDF和完整工作簿仅在本地留存，不随代码发布。

## 显式缓存准备

缓存缺失不会自动重扫原始矩阵。取得并核对原始文件后，从项目根目录显式构建新版本：

```bash
# 两WT：任务二/五等共用；已有缓存时直接只读复用
conda run --no-capture-output -n microcture python -B run.py prepare-bands \\
  --data-root . --output-dir data/shared/cache/wt_next_v1
# 六样本：任务三/四；不能用两WT缓存替代
conda run --no-capture-output -n microcture python -B run.py prepare-bands \\
  --data-root . --conditions --output-dir data/shared/cache/conditions_next_v1
```

每样本主带约46,417×256 float32，约47.5 MB；构建另需float64累加、coverage及HDF5块，建议至少2 GB可用内存。两WT原始合计约5.07亿条非零记录，耗时依WSL磁盘与CPU而变，完整新入口没有重新计时。成功为预期2或6份NPZ及完整manifest；将新目录作为相应任务的`--band-cache-dir`。任务一独立分类缓存与任务六5 kb缓存分别见TASK1、TASK6，原始预处理不能视为新增独立数据集。
