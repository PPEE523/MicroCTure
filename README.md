# MicroCTure
Deep learning-based Micro-C contact matrix processing and novel chromatin structure discovery

## 当前进度

步骤一：样本登记、结构标注整理、数据检查与独立环境。尚未训练模型。

- [实施路线](docs/项目分析与实施路线.md)
- [数据准备说明](docs/数据准备.md)
- [样本与坐标配置](configs/data.json)
- 标准输出：`data/processed/structures.csv`、`samples.csv`、`structure_overlaps.csv`
- 检查报告：`reports/data_audit.json`

## 环境与复现

本次环境为 Linux、Python 3.14。`requirements.txt` 固定实际安装的全部依赖；
`requirements.in` 列出直接依赖。模型训练环境将在后续步骤验证。

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-build.txt
.venv/bin/python -m pip install --no-build-isolation -r requirements.txt
.venv/bin/python scripts/prepare_data.py
.venv/bin/python -m unittest discover -s tests -v
```

原始文件按 `configs/data.json` 放入 `micro-c数据/`，不提交 Git。
脚本不会修改原始文件；从任意工作目录执行都会向本项目写出结果。
全基因组只读取 bin 和索引，接触矩阵仅做 1 kb 局部读取检查。

**坐标限制：**原表没有声明坐标约定。目前保留原始数值，暂按 0-based 左闭右开解释，
每条输出均标记 `coordinate_status=provisional`。这不代表来源约定已核实。
需要已核实坐标的流程可运行 `scripts/prepare_data.py --require-verified-coordinates`，
当前会明确拒绝执行。确认来源后应修改配置并重新生成结果，不能仅删除警告。
