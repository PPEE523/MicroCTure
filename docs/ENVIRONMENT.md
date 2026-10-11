# WSL 本地 Conda 环境

项目环境名为 `microcture`，安装在 `/home/pppe/miniconda3/envs/microcture`。Conda 管理独立的 Python 环境，数据分析包和官方 CUDA 版 PyTorch 通过该环境内的 pip 安装。

日常使用：

```bash
cd /home/pppe/MicroCTure
conda activate microcture
python scripts/check_environment.py
```

无需激活也可运行：

```bash
conda run --no-capture-output -n microcture python scripts/check_environment.py
```

检查读取原始数据的小窗口与 Excel 标注，并执行 GPU 卷积、反向传播和优化器更新。结果写入 `reports/environment_check.json`；它验证环境可用，不代表任务一模型达标。

在另一套 WSL/Linux 安装时使用：

```bash
conda env create -f environment.yml
conda activate microcture
python scripts/check_environment.py
```

`environment.yml` 固定直接依赖版本；`environment.lock.yml` 记录本次完整依赖版本，供同平台复现。项目旧环境的依赖清单仅存于历史方法快照。

PyTorch 使用官方 CUDA 13.0 wheel，安装命令依据 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/)。当前 WSL 的 NVIDIA 驱动显示支持 CUDA 13.1；本次 GPU 是否可实际训练以环境检查记录为准。

更新依赖时先修改环境配置，再完成数据读取和 GPU 训练检查，并重新生成完整版本锁定记录。
