"""Build a standalone coursework ZIP from the checked source/evidence inventory."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from check_repository import public_files, ROOT


DELIVERY_GUIDE = """# MicroCTure 代码交付说明

本包包括六项任务的源码、配置、环境版本、测试和精选结果，解压后的唯一项目根目录为 `MicroCTure/`。项目概览见 [README](README.md)，完整方法和实验命令见 [六任务文档](docs/README.md)。

## 先核验和浏览

在 Linux / WSL 终端进入解压后的 `MicroCTure/`，首次创建环境；已有 `microcture` 环境时跳过安装：

```bash
conda env create -f environment.yml
conda run --no-capture-output -n microcture python -B run.py check
conda run --no-capture-output -n microcture python -B run.py demo
conda run --no-capture-output -n microcture python -B run.py serve
```

`check`核验保存证据、文档链接及合成/数值测试；`demo`以CPU执行随包M2+固定三例推理。这两步不需要原始矩阵，不训练模型。测试若依赖未分发的历史实验文件，会明确标记跳过。浏览打开 `http://localhost:8000/site/`；完整图集位于 `http://localhost:8000/submission/task3/atlas/v1/`，含465张实际PNG和对应浏览页。关闭服务用 Ctrl+C。

环境安装需要联网；随包结果浏览不依赖CDN。`environment.yml`指定环境，`environment.lock.yml`记录版本。当前运行与验收使用WSL的Conda `microcture`环境，其他系统的安装兼容性未逐一实测。

## 六任务代码入口

下列命令只查看帮助，不启动实验。完整参数、输入和成功判据分别见任务文档。

| 任务 | 命令 | 主要实现与说明 |
| --- | --- | --- |
| 任务一：结构分类 | `python -B run.py task1 --help` | `microcture/tasks/task1/`；模型比较集中在`comparison.py`，消融集中在`ablation.py`；[TASK1](docs/TASK1.md) |
| 任务二：候选发现与验证 | `python -B run.py task2 --help` | `scripts/task2/`与`microcture/tasks/task2/`；固定候选验证另用`validate-task2`；[TASK2](docs/TASK2.md) |
| 任务三：多条件轨道 | `python -B run.py task34 --help`、`python -B run.py task3-atlas --help` | `scripts/task3/`与`microcture/tasks/task3/`；[TASK3](docs/TASK3.md) |
| 任务四：条件差异 | `python -B run.py task34 --help` | 与任务三共用六样本读取，区间差异实现在`scripts/task3/analyze_conditions.py`，不是缺少任务四代码；[TASK4](docs/TASK4.md) |
| 任务五：合成超分辨率 | `python -B run.py task5 --help`、`python -B run.py task5-analysis --help` | `scripts/task5/`与`microcture/tasks/task5/`；[TASK5](docs/TASK5.md) |
| 任务六：三维接触嵌入 | `python -B run.py task6 --help` | `scripts/task6/`；[TASK6](docs/TASK6.md) |

## 文件分区

- `microcture/tasks/`、`scripts/taskN/`：当前任务实现与入口。
- `scripts/shared/`、`scripts/project/`：共享数据处理、核验和打包工具。
- `configs/`：任务与实验配置；`layout.json`解析历史路径，保留以支持历史签名和测试。
- `microcture/history/`、`scripts/task1/history/`、`configs/task1/history/`：原字节保留的历史实现，不是当前默认训练入口。
- `data/`：小型注释、样本信息与固定划分；数据来源与目录见[data/README](data/README.md)。
- `submission/`：15份M2+权重、固定推理输入、六任务精选表图、任务二候选核查和465页图集。
- `showcase/`、`site/`：结果快照和交互浏览；`tests/`：可运行的规则与数值测试。

## 完整重跑所需输入与复现范围

原始Cooler矩阵、工作簿、大型缓存和完整历史`results/`不在代码包内。完整训练或结果数组复算前，应按数据说明取得输入，核对配置，并使用尚不存在的实验版本目录。不要将缺输入的训练命令当作安装自检；`run.py check`才是随包自检入口。

任务二、五、六各6份历史checkpoint缺失，共18份；保存的表图可以核验，但不能直接重放这些原模型。任务一随包提供15份M2+权重及固定推理演示。完整本地冻结证据的`verify-evidence --all`和带`--require-local-inputs`的复核要求额外本地文件，不能用来验收这个精简代码包。

本包保留负结果和NA：任务二候选不等同于确认新结构，任务四使用描述性规则，任务五像素改善与边界偏差并存，任务六提供相对接触嵌入。各任务文档给出具体证据和解释范围。

`DELIVERY_MANIFEST.json`列出随包文件大小与SHA-256；`submission/manifest.json`与`submission/results_delivery_manifest.json`保存精选科学证据的来源和校验。重新打包用`python -B run.py package --output 新文件名.zip`，拒绝覆盖已有ZIP。原始Word报告、PPT和视频应单独交付。
"""


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new ZIP path; refusing to overwrite {output}')
    if not output.is_relative_to(ROOT):
        raise ValueError('Write optional archives inside the existing MicroCTure directory')
    if not output.parent.is_dir():
        raise ValueError('Archive parent must be an existing project directory')
    subprocess.run([sys.executable, '-B', str(_layout_resolve(ROOT/'scripts/check_repository.py'))], cwd=ROOT, check=True)
    files = [p for p in public_files()
             if p.relative_to(ROOT).as_posix() not in
             {'AGENTS.md', 'DELIVERY_MANIFEST.json', 'START_HERE.md'}]
    records = [{'path':p.relative_to(ROOT).as_posix(), 'bytes':p.stat().st_size,
                'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    guide = DELIVERY_GUIDE.encode('utf-8')
    records.append({'path':'START_HERE.md', 'bytes':len(guide),
                    'sha256':hashlib.sha256(guide).hexdigest()})
    manifest = {'schema_version':1, 'description':'Final coursework sources and selected computed evidence; no raw matrices or training caches.',
                'start_here':'START_HERE.md', 'tasks':[1, 2, 3, 4, 5, 6],
                'default_classifier':'I3/H3 M2+, lr1e-4, dropout0.2, final40',
                'biological_morphology_reliability_established':False,
                'files':records}
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open('xb')
    owned_file = os.fstat(stream.fileno())
    try:
        with stream:
            with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for record, path in zip(records, files):
                    payload = path.read_bytes()
                    if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
                        raise ValueError(f'Source changed while packaging: {record["path"]}')
                    item = zipfile.ZipInfo('MicroCTure/'+record['path'], date_time=(2026,10,14,0,0,0))
                    item.compress_type = zipfile.ZIP_DEFLATED
                    item.external_attr = 0o100644 << 16
                    archive.writestr(item, payload)
                archive.writestr('MicroCTure/START_HERE.md', guide)
                archive.writestr('MicroCTure/DELIVERY_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    except BaseException:
        if output.exists():
            current = output.stat()
            if (current.st_dev, current.st_ino) == (owned_file.st_dev, owned_file.st_ino):
                output.unlink()
        raise
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({'zip':str(output),'files':len(records)+1,'bytes':output.stat().st_size,'sha256':digest}, ensure_ascii=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New ZIP file inside the existing project directory')
    args=parser.parse_args(argv)
    build(args.output)


if __name__=='__main__':
    main()
