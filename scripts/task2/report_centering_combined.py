"""Render primary and exploratory supplement separately without modifying inputs."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import argparse
import json
from pathlib import Path
from report_centering import ROOT, read_rows, write_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(_layout_resolve(ROOT/'configs/centering_sensitivity.json')))
    parser.add_argument('--supplement-config', default=str(_layout_resolve(ROOT/'configs/centering_supplement.json')))
    parser.add_argument('--primary-dir')
    parser.add_argument('--supplement-dir')
    parser.add_argument('--output-dir', default=str(_layout_resolve(ROOT/'results/task2_centering_combined_review_v2')))
    args = parser.parse_args()
    cfg = json.loads(_layout_resolve(args.config).read_text())
    extra = json.loads(_layout_resolve(args.supplement_config).read_text())
    primary = _layout_resolve(args.primary_dir or _layout_resolve(ROOT/cfg['output_dir']))
    supplement = _layout_resolve(args.supplement_dir or _layout_resolve(ROOT/extra['output_dir']))
    output = _layout_resolve(args.output_dir)
    primary_report = write_review(primary, output/'primary', cfg, title='Primary matched centering evidence')
    supplemental_report = write_review(supplement, output/'supplement', cfg, read_rows(primary/'variants.csv'),
                                      title='Exploratory calibration supplement (previously viewed regions)')
    summary = dict(primary=primary_report, exploratory_supplement=supplemental_report,
                   independent_confirmation=False, interpretation='Supplement reuses previously viewed evaluation regions; it does not repair an independent test set.')
    (output/'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>主实验与探索性补充</h1><p>NA表示未评估，不是阴性。补充组使用已查看区域，不能当作独立确认。</p><a href="primary/index.html">主实验</a> · <a href="supplement/index.html">探索性补充</a>')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
