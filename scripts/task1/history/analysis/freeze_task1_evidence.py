#!/usr/bin/env python3
"""Freeze the failed comparison in place, or check its frozen SHA-256 inventory."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = _layout_root
sys.path.insert(0, str(ROOT))
from microcture.task1_controlled.evidence import freeze_failed_experiment, verify_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify"))
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "evidence/task1_failed_v1")
    args = parser.parse_args()
    operation = freeze_failed_experiment if args.command == "freeze" else verify_evidence
    print(json.dumps(operation(ROOT, args.evidence_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
