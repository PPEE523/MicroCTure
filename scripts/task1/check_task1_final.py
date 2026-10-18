#!/usr/bin/env python3
"""Read-only check of the portable final task-one workflow and optional demo."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
_PROJECT_DIRECTORY = next(parent for parent in Path(__file__).resolve().parents
                          if (parent / "microcture/layout.py").is_file())
sys.path.insert(0, str(_PROJECT_DIRECTORY))
from microcture.layout import bootstrap
bootstrap()
from run_task1_local import main

if __name__ == "__main__":
    if "check" not in sys.argv[1:]:
        sys.argv.append("check")
    raise SystemExit(main())
