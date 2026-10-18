"""Portable, versioned task-3 atlas; uses saved signals, not contact scans."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from microcture.tasks.task3.atlas import main

if __name__ == "__main__":
    main()
