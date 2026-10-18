"""Read saved predictions only; never train or scan raw contacts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from microcture.tasks.task5.analysis import main

if __name__ == '__main__':
    main()
