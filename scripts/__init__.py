"""Organized command libraries, including import compatibility for retained analyses."""
from pathlib import Path
_base=Path(__file__).resolve().parent
__path__.extend(str(p) for p in _base.rglob("*") if p.is_dir() and "__pycache__" not in p.parts)
