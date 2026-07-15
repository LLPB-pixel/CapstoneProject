#!/usr/bin/env python3
from pathlib import Path
import runpy

SCRIPT = Path(__file__).resolve().parent / "2-fase2" / "train_distilbert.py"
runpy.run_path(str(SCRIPT), run_name="__main__")
