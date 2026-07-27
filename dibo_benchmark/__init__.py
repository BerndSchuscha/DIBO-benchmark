from pathlib import Path
import os
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
# --- Detect environment ---
if "SCRATCH" in os.environ and Path(os.environ["SCRATCH"]).exists():
    SAVE = Path(os.environ["SCRATCH"]) / "pibob"
else:
    SAVE = PROJECT_ROOT
