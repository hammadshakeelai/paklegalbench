import os
import sys
from pathlib import Path

# Ensure fast test runs without downloading or encoding dense models on CPU
os.environ.setdefault("PLB_NO_DENSE", "1")

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
