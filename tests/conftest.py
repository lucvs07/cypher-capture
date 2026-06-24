"""Make the project root importable so `from src...` works under pytest
regardless of the working directory it is launched from."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
