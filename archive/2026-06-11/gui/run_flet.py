"""Launch the TALOS Flet desktop app."""
import sys
from pathlib import Path

# Ensure workspace root is on path
_WS = Path(__file__).resolve().parents[1]
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))

from flet_app.main import run

if __name__ == "__main__":
    run()
