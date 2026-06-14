"""List \\ref{} targets in the thesis chapters with no \\newlabel in the aux files (post-compile)."""
import re
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "docs/thesis_working_distilling_overleaf"
CH = ["introduction", "related_work", "methodology", "empirical", "conclusion", "appendices"]
aux = set()
for f in [D / "main.aux"] + [D / f"chapters/{c}.aux" for c in CH]:
    if f.exists():
        aux |= {m.group(1) for m in re.finditer(r"\\newlabel\{([^}]*)\}", f.read_text(encoding="utf-8", errors="ignore"))}
refs = set()
for c in CH:
    t = (D / f"chapters/{c}.tex").read_text(encoding="utf-8", errors="ignore")
    refs |= {m.group(1) for m in re.finditer(r"\\ref\{([^}]*)\}", t)}
missing = sorted(r for r in refs if r not in aux)
print("missing:", missing if missing else "none")
