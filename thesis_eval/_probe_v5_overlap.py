"""Read-only probe: stem overlap + label-state transitions between IR corpus versions."""
from pathlib import Path

def lab(root):
    idx = {}
    for s in ("train", "val", "test"):
        d = Path(root) / s / "labels"
        if d.exists():
            for p in d.iterdir():
                if p.suffix == ".txt":
                    idx.setdefault(p.stem, p)
    return idx

V5, V6, FIN = lab("G:/drone/IR_dsetV5"), lab("G:/drone/IR_dsetV6"), lab("G:/drone/IR_dset_final")
print(f"V5 {len(V5)}  V6 {len(V6)}  FINAL {len(FIN)}")
print(f"V5∩V6 {len(V5.keys() & V6.keys())}   V5∩FIN {len(V5.keys() & FIN.keys())}   V6∩FIN {len(V6.keys() & FIN.keys())}")

def transitions(a, b, na, nb):
    shared = a.keys() & b.keys()
    e2f = f2e = 0
    examples = []
    for s in shared:
        ea = a[s].read_text().strip() == ""
        eb = b[s].read_text().strip() == ""
        if ea and not eb:
            e2f += 1
            if len(examples) < 5:
                examples.append(s)
        if eb and not ea:
            f2e += 1
    print(f"{na}->{nb}: shared {len(shared)}, empty->filled {e2f}, filled->empty {f2e}, ex: {examples}")

transitions(V5, V6, "V5", "V6")
transitions(V5, FIN, "V5", "FIN")
transitions(V6, FIN, "V6", "FIN")
