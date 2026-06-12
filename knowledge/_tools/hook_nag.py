#!/usr/bin/env python3
"""hook_nag.py — knowledge-system backstop hook (advisory, never blocking).

Two modes, wired to Claude Code hooks (see .claude/settings.json):
  snapshot   (SessionStart)  record the current `git status` so we can diff later
  check      (Stop)          if this session created/changed .py scripts but did NOT
                             record into knowledge/*.csv, print a gentle reminder

Session-scoped: it only looks at changes *since* the session-start snapshot, so it does
NOT fire on the repo's pre-existing dirty state. Always exits 0 — it never blocks.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SNAP = REPO / ".claude" / ".kb_session_snapshot"


def _porcelain() -> set[str]:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return set()
    return set(ln for ln in out.splitlines() if ln.strip())


def snapshot() -> None:
    try:
        SNAP.parent.mkdir(parents=True, exist_ok=True)
        SNAP.write_text("\n".join(sorted(_porcelain())), encoding="utf-8")
    except Exception:
        pass


def _path_of(line: str) -> str:
    # porcelain line: "XY <path>" (path may be quoted or contain " -> " for renames)
    p = line[3:].strip()
    if " -> " in p:
        p = p.split(" -> ", 1)[1]
    return p.strip().strip('"')


def check() -> None:
    before = set()
    if SNAP.exists():
        before = set(l for l in SNAP.read_text(encoding="utf-8").splitlines() if l.strip())
    now = _porcelain()
    new_changes = now - before
    code_changed, recorded = [], False
    for ln in new_changes:
        path = _path_of(ln).replace("\\", "/")
        if path.startswith("knowledge/") and path.endswith(".csv"):
            recorded = True
        elif path.endswith(".py") and not path.startswith("knowledge/_tools/"):
            code_changed.append(path)
    if code_changed and not recorded:
        shown = ", ".join(code_changed[:5]) + (" ..." if len(code_changed) > 5 else "")
        print(f"[knowledge reminder] this session touched script(s) [{shown}] but nothing "
              f"was recorded into knowledge/. If they're keepers, run /record "
              f"(or `py knowledge/_tools/kb.py record scripts ...`). See CLAUDE.md.")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    try:
        (snapshot if mode == "snapshot" else check)()
    except Exception:
        pass  # advisory only — never break the session
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
