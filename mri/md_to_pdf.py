"""md_to_pdf.py - Markdown -> LaTeX -> PDF for MRI reports, via MiKTeX pdflatex.

  py mri/md_to_pdf.py <report.md>

Handles ATX headers, the '>' verdict quote, GitHub pipe tables, '-' bullet lists,
images (![](images/x.png)), **bold**, `code`, _italic_. Escapes LaTeX specials and
uses tabularx so wide tables wrap instead of overflowing. Drops the 'Delivered'
section (local file paths) for a clean shareable PDF. Compiles in the .md's own
directory so relative image paths resolve. No external deps beyond MiKTeX.
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

BIN = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64"

_SPECIAL = [("\\", r"\textbackslash{}"), ("{", r"\{"), ("}", r"\}"), ("&", r"\&"),
            ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
            ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]


def esc(s: str) -> str:
    for a, b in _SPECIAL:
        s = s.replace(a, b)
    return s


def inline(s: str) -> str:
    """Render inline `code`, **bold**, _italic_; escape everything else."""
    out = []
    for seg in re.split(r"(`[^`]*`)", s):
        if len(seg) >= 2 and seg[0] == "`" and seg[-1] == "`":
            out.append(r"\texttt{" + esc(seg[1:-1]) + "}")
            continue
        for b in re.split(r"(\*\*[^*]+\*\*)", seg):
            if b.startswith("**") and b.endswith("**"):
                out.append(r"\textbf{" + esc(b[2:-2]) + "}")
            else:
                for it in re.split(r"(_[^_]+_)", b):
                    if len(it) >= 2 and it[0] == "_" and it[-1] == "_":
                        out.append(r"\emph{" + esc(it[1:-1]) + "}")
                    else:
                        out.append(esc(it))
    return "".join(out)


def main():
    md_path = Path(sys.argv[1]).resolve()
    lines = md_path.read_text(encoding="utf-8").splitlines()
    body, i, n, title = [], 0, len(lines), None

    while i < n:
        s = lines[i].strip()
        if re.match(r"^#{1,6}\s+Delivered\b", s):
            break
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", s)
        if m:
            body.append(r"\begin{center}\includegraphics[width=0.86\linewidth,"
                        r"height=0.42\textheight,keepaspectratio]{" + m.group(2) + r"}\end{center}")
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            level, txt = len(m.group(1)), inline(m.group(2))
            if level == 1 and title is None:
                title = txt
                body.append(r"\begin{center}{\Large\textbf{" + txt + r"}}\end{center}\vspace{0.4em}")
            elif level <= 2:
                body.append(r"\section*{" + txt + "}")
            else:
                body.append(r"\subsection*{" + txt + "}")
            i += 1
            continue
        if s.startswith(">"):
            body.append(r"\begin{quote}\itshape " + inline(s.lstrip("> ").strip()) + r"\end{quote}")
            i += 1
            continue
        if "|" in s and i + 1 < n and "-" in lines[i + 1] and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            header = [c.strip() for c in s.strip().strip("|").split("|")]
            ncol = len(header)
            i += 2
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append((cells + [""] * ncol)[:ncol])
                i += 1
            colspec = "|" + "X|" * ncol
            t = [r"\begin{center}\footnotesize", r"\begin{tabularx}{\linewidth}{" + colspec + "}", r"\hline"]
            t.append(" & ".join(r"\textbf{" + inline(h) + "}" for h in header) + r" \\ \hline")
            for r in rows:
                t.append(" & ".join(inline(c) for c in r) + r" \\ \hline")
            t += [r"\end{tabularx}", r"\end{center}"]
            body.append("\n".join(t))
            continue
        if s.startswith("- "):
            items = []
            while i < n and lines[i].strip().startswith("- "):
                items.append(r"\item " + inline(lines[i].strip()[2:]))
                i += 1
            body.append(r"\begin{itemize}\itemsep2pt" + "\n" + "\n".join(items) + "\n" + r"\end{itemize}")
            continue
        if s in ("", "---"):
            body.append("")
            i += 1
            continue
        body.append(inline(s) + r"\\")
        i += 1

    doc = "\n".join([
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}", r"\usepackage{textcomp}",
        r"\usepackage[margin=2cm]{geometry}", r"\usepackage{graphicx}",
        r"\usepackage{array}", r"\usepackage{tabularx}", r"\usepackage{hyperref}",
        r"\setlength{\parindent}{0pt}", r"\sloppy", r"\begin{document}",
        "\n\n".join(body), r"\end{document}",
    ])

    out_dir = md_path.parent
    tex_path = out_dir / (md_path.stem + ".tex")
    tex_path.write_text(doc, encoding="utf-8")
    env = dict(os.environ)
    env["PATH"] = str(BIN) + os.pathsep + env["PATH"]
    res = None
    for _ in range(2):
        res = subprocess.run([str(BIN / "pdflatex.exe"), "-interaction=nonstopmode", tex_path.name],
                             cwd=str(out_dir), env=env, capture_output=True, text=True)
    pdf = out_dir / (md_path.stem + ".pdf")
    if pdf.exists():
        print(f"OK -> {pdf}  ({round(pdf.stat().st_size / 1024)} KB)")
    else:
        print("FAILED. Tail of pdflatex log:")
        print((res.stdout or "")[-3000:])
        sys.exit(1)


if __name__ == "__main__":
    main()
