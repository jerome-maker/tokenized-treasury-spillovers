# -*- coding: utf-8 -*-
"""Verify that the notebook's math survives GitHub's markdown pipeline.

GitHub runs a CommonMark backslash-escape pass BEFORE it extracts math, so a
sequence like a backslash followed by punctuation is silently replaced by the
bare punctuation. The page then renders a clean-looking equation that says
something different from what was written -- no error, no red box, nothing to
notice. "It looks fine on the page" is not evidence here.

This script asks GitHub's own renderer what it actually received:

    python check_github_math.py

It POSTs every markdown cell to the /markdown API and compares the number of
LaTeX commands in the source against the number surviving in the returned
HTML. Exits non-zero if anything was eaten, so it works as a pre-commit hook.

Requires the `gh` CLI to be authenticated.
"""
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import nbformat

# Windows consoles default to cp950/cp1252, which cannot encode this file's
# output. Force UTF-8 so the script is runnable on every platform.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BS = chr(92)
HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "research_methods.ipynb"

# Sequences GitHub's escape pass destroys. Each maps to what the reader ends up
# seeing, which is why none of them may appear in a markdown cell.
FORBIDDEN = {
    BS + ",": "thin space becomes a comma",
    BS + ";": "medium space becomes a semicolon",
    BS + ":": "medium space becomes a colon",
    BS + "!": "negative space becomes a factorial",
    BS + "%": "percent becomes a TeX comment and eats the rest of the line",
    BS + "_": "escaped underscore becomes a subscript",
    BS + "&": "escaped ampersand becomes an alignment tab",
    BS + "#": "escaped hash becomes a macro parameter",
    BS + BS: "row separator is halved, breaking cases/aligned/matrix",
    BS + "operatorname": "rejected outright by GitHub's KaTeX",
}


def markdown_source(nb):
    """All markdown cells joined, and the same with code fences stripped."""
    cells = [c.source for c in nb.cells if c.cell_type == "markdown"]
    joined = "\n\n---\n\n".join(cells)
    # Fenced code is not subject to inline processing, so it is exempt.
    return joined, re.sub(r"```.*?```", "", joined, flags=re.S)


def check_forbidden(src_no_fence):
    print("=== 靜態檢查：會被 CommonMark 轉義吃掉的序列 ===")
    bad = 0
    for seq, why in FORBIDDEN.items():
        n = src_no_fence.count(seq)
        bad += n
        status = "← 需處理" if n else "OK"
        print(f"  {seq!r:18} {n:3d}  {status:8} {why}")
    return bad


def check_table_pipes(nb):
    """A pipe inside a table cell splits the row, silently destroying whatever
    follows it in that row -- including entire math expressions."""
    print("\n=== 靜態檢查：表格儲存格內的豎線 ===")
    bad = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "markdown":
            continue
        for line in c.source.splitlines():
            t = line.strip()
            if not t.startswith("|"):
                continue
            for m in re.finditer(r"\$([^$]+)\$", t):      # inline math
                if "|" in m.group(1):
                    bad.append((i, m.group(0)[:60]))
            for m in re.finditer(r"`([^`]*)`", t):        # inline code
                if "|" in m.group(1):
                    bad.append((i, m.group(0)[:60]))
    for i, frag in bad:
        print(f"  cell {i}: {frag}")
    print(f"  問題數：{len(bad)}")
    return len(bad)


def check_against_github(src, src_no_fence):
    """The decisive check: what did the renderer actually receive?"""
    print("\n=== 動態檢查：送進 GitHub markdown API ===")
    with tempfile.TemporaryDirectory() as td:
        payload = Path(td) / "payload.json"
        payload.write_text(json.dumps({"text": src, "mode": "gfm"}), encoding="utf-8")
        try:
            out = subprocess.run(
                ["gh", "api", "markdown", "-X", "POST", "--input", str(payload)],
                capture_output=True, text=True, encoding="utf-8", timeout=120,
            )
        except FileNotFoundError:
            print("  略過：找不到 gh CLI")
            return 0
    if out.returncode != 0:
        print(f"  略過：gh api 失敗 ({out.stderr.strip()[:120]})")
        return 0

    html = out.stdout
    toks = Counter(re.findall(rf"{re.escape(BS)}([a-zA-Z]+)", src_no_fence))
    lost = [(t, n, html.count(BS + t)) for t, n in toks.items() if html.count(BS + t) < n]
    # A failed expression renders as an element carrying the flash-error class.
    # Match the class attribute, not the bare string -- a document that merely
    # mentions "flash-error" in its prose (this one does) would otherwise
    # report failures that are not there.
    errors = len(re.findall(r'class="[^"]*flash-error', html))

    print(f"  LaTeX 指令：{len(toks)} 種、{sum(toks.values())} 次出現")
    print(f"  被吃掉的  ：{len(lost)}")
    for t, n, g in lost:
        print(f"      {BS + t}: {n} → {g}")
    print(f"  渲染失敗（flash-error）：{errors}")
    return len(lost) + errors


def main():
    if not NOTEBOOK.exists():
        sys.exit(f"找不到 {NOTEBOOK.name}，請先執行 build_notebook.py")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    src, src_no_fence = markdown_source(nb)

    problems = check_forbidden(src_no_fence)
    problems += check_table_pipes(nb)
    problems += check_against_github(src, src_no_fence)

    print()
    if problems:
        print(f"[FAIL] 共 {problems} 個問題，GitHub 上會渲染錯誤")
        sys.exit(1)
    print("[PASS] 全數通過，GitHub 渲染安全")


if __name__ == "__main__":
    main()
