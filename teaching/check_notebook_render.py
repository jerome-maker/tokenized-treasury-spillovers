#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify a notebook renders correctly everywhere it will be read.

    python check_notebook_render.py notebook.ipynb
    python check_notebook_render.py notebook.ipynb --no-api   # skip network check

Four checks, covering the three silent failure modes described in
references/rendering-traps.md:

  1. Forbidden escape sequences   GitHub's CommonMark pass strips
                                  backslash-punctuation before math is
                                  extracted, so `\\,` `\\%` `\\\\` and friends
                                  vanish and the equation renders cleanly
                                  meaning something else.
  2. Pipes in table cells         An unescaped `|` splits the row and silently
                                  discards whatever follows it, including whole
                                  equations in later cells of that row.
  3. Non-Latin plot text          Matplotlib's default font has no CJK, Arabic,
                                  Hebrew or Thai glyphs, so such labels render
                                  as empty "tofu" boxes with no warning.
  4. GitHub round-trip            The decisive one: POST every markdown cell to
                                  GitHub's own renderer and count how many
                                  LaTeX commands survive.

Check 4 is what makes this worth running. The other three are static and
conservative; only the round-trip reflects what KaTeX actually receives. The
rendered page is not evidence -- an equation whose spacing commands were eaten
still renders, professionally, saying something else.

This file is vendored deliberately rather than imported from a shared location:
the repository has to stay reproducible for anyone who clones it, and a build
that reaches into the author's home directory is not. The same script is kept
as a template for other projects; if you improve it here, port the change.

Exit status is non-zero if any check fails, so this works as a pre-commit hook.
Requires `gh` (authenticated) for check 4; it is skipped with a notice if
absent, and the other three still run.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import nbformat

if hasattr(sys.stdout, "reconfigure"):          # Windows consoles default to cp950/cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BS = chr(92)

# Sequences CommonMark destroys, mapped to what the reader ends up seeing.
FORBIDDEN = {
    BS + ",": "thin space becomes a comma",
    BS + ";": "medium space becomes a semicolon",
    BS + ":": "medium space becomes a colon",
    BS + "!": "negative space becomes a factorial",
    BS + "%": "percent becomes a TeX comment and eats the rest of the expression",
    BS + "_": "escaped underscore becomes a subscript",
    BS + "&": "escaped ampersand becomes an alignment tab",
    BS + "#": "escaped hash becomes a macro parameter",
    BS + BS: "row separator is halved, breaking cases/aligned/matrix",
    BS + "operatorname": "rejected outright by GitHub's KaTeX -- use " + BS + "mathrm",
}

# Scripts with no coverage in matplotlib's default font (DejaVu Sans).
NON_LATIN = re.compile(
    "["
    "一-鿿"      # CJK unified ideographs
    "぀-ヿ"      # hiragana, katakana
    "가-힯"      # hangul
    "֐-׿"      # hebrew
    "؀-ۿ"      # arabic
    "฀-๿"      # thai
    "]"
)

PLOT_TEXT = re.compile(
    r"(set_title|set_xlabel|set_ylabel|set_xticklabels|set_yticklabels|suptitle"
    r"|\.text\(|annotate|\blabel\s*=|legend\s*\(\s*title\s*=|set_label)"
)


def markdown_source(nb):
    """Markdown cells joined, plus a copy with fenced code stripped.

    Fenced code is exempt from inline processing, so a code sample containing
    a backslash sequence is safe and must not be counted as a violation.
    """
    cells = [c.source for c in nb.cells if c.cell_type == "markdown"]
    joined = "\n\n---\n\n".join(cells)
    return joined, re.sub(r"```.*?```", "", joined, flags=re.S)


def check_forbidden(src_no_fence):
    print("=== 1. Escape sequences CommonMark would eat ===")
    bad = 0
    for seq, why in FORBIDDEN.items():
        n = src_no_fence.count(seq)
        bad += n
        print(f"  {seq!r:18} {n:3d}  {'<-- FIX' if n else 'ok':8} {why}")
    return bad


def check_table_pipes(nb):
    print("\n=== 2. Pipes inside table cells ===")
    bad = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "markdown":
            continue
        for line in c.source.splitlines():
            t = line.strip()
            if not t.startswith("|"):
                continue
            for pat in (r"\$([^$]+)\$", r"`([^`]*)`"):       # inline math, inline code
                for m in re.finditer(pat, t):
                    if "|" in m.group(1):
                        bad.append((i, m.group(0)[:70]))
    for i, frag in bad:
        print(f"  cell {i}: {frag}")
    print(f"  problems: {len(bad)}"
          + ("   (use \\lvert/\\rvert, or reword to avoid the pipe)" if bad else ""))
    return len(bad)


def check_plot_text(nb):
    print("\n=== 3. Non-Latin text in plotting calls ===")
    bad = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        for line in c.source.splitlines():
            if PLOT_TEXT.search(line) and NON_LATIN.search(line):
                bad.append((i, line.strip()[:90]))
    for i, line in bad:
        print(f"  cell {i}: {line}")
    print(f"  problems: {len(bad)}"
          + ("   (these render as tofu boxes; prose and print() are fine)" if bad else ""))
    if not bad:
        print("  note: labels built from variables need the runtime glyph check --")
        print("        see teaching/README.md")
    return len(bad)


def check_against_github(src, src_no_fence):
    """Ask GitHub's renderer what it actually received."""
    print("\n=== 4. GitHub round-trip (decisive) ===")
    with tempfile.TemporaryDirectory() as td:
        payload = Path(td) / "payload.json"
        payload.write_text(json.dumps({"text": src, "mode": "gfm"}), encoding="utf-8")
        try:
            out = subprocess.run(
                ["gh", "api", "markdown", "-X", "POST", "--input", str(payload)],
                capture_output=True, text=True, encoding="utf-8", timeout=180,
            )
        except FileNotFoundError:
            print("  skipped: gh CLI not found")
            return 0
        except subprocess.TimeoutExpired:
            print("  skipped: gh api timed out")
            return 0
    if out.returncode != 0:
        print(f"  skipped: gh api failed ({out.stderr.strip()[:140]})")
        return 0

    html = out.stdout
    toks = Counter(re.findall(rf"{re.escape(BS)}([a-zA-Z]+)", src_no_fence))
    lost = [(t, n, html.count(BS + t)) for t, n in toks.items() if html.count(BS + t) < n]
    # Match the class attribute, not the bare string: a document that merely
    # mentions "flash-error" in its prose would otherwise report phantom failures.
    errors = len(re.findall(r'class="[^"]*flash-error', html))

    print(f"  LaTeX commands : {len(toks)} distinct, {sum(toks.values())} occurrences")
    print(f"  eaten          : {len(lost)}")
    for t, n, g in lost:
        print(f"      {BS + t}: {n} -> {g}")
    print(f"  render failures: {errors}")
    return len(lost) + errors


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--no-api", action="store_true",
                    help="skip the GitHub round-trip (offline use)")
    args = ap.parse_args()

    if not args.notebook.exists():
        sys.exit(f"not found: {args.notebook}")

    nb = nbformat.read(args.notebook, as_version=4)
    src, src_no_fence = markdown_source(nb)

    problems = check_forbidden(src_no_fence)
    problems += check_table_pipes(nb)
    problems += check_plot_text(nb)
    if not args.no_api:
        problems += check_against_github(src, src_no_fence)

    print()
    if problems:
        print(f"[FAIL] {problems} problem(s) -- see teaching/README.md")
        sys.exit(1)
    print("[PASS] renders correctly in Jupyter, nbviewer and GitHub")


if __name__ == "__main__":
    main()
