# -*- coding: utf-8 -*-
"""Assemble the teaching notebook from the per-part cell modules.

Each module in parts/ exposes a CELLS list. Keeping them separate makes the
material editable one section at a time instead of one 4000-line JSON blob.

    python build_notebook.py            # write research_methods.ipynb
    python build_notebook.py --check    # validate only, do not write
    python build_notebook.py --execute  # write, then run every cell and store outputs
"""
import re
import sys
from pathlib import Path

import nbformat as nbf

# The guard below prints the offending line, which by definition contains
# non-Latin characters. On a cp950/cp1252 console that raises UnicodeEncodeError
# and buries a clear message under a traceback -- so force UTF-8 first.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from parts import part0, part1, part2, part3, part4, part5, part6, part7, part8, part9  # noqa: E402

PARTS = [part0, part1, part2, part3, part4, part5, part6, part7, part8, part9]
OUT = HERE / "research_methods.ipynb"

# Matplotlib's default font (DejaVu Sans) carries no glyphs for CJK, Hangul,
# Hebrew, Arabic or Thai, so a label in any of those scripts renders as a row of
# tofu boxes -- silently, with no warning from either matplotlib or the notebook.
# Figure text must therefore be Latin even though the surrounding narrative is
# Chinese. Printed output and DataFrames are unaffected: they go through the HTML
# layer, which renders every script correctly.
#
# The range list is broader than this project strictly needs. Narrowing a guard
# to exactly what has bitten you so far is how the next variant gets through.
NON_LATIN = re.compile(
    "["
    + chr(0x4E00) + "-" + chr(0x9FFF)      # CJK unified ideographs
    + chr(0x3040) + "-" + chr(0x30FF)      # hiragana, katakana
    + chr(0xAC00) + "-" + chr(0xD7AF)      # hangul
    + chr(0x0590) + "-" + chr(0x05FF)      # hebrew
    + chr(0x0600) + "-" + chr(0x06FF)      # arabic
    + chr(0x0E00) + "-" + chr(0x0E7F)      # thai
    + "]"
)
PLOT_TEXT = re.compile(
    r"(set_title|set_xlabel|set_ylabel|set_xticklabels|set_yticklabels|suptitle"
    r"|\.text\(|annotate|label\s*=|legend\s*\(\s*title\s*=|set_label)"
)


def check_figure_text(nb):
    """Fail the build if any matplotlib text argument uses a non-Latin script.

    Static scanning misses labels passed through variables; the runtime glyph
    check in check_notebook_render.py's companion recipe catches those. This
    guard exists to stop the common case cheaply, at build time.
    """
    bad = []
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for line in cell.source.splitlines():
            if PLOT_TEXT.search(line) and NON_LATIN.search(line):
                bad.append((idx, line.strip()))
    if bad:
        print("\nERROR: non-Latin text in matplotlib calls -- renders as tofu boxes:")
        for idx, line in bad:
            print(f"  cell {idx}: {line[:100]}")
        raise SystemExit(f"{len(bad)} offending line(s); figure text must be Latin script")
    return True


def execute(path):
    """Run every cell and store the outputs.

    allow_errors=True then counting failures ourselves is deliberate: raising on
    the first error tells you nothing about the other forty cells, and one pass
    that surfaces every failure is worth several that stop early.
    """
    from nbclient import NotebookClient

    nb = nbf.read(path, as_version=4)
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}},
                   allow_errors=True).execute()
    nbf.write(nb, path)

    n = fails = figs = 0
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        n += 1
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                fails += 1
                print(f"  ERROR: {o['ename']}: {o['evalue'][:90]}")
            if "image/png" in o.get("data", {}):
                figs += 1
    print(f"executed: {n} cells | failures: {fails} | figures: {figs}")
    return fails


def build():
    nb = nbf.v4.new_notebook()
    cells = []
    for mod in PARTS:
        cells.extend(mod.CELLS)
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "title": "代幣化國債的跨資產波動關聯：研究方法完整教材",
    }
    nbf.validate(nb)
    check_figure_text(nb)
    return nb


if __name__ == "__main__":
    nb = build()
    n_md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    units = sum(1 for c in nb.cells
                if c.cell_type == "markdown" and c.source.lstrip("-\n").startswith("## 單元"))
    print(f"cells   : {len(nb.cells)}  ({n_md} markdown, {n_code} code)")
    print(f"units   : {units}")
    print("figures : text checked, Latin script only")
    if "--check" in sys.argv:
        print("validate: OK (not written)")
        sys.exit(0)

    nbf.write(nb, OUT)
    print(f"written : {OUT.name}  ({OUT.stat().st_size / 1024:.0f} KB)")

    if "--execute" in sys.argv:
        if execute(OUT):
            sys.exit(1)
    else:
        # A rebuilt notebook has empty outputs. Committing that silently strips
        # every figure from the version readers see on GitHub.
        print("\nreminder: outputs are now empty -- rerun with --execute before committing")
