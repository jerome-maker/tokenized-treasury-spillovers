# -*- coding: utf-8 -*-
"""Assemble the teaching notebook from the per-part cell modules.

Each module in parts/ exposes a CELLS list. Keeping them separate makes the
material editable one section at a time instead of one 4000-line JSON blob.

    python build_notebook.py            # write research_methods.ipynb
    python build_notebook.py --check    # validate only, do not write
"""
import re
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from parts import part0, part1, part2, part3, part4, part5, part6, part7, part8, part9  # noqa: E402

PARTS = [part0, part1, part2, part3, part4, part5, part6, part7, part8, part9]
OUT = HERE / "research_methods.ipynb"

# Matplotlib's default font (DejaVu Sans) has no CJK glyphs, so a Chinese title
# or axis label renders as a row of tofu boxes -- silently, with no warning from
# either matplotlib or the notebook. Figure text must therefore be English even
# though the surrounding narrative is Chinese. Printed output is unaffected: it
# goes through the HTML layer, which renders CJK correctly.
CJK = re.compile("[" + chr(0x4E00) + "-" + chr(0x9FFF) + "]")
PLOT_TEXT = re.compile(
    r"(set_title|set_xlabel|set_ylabel|set_xticklabels|set_yticklabels|suptitle"
    r"|\.text\(|annotate|label\s*=|legend\s*\(\s*title\s*=|set_label)"
)


def check_figure_text(nb):
    """Fail the build if any matplotlib text argument contains CJK."""
    bad = []
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for line in cell.source.splitlines():
            if PLOT_TEXT.search(line) and CJK.search(line):
                bad.append((idx, line.strip()))
    if bad:
        print("\nERROR: CJK found in matplotlib text -- these render as tofu boxes:")
        for idx, line in bad:
            print(f"  cell {idx}: {line[:100]}")
        raise SystemExit(f"{len(bad)} offending line(s); figure text must be English")
    return True


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
    print("figures : text checked, no CJK (would render as tofu)")
    if "--check" in sys.argv:
        print("validate: OK (not written)")
    else:
        nbf.write(nb, OUT)
        print(f"written : {OUT.name}  ({OUT.stat().st_size / 1024:.0f} KB)")
