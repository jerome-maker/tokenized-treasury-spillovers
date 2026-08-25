# -*- coding: utf-8 -*-
"""Assemble the teaching notebook from the per-part cell modules.

Each module in parts/ exposes a CELLS list. Keeping them separate makes the
material editable one section at a time instead of one 4000-line JSON blob.

    python build_notebook.py            # write research_methods.ipynb
    python build_notebook.py --check    # validate only, do not write
"""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from parts import part0, part1, part2, part3, part4, part5, part6, part7, part8, part9  # noqa: E402

PARTS = [part0, part1, part2, part3, part4, part5, part6, part7, part8, part9]
OUT = HERE / "research_methods.ipynb"


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
    return nb


if __name__ == "__main__":
    nb = build()
    n_md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    units = sum(1 for c in nb.cells
                if c.cell_type == "markdown" and c.source.lstrip("-\n").startswith("## 單元"))
    print(f"cells   : {len(nb.cells)}  ({n_md} markdown, {n_code} code)")
    print(f"units   : {units}")
    if "--check" in sys.argv:
        print("validate: OK (not written)")
    else:
        nbf.write(nb, OUT)
        print(f"written : {OUT.name}  ({OUT.stat().st_size / 1024:.0f} KB)")
