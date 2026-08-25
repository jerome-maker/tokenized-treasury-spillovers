# -*- coding: utf-8 -*-
"""Cell constructors shared by every part module.

Math conventions for this notebook
----------------------------------
The notebook is read in three places with three different math renderers:
Jupyter/VS Code, nbviewer/Colab, and GitHub's .ipynb viewer. Only the plain
`$...$` / `$$...$$` delimiters work in all three, so those are what we use --
GitHub's protected `` $`...`$ `` form would show up as literal code in Jupyter.

The cost of that choice is that GitHub runs a CommonMark backslash-escape pass
before extracting math, which silently eats backslash-punctuation pairs. We
therefore avoid them entirely in markdown cells:

    \\,  \\;  \\:  \\!   spacing      -> use a plain space, or nothing
    \\%                  percent      -> write the word, or put it outside math
    \\_  \\&  \\#                      -> avoid in math; use \\text{} if needed

`\\\\` (row separator) is unavoidable inside aligned/matrix/cases and is the one
exception; those blocks are checked visually on GitHub after publishing.
`\\operatorname` is rejected by GitHub's KaTeX -- use `\\mathrm` instead.
"""

import nbformat as nbf


def md(text):
    """A markdown cell. Leading/trailing blank lines are trimmed."""
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    """A code cell."""
    return nbf.v4.new_code_cell(text.strip("\n"))


def unit_header(number, title, technique, script, refs):
    """Standard opening block for a teaching unit.

    number    : e.g. "單元 7"
    title     : the unit's topic
    technique : the statistical technique in one line
    script    : which pipeline script this unit corresponds to
    refs      : list of canonical references
    """
    ref_lines = "\n".join(f"> - {r}" for r in refs)
    return md(f"""
---

## {number}｜{title}

> **統計技術**：{technique}
>
> **對應程式**：`{script}`
>
> **原始文獻**：
{ref_lines}
""")
