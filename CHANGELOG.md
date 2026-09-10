# Changelog

All notable changes to this dataset and pipeline. Versions correspond to
Zenodo archived releases under concept DOI
[10.5281/zenodo.22092666](https://doi.org/10.5281/zenodo.22092666).

## v1.2.0 — 2026-09-10

Version DOI: pending (assigned by Zenodo when this release is archived).

Additive release. **No reported number changes.** `code/`, `data/`, `tables/`
and `figures/` are byte-identical to v1.1.0; everything below is new material
or tooling. Anyone who pulled v1.1.0 for the results does not need to re-pull.

### Added

- **`teaching/research_methods.ipynb`, a 26-unit graduate teaching notebook**
  covering the full pipeline, organised by research stage and by the
  statistical technique each stage uses. Each unit works through four layers:
  the principle (what problem the step solves and why the previous step is
  insufficient), the formulas with every symbol defined, the algorithm
  including the numerical details that matter, and the program logic tied back
  to the actual pipeline code. Committed with executed output and 19 figures,
  so it reads on GitHub without running anything; all 54 code cells execute
  against the `data/processed/` in this same release.
- Code cells are of two kinds: demonstrations that read existing pipeline
  output and interpret it, and from-scratch reimplementations of the core
  algorithms (the DCC `Q_t` recursion, the GFEVD matrix algebra, Jarque-Bera,
  Ljung-Box, the moving block bootstrap, Bai-Perron's `F(lambda)` curve)
  written for readability rather than speed, so the formula-to-code
  correspondence stays visible.
- The final unit dissects the four errors found in the v1.1.0 pre-submission
  audit — each of which passed the author's own review, raised no exception and
  produced plausible-looking output — and pairs each with the check that would
  have caught it.

### Added — tooling

- **`teaching/build_notebook.py`** assembles the notebook from `parts/*.py`
  modules rather than hand-edited JSON, and executes it (`--execute`) so the
  committed copy always carries real output. Without that flag an earlier
  rebuild produced a notebook with 0 of 54 outputs and no figures while the
  render check still passed, because the check reads cell source, not output.
- **`teaching/check_notebook_render.py`** verifies four things a rendered page
  will not reveal: forbidden backslash-punctuation sequences, pipes inside
  table cells, non-Latin text in plot calls, and — the decisive one — a
  round-trip through GitHub's own `/markdown` endpoint that counts how many
  LaTeX commands survive.

### Fixed — presentation only

- **The notebook's math was being silently altered by GitHub's renderer.**
  Measured against GitHub's markdown API rather than by eye: a CommonMark
  backslash-escape pass runs before math extraction, so a broken equation
  renders cleanly while saying something other than what was written. Row
  separators were being halved (`\\\\` to `\\`), which breaks every
  `cases`/`pmatrix`/`aligned` block. Rewritten into equivalent forms that need
  no row separator (min/max expressions, prose, separate display equations,
  markdown tables) rather than double-escaped, which would then break in
  Jupyter. A second measurement caught an inline code span whose unescaped
  pipes split a table row and destroyed the math in the following cell.
- **All figure text in the notebook is now English.** The matplotlib default
  font has no CJK glyphs, so every Chinese label in all 19 figures had been
  rendering as blank boxes. 61 text arguments translated; the build now fails
  on any non-Latin string passed to a plotting call.
- A figure caption that contradicted its own axes, and a "very narrow" band
  description that disagreed with the statistic the pipeline computes (57% vs
  75% coverage), both corrected.

## v1.1.0 — 2026-08-25

Version DOI [10.5281/zenodo.22094343](https://doi.org/10.5281/zenodo.22094343).

Correctness release, produced during a pre-submission audit. Several reported
numbers change. Anyone who used v1.0.0 should re-pull.

### Fixed — results affected

- **Transaction-cost overlay in `08_hedge.py` was wrong in three ways** and is
  rewritten. It converted basis points with `bps/10000` while the return series
  are `*100` log returns (100x too small); it multiplied the cost by
  `|r_hedge|`, which is dimensionally wrong; and it folded the cost into the
  Ederington variance-reduction ratio, which made cost-adjusted effectiveness
  *higher* than unadjusted effectiveness in every single row. Costs are now
  `c * |Delta beta|` in matching percent units and are reported as an
  annualised drag on the mean in basis points (`cost_bp_ann_5bp` / `_10bp` /
  `_20bp`), separately from `HE`. `HE_net_*` retains the post-drag variance
  ratio for completeness. Consequence: the conventional Treasury ETF's hedge
  turns out to cost 300-545 bp/year to rebalance against 4-132 bp/year for the
  tokens, a difference that was invisible before.
- **`11_r_baiperron_bootstrap.R` silently covered only 31 of 34 DCC pairs.**
  It had been run before `dcc_IEF_ETH.csv`, `dcc_SHV_BTC.csv` and
  `dcc_SHV_ETH.csv` existed, and nothing downstream noticed. All 34 pairs are
  now present; the previously reported 31 rows are numerically unchanged. The
  newly covered IEF-ETH pair shows 3 structural breaks (supF = 28.7,
  p = 3e-6), a result that was missing entirely. Block-bootstrap intervals for
  shared pairs shift by at most 0.0035 because adding pairs advances the
  seeded RNG stream.
- **`03_pretests.py` omitted five series.** Descriptive statistics and
  unit-root tests now cover BTC, ETH, UNI, AAVE and GLD alongside the original
  eleven, so `descriptive_stats.csv` and `unitroot_tests.csv` describe every
  series the downstream stages consume.
- **`tables/*.tex` for the pre-test stage were stale**, generated from an
  earlier vintage of `returns_panel.csv` (n = 907 vs the current 906). They now
  match the panel that every other stage was estimated on.

### Fixed — tooling

- `11_r_baiperron_bootstrap.R` gains a coverage guard that aborts if any DCC
  pair is missing from its output, instead of writing a quietly incomplete
  table, and no longer treats `dcc_adcc_summary.csv` as a pair file.
- `figures/fig15_bootstrap_forest.pdf` regenerated: 23 non-degenerate pairs,
  up from 20.

### Documentation

- README documents that `R_BUIDL_nav` and `R_OUSG_nav` are the identical series
  by construction (both are the same 3-month T-bill accrual), that 18 of 34 DCC
  fits converge on a parameter boundary and 11 produce a numerically constant
  correlation path, that transaction costs are deliberately not folded into
  `HE`, and that the Python stages must run before the R stages.

## v1.0.0 — 2026-08-25

Initial release. Version DOI
[10.5281/zenodo.22092667](https://doi.org/10.5281/zenodo.22092667).
