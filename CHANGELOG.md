# Changelog

All notable changes to this dataset and pipeline. Versions correspond to
Zenodo archived releases under concept DOI
[10.5281/zenodo.22092666](https://doi.org/10.5281/zenodo.22092666).

## v1.1.0 — 2026-08-25

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
