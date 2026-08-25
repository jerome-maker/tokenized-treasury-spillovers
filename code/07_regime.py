"""
07_regime.py
Two-state Markov-switching model (Hamilton 1989) fit to each pair's DCC
correlation series rho_t, via statsmodels'
`MarkovRegression` (switching mean and variance). This substitutes for the R
packages MSwM/MSGARCH named in the underlying research design (documented
in the supplementary appendix).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

PAIRS = ["BUIDL_FANG", "OUSG_FANG", "sUSDS_FANG", "BUIDL_SOXX", "OUSG_SOXX",
         "sUSDS_SOXX", "BUIDL_SMH", "OUSG_SMH", "sUSDS_SMH", "SHV_FANG", "IEF_FANG",
         "BUIDL_BTC", "OUSG_BTC", "sUSDS_BTC", "OUSG_AAVE", "OUSG_UNI",
         "IEF_BTC", "IEF_GLD", "SHV_GLD"]

rows = []
for pair in PAIRS:
    path = PROC / f"dcc_{pair}.csv"
    if not path.exists():
        continue
    d = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    rho = d["rho_dcc"].dropna()
    if len(rho) < 100:
        print(f"[regime] skip {pair}: only {len(rho)} obs")
        continue
    try:
        mod = MarkovRegression(rho, k_regimes=2, trend="c", switching_variance=True)
        res = mod.fit(search_reps=20)
    except Exception as e:
        print(f"[regime] {pair}: FAILED {e}")
        continue

    means = res.params[["const[0]", "const[1]"]].values
    lo_idx, hi_idx = np.argsort(means)  # regime with lower mean correlation first
    p = res.regime_transition
    # statsmodels stores the transition matrix as p[to, from, t]; use the
    # time-invariant estimate (first slice) for a constant-transition model.
    P = np.array(res.regime_transition)[:, :, 0]
    smoothed = res.smoothed_marginal_probabilities

    duration_lo = 1 / (1 - P[lo_idx, lo_idx]) if P[lo_idx, lo_idx] < 1 else np.inf
    duration_hi = 1 / (1 - P[hi_idx, hi_idx]) if P[hi_idx, hi_idx] < 1 else np.inf

    smoothed.to_csv(PROC / f"regime_probs_{pair}.csv")

    rows.append({
        "pair": pair, "n_obs": len(rho),
        "mean_rho_low_regime": means[lo_idx], "mean_rho_high_regime": means[hi_idx],
        "P_stay_low": P[lo_idx, lo_idx], "P_stay_high": P[hi_idx, hi_idx],
        "expected_duration_low_days": duration_lo, "expected_duration_high_days": duration_hi,
        "frac_time_high_regime": float((smoothed.iloc[:, hi_idx] > 0.5).mean()),
        "loglik": res.llf, "AIC": res.aic,
    })
    print(f"[regime] {pair}: mean_low={means[lo_idx]:.4f}, mean_high={means[hi_idx]:.4f}, "
          f"P(stay_high)={P[hi_idx,hi_idx]:.3f}, frac_high={rows[-1]['frac_time_high_regime']:.3f}")

regime_df = pd.DataFrame(rows).set_index("pair")
regime_df.to_csv(PROC / "regime_summary.csv")
print(regime_df.round(4))
