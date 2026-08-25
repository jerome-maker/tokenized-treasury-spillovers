"""
08_hedge.py
Time-varying optimal hedge ratio (Kroner and Sultan 1993) and hedging
effectiveness (Ederington 1979) using the DCC conditional covariance
h_ij,t = rho_t * sigma_i,t * sigma_j,t already saved by 04_garch_dcc.py.
Hedge ratios are computed with a one-day lag (beta*_{t-1}) to avoid
look-ahead bias, per eq. (3.29) of the underlying research design.
Traditional Treasury ETFs (SHV, IEF) are estimated the same way as a
benchmark for comparison.

A simple transaction-cost scenario (one-way costs of 5/10/20 bp on hedge-
ratio turnover) is layered on top, following eq. in Sec. 3.10.4 of the
research design.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

EQUITY_HEDGED = {"FANG": "R_FANG", "SOXX": "R_SOXX", "SMH": "R_SMH", "BTC": "R_BTC", "ETH": "R_ETH"}
HEDGE_INSTRUMENTS = ["BUIDL", "OUSG", "sUSDS", "SHV", "IEF"]
COST_BPS = [5, 10, 20]

df = pd.read_csv(PROC / "returns_panel.csv", parse_dates=["date"]).set_index("date")

rows = []
for eq_name, eq_col in EQUITY_HEDGED.items():
    r_eq = df[eq_col]
    for hedge in HEDGE_INSTRUMENTS:
        path = PROC / f"dcc_{hedge}_{eq_name}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path, parse_dates=["date"]).set_index("date")

        # h_ij,t: covariance between the hedged asset (equity, "eq") and the
        # hedge instrument (token/ETF, "tok"); h_jj,t: variance of the hedge
        # instrument. beta*_t = h_ij,t / h_jj,t (Kroner-Sultan 1993).
        h_ij = d["rho_dcc"] * d["sigma_tok"] * d["sigma_eq"]
        h_jj = d["sigma_tok"] ** 2
        beta_star = (h_ij / h_jj).clip(-5, 5)  # clip extreme ratios from thin-sample noise
        beta_lag = beta_star.shift(1)

        hedge_col = f"R_{hedge}" if hedge in ("SHV", "IEF") else f"R_{hedge}_mkt"
        r_hedge = df[hedge_col].reindex(d.index).values
        r_eq_aligned = r_eq.reindex(d.index).values

        combined = pd.DataFrame({
            "r_eq": r_eq_aligned, "r_hedge": r_hedge, "beta_lag": beta_lag.values,
        }, index=pd.to_datetime(d.index)).dropna()

        r_hedged = combined["r_eq"] - combined["beta_lag"] * combined["r_hedge"]
        var_unhedged = combined["r_eq"].var()
        var_hedged = r_hedged.var()
        HE = 1 - var_hedged / var_unhedged if var_unhedged > 0 else np.nan

        # turnover / transaction-cost adjustment
        turnover = combined["beta_lag"].diff().abs()
        cost_results = {}
        for bps in COST_BPS:
            c = bps / 10000.0
            r_hedged_cost = r_hedged - c * turnover.reindex(r_hedged.index).fillna(0) * combined["r_hedge"].abs()
            var_hedged_cost = r_hedged_cost.var()
            HE_cost = 1 - var_hedged_cost / var_unhedged if var_unhedged > 0 else np.nan
            cost_results[f"HE_cost_{bps}bp"] = HE_cost

        # downside-risk metrics
        var95_unhedged = combined["r_eq"].quantile(0.05)
        var95_hedged = r_hedged.quantile(0.05)
        cvar95_unhedged = combined["r_eq"][combined["r_eq"] <= var95_unhedged].mean()
        cvar95_hedged = r_hedged[r_hedged <= var95_hedged].mean()

        rows.append({
            "hedged_equity": eq_name, "hedge_instrument": hedge, "n_obs": len(combined),
            "mean_beta_star": combined["beta_lag"].mean(), "sd_beta_star": combined["beta_lag"].std(),
            "var_unhedged": var_unhedged, "var_hedged": var_hedged, "HE": HE,
            **cost_results,
            "VaR95_unhedged": var95_unhedged, "VaR95_hedged": var95_hedged,
            "CVaR95_unhedged": cvar95_unhedged, "CVaR95_hedged": cvar95_hedged,
            "mean_turnover": turnover.mean(),
        })
        print(f"[hedge] {hedge}->{eq_name}: n={len(combined)}, mean_beta={combined['beta_lag'].mean():.4f}, "
              f"HE={HE:.4f}, HE_10bp={cost_results['HE_cost_10bp']:.4f}")

hedge_df = pd.DataFrame(rows)
hedge_df.to_csv(PROC / "hedge_summary.csv", index=False)
print(hedge_df.round(4))
