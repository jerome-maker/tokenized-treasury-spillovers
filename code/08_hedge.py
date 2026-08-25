"""
08_hedge.py
Time-varying optimal hedge ratio (Kroner and Sultan 1993) and hedging
effectiveness (Ederington 1979) using the DCC conditional covariance
h_ij,t = rho_t * sigma_i,t * sigma_j,t already saved by 04_garch_dcc.py.
Hedge ratios are computed with a one-day lag (beta*_{t-1}) to avoid
look-ahead bias, per eq. (3.29) of the underlying research design.
Traditional Treasury ETFs (SHV, IEF) are estimated the same way as a
benchmark for comparison.

A transaction-cost scenario (one-way costs of 5/10/20 bp applied to
hedge-ratio turnover |Delta beta|) is layered on top. Costs are reported
as a mean/level drag in basis points per year, NOT folded into the
Ederington variance-reduction ratio, which they do not belong in.
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

        # --- turnover and transaction costs ---------------------------------
        # beta_lag is the notional weight held in the hedge instrument per unit
        # of the hedged asset, so rebalancing the hedge trades |Delta beta|
        # units of notional and costs c * |Delta beta| in the SAME percent
        # return units as the series themselves (all returns are *100 log
        # returns, so bp -> percent is /100, not /10000).
        #
        # Transaction costs are a MEAN (level) effect, not a variance effect:
        # Ederington HE is a variance-reduction ratio and costs do not belong
        # inside it. We therefore keep HE as the pure variance measure and
        # report the cost drag separately, on the mean, in basis points.
        turnover = combined["beta_lag"].diff().abs().fillna(0.0)
        mean_ret_unhedged = combined["r_eq"].mean()
        mean_ret_hedged = r_hedged.mean()
        cost_results = {}
        for bps in COST_BPS:
            c = bps / 100.0  # basis points -> percent, matching the *100 log returns
            cost_t = c * turnover
            r_hedged_net = r_hedged - cost_t
            cost_results[f"cost_bp_day_{bps}bp"] = 100.0 * cost_t.mean()
            cost_results[f"cost_bp_ann_{bps}bp"] = 100.0 * cost_t.mean() * 252.0
            cost_results[f"mean_ret_net_{bps}bp"] = r_hedged_net.mean()
            # variance ratio after the cost drag, reported for completeness only
            cost_results[f"HE_net_{bps}bp"] = (
                1 - r_hedged_net.var() / var_unhedged if var_unhedged > 0 else np.nan
            )

        # downside-risk metrics
        var95_unhedged = combined["r_eq"].quantile(0.05)
        var95_hedged = r_hedged.quantile(0.05)
        cvar95_unhedged = combined["r_eq"][combined["r_eq"] <= var95_unhedged].mean()
        cvar95_hedged = r_hedged[r_hedged <= var95_hedged].mean()

        rows.append({
            "hedged_equity": eq_name, "hedge_instrument": hedge, "n_obs": len(combined),
            "mean_beta_star": combined["beta_lag"].mean(), "sd_beta_star": combined["beta_lag"].std(),
            "var_unhedged": var_unhedged, "var_hedged": var_hedged, "HE": HE,
            "mean_ret_unhedged": mean_ret_unhedged, "mean_ret_hedged": mean_ret_hedged,
            **cost_results,
            "VaR95_unhedged": var95_unhedged, "VaR95_hedged": var95_hedged,
            "CVaR95_unhedged": cvar95_unhedged, "CVaR95_hedged": cvar95_hedged,
            "mean_turnover": turnover.mean(),
        })
        print(f"[hedge] {hedge}->{eq_name}: n={len(combined)}, mean_beta={combined['beta_lag'].mean():.4f}, "
              f"HE={HE:.4f}, cost@10bp={cost_results['cost_bp_ann_10bp']:.1f}bp/yr, "
              f"HE_net_10bp={cost_results['HE_net_10bp']:.4f}")

hedge_df = pd.DataFrame(rows)
hedge_df.to_csv(PROC / "hedge_summary.csv", index=False)
print(hedge_df.round(4))
