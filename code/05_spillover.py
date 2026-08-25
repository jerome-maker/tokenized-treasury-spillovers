"""
05_spillover.py
Diebold-Yilmaz (2012, 2014) generalized-FEVD connectedness, computed from a
levels VAR fit with statsmodels (no R / ConnectednessApproach package is
available in this environment, so the generalized FEVD, TCI, TO/FROM/NET,
and NPDC formulas are implemented directly from Koop-Pesaran-Potter (1996)
/ Pesaran-Shin (1998), matching eqs. (3.15)-(3.21) of the underlying
research design).

Two systems are estimated:
  SYS_A (longer sample, 2024-07 to present): FANG, SOXX, SMH, SHV, IEF,
         BUIDL, OUSG  -- excludes sUSDS, which only starts 2025-02.
  SYS_B (shorter, all-three-tokens sample, 2025-02 to present): adds sUSDS.

For each system: a static full-sample spillover table, and a rolling-window
(window=100 trading days, step=5) time-varying total/net spillover series
(used in place of the original design's TVP-VAR/Kalman-filter connectedness,
which is not implementable without rmgarch/ConnectednessApproach; documented
as a simplification in the supplementary appendix).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "tables"

df = pd.read_csv(PROC / "returns_panel.csv", parse_dates=["date"]).set_index("date")

H = 10  # forecast horizon, matches proposal default

SYS_A_COLS = {"FANG": "R_FANG", "SOXX": "R_SOXX", "SMH": "R_SMH", "SHV": "R_SHV",
              "IEF": "R_IEF", "BUIDL": "R_BUIDL_mkt", "OUSG": "R_OUSG_mkt"}
SYS_B_COLS = dict(SYS_A_COLS, **{"sUSDS": "R_sUSDS_mkt"})
# System C: cross-asset system matching the R TVP-VAR System R1 (script 10),
# used as an independent (rolling-VAR GFEVD vs. Kalman-filtered TVP-VAR)
# cross-check on the same variable set.
SYS_C_COLS = dict(SYS_A_COLS, **{"BTC": "R_BTC", "ETH": "R_ETH", "UNI": "R_UNI",
                                  "AAVE": "R_AAVE", "GLD": "R_GLD"})


def gfevd(var_res, H):
    names = var_res.names
    N = len(names)
    Sigma = var_res.sigma_u.values
    A = var_res.ma_rep(maxn=H - 1)  # A[0..H-1], each N x N
    theta = np.zeros((N, N))
    for i in range(N):
        e_i = np.zeros(N); e_i[i] = 1.0
        denom = sum((e_i @ A[h] @ Sigma @ A[h].T @ e_i) for h in range(H))
        for j in range(N):
            e_j = np.zeros(N); e_j[j] = 1.0
            num = sum((e_i @ A[h] @ Sigma @ e_j) ** 2 for h in range(H)) / Sigma[j, j]
            theta[i, j] = num / denom if denom > 0 else np.nan
    theta_tilde = theta / theta.sum(axis=1, keepdims=True)
    return theta_tilde, names


def connectedness_table(theta_tilde, names):
    N = len(names)
    TO = 100 * (theta_tilde.sum(axis=0) - np.diag(theta_tilde))
    FROM = 100 * (theta_tilde.sum(axis=1) - np.diag(theta_tilde))
    NET = TO - FROM
    TCI = 100 * (theta_tilde.sum() - np.trace(theta_tilde)) / N
    npdc = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            npdc.loc[ni, nj] = 100 * (theta_tilde[j, i] - theta_tilde[i, j]) / N
    table = pd.DataFrame({"TO": TO, "FROM": FROM, "NET": NET}, index=names)
    return table, TCI, npdc


def fit_var_and_spillover(data, H, maxlags=5):
    model = VAR(data)
    sel = model.select_order(maxlags=maxlags)
    p = max(1, sel.aic)
    res = model.fit(p)
    theta_tilde, names = gfevd(res, H)
    table, tci, npdc = connectedness_table(theta_tilde, names)
    return table, tci, npdc, p


def run_system(cols_map, tag):
    cols = list(cols_map.values())
    data = df[cols].dropna()
    data.columns = list(cols_map.keys())
    print(f"[{tag}] system n_obs={len(data)}, vars={list(cols_map.keys())}")

    table, tci, npdc, p = fit_var_and_spillover(data, H)
    table.to_csv(PROC / f"spillover_static_{tag}.csv")
    npdc.to_csv(PROC / f"spillover_npdc_{tag}.csv")
    with open(PROC / f"spillover_tci_{tag}.txt", "w") as f:
        f.write(f"VAR lag order (AIC): {p}\nTotal Connectedness Index (static, H={H}): {tci:.4f}\n")
    print(table.round(3))
    print(f"TCI = {tci:.3f}, VAR(p={p})")

    # rolling-window time-varying spillover
    window, step = 100, 5
    roll_rows = []
    for start in range(0, len(data) - window, step):
        chunk = data.iloc[start:start + window]
        try:
            m = VAR(chunk)
            p_r = 1  # fixed lag for rolling windows: keeps the estimator stable at small T
            res_r = m.fit(p_r)
            theta_tilde, names = gfevd(res_r, H)
            table_r, tci_r, _ = connectedness_table(theta_tilde, names)
            row = {"date": chunk.index[-1], "TCI": tci_r}
            for n in names:
                row[f"NET_{n}"] = table_r.loc[n, "NET"]
            roll_rows.append(row)
        except Exception:
            continue
    roll_df = pd.DataFrame(roll_rows).set_index("date")
    roll_df.to_csv(PROC / f"spillover_rolling_{tag}.csv")
    print(f"[{tag}] rolling spillover: {len(roll_df)} windows")
    return table, tci, npdc, roll_df


table_A, tci_A, npdc_A, roll_A = run_system(SYS_A_COLS, "A")
table_B, tci_B, npdc_B, roll_B = run_system(SYS_B_COLS, "B")
table_C, tci_C, npdc_C, roll_C = run_system(SYS_C_COLS, "C")
