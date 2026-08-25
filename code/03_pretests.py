"""
03_pretests.py
Descriptive statistics and pre-estimation diagnostics on the return series
that feed the GARCH/DCC stage: normality (Jarque-Bera), stationarity
(ADF, Phillips-Perron, KPSS via the `arch` package's unitroot module),
serial correlation (Ljung-Box on returns and squared returns), and
conditional heteroskedasticity (ARCH-LM).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch.unitroot import ADF, KPSS, PhillipsPerron
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROC / "returns_panel.csv", parse_dates=["date"]).set_index("date")

SERIES = [
    "R_FANG", "R_SOXX", "R_SMH", "R_SHV", "R_IEF",
    "R_BUIDL_mkt", "R_BUIDL_nav",
    "R_OUSG_mkt", "R_OUSG_nav",
    "R_sUSDS_mkt", "R_sUSDS_nav",
]

desc_rows = []
unitroot_rows = []
diag_rows = []

for col in SERIES:
    s = df[col].dropna()
    if len(s) < 30:
        continue

    # --- descriptive stats ---
    jb_stat, jb_p = jarque_bera(s)
    desc_rows.append({
        "series": col, "n": len(s), "mean": s.mean(), "sd": s.std(),
        "skew": skew(s), "kurtosis": kurtosis(s, fisher=False),
        "min": s.min(), "max": s.max(),
        "JB_stat": jb_stat, "JB_pvalue": jb_p,
    })

    # --- unit root / stationarity ---
    try:
        adf = ADF(s)
        pp = PhillipsPerron(s)
        kpss = KPSS(s)
        unitroot_rows.append({
            "series": col,
            "ADF_stat": adf.stat, "ADF_pvalue": adf.pvalue,
            "PP_stat": pp.stat, "PP_pvalue": pp.pvalue,
            "KPSS_stat": kpss.stat, "KPSS_pvalue": kpss.pvalue,
        })
    except Exception as e:
        unitroot_rows.append({"series": col, "ADF_stat": np.nan, "ADF_pvalue": np.nan,
                               "PP_stat": np.nan, "PP_pvalue": np.nan,
                               "KPSS_stat": np.nan, "KPSS_pvalue": np.nan, "error": str(e)})

    # --- serial correlation & ARCH effects ---
    lb = acorr_ljungbox(s, lags=[10], return_df=True)
    lb_sq = acorr_ljungbox(s**2, lags=[10], return_df=True)
    try:
        arch_lm = het_arch(s, nlags=10)
        arch_lm_stat, arch_lm_p = arch_lm[0], arch_lm[1]
    except Exception:
        arch_lm_stat, arch_lm_p = np.nan, np.nan

    diag_rows.append({
        "series": col,
        "LB10_stat": lb["lb_stat"].iloc[0], "LB10_pvalue": lb["lb_pvalue"].iloc[0],
        "LB10_sq_stat": lb_sq["lb_stat"].iloc[0], "LB10_sq_pvalue": lb_sq["lb_pvalue"].iloc[0],
        "ARCH_LM_stat": arch_lm_stat, "ARCH_LM_pvalue": arch_lm_p,
    })

desc_df = pd.DataFrame(desc_rows).set_index("series")
unitroot_df = pd.DataFrame(unitroot_rows).set_index("series")
diag_df = pd.DataFrame(diag_rows).set_index("series")

desc_df.to_csv(PROC / "descriptive_stats.csv")
unitroot_df.to_csv(PROC / "unitroot_tests.csv")
diag_df.to_csv(PROC / "diagnostics.csv")

print(desc_df.round(4))
print()
print(unitroot_df.round(4))
print()
print(diag_df.round(4))
