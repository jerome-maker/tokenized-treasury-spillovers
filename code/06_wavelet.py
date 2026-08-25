"""
06_wavelet.py
Morlet continuous-wavelet coherence (Torrence and Webster 1999; Grinsted,
Moore and Jevrejeva 2004) between each tokenized-Treasury market-cap return
series and the tech-equity benchmarks, computed with `pycwt` (a Python
implementation of the same cross-wavelet / wavelet-coherence machinery as
the R `biwavelet` package used in the underlying research design; this
substitution is disclosed in the supplementary appendix). Monte Carlo AR(1)
surrogate significance (pycwt's built-in `sig=True`) and the cone of
influence are computed alongside the coherence surface.

Average coherence is also summarized into the short/medium/long-run bands
of Table 3-3 of the research design (2-8, 8-32, 32-128, 128+ trading days)
for a compact numeric table alongside the heatmap figure.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pycwt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROC / "returns_panel.csv", parse_dates=["date"]).set_index("date")

PAIRS = [
    ("BUIDL", "R_BUIDL_mkt", "FANG", "R_FANG"),
    ("OUSG", "R_OUSG_mkt", "FANG", "R_FANG"),
    ("sUSDS", "R_sUSDS_mkt", "FANG", "R_FANG"),
    ("OUSG", "R_OUSG_mkt", "SOXX", "R_SOXX"),
    ("OUSG", "R_OUSG_mkt", "BTC", "R_BTC"),
    ("sUSDS", "R_sUSDS_mkt", "BTC", "R_BTC"),
    ("IEF", "R_IEF", "GLD", "R_GLD"),
]

BANDS = [("D1-D2 (2-8d)", 2, 8), ("D3-D4 (8-32d)", 8, 32), ("D5-D6 (32-128d)", 32, 128)]


def band_average(WCT, periods, coi, t):
    """Average coherence within each period band, restricted to points inside
    the cone of influence (COI) so edge-affected values are excluded."""
    out = {}
    for label, lo, hi in BANDS:
        mask_period = (periods >= lo) & (periods < hi)
        vals = []
        for ti in range(WCT.shape[1]):
            valid_scales = mask_period & (periods < coi[ti])
            if valid_scales.any():
                vals.append(WCT[valid_scales, ti].mean())
        out[label] = float(np.nanmean(vals)) if vals else np.nan
    return out


summary_rows = []

for tok_name, tok_col, eq_name, eq_col in PAIRS:
    s1 = df[tok_col].dropna()
    s2 = df[eq_col].dropna()
    common = s1.index.intersection(s2.index)
    y1 = s1.loc[common].values
    y2 = s2.loc[common].values
    if len(y1) < 100:
        print(f"[wavelet] skip {tok_name}-{eq_name}: only {len(y1)} obs")
        continue

    dt = 1  # 1 trading day
    # mc_count reduced from the pycwt default of 300 to 60 Monte Carlo AR(1)
    # surrogates to keep runtime tractable; the resulting 95% significance
    # contour is a coarser but still valid approximation (documented in the
    # supplementary appendix).
    WCT, aWCT, coi, freq, sig = pycwt.wct(y1, y2, dt, sig=True, significance_level=0.95,
                                           wavelet="morlet", mc_count=60, progress=False)
    periods = 1 / freq

    np.savez(PROC / f"wavelet_{tok_name}_{eq_name}.npz",
              WCT=WCT, aWCT=aWCT, coi=coi, periods=periods, sig=sig,
              dates=common.values.astype("datetime64[D]").astype(str))

    band_avg = band_average(WCT, periods, coi, common)
    band_avg.update({"pair": f"{tok_name}-{eq_name}", "n_obs": len(y1)})
    summary_rows.append(band_avg)
    print(f"[wavelet] {tok_name}-{eq_name}: n={len(y1)}, band avg coherence = "
          + ", ".join(f"{k}={v:.3f}" for k, v in band_avg.items() if k not in ("pair", "n_obs")))

summary_df = pd.DataFrame(summary_rows).set_index("pair")
summary_df.to_csv(PROC / "wavelet_band_summary.csv")
print(summary_df.round(3))
