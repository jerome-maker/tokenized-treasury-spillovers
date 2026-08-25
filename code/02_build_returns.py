"""
02_build_returns.py
Trading-day alignment + dual-track (fundamental / market-cap) return construction.

Methodological notes (must be reflected honestly in the manuscript):
  - BUIDL and OUSG have no publicly free, keyless daily NAV-per-share feed.
    Both are short-dated US Treasury / cash money-market vehicles, so their
    "fundamental" (NAV) daily return is proxied by the 3-month Treasury bill
    yield (FRED DGS3MO) accrued daily: r_fund,t = (DGS3MO_t/100)/365. This is
    an approximation, not an official NAV series, and is disclosed as such.
  - sUSDS has an actual product-specific yield series (the Sky Savings Rate,
    SSR) pulled from the sUSDS lending pool on DefiLlama's yields API, so its
    fundamental return uses the real reported APY, not a proxy.
  - The "market-cap" track for all three tokens is the log change in protocol
    TVL (BUIDL, OUSG) or pool TVL (sUSDS), which mixes valuation and net
    flow effects (mint/redemption). Because no Dune/Etherscan API key is
    available in this environment, flow and price effects cannot be
    separated event-by-event as in the original thesis design; the
    difference between the market-cap and fundamental tracks is instead
    used directly as a liquidity-premium shock (LIQ), a coarser but still
    informative proxy.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

WINSOR_LOW, WINSOR_HIGH = 0.005, 0.995


def load_yf(name):
    df = pd.read_csv(RAW / f"yf_{name}.csv", parse_dates=["Date"])
    df = df.rename(columns={"Date": "date"}).set_index("date")[name]
    return df


def load_fred(series):
    df = pd.read_csv(RAW / f"fred_{series}.csv", parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date"}).set_index("date")[series]
    df = pd.to_numeric(df, errors="coerce")
    return df


def log_ret(series, mult=100):
    return mult * np.log(series / series.shift(1))


def winsorize(s):
    lo, hi = s.quantile(WINSOR_LOW), s.quantile(WINSOR_HIGH)
    return s.clip(lo, hi)


# ---------------------------------------------------------------------------
# master trading-day calendar = US equity trading days (from NYFANG)
# ---------------------------------------------------------------------------
nyfang = load_yf("NYFANG")
calendar = nyfang.index.sort_values()

# ---------------------------------------------------------------------------
# traditional market series -> log returns on the equity calendar
# ---------------------------------------------------------------------------
price_series = {
    "FANG": "NYFANG",
    "FNGS": "FNGS",
    "SOXX": "SOXX",
    "SMH": "SMH",
    "SHV": "SHV",
    "SGOV": "SGOV",
    "BIL": "BIL",
    "IEF": "IEF",
    "TLT": "TLT",
    "BTC": "BTC",
    "ETH": "ETH",
    "UNI": "UNI",
    "AAVE": "AAVE",
    "GLD": "GLD",
}

prices = pd.DataFrame(index=calendar)
for label, fname in price_series.items():
    s = load_yf(fname)
    prices[label] = s.reindex(calendar).ffill(limit=2)

vix = load_yf("VIX").reindex(calendar).ffill(limit=2)

returns = pd.DataFrame(index=calendar)
for label in price_series:
    returns[f"R_{label}"] = log_ret(prices[label])
returns["VIX_chg"] = log_ret(vix)

# ---------------------------------------------------------------------------
# FRED macro / rate series -> aligned to equity calendar, forward filled
# (rates are step functions between releases, so ffill is the correct join)
# ---------------------------------------------------------------------------
dgs3mo = load_fred("DGS3MO").reindex(calendar).ffill(limit=5)
dgs10 = load_fred("DGS10").reindex(calendar).ffill(limit=5)
slope = load_fred("T10Y2Y").reindex(calendar).ffill(limit=5)
dfedtaru = load_fred("DFEDTARU").reindex(calendar).ffill(limit=5)

returns["Y3M_chg"] = dgs3mo.diff()
returns["Y10_chg"] = dgs10.diff()
returns["SLOPE_chg"] = slope.diff()
returns["FEDTARU"] = dfedtaru

# ---------------------------------------------------------------------------
# tokenized Treasury data
# ---------------------------------------------------------------------------
buidl_tvl = pd.read_csv(RAW / "defillama_BUIDL_tvl.csv", parse_dates=["date"]).set_index("date")["BUIDL_TVL"]
ousg_tvl = pd.read_csv(RAW / "defillama_OUSG_tvl.csv", parse_dates=["date"]).set_index("date")["OUSG_TVL"]
susds_pool = pd.read_csv(RAW / "defillama_sUSDS_pool.csv", parse_dates=["date"]).set_index("date")

# DefiLlama TVL series can have duplicate/near-duplicate intraday-snapshot
# timestamps near "today"; keep the last snapshot per calendar date.
buidl_tvl = buidl_tvl.groupby(buidl_tvl.index.normalize()).last()
ousg_tvl = ousg_tvl.groupby(ousg_tvl.index.normalize()).last()
susds_pool.index = susds_pool.index.normalize()
susds_pool = susds_pool.groupby(susds_pool.index).last()

buidl_tvl_c = buidl_tvl.reindex(calendar).ffill(limit=3)
ousg_tvl_c = ousg_tvl.reindex(calendar).ffill(limit=3)
susds_tvl_c = susds_pool["sUSDS_TVL"].reindex(calendar).ffill(limit=3)
susds_apy_c = susds_pool["sUSDS_APY"].reindex(calendar).ffill(limit=3)

# --- fundamental (NAV-proxy) daily returns ---
# BUIDL / OUSG: proxied by the 3-month T-bill yield accrued daily (see module docstring)
fund_buidl = (dgs3mo / 100.0) / 365.0 * 100  # scaled like other *100 log-returns
fund_ousg = (dgs3mo / 100.0) / 365.0 * 100
# sUSDS: actual reported Sky Savings Rate (SSR), accrued daily
fund_susds = (susds_apy_c / 100.0) / 365.0 * 100

# --- market-cap (TVL) track: log change in protocol/pool TVL ---
mkt_buidl = log_ret(buidl_tvl_c)
mkt_ousg = log_ret(ousg_tvl_c)
mkt_susds = log_ret(susds_tvl_c)

returns["R_BUIDL_nav"] = fund_buidl
returns["R_BUIDL_mkt"] = mkt_buidl
returns["LIQ_BUIDL"] = mkt_buidl - fund_buidl

returns["R_OUSG_nav"] = fund_ousg
returns["R_OUSG_mkt"] = mkt_ousg
returns["LIQ_OUSG"] = mkt_ousg - fund_ousg

returns["R_sUSDS_nav"] = fund_susds
returns["R_sUSDS_mkt"] = mkt_susds
returns["LIQ_sUSDS"] = mkt_susds - fund_susds

# ---------------------------------------------------------------------------
# OUSG incubation trim: TVL was economically negligible (<$10M) for the first
# ~1.5 weeks after the 2023-01-27 launch, so percentage TVL changes over that
# window are a small-denominator artifact, not genuine returns. Drop
# observations before TVL first exceeds $10M (2023-02-04). This is disclosed
# in the manuscript as a data-cleaning rule, not a silent trim.
INCUBATION_CUTOFF = ousg_tvl_c[ousg_tvl_c > 10_000_000].index.min()
for c in ["R_OUSG_mkt", "LIQ_OUSG"]:
    returns.loc[returns.index < INCUBATION_CUTOFF, c] = np.nan

# ---------------------------------------------------------------------------
# winsorize all return-type columns; the tokenized-treasury market-cap track
# is winsorized more tightly (1%/99%) than traditional-asset returns
# (0.5%/99.5%) because DefiLlama TVL series exhibit large discrete jumps
# tied to lumpy institutional subscription/redemption activity (consistent
# with Mafrur 2025's finding that tokenized RWA secondary markets are thin);
# this is a data-quality control, disclosed explicitly, not a device to
# manufacture significance.
# ---------------------------------------------------------------------------
tight_cols = ["R_BUIDL_mkt", "LIQ_BUIDL", "R_OUSG_mkt", "LIQ_OUSG", "R_sUSDS_mkt", "LIQ_sUSDS"]
return_cols = [c for c in returns.columns if c.startswith("R_") or c.startswith("LIQ_") or c == "VIX_chg"]
for c in return_cols:
    if c in tight_cols:
        lo, hi = returns[c].quantile(0.01), returns[c].quantile(0.99)
        returns[c] = returns[c].clip(lo, hi)
    else:
        returns[c] = winsorize(returns[c])

# ---------------------------------------------------------------------------
# sample-window bookkeeping: report, per series, the first non-NaN date so
# the manuscript's data section can state exact usable windows honestly
# ---------------------------------------------------------------------------
availability = {}
for c in returns.columns:
    s = returns[c].dropna()
    if len(s):
        availability[c] = (str(s.index.min().date()), str(s.index.max().date()), int(len(s)))
    else:
        availability[c] = (None, None, 0)

avail_df = pd.DataFrame(availability, index=["first_date", "last_date", "n_obs"]).T
avail_df.to_csv(PROC / "series_availability.csv")

returns.index.name = "date"
returns.to_csv(PROC / "returns_panel.csv")

print("Saved returns panel:", returns.shape)
print(avail_df)
