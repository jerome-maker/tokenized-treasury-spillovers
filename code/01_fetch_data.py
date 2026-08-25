"""
01_fetch_data.py
Pull real, public, no-API-key data for the JAM manuscript.

Sources:
  - yfinance: equity indices/ETFs, BTC, VIX, USD index proxy
  - FRED (fredgraph.csv, no key required): rates
  - DefiLlama (api.llama.fi / stablecoins.llama.fi / yields.llama.fi, no key required):
      tokenized Treasury protocol TVL, USDS circulating supply, sUSDS savings-rate pool

All raw pulls are saved verbatim to data/raw/ so the pipeline is auditable and
re-runnable. No values in this script are fabricated; anything that fails to
download is logged and left out (never backfilled with invented numbers).
"""
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

START = "2023-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")

LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


# ---------------------------------------------------------------------------
# 1. yfinance: equities / ETFs / indices / crypto / vol / fx
# ---------------------------------------------------------------------------
YF_TICKERS = {
    "NYFANG": "^NYFANG",
    "SOXX": "SOXX",
    "SMH": "SMH",
    "SHV": "SHV",
    "SGOV": "SGOV",
    "BIL": "BIL",
    "IEF": "IEF",
    "TLT": "TLT",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "UNI": "UNI7083-USD",  # Uniswap governance token, CMC-ID-disambiguated ticker: the
                            # plain "UNI-USD" symbol on Yahoo Finance resolves to an
                            # unrelated, near-worthless, delisted-since-2025-04 token
                            # (price ~$0.0001, near-zero volume) -- confirmed by manual
                            # inspection before use, not assumed from the ticker name
    "AAVE": "AAVE-USD",  # Aave governance token: DeFi blue chip, lending
    "GLD": "GLD",        # SPDR Gold Shares ETF: conventional safe-haven benchmark
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "DXY_ETF": "UUP",  # fallback proxy for USD index if DX-Y.NYB is unreliable
    "FNGS": "FNGS",  # MicroSectors FANG+ ETN, tradable proxy for NYFANG
}

for name, ticker in YF_TICKERS.items():
    try:
        df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
        if df is None or df.empty:
            log(f"[yfinance] {name} ({ticker}): EMPTY result")
            continue
        # flatten possible MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()[["Date", "Close"]].rename(columns={"Close": name})
        df.to_csv(RAW / f"yf_{name}.csv", index=False)
        log(f"[yfinance] {name} ({ticker}): {len(df)} rows, {df['Date'].min()} -> {df['Date'].max()}")
    except Exception as e:
        log(f"[yfinance] {name} ({ticker}): ERROR {e}")
    time.sleep(1)

# ---------------------------------------------------------------------------
# 2. FRED public CSV endpoint (no API key needed)
# ---------------------------------------------------------------------------
FRED_SERIES = ["DGS3MO", "DGS10", "T10Y2Y", "DFEDTARU"]

for series in FRED_SERIES:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={START}&coed={END}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        out_path = RAW / f"fred_{series}.csv"
        out_path.write_text(r.text, encoding="utf-8")
        df = pd.read_csv(out_path)
        log(f"[FRED] {series}: {len(df)} rows")
    except Exception as e:
        log(f"[FRED] {series}: ERROR {e}")

# ---------------------------------------------------------------------------
# 3. DefiLlama: tokenized Treasury protocol TVL (BUIDL, OUSG)
# ---------------------------------------------------------------------------
DEFILLAMA_PROTOCOLS = {
    "BUIDL": "blackrock-buidl",
    "OUSG": "ondo-yield-assets",
}

for name, slug in DEFILLAMA_PROTOCOLS.items():
    url = f"https://api.llama.fi/protocol/{slug}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        (RAW / f"defillama_{name}_raw.json").write_text(json.dumps(data), encoding="utf-8")
        tvl = data.get("tvl", [])
        df = pd.DataFrame(tvl)
        df["date"] = pd.to_datetime(df["date"], unit="s")
        df = df.rename(columns={"totalLiquidityUSD": f"{name}_TVL"})[["date", f"{name}_TVL"]]
        df.to_csv(RAW / f"defillama_{name}_tvl.csv", index=False)
        log(f"[DefiLlama] {name} TVL: {len(df)} rows, {df['date'].min()} -> {df['date'].max()}")
    except Exception as e:
        log(f"[DefiLlama] {name}: ERROR {e}")

# ---------------------------------------------------------------------------
# 4. DefiLlama stablecoins: USDS circulating supply (id 209)
# ---------------------------------------------------------------------------
try:
    url = "https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=209"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    (RAW / "defillama_USDS_raw.json").write_text(json.dumps(data), encoding="utf-8")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"].astype(int), unit="s")
    df["USDS_supply"] = df["totalCirculatingUSD"].apply(
        lambda x: x.get("peggedUSD") if isinstance(x, dict) else None
    )
    df = df[["date", "USDS_supply"]].dropna()
    df.to_csv(RAW / "defillama_USDS_supply.csv", index=False)
    log(f"[DefiLlama] USDS supply: {len(df)} rows, {df['date'].min()} -> {df['date'].max()}")
except Exception as e:
    log(f"[DefiLlama] USDS supply: ERROR {e}")

# ---------------------------------------------------------------------------
# 5. DefiLlama yields: sUSDS (Sky Savings Rate) pool history
# ---------------------------------------------------------------------------
SUSDS_POOL_ID = "d8c4eff5-c8a9-46fc-a888-057c4c668e72"  # sky-lending SUSDS, Ethereum
try:
    url = f"https://yields.llama.fi/chart/{SUSDS_POOL_ID}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    df = df[["date", "tvlUsd", "apy", "apyBase"]].rename(
        columns={"tvlUsd": "sUSDS_TVL", "apy": "sUSDS_APY", "apyBase": "sUSDS_APY_base"}
    )
    df.to_csv(RAW / "defillama_sUSDS_pool.csv", index=False)
    log(f"[DefiLlama] sUSDS pool: {len(df)} rows, {df['date'].min()} -> {df['date'].max()}")
except Exception as e:
    log(f"[DefiLlama] sUSDS pool: ERROR {e}")

# ---------------------------------------------------------------------------
# write fetch log
# ---------------------------------------------------------------------------
(RAW / "_fetch_log.txt").write_text("\n".join(LOG), encoding="utf-8")
log("\nDone. See data/raw/_fetch_log.txt for the full log.")
