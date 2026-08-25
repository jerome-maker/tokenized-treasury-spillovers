"""
09_make_outputs.py
Consolidate all pipeline results into publication-ready figures (PDF,
vector, journal-safe) and LaTeX booktabs tables for the manuscript and
supplementary appendix to \\input{} directly -- this keeps every number in
the paper numerically traceable to the pipeline output rather than
hand-transcribed.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
TAB = ROOT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "figure.dpi": 300, "savefig.dpi": 300,
                      "font.family": "DejaVu Sans"})


def to_booktabs(df, path, float_fmt="%.4f", index=True, caption=None, label=None):
    tex = df.to_latex(float_format=float_fmt, index=index, escape=True)
    tex = tex.replace("\\toprule", "\\toprule").replace("\\midrule", "\\midrule")
    (TAB / path).write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------------------
# TABLES
# ---------------------------------------------------------------------------
desc = pd.read_csv(PROC / "descriptive_stats.csv", index_col=0)
cols = ["n", "mean", "sd", "skew", "kurtosis", "JB_stat"]
to_booktabs(desc[cols].round(4), "table_descriptive.tex")

unitroot = pd.read_csv(PROC / "unitroot_tests.csv", index_col=0)
cols = ["ADF_stat", "ADF_pvalue", "PP_stat", "PP_pvalue", "KPSS_stat", "KPSS_pvalue"]
to_booktabs(unitroot[cols].round(4), "table_unitroot.tex")

diag = pd.read_csv(PROC / "diagnostics.csv", index_col=0)
to_booktabs(diag.round(4), "table_diagnostics.tex")

garch = pd.read_csv(PROC / "garch_params.csv", index_col=0)
cols = ["model", "n_obs", "persistence", "gamma_asym", "nu_df", "AIC"]
to_booktabs(garch[cols], "table_garch.tex", float_fmt="%.4f")

dcc = pd.read_csv(PROC / "dcc_adcc_summary.csv", index_col=0)
cols = ["n_obs", "a_DCC", "b_DCC", "mean_rho_DCC", "sd_rho_DCC", "g_ADCC", "LR_stat_vs_CCC", "LR_pvalue_vs_CCC"]
dcc_disp = dcc[cols].rename(columns={
    "mean_rho_DCC": "mean_rho", "sd_rho_DCC": "sd_rho",
    "LR_stat_vs_CCC": "LR_stat", "LR_pvalue_vs_CCC": "LR_pval",
})
# main-text table: core token-vs-tech-equity + ETF-benchmark pairs only
core_pairs_main = ["BUIDL-FANG", "OUSG-FANG", "sUSDS-FANG", "BUIDL-SOXX", "OUSG-SOXX", "sUSDS-SOXX",
                    "BUIDL-SMH", "OUSG-SMH", "sUSDS-SMH", "SHV-FANG", "IEF-FANG", "SHV-SOXX", "IEF-SOXX"]
to_booktabs(dcc_disp.loc[[p for p in core_pairs_main if p in dcc_disp.index]].round(4), "table_dcc.tex")
# full 34-pair table (incl. crypto/DeFi/gold) -> supplementary appendix
to_booktabs(dcc_disp.round(4), "table_dcc_full.tex")

# condensed cross-asset-class summary: one row per token/benchmark, one
# column per asset class, mean DCC correlation (tech-equity column is the
# average across FANG/SOXX/SMH)
asset_classes = {"BTC": "BTC", "ETH": "ETH", "UNI": "UNI", "AAVE": "AAVE", "GLD": "Gold"}
rows = {}
for base in ["BUIDL", "OUSG", "sUSDS", "SHV", "IEF"]:
    tech_pairs = [f"{base}-{e}" for e in ["FANG", "SOXX", "SMH"] if f"{base}-{e}" in dcc.index]
    row = {"Tech equity (avg)": dcc.loc[tech_pairs, "mean_rho_DCC"].mean() if tech_pairs else np.nan}
    for a, label in asset_classes.items():
        pair = f"{base}-{a}"
        row[label] = dcc.loc[pair, "mean_rho_DCC"] if pair in dcc.index else np.nan
    rows[base] = row
dcc_summary = pd.DataFrame(rows).T
to_booktabs(dcc_summary.round(4), "table_dcc_crossasset.tex")

# ---- R: TVP-VAR time-domain connectedness (System R1) ----
tvpvar_path = PROC / "r_tvpvar_static_table.csv"
if tvpvar_path.exists():
    tvpvar = pd.read_csv(tvpvar_path).set_index("variable")
    to_booktabs(tvpvar.round(2), "table_tvpvar.tex")

kappa_path = PROC / "r_tvpvar_kappa_sensitivity.csv"
if kappa_path.exists():
    kappa_df = pd.read_csv(kappa_path)
    to_booktabs(kappa_df.round(2), "table_tvpvar_kappa.tex", index=False)

# ---- R: frequency-domain (Baruník-Křehlík) decomposition ----
freq_path = PROC / "r_frequency_static_table.csv"
if freq_path.exists():
    freq = pd.read_csv(freq_path)
    to_booktabs(freq.round(2), "table_frequency_full.tex", index=False)
    # main-text: NET only, wide by band (excluding the "Total" aggregate column)
    freq_net = freq[freq["band"] != "Total"].pivot(index="variable", columns="band", values="NET")
    band_order = [b for b in ["1-5", "5-20", "20-Inf"] if b in freq_net.columns]
    freq_net = freq_net[band_order]
    to_booktabs(freq_net.round(2), "table_frequency_net.tex")

# ---- Python: System C (cross-asset) static spillover, cross-check vs R TVP-VAR ----
spill_c_path = PROC / "spillover_static_C.csv"
if spill_c_path.exists():
    spill_c = pd.read_csv(spill_c_path, index_col=0)
    spill_c.index.name = "variable"
    to_booktabs(spill_c.round(2), "table_spillover_C.tex")

# ---- R: Bai-Perron structural breaks ----
bp_path = PROC / "r_baiperron_results.csv"
if bp_path.exists():
    bp = pd.read_csv(bp_path)
    to_booktabs(bp.round(4), "table_baiperron_full.tex", index=False)
    bp_main = bp[~bp["degenerate"]][["pair", "n_obs", "opt_num_breaks", "supF_stat", "supF_pvalue", "break_dates"]]
    to_booktabs(bp_main.round(3), "table_baiperron_main.tex", index=False)

# ---- R: block-bootstrap confidence intervals ----
boot_path = PROC / "r_blockbootstrap_results.csv"
if boot_path.exists():
    bo = pd.read_csv(boot_path)
    to_booktabs(bo.round(4), "table_bootstrap_full.tex", index=False)
    bo_main = bo[~bo["degenerate"]][["pair", "block_length", "point_mean", "ci95_lo", "ci95_hi"]]
    to_booktabs(bo_main.round(4), "table_bootstrap_main.tex", index=False)

spill_a = pd.read_csv(PROC / "spillover_static_A.csv", index_col=0)
spill_b = pd.read_csv(PROC / "spillover_static_B.csv", index_col=0)
to_booktabs(spill_a.round(3), "table_spillover_A.tex")
to_booktabs(spill_b.round(3), "table_spillover_B.tex")

wv_path = PROC / "wavelet_band_summary.csv"
if wv_path.exists():
    wv = pd.read_csv(wv_path, index_col=0)
    to_booktabs(wv.round(3), "table_wavelet.tex")

# ---- R: partial wavelet coherence (controlling for the 3-month T-bill rate factor) ----
pwtc_path = PROC / "r_partial_wavelet.csv"
if pwtc_path.exists():
    pwtc_df = pd.read_csv(pwtc_path)
    to_booktabs(pwtc_df.round(4), "table_partial_wavelet.tex", index=False)

regime_path = PROC / "regime_summary.csv"
if regime_path.exists():
    regime = pd.read_csv(regime_path, index_col=0)
    cols = ["n_obs", "mean_rho_low_regime", "mean_rho_high_regime",
            "P_stay_high", "expected_duration_high_days", "frac_time_high_regime"]
    regime_disp = regime[cols].rename(columns={
        "mean_rho_low_regime": "mean_rho_low", "mean_rho_high_regime": "mean_rho_high",
        "expected_duration_high_days": "E[dur_high] (days)", "frac_time_high_regime": "frac_high",
    })
    to_booktabs(regime_disp.round(4), "table_regime.tex")

hedge = pd.read_csv(PROC / "hedge_summary.csv")
hedge_disp = hedge[hedge["hedge_instrument"] != "SHV"].copy()  # SHV excluded: near-zero variance destabilizes h_ij/h_jj
cols = ["hedged_equity", "hedge_instrument", "n_obs", "mean_beta_star", "HE", "HE_cost_10bp", "CVaR95_hedged"]
hedge_disp = hedge_disp[cols].round(4)
to_booktabs(hedge_disp, "table_hedge.tex", index=False)

# supplementary: full hedge table incl. SHV and all three cost scenarios
full_cols = ["hedged_equity", "hedge_instrument", "n_obs", "mean_beta_star", "sd_beta_star",
             "HE", "HE_cost_5bp", "HE_cost_10bp", "HE_cost_20bp", "mean_turnover",
             "VaR95_unhedged", "VaR95_hedged", "CVaR95_unhedged", "CVaR95_hedged"]
to_booktabs(hedge[full_cols].round(4), "table_hedge_full.tex", index=False)

# supplementary: full GARCH model-selection comparison (all 3 candidates per series)
sel_path = PROC / "garch_model_selection.csv"
if sel_path.exists():
    sel = pd.read_csv(sel_path)
    to_booktabs(sel.round(3), "table_garch_selection.tex", index=False)

# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------

# Fig 1: DCC correlation paths, tokens vs FANG
fig, axes = plt.subplots(3, 1, figsize=(6.5, 7), sharex=True)
for ax, tok in zip(axes, ["BUIDL", "OUSG", "sUSDS"]):
    path = PROC / f"dcc_{tok}_FANG.csv"
    if not path.exists():
        continue
    d = pd.read_csv(path, parse_dates=["date"])
    ax.plot(d["date"], d["rho_dcc"], color="#1b4965", lw=0.9, label="DCC")
    ax.plot(d["date"], d["rho_adcc"], color="#bc4749", lw=0.8, ls="--", label="ADCC", alpha=0.8)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(f"{tok} vs. NYSE FANG+", fontsize=9, loc="left")
    ax.set_ylabel(r"$\rho_t$")
axes[0].legend(fontsize=7, loc="upper right")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG / "fig1_dcc_paths.pdf")
plt.close(fig)

# Fig 2: conditional volatility, tokens vs FANG (from garch stage; recompute quickly)
# (loaded from dcc files' sigma columns, which store both series' conditional vol)
fig, axes = plt.subplots(3, 1, figsize=(6.5, 7), sharex=True)
for ax, tok in zip(axes, ["BUIDL", "OUSG", "sUSDS"]):
    path = PROC / f"dcc_{tok}_FANG.csv"
    if not path.exists():
        continue
    d = pd.read_csv(path, parse_dates=["date"])
    ax2 = ax.twinx()
    ax.plot(d["date"], d["sigma_tok"], color="#52796f", lw=0.9, label=f"{tok} (left)")
    ax2.plot(d["date"], d["sigma_eq"], color="#e07a5f", lw=0.8, label="FANG+ (right)")
    ax.set_ylabel(f"$\\sigma_{{{tok}}}$", color="#52796f")
    ax2.set_ylabel(r"$\sigma_{FANG+}$", color="#e07a5f")
    ax.set_title(f"{tok} vs. FANG+ conditional volatility", fontsize=9, loc="left")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG / "fig2_volatility.pdf")
plt.close(fig)

# Fig 3: wavelet coherence heatmaps (main text: tech-equity pairs)
wavelet_pairs = [("BUIDL", "FANG"), ("OUSG", "FANG"), ("sUSDS", "FANG"), ("OUSG", "SOXX")]
existing = [(t, e) for t, e in wavelet_pairs if (PROC / f"wavelet_{t}_{e}.npz").exists()]
if existing:
    # 2.15in/panel (not 3in): at the widths used in manuscript.tex, a taller
    # figure plus its double-spaced caption can overflow past \textheight
    # onto the same page as the page number (observed with 4 stacked panels
    # at 3in each). Keep this comfortably short enough to leave room below.
    fig, axes = plt.subplots(len(existing), 1, figsize=(7, 2.15 * len(existing)))
    if len(existing) == 1:
        axes = [axes]
    for ax, (tok, eq) in zip(axes, existing):
        z = np.load(PROC / f"wavelet_{tok}_{eq}.npz", allow_pickle=True)
        WCT, coi, periods = z["WCT"], z["coi"], z["periods"]
        dates = pd.to_datetime(z["dates"])
        t_num = mdates.date2num(dates)
        im = ax.contourf(t_num, periods, WCT, levels=np.linspace(0, 1, 11), cmap="viridis")
        ax.plot(t_num, coi, color="white", lw=1.2)
        ax.fill_between(t_num, coi, periods.max(), color="white", alpha=0.35)
        ax.set_yscale("log")
        ax.set_ylim(periods.min(), periods.max())
        ax.set_ylabel("Period (days)")
        ax.set_title(f"{tok} vs. {eq}: wavelet coherence", fontsize=9, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.subplots_adjust(hspace=0.7)
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, label="Coherence")
    fig.savefig(FIG / "fig3_wavelet_coherence.pdf")
    plt.close(fig)

# Fig 4: rolling total connectedness index
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for tag, label, color in [("A", "System A (BUIDL+OUSG, 2024-07 to present)", "#1b4965"),
                           ("B", "System B (+sUSDS, 2025-02 to present)", "#bc4749")]:
    p = PROC / f"spillover_rolling_{tag}.csv"
    if p.exists():
        d = pd.read_csv(p, parse_dates=["date"])
        ax.plot(d["date"], d["TCI"], lw=1.0, label=label, color=color)
ax.set_ylabel("Total Connectedness Index")
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG / "fig4_rolling_tci.pdf")
plt.close(fig)

# Fig 5: net spillover for tokens, rolling
fig, ax = plt.subplots(figsize=(6.5, 3.2))
p = PROC / "spillover_rolling_B.csv"
if p.exists():
    d = pd.read_csv(p, parse_dates=["date"])
    for tok, color in [("BUIDL", "#52796f"), ("OUSG", "#e07a5f"), ("sUSDS", "#3d5a80")]:
        col = f"NET_{tok}"
        if col in d.columns:
            ax.plot(d["date"], d[col], lw=1.0, label=tok, color=color)
ax.axhline(0, color="grey", lw=0.6)
ax.set_ylabel("Net directional spillover (NET$_i$)")
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG / "fig5_net_spillover.pdf")
plt.close(fig)

# Fig 6: hedge ratio paths, tokens & IEF vs FANG
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for hedge, color in [("BUIDL", "#52796f"), ("OUSG", "#e07a5f"), ("sUSDS", "#3d5a80"), ("IEF", "#6d6875")]:
    p = PROC / f"dcc_{hedge}_FANG.csv"
    if p.exists():
        d = pd.read_csv(p, parse_dates=["date"])
        h_ij = d["rho_dcc"] * d["sigma_tok"] * d["sigma_eq"]
        h_jj = d["sigma_tok"] ** 2
        beta = (h_ij / h_jj).clip(-5, 5)
        ax.plot(d["date"], beta, lw=0.9, label=hedge, color=color)
ax.axhline(0, color="grey", lw=0.6)
ax.set_ylabel(r"Optimal hedge ratio $\beta^{*}_t$ (vs. FANG+)")
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG / "fig6_hedge_ratio.pdf")
plt.close(fig)

# Fig 7: regime smoothed probabilities (high-correlation state)
regime_pairs = ["sUSDS_FANG", "IEF_FANG", "OUSG_FANG"]
existing_r = [p for p in regime_pairs if (PROC / f"regime_probs_{p}.csv").exists()]
if existing_r:
    fig, axes = plt.subplots(len(existing_r), 1, figsize=(6.5, 2.2 * len(existing_r)), sharex=True)
    if len(existing_r) == 1:
        axes = [axes]
    for ax, pair in zip(axes, existing_r):
        d = pd.read_csv(PROC / f"regime_probs_{pair}.csv", index_col=0, parse_dates=[0])
        hi_col = d.columns[np.argmax(d.mean().values)]
        ax.fill_between(d.index, 0, d[hi_col], color="#bc4749", alpha=0.6, step="mid")
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(high-corr.)")
        ax.set_title(pair.replace("_", " vs. "), fontsize=9, loc="left")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "fig7_regime_prob.pdf")
    plt.close(fig)

# Fig 8 (appendix): wavelet coherence, cross-asset pairs (crypto/gold)
wavelet_pairs_x = [("OUSG", "BTC"), ("sUSDS", "BTC"), ("IEF", "GLD")]
existing_x = [(t, e) for t, e in wavelet_pairs_x if (PROC / f"wavelet_{t}_{e}.npz").exists()]
if existing_x:
    fig, axes = plt.subplots(len(existing_x), 1, figsize=(7, 3 * len(existing_x)))
    if len(existing_x) == 1:
        axes = [axes]
    for ax, (tok, eq) in zip(axes, existing_x):
        z = np.load(PROC / f"wavelet_{tok}_{eq}.npz", allow_pickle=True)
        WCT, coi, periods = z["WCT"], z["coi"], z["periods"]
        dates = pd.to_datetime(z["dates"])
        t_num = mdates.date2num(dates)
        im = ax.contourf(t_num, periods, WCT, levels=np.linspace(0, 1, 11), cmap="viridis")
        ax.plot(t_num, coi, color="white", lw=1.2)
        ax.fill_between(t_num, coi, periods.max(), color="white", alpha=0.35)
        ax.set_yscale("log")
        ax.set_ylim(periods.min(), periods.max())
        ax.set_ylabel("Period (days)")
        ax.set_title(f"{tok} vs. {eq}: wavelet coherence", fontsize=9, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, label="Coherence")
    fig.savefig(FIG / "fig8_wavelet_crossasset.pdf")
    plt.close(fig)

# Fig 9: TVP-VAR time-domain rolling TCI + frequency-band TCI (R outputs)
tvpvar_tci_path = PROC / "r_tvpvar_tci_series.csv"
freq_tci_path = PROC / "r_frequency_tci_series.csv"
if tvpvar_tci_path.exists() and freq_tci_path.exists():
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)
    tvp = pd.read_csv(tvpvar_tci_path, parse_dates=["date"])
    axes[0].plot(tvp["date"], tvp["TCI"], color="#1b4965", lw=1.0)
    axes[0].set_ylabel("Total Connectedness Index")
    axes[0].set_title("TVP-VAR time-domain connectedness (System R1, $\\kappa$=0.99/0.99)",
                       fontsize=9, loc="left")

    freq_tci = pd.read_csv(freq_tci_path, parse_dates=["date"])
    colors = {"1-5": "#52796f", "5-20": "#e07a5f", "20-Inf": "#3d5a80"}
    for band, color in colors.items():
        sub = freq_tci[freq_tci["band"] == band]
        if len(sub):
            axes[1].plot(sub["date"], sub["TCI"], lw=1.0, label=f"{band} days", color=color)
    axes[1].set_ylabel("Band TCI")
    axes[1].legend(fontsize=7, title="Frequency band")
    axes[1].set_title("Baruník-Křehlík frequency-domain connectedness", fontsize=9, loc="left")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "fig9_tvpvar_frequency.pdf")
    plt.close(fig)

# Fig 10: hedge ratio vs. BTC (cross-asset hedge comparison)
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for hedge, color in [("BUIDL", "#52796f"), ("OUSG", "#e07a5f"), ("sUSDS", "#3d5a80"), ("IEF", "#6d6875")]:
    p = PROC / f"dcc_{hedge}_BTC.csv"
    if p.exists():
        d = pd.read_csv(p, parse_dates=["date"])
        h_ij = d["rho_dcc"] * d["sigma_tok"] * d["sigma_eq"]
        h_jj = d["sigma_tok"] ** 2
        beta = (h_ij / h_jj).clip(-5, 5)
        ax.plot(d["date"], beta, lw=0.9, label=hedge, color=color)
ax.axhline(0, color="grey", lw=0.6)
ax.set_ylabel(r"Optimal hedge ratio $\beta^{*}_t$ (vs. BTC)")
ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG / "fig10_hedge_ratio_btc.pdf")
plt.close(fig)

# Fig 11: cross-asset DCC correlation heatmap (Table 4 in the main text, visualized)
dcc_summary_path = TAB / "table_dcc_crossasset.tex"
dcc_x_data = None
dcc_full = pd.read_csv(PROC / "dcc_adcc_summary.csv", index_col=0)
asset_classes = {"BTC": "BTC", "ETH": "ETH", "UNI": "UNI", "AAVE": "AAVE", "GLD": "Gold"}
rows = {}
for base in ["BUIDL", "OUSG", "sUSDS", "SHV", "IEF"]:
    tech_pairs = [f"{base}-{e}" for e in ["FANG", "SOXX", "SMH"] if f"{base}-{e}" in dcc_full.index]
    row = {"Tech\nequity": dcc_full.loc[tech_pairs, "mean_rho_DCC"].mean() if tech_pairs else np.nan}
    for a, label in asset_classes.items():
        pair = f"{base}-{a}"
        row[label] = dcc_full.loc[pair, "mean_rho_DCC"] if pair in dcc_full.index else np.nan
    rows[base] = row
dcc_x_data = pd.DataFrame(rows).T

fig, ax = plt.subplots(figsize=(6, 3.2))
vmax = np.nanmax(np.abs(dcc_x_data.values))
im = ax.imshow(dcc_x_data.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(dcc_x_data.columns)))
ax.set_xticklabels(dcc_x_data.columns, fontsize=8)
ax.set_yticks(range(len(dcc_x_data.index)))
ax.set_yticklabels(dcc_x_data.index, fontsize=8)
for i in range(dcc_x_data.shape[0]):
    for j in range(dcc_x_data.shape[1]):
        v = dcc_x_data.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                     color="white" if abs(v) > vmax * 0.6 else "black")
fig.colorbar(im, ax=ax, label="Mean DCC correlation", fraction=0.046, pad=0.04)
ax.set_title("Mean conditional correlation by asset class", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig11_dcc_heatmap.pdf")
plt.close(fig)

# Fig 12: spillover network diagram (TVP-VAR time-averaged NPDC, System R1)
npdc_path = PROC / "r_tvpvar_npdc.csv"
tvp_path = PROC / "r_tvpvar_static_table.csv"
if npdc_path.exists() and tvp_path.exists():
    npdc = pd.read_csv(npdc_path, index_col=0)
    tvp = pd.read_csv(tvp_path).set_index("variable")
    # sign convention (verified against NET): npdc.loc[i, j] > 0 means net flow
    # FROM column-asset j TO row-asset i (i is the net receiver in that pair)
    G = nx.DiGraph()
    for v in npdc.index:
        G.add_node(v, net=tvp.loc[v, "NET"])
    edges = []
    for i in npdc.index:
        for j in npdc.columns:
            if i == j:
                continue
            val = npdc.loc[i, j]
            if val > 0:
                edges.append((j, i, val))  # j -> i
    edges.sort(key=lambda e: -e[2])
    top_edges = edges[:18]  # keep the strongest pairwise links only, for legibility
    for j, i, val in top_edges:
        G.add_edge(j, i, weight=val)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    pos = nx.circular_layout(G)
    nets = np.array([G.nodes[n]["net"] for n in G.nodes])
    node_colors = ["#bc4749" if G.nodes[n]["net"] > 0 else "#1b4965" for n in G.nodes]
    node_sizes = 400 + 60 * np.abs(nets)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", font_weight="bold", ax=ax)
    weights = [G[u][v]["weight"] for u, v in G.edges]
    wmax = max(weights) if weights else 1
    nx.draw_networkx_edges(G, pos, width=[1 + 3 * w / wmax for w in weights],
                            edge_color="grey", alpha=0.5, arrowsize=12,
                            connectionstyle="arc3,rad=0.08", ax=ax)
    ax.set_title("Dynamic spillover network (TVP-VAR, mean NPDC; arrow = net direction of transmission)\n"
                  "red = net transmitter, blue = net receiver; node size $\\propto$ |NET|", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "fig12_spillover_network.pdf")
    plt.close(fig)

# Fig 13: frequency-domain NET spillover by band (bar chart, Table 8 visualized)
if freq_path.exists():
    freq = pd.read_csv(freq_path)
    freq_net_plot = freq[freq["band"] != "Total"].pivot(index="variable", columns="band", values="NET")
    band_order = [b for b in ["1-5", "5-20", "20-Inf"] if b in freq_net_plot.columns]
    freq_net_plot = freq_net_plot[band_order].sort_values("1-5")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(freq_net_plot))
    width = 0.25
    colors = {"1-5": "#52796f", "5-20": "#e07a5f", "20-Inf": "#3d5a80"}
    for k, band in enumerate(band_order):
        ax.bar(x + (k - 1) * width, freq_net_plot[band], width, label=f"{band}d", color=colors[band])
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(freq_net_plot.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Net spillover (NET)")
    ax.legend(fontsize=7, title="Band")
    fig.tight_layout()
    fig.savefig(FIG / "fig13_frequency_net_bars.pdf")
    plt.close(fig)

# Fig 14 (appendix): Bai-Perron breaks overlaid on the DCC path, for the pairs with the
# clearest structural-break evidence
bp_path = PROC / "r_baiperron_results.csv"
if bp_path.exists():
    bp = pd.read_csv(bp_path)
    highlight_pairs = ["IEF_GLD", "IEF_FANG", "OUSG_ETH"]
    existing_bp = [p for p in highlight_pairs if (PROC / f"dcc_{p}.csv").exists()]
    if existing_bp:
        fig, axes = plt.subplots(len(existing_bp), 1, figsize=(6.5, 2.3 * len(existing_bp)), sharex=False)
        if len(existing_bp) == 1:
            axes = [axes]
        for ax, pair in zip(axes, existing_bp):
            d = pd.read_csv(PROC / f"dcc_{pair}.csv", parse_dates=["date"])
            ax.plot(d["date"], d["rho_dcc"], color="#1b4965", lw=0.9)
            row = bp[bp["pair"] == pair]
            if len(row) and isinstance(row["break_dates"].iloc[0], str) and row["break_dates"].iloc[0]:
                for bd in row["break_dates"].iloc[0].split(";"):
                    ax.axvline(pd.Timestamp(bd), color="#bc4749", lw=1.0, ls="--")
            ax.set_title(pair.replace("_", " vs. ") + ": DCC path with Bai-Perron breaks", fontsize=9, loc="left")
            ax.set_ylabel(r"$\rho_t$")
        fig.tight_layout()
        fig.savefig(FIG / "fig14_baiperron_breaks.pdf")
        plt.close(fig)

# Fig 16: partial wavelet coherence, WTC vs PWC by band (controlling for the
# 3-month T-bill rate factor)
if pwtc_path.exists():
    pwtc_df = pd.read_csv(pwtc_path)
    pair_order = ["BUIDL_FANG", "OUSG_FANG", "sUSDS_FANG", "OUSG_SOXX", "IEF_GLD"]
    band_order_pwtc = ["D1-D2 (2-8d)", "D3-D4 (8-32d)", "D5-D6 (32-128d)"]
    fig, axes = plt.subplots(1, len(pair_order), figsize=(11, 3.3), sharey=True)
    bars_for_legend = None
    for ax, pair in zip(axes, pair_order):
        sub = pwtc_df[pwtc_df["pair"] == pair].set_index("band").loc[band_order_pwtc]
        x = np.arange(len(band_order_pwtc))
        width = 0.35
        b1 = ax.bar(x - width / 2, sub["WTC"], width, label="WTC (raw)", color="#3d5a80")
        b2 = ax.bar(x + width / 2, sub["PWC"], width, label="PWC (rate-controlled)", color="#e07a5f")
        bars_for_legend = (b1, b2)
        ax.set_xticks(x)
        ax.set_xticklabels(["2-8d", "8-32d", "32-128d"], fontsize=7, rotation=30, ha="right")
        ax.set_title(pair.replace("_", " vs. "), fontsize=8)
        ax.set_ylim(0, 0.55)
    axes[0].set_ylabel("Coherence")
    axes[0].legend(bars_for_legend, ["WTC (raw)", "PWC (rate-controlled)"], fontsize=6.5,
                    loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig16_partial_wavelet.pdf")
    plt.close(fig)

# Fig 15 (appendix): block-bootstrap forest plot for non-degenerate pairs
boot_path = PROC / "r_blockbootstrap_results.csv"
if boot_path.exists():
    bo = pd.read_csv(boot_path)
    bo_nd = bo[~bo["degenerate"]].sort_values("point_mean")
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(bo_nd) + 1))
    y = np.arange(len(bo_nd))
    ax.errorbar(bo_nd["point_mean"], y,
                xerr=[bo_nd["point_mean"] - bo_nd["ci95_lo"], bo_nd["ci95_hi"] - bo_nd["point_mean"]],
                fmt="o", color="#1b4965", ecolor="#6d6875", elinewidth=1.2, capsize=3, markersize=4)
    ax.axvline(0, color="grey", lw=0.7, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(bo_nd["pair"].str.replace("_", " vs. "), fontsize=7)
    ax.set_xlabel("Mean DCC correlation (block-bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(FIG / "fig15_bootstrap_forest.pdf")
    plt.close(fig)

print("All figures and tables written to figures/ and tables/.")
