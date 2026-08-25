"""
04_garch_dcc.py
Stage 1: for each series, fit three candidate skew-t marginal volatility
models -- GARCH(1,1), GJR-GARCH(1,1), EGARCH(1,1), each with an AR(1) mean
-- and select the best by AIC among converged, non-boundary-pinned fits
(this mirrors the model-selection procedure described in the underlying
research design). Several tokenized-Treasury market-cap series turn out to
push the GJR asymmetry term to its box constraint (alpha or gamma pinned at
its bound), which is a genuine, reportable feature of thin, jump-driven
on-chain TVL data -- not a bug -- but it means GJR is not the selected model
for every series, so a real comparison is run rather than assumed.

Stage 2: two-stage QMLE DCC (Engle 2002) and asymmetric ADCC (Cappiello,
Engle and Sheppard 2006) on the standardized residuals from the selected
Stage-1 models, implemented from scratch with scipy.optimize since no R /
rmgarch is available in this environment. The ADCC positive-definiteness
condition a+b+delta*g<1 (delta = max eigenvalue of Nbar) is approximated by
the simpler sufficient-in-practice constraint a+b+g<1 (documented in the
supplementary appendix).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
from scipy.stats import chi2

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "tables"
PROC.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROC / "returns_panel.csv", parse_dates=["date"]).set_index("date")

UNIVARIATE_SERIES = {
    "FANG": "R_FANG",
    "SOXX": "R_SOXX",
    "SMH": "R_SMH",
    "SHV": "R_SHV",
    "IEF": "R_IEF",
    "BUIDL": "R_BUIDL_mkt",
    "OUSG": "R_OUSG_mkt",
    "sUSDS": "R_sUSDS_mkt",
    "BTC": "R_BTC",
    "ETH": "R_ETH",
    "UNI": "R_UNI",
    "AAVE": "R_AAVE",
    "GLD": "R_GLD",
}

TOKENS = ["BUIDL", "OUSG", "sUSDS"]
TECH_EQUITY = ["FANG", "SOXX", "SMH"]
CROSS_ASSETS = ["BTC", "ETH", "UNI", "AAVE", "GLD"]

CORE_PAIRS = (
    [(t, e) for t in TOKENS for e in TECH_EQUITY]          # tokens vs tech equities
    + [(t, a) for t in TOKENS for a in CROSS_ASSETS]        # tokens vs crypto/DeFi/gold
    + [("SHV", "FANG"), ("IEF", "FANG"), ("SHV", "SOXX"), ("IEF", "SOXX")]  # ETF benchmarks vs tech
    + [("IEF", "BTC"), ("IEF", "ETH"), ("IEF", "GLD"), ("SHV", "BTC"), ("SHV", "ETH"), ("SHV", "GLD")]
    # ^ ETF benchmarks vs crypto/gold, needed both for context and so 08_hedge.py
    #   can compare token-vs-BTC/ETH hedges against the conventional SHV/IEF benchmark
)

CANDIDATES = [
    ("GARCH", dict(vol="GARCH", p=1, o=0, q=1)),
    ("GJR-GARCH", dict(vol="GARCH", p=1, o=1, q=1)),
    ("EGARCH", dict(vol="EGARCH", p=1, o=1, q=1)),
]


def is_boundary_pinned(res):
    p = res.params
    tol = 1e-6
    for key, lo, hi in [("alpha[1]", 0.0, 1.0), ("beta[1]", 0.0, 1.0), ("gamma[1]", -1.0, 2.0)]:
        if key in p.index:
            v = p[key]
            if abs(v - lo) < tol or abs(v - hi) < tol:
                return True
    return False


def fit_best_marginal(s, colname):
    fitted = []
    for vol_name, kwargs in CANDIDATES:
        try:
            am = arch_model(s, mean="AR", lags=1, dist="skewt", rescale=False, **kwargs)
            res = am.fit(disp="off", show_warning=False)
            fitted.append((vol_name, res))
        except Exception as e:
            print(f"    [{colname}] {vol_name} failed: {e}")
    converged_clean = [(v, r) for v, r in fitted if r.convergence_flag == 0 and not is_boundary_pinned(r)]
    pool = converged_clean if converged_clean else [(v, r) for v, r in fitted if r.convergence_flag == 0]
    pool = pool if pool else fitted
    best_name, best_res = min(pool, key=lambda x: x[1].aic)
    return best_name, best_res, fitted


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------
garch_results = {}
garch_param_rows = []
model_selection_rows = []

for name, col in UNIVARIATE_SERIES.items():
    s = df[col].dropna()
    best_name, res, all_fits = fit_best_marginal(s, name)
    for vol_name, r in all_fits:
        model_selection_rows.append({
            "series": name, "model": vol_name, "AIC": r.aic, "BIC": r.bic,
            "loglik": r.loglikelihood, "convergence_flag": r.convergence_flag,
            "boundary_pinned": is_boundary_pinned(r), "selected": vol_name == best_name,
        })

    std_resid = res.resid / res.conditional_volatility
    garch_results[name] = {"std_resid": std_resid, "sigma": res.conditional_volatility,
                            "res": res, "model": best_name}
    p = res.params
    alpha = p.get("alpha[1]", np.nan)
    beta = p.get("beta[1]", np.nan)
    gamma = p.get("gamma[1]", np.nan)
    if best_name == "GJR-GARCH":
        persistence = alpha + (gamma if not np.isnan(gamma) else 0) / 2 + beta
    elif best_name == "GARCH":
        persistence = alpha + beta
    else:  # EGARCH: persistence of log-variance is beta alone
        persistence = beta

    garch_param_rows.append({
        "series": name, "model": best_name, "n_obs": int(s.shape[0]),
        "mu": p.get("Const", np.nan),
        "phi_AR1": next((p[k] for k in p.index if k.endswith("[1]") and "R_" in k), np.nan),
        "omega": p.get("omega", np.nan), "alpha": alpha,
        "gamma_asym": gamma, "beta": beta,
        "nu_df": p.get("eta", p.get("nu", np.nan)), "lambda_skew": p.get("lambda", np.nan),
        "persistence": persistence,
        "loglik": res.loglikelihood, "AIC": res.aic, "BIC": res.bic,
    })
    print(f"[GARCH] {name}: model={best_name}, n={s.shape[0]}, persistence={persistence:.4f}, "
          f"gamma={gamma if not np.isnan(gamma) else float('nan'):.4f}")

garch_param_df = pd.DataFrame(garch_param_rows).set_index("series")
garch_param_df.to_csv(PROC / "garch_params.csv")
pd.DataFrame(model_selection_rows).to_csv(PROC / "garch_model_selection.csv", index=False)

# ---------------------------------------------------------------------------
# Stage 2: DCC / ADCC on standardized residuals, per pair
# ---------------------------------------------------------------------------


def build_Q_path(U, a, b, g, asymmetric):
    T, N = U.shape
    Qbar = np.cov(U.T, bias=True)
    if asymmetric:
        Neg = np.where(U < 0, U, 0.0)
        Nbar = (Neg.T @ Neg) / T
    Qt = Qbar.copy()
    Q_path = np.zeros((T, N, N))
    Q_path[0] = Qt
    for t in range(1, T):
        u_prev = U[t - 1, :].reshape(-1, 1)
        if asymmetric:
            n_prev = np.where(u_prev < 0, u_prev, 0.0)
            Qt = (Qbar - a * Qbar - b * Qbar - g * Nbar) + a * (u_prev @ u_prev.T) \
                 + g * (n_prev @ n_prev.T) + b * Qt
        else:
            Qt = (1 - a - b) * Qbar + a * (u_prev @ u_prev.T) + b * Qt
        Q_path[t] = Qt
    return Q_path


def neg_corr_loglik(params, U, asymmetric):
    a, b = params[0], params[1]
    g = params[2] if asymmetric else 0.0
    if a < 0 or b < 0 or g < 0 or (a + b + g) >= 0.999:
        return 1e10
    Q_path = build_Q_path(U, a, b, g, asymmetric)
    T = U.shape[0]
    ll = 0.0
    for t in range(T):
        Qt = Q_path[t]
        d = np.sqrt(np.diag(Qt))
        if np.any(d <= 0):
            return 1e10
        Rt = Qt / np.outer(d, d)
        sign, logdet = np.linalg.slogdet(Rt)
        if sign <= 0:
            return 1e10
        try:
            Rinv = np.linalg.inv(Rt)
        except np.linalg.LinAlgError:
            return 1e10
        u_t = U[t, :]
        quad = u_t @ Rinv @ u_t
        ll += logdet + quad - (u_t @ u_t)
    return 0.5 * ll


def fit_dcc(U, asymmetric):
    # a (short-run news) is bounded well below b (persistence) as is standard
    # DCC practice (Engle 2002; Cappiello, Engle and Sheppard 2006), which
    # also keeps the correlation path from degenerating into a noisy,
    # near-ARCH(1)-only process on these short, fat-tailed token samples.
    bounds = [(1e-6, 0.3), (1e-6, 0.998)] + ([(1e-6, 0.3)] if asymmetric else [])
    cons = [{"type": "ineq", "fun": lambda x: 0.998 - (x[0] + x[1] + (x[2] if asymmetric else 0.0))}]
    starts = [[0.03, 0.90, 0.02], [0.05, 0.70, 0.05], [0.02, 0.50, 0.02], [0.01, 0.20, 0.01]]
    if not asymmetric:
        starts = [s[:2] for s in starts]
    best = None
    for x0 in starts:
        res = minimize(neg_corr_loglik, x0, args=(U, asymmetric), method="SLSQP",
                        bounds=bounds, constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        best = res  # fall back to the last attempt even if not flagged successful
    return best


pair_summary_rows = []

for tok, eq in CORE_PAIRS:
    z1 = garch_results[tok]["std_resid"]
    z2 = garch_results[eq]["std_resid"]
    common = z1.index.intersection(z2.index)
    Z = pd.concat([z1.loc[common], z2.loc[common]], axis=1).dropna()
    Z.columns = [tok, eq]
    if len(Z) < 60:
        print(f"[DCC] skip {tok}-{eq}: only {len(Z)} overlapping obs")
        continue
    U = Z.values

    dcc_res = fit_dcc(U, asymmetric=False)
    adcc_res = fit_dcc(U, asymmetric=True)

    a_dcc, b_dcc = dcc_res.x
    a_adcc, b_adcc, g_adcc = adcc_res.x

    # Constant-correlation (CCC) benchmark loglik (a=b=0 -> R_t = Qbar for all t),
    # used for a likelihood-ratio test of H2 (Engle-Sheppard-style test of
    # constant vs. dynamic conditional correlation; see supplementary
    # appendix for the exact-score-test caveat).
    loglik_ccc = -neg_corr_loglik([0.0, 0.0], U, False)
    loglik_dcc = -dcc_res.fun
    lr_stat = max(0.0, 2 * (loglik_dcc - loglik_ccc))
    lr_pvalue = 1 - chi2.cdf(lr_stat, df=2)

    Q_dcc = build_Q_path(U, a_dcc, b_dcc, 0.0, False)
    Q_adcc = build_Q_path(U, a_adcc, b_adcc, g_adcc, True)

    def rho_series(Q_path):
        rhos = []
        for Qt in Q_path:
            d = np.sqrt(np.diag(Qt))
            Rt = Qt / np.outer(d, d)
            rhos.append(Rt[0, 1])
        return np.array(rhos)

    rho_dcc = rho_series(Q_dcc)
    rho_adcc = rho_series(Q_adcc)

    out = pd.DataFrame({
        "date": Z.index, "rho_dcc": rho_dcc, "rho_adcc": rho_adcc,
        "sigma_tok": garch_results[tok]["sigma"].reindex(Z.index).values,
        "sigma_eq": garch_results[eq]["sigma"].reindex(Z.index).values,
    })
    out["h_cov_dcc"] = out["rho_dcc"] * out["sigma_tok"] * out["sigma_eq"]
    fname = f"dcc_{tok}_{eq}.csv"
    out.to_csv(PROC / fname, index=False)

    pair_summary_rows.append({
        "pair": f"{tok}-{eq}", "n_obs": len(Z),
        "a_DCC": a_dcc, "b_DCC": b_dcc, "loglik_DCC": loglik_dcc,
        "a_ADCC": a_adcc, "b_ADCC": b_adcc, "g_ADCC": g_adcc, "loglik_ADCC": -adcc_res.fun,
        "mean_rho_DCC": rho_dcc.mean(), "sd_rho_DCC": rho_dcc.std(),
        "min_rho_DCC": rho_dcc.min(), "max_rho_DCC": rho_dcc.max(),
        "corr_const_naive": np.corrcoef(U[:, 0], U[:, 1])[0, 1],
        "loglik_CCC": loglik_ccc, "LR_stat_vs_CCC": lr_stat, "LR_pvalue_vs_CCC": lr_pvalue,
    })
    print(f"[DCC/ADCC] {tok}-{eq}: n={len(Z)}, a_DCC={a_dcc:.4f}, b_DCC={b_dcc:.4f}, "
          f"g_ADCC={g_adcc:.4f}, mean_rho={rho_dcc.mean():.4f}")

pair_summary_df = pd.DataFrame(pair_summary_rows).set_index("pair")
pair_summary_df.to_csv(PROC / "dcc_adcc_summary.csv")
print(pair_summary_df.round(4))
