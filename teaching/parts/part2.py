# -*- coding: utf-8 -*-
"""第 2 篇：時間序列前置檢定（單元 4–6），對應 03_pretests.py。"""

from ._helpers import md, code, unit_header

CELLS = [

md("""
---
---

# 第 2 篇　時間序列前置檢定

前置檢定的目的不是「跑一遍讓表格好看」，而是**確認後續模型的前提成立**。
三個單元對應三個前提：

| 檢定 | 確認的前提 | 若不成立會發生什麼 |
|---|---|---|
| Jarque–Bera | 條件分配該不該用常態 | 用常態 GARCH 會低估尾部風險 |
| ADF / PP / KPSS | 序列定態 | VAR 與 GARCH 的漸近理論全部失效 |
| Ljung–Box / ARCH-LM | 存在條件異質變異 | 若沒有 ARCH 效果，GARCH 就是多餘的 |

第三項尤其常被忽略：**先證明有 ARCH 效果，才有理由配適 GARCH**。
"""),

unit_header("單元 4", "描述統計與 Jarque–Bera 常態性檢定",
            "動差估計；Jarque–Bera 檢定",
            "03_pretests.py",
            ["Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial independence of regression residuals. Economics Letters, 6(3), 255–259."]),

md("""
### 原理與公式

給定樣本 $\\{r_t\\}_{t=1}^{T}$，樣本偏態與峰態為

$$
S = \\frac{\\frac{1}{T}\\sum_{t=1}^{T}(r_t - \\bar{r})^3}{\\hat{\\sigma}^3},
\\qquad
K = \\frac{\\frac{1}{T}\\sum_{t=1}^{T}(r_t - \\bar{r})^4}{\\hat{\\sigma}^4}
$$

（本研究用 $K$ 的**非中心化**定義，常態分配下 $K = 3$。）

Jarque–Bera 統計量同時檢定「偏態為 0」與「峰態為 3」：

$$
\\mathrm{JB} = \\frac{T}{6}\\left(S^2 + \\frac{(K-3)^2}{4}\\right)
\\xrightarrow{d} 
\\chi^2(2) \\quad \\text{在 } H_0: \\text{常態}
$$

### 為什麼這個檢定幾乎一定會拒絕

日頻金融報酬**必然**呈現厚尾。JB 拒絕常態不是發現，而是預期。
真正有資訊的是**拒絕的程度與方向**：

- 峰態 $K \\approx 4$–5：溫和厚尾，$t$ 分配自由度約 5–8
- 峰態 $K > 15$：極端厚尾，通常意味著少數幾個跳躍主導了整個樣本
- 偏態 $S$ 顯著為正：右尾長 —— 對代幣的 TVL 序列，這對應機構整筆申購

### 對建模的直接影響

本研究的 GARCH 邊際模型統一使用**偏態 $t$ 分配**（`dist="skewt"`），
而非常態。原因就在描述統計表：代幣市值軌的峰態高達 16–17，偏態超過 2.4。
用常態分配的 QMLE 雖然參數估計仍有一致性，但：
- 標準誤會被嚴重低估
- VaR / CVaR 這類尾部量會系統性偏低
- 概似比檢定（單元 13 要用）的分配會扭曲
"""),

code("""
# ---------------------------------------------------------------------------
# 描述統計：讀取管線輸出並解讀
# ---------------------------------------------------------------------------
desc = pd.read_csv(PROC / "descriptive_stats.csv", index_col=0)
display(desc[["n", "mean", "sd", "skew", "kurtosis", "JB_stat", "JB_pvalue"]].round(4))

print("解讀重點：")
print(f"  1. 全部 {len(desc)} 個序列的 JB p 值皆 < 0.05 → 常態全數遭拒（預期之內）")
print(f"  2. 代幣市值軌峰態最高：R_OUSG_mkt = {desc.loc['R_OUSG_mkt','kurtosis']:.1f}, "
      f"R_BUIDL_mkt = {desc.loc['R_BUIDL_mkt','kurtosis']:.1f}")
print(f"  3. 對應的偏態亦為正且大：{desc.loc['R_OUSG_mkt','skew']:.2f}, {desc.loc['R_BUIDL_mkt','skew']:.2f}")
print("     → 右尾長，符合『機構整筆申購造成 TVL 向上跳躍』的解釋")
print(f"  4. 對照 R_FANG 峰態僅 {desc.loc['R_FANG','kurtosis']:.2f}，偏態 {desc.loc['R_FANG','skew']:.2f}（左偏，股市典型）")
"""),

code("""
# ---------------------------------------------------------------------------
# 從零實作 Jarque-Bera，驗證與 scipy 一致
# ---------------------------------------------------------------------------
from scipy import stats

def jarque_bera_manual(x):
    x = np.asarray(x)
    T = len(x)
    m = x.mean()
    s2 = ((x - m) ** 2).mean()          # 有偏（除以 T），與定義一致
    S = ((x - m) ** 3).mean() / s2 ** 1.5
    K = ((x - m) ** 4).mean() / s2 ** 2
    JB = T / 6 * (S ** 2 + (K - 3) ** 2 / 4)
    p = 1 - stats.chi2.cdf(JB, df=2)
    return JB, p, S, K

r = panel["R_FANG"].dropna()
jb_manual = jarque_bera_manual(r)
jb_scipy = stats.jarque_bera(r)

print(f"手動實作 : JB = {jb_manual[0]:9.4f}, p = {jb_manual[1]:.3e}, S = {jb_manual[2]:+.4f}, K = {jb_manual[3]:.4f}")
print(f"scipy    : JB = {jb_scipy[0]:9.4f}, p = {jb_scipy[1]:.3e}")
print(f"差異     : {abs(jb_manual[0] - jb_scipy[0]):.2e}")
"""),

unit_header("單元 5", "單根與定態：ADF、Phillips–Perron、KPSS",
            "單根檢定（ADF, PP）；定態檢定（KPSS）；虛無假設方向相反的互補設計",
            "03_pretests.py",
            ["Dickey, D. A., & Fuller, W. A. (1979)", "Phillips, P. C. B., & Perron, P. (1988)",
             "Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992)"]),

md("""
### 原理：為什麼定態如此關鍵

VAR、GARCH、DCC 的漸近理論全部建立在**弱定態**（covariance stationarity）之上：

$$
\\mathbb{E}[r_t] = \\mu, \\quad
\\mathrm{Var}(r_t) = \\sigma^2 < \\infty, \\quad
\\mathrm{Cov}(r_t, r_{t-k}) = \\gamma_k  \\text{（僅依賴 } k\\text{）}
$$

若序列有單根，OLS 的 $t$ 統計量不再收斂到常態，而是收斂到非標準的泛函布朗運動分配 ——
所有的 $p$ 值都會失去意義（Granger–Newbold 的偽迴歸問題）。

### 三個檢定的公式與虛無假設

**ADF（Augmented Dickey–Fuller）**

檢定迴歸：

$$
\\Delta r_t = \\alpha + \\gamma r_{t-1} + \\sum_{i=1}^{p} \\delta_i \\Delta r_{t-i} + \\varepsilon_t
$$

$$
H_0: \\gamma = 0  (\\text{有單根}) \\quad\\text{vs}\\quad H_1: \\gamma < 0  (\\text{定態})
$$

落後差分項 $\\sum \\delta_i \\Delta r_{t-i}$ 的作用是**吸收殘差中的序列相關**，讓 $\\varepsilon_t$ 接近白噪音。

**Phillips–Perron**

與 ADF 相同的虛無假設，但處理序列相關的方式不同：不加落後項，
而是對統計量做**非參數的 Newey–West 型修正**。

$$
Z_\\tau = \\left(\\frac{\\hat{\\gamma}_0}{\\hat{\\lambda}^2}\\right)^{1/2} t_{\\hat{\\gamma}}
- \\frac{(\\hat{\\lambda}^2 - \\hat{\\gamma}_0) T  \\mathrm{se}(\\hat{\\gamma})}{2 \\hat{\\lambda}^2 \\hat{\\sigma}}
$$

其中 $\\hat{\\lambda}^2$ 是長期變異數的一致估計。PP 對**異質變異**比 ADF 穩健。

**KPSS —— 虛無假設方向相反**

$$
r_t = \\xi t + \\rho_t + \\varepsilon_t, \\qquad \\rho_t = \\rho_{t-1} + u_t
$$

$$
H_0: \\sigma_u^2 = 0  (\\text{定態}) \\quad\\text{vs}\\quad H_1: \\sigma_u^2 > 0  (\\text{有單根})
$$

統計量為部分和的平方累積：

$$
\\mathrm{KPSS} = \\frac{1}{T^2 \\hat{\\lambda}^2} \\sum_{t=1}^{T} S_t^2,
\\qquad S_t = \\sum_{i=1}^{t} \\hat{\\varepsilon}_i
$$
"""),

md("""
### 為什麼要同時做三個：虛無假設方向的互補性

這是本單元的核心教學點。ADF / PP 與 KPSS 的虛無假設**方向相反**：

| | ADF / PP | KPSS |
|---|---|---|
| $H_0$ | 有單根（非定態） | 定態 |
| 拒絕代表 | 定態 | 非定態 |

「無法拒絕 $H_0$」**不等於**「$H_0$ 成立」—— 它也可能只是檢定力不足。
把兩個方向相反的檢定並用，就能區分四種情況：

| ADF 拒絕? | KPSS 拒絕? | 結論 |
|---|---|---|
| 是 | 否 | **定態**（兩者一致，最理想） |
| 否 | 是 | **非定態**（兩者一致） |
| 是 | 是 | 矛盾 → 可能是結構斷點或分數整合 |
| 否 | 否 | 資訊不足 → 樣本太短或檢定力太低 |

### 本研究的實際結果與其意義

代幣的 **NAV 軌明確地無法拒絕單根**（ADF $p \\approx 0.95$，KPSS $p \\approx 0.0001$，
即兩個檢定一致指向非定態）。這**不是資料錯誤**，而是建構方式的必然後果：

NAV 軌是 3 個月期國庫券殖利率除以 365，而**殖利率水準本身就是高度持續的近單根序列**。
把一個近單根序列做線性縮放，當然還是近單根。

這在論文中被明確揭露為限制：NAV 軌比市場決定的序列平滑得多，
因此**所有主要結果都建立在市值軌上**，NAV 軌僅作為流動性溢價的參照。
""")
,

code("""
# ---------------------------------------------------------------------------
# 單根檢定結果的四象限解讀
# ---------------------------------------------------------------------------
ur = pd.read_csv(PROC / "unitroot_tests.csv", index_col=0)

def verdict(row):
    adf_rej = row["ADF_pvalue"] < 0.05
    kpss_rej = row["KPSS_pvalue"] < 0.05
    if adf_rej and not kpss_rej:
        return "定態（一致）"
    if not adf_rej and kpss_rej:
        return "非定態（一致）"
    if adf_rej and kpss_rej:
        return "矛盾（斷點?）"
    return "資訊不足"

ur["判定"] = ur.apply(verdict, axis=1)
display(ur[["ADF_stat", "ADF_pvalue", "PP_pvalue", "KPSS_stat", "KPSS_pvalue", "判定"]].round(4))

print("\\n分類統計：")
print(ur["判定"].value_counts().to_string())
print("\\n注意 R_BUIDL_nav 與 R_OUSG_nav：兩列數字完全相同（同一代理序列），")
print("且皆為『非定態（一致）』—— 殖利率水準的近單根性質被線性縮放後保留下來。")
"""),

code("""
# ---------------------------------------------------------------------------
# 視覺對照：定態的市值軌 vs 近單根的 NAV 軌
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

s_nav = panel["R_OUSG_nav"].dropna()
s_mkt = panel["R_OUSG_mkt"].dropna()

axes[0].plot(s_nav.index, s_nav.values, lw=1, color="#1b4965")
axes[0].set_title(f"NAV 軌：ADF p = {ur.loc['R_OUSG_nav','ADF_pvalue']:.3f}（無法拒絕單根）")
axes[0].set_ylabel("% / 日")

axes[1].plot(s_mkt.index, s_mkt.values, lw=0.5, color="#bc4749")
axes[1].axhline(0, color="grey", lw=0.7)
axes[1].set_title(f"市值軌：ADF p = {ur.loc['R_OUSG_mkt','ADF_pvalue']:.4f}（定態）")

fig.tight_layout()
plt.show()

print("左圖呈現緩慢的水準漂移（跟隨聯準會政策路徑），右圖繞著零均值震盪。")
print("肉眼即可分辨，檢定只是把這個判斷形式化。")
"""),

unit_header("單元 6", "序列相關與條件異質變異：Ljung–Box、ARCH-LM",
            "Ljung–Box Q 檢定；Engle ARCH-LM 檢定",
            "03_pretests.py",
            ["Ljung, G. M., & Box, G. E. P. (1978)", "Engle, R. F. (1982)"]),

md("""
### Ljung–Box Q 檢定

檢定「前 $m$ 階自相關是否全為零」：

$$
Q(m) = T(T+2) \\sum_{k=1}^{m} \\frac{\\hat{\\rho}_k^2}{T-k}
\\xrightarrow{d}  \\chi^2(m)
\\qquad H_0: \\rho_1 = \\cdots = \\rho_m = 0
$$

其中 $\\hat{\\rho}_k$ 是第 $k$ 階樣本自相關。$T(T+2)/(T-k)$ 是相對於原始 Box–Pierce 統計量的
小樣本修正。

**關鍵：對 $r_t$ 與對 $r_t^2$ 各做一次**

| 檢定對象 | 拒絕代表 | 建模含意 |
|---|---|---|
| $Q(m)$ on $r_t$ | 報酬本身可預測 | 需要 AR 或 ARMA 的均值方程 |
| $Q(m)$ on $r_t^2$ | **波動叢聚** | 需要 GARCH 族的變異數方程 |

本研究的均值方程統一用 **AR(1)**（`mean="AR", lags=1`），正是為了吸收第一項。

### Engle ARCH-LM 檢定

更直接地檢定 ARCH 效果。對平方殘差做輔助迴歸：

$$
\\hat{\\varepsilon}_t^2 = \\alpha_0 + \\alpha_1 \\hat{\\varepsilon}_{t-1}^2 + \\cdots
+ \\alpha_q \\hat{\\varepsilon}_{t-q}^2 + u_t
$$

$$
\\mathrm{LM} = T \\cdot R^2 \\xrightarrow{d}  \\chi^2(q)
\\qquad H_0: \\alpha_1 = \\cdots = \\alpha_q = 0  (\\text{無 ARCH 效果})
$$

### 為什麼這一步不能跳過

這是**配適 GARCH 的正當性依據**。如果 ARCH-LM 無法拒絕，那條件變異數就是常數，
GARCH 模型會退化成同方差模型，估出來的 $\\alpha, \\beta$ 只是在配適雜訊。

本研究的結果顯示一個值得注意的差異：
- 傳統資產（FANG、SOXX、GLD）與代幣市值軌：ARCH-LM **強烈拒絕**，GARCH 有正當性
- 但 **OUSG 市值軌的 ARCH-LM $p \\approx 0.26$，無法拒絕**

後者是誠實的訊號：OUSG 的 TVL 變動比較接近「隨機的大跳躍」而非「波動叢聚」。
這與單元 10 會看到的現象一致 —— 代幣序列的 GARCH 配適常常落在參數邊界上。
"""),

code("""
# ---------------------------------------------------------------------------
# 診斷檢定結果：報酬 vs 平方報酬
# ---------------------------------------------------------------------------
diag = pd.read_csv(PROC / "diagnostics.csv", index_col=0)
display(diag.round(4))

print("解讀：")
print("  LB10        = Ljung-Box on r_t      → 報酬本身的可預測性（AR(1) 均值方程要處理的）")
print("  LB10_sq     = Ljung-Box on r_t^2    → 波動叢聚（GARCH 要處理的）")
print("  ARCH_LM     = Engle LM              → ARCH 效果的直接檢定")
print()
no_arch = diag[diag["ARCH_LM_pvalue"] > 0.05]
print(f"ARCH-LM 無法拒絕（p > 0.05）的序列共 {len(no_arch)} 個：")
for s, row in no_arch.iterrows():
    print(f"    {s:14} ARCH-LM p = {row['ARCH_LM_pvalue']:.4f}")
print()
print("→ 這些序列配適 GARCH 的正當性較弱，其參數估計應謹慎解讀（見單元 10）。")
"""),

code("""
# ---------------------------------------------------------------------------
# 從零實作 Ljung-Box，並視覺化波動叢聚
# ---------------------------------------------------------------------------
def ljung_box(x, m=10):
    x = np.asarray(x); T = len(x)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    Q = 0.0
    for k in range(1, m + 1):
        rho_k = (xc[k:] * xc[:-k]).sum() / denom
        Q += rho_k ** 2 / (T - k)
    Q *= T * (T + 2)
    return Q, 1 - stats.chi2.cdf(Q, df=m)

r = panel["R_FANG"].dropna()
q_lvl, p_lvl = ljung_box(r, 10)
q_sq, p_sq = ljung_box(r ** 2, 10)

print(f"R_FANG   Q(10) on r_t   = {q_lvl:8.3f}, p = {p_lvl:.4f}  → "
      f"{'報酬可預測' if p_lvl < 0.05 else '報酬近似不可預測'}")
print(f"R_FANG   Q(10) on r_t^2 = {q_sq:8.3f}, p = {p_sq:.4f}  → "
      f"{'存在波動叢聚' if p_sq < 0.05 else '無明顯波動叢聚'}")
print()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(r.index, r.values, lw=0.5, color="#1b4965")
axes[0].set_title("報酬 $r_t$：均值附近震盪，方向難以預測")
axes[1].plot(r.index, (r ** 2).values, lw=0.5, color="#bc4749")
axes[1].set_title("平方報酬 $r_t^2$：明顯的叢聚結構 → GARCH 的正當性")
fig.tight_layout()
plt.show()
"""),

]
