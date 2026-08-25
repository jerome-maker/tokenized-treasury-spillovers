# -*- coding: utf-8 -*-
"""第 3 篇：邊際波動模型（單元 7–10），對應 04_garch_dcc.py 階段一。"""

from ._helpers import md, code, unit_header

CELLS = [

md("""
---
---

# 第 3 篇　邊際波動模型

DCC 是一個**兩階段**估計程序。階段一為每個序列各自配適一個單變量波動模型，
取出標準化殘差；階段二才在標準化殘差上估計相關動態。

這一篇處理階段一。三個候選模型（GARCH、GJR-GARCH、EGARCH）不是隨意並列，
而是三種對「波動如何回應衝擊」的不同假設。單元 10 說明如何在它們之間做出**有依據的**選擇，
以及當最佳化落在參數邊界時該怎麼辦 —— 後者在本研究的代幣序列上頻繁發生，
是必須誠實報告的實證特徵。
"""),

unit_header("單元 7", "GARCH(1,1)：條件變異數的遞迴結構",
            "廣義自迴歸條件異質變異；QMLE",
            "04_garch_dcc.py（階段一）",
            ["Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307–327.",
             "Engle, R. F. (1982). Autoregressive conditional heteroscedasticity. Econometrica, 50(4), 987–1008."]),

md("""
### 原理

單元 6 已證明報酬存在波動叢聚：大波動之後跟著大波動。GARCH 用一條遞迴式把這件事形式化。

模型分成**均值方程**與**變異數方程**兩部分。本研究的均值方程為 AR(1)：

$$
r_t = \\mu + \\phi r_{t-1} + \\varepsilon_t, \\qquad \\varepsilon_t = \\sigma_t z_t, \\qquad z_t \\sim \\text{skew-}t(\\nu, \\lambda)
$$

變異數方程為 GARCH(1,1)：

$$
\\sigma_t^2 = \\omega + \\alpha \\varepsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2
$$

### 每個參數的意義

| 參數 | 名稱 | 解讀 |
|---|---|---|
| $\\omega > 0$ | 常數項 | 決定長期變異數的水準 |
| $\\alpha \\ge 0$ | ARCH 項／**新聞衝擊** | 昨日「意外」的平方對今日波動的影響。大 $\\alpha$ = 波動對新資訊反應劇烈 |
| $\\beta \\ge 0$ | GARCH 項／**持續性** | 昨日波動水準的延續。大 $\\beta$ = 波動有長記憶 |

### 定態條件與長期變異數

弱定態要求

$$
\\alpha + \\beta < 1
$$

此時無條件（長期）變異數存在且為

$$
\\bar{\\sigma}^2 = \\mathbb{E}[\\sigma_t^2] = \\frac{\\omega}{1 - \\alpha - \\beta}
$$

$\\alpha + \\beta$ 稱為**持續性（persistence）**。它的直觀意義是「衝擊的半衰期」：

$$
\\text{半衰期} = \\frac{\\ln 0.5}{\\ln(\\alpha + \\beta)}  \\text{（交易日）}
$$

- $\\alpha + \\beta = 0.95$ → 半衰期約 13.5 天
- $\\alpha + \\beta = 0.99$ → 半衰期約 69 天
- $\\alpha + \\beta \\to 1$ → IGARCH，衝擊永不衰減，無條件變異數不存在

### 演算法：遞迴的初始化

$\\sigma_1^2$ 沒有前一期可用，必須初始化。常見做法是用樣本無條件變異數：

$$
\\sigma_1^2 = \\frac{1}{T}\\sum_{t=1}^{T} \\varepsilon_t^2
$$

初始值的影響隨 $\\beta^t$ 幾何衰減，在 $T$ 夠大時可忽略 —— 但在本研究 sUSDS 這種
只有 371 個觀測的短樣本上，這個「可忽略」的假設本身就值得警惕。

### 估計：準最大概似（QMLE）

在條件常態假設下，對數概似為

$$
\\ell(\\theta) = -\\frac{1}{2}\\sum_{t=1}^{T}\\left[\\ln(2\\pi) + \\ln \\sigma_t^2(\\theta) + \\frac{\\varepsilon_t^2}{\\sigma_t^2(\\theta)}\\right]
$$

即使真實分配不是常態，這個估計量仍有一致性（此即「準」的來源），
但標準誤需要用夾心估計量修正。本研究直接改用偏態 $t$ 概似（單元 10），避免這個問題。
"""),

code("""
# ---------------------------------------------------------------------------
# 從零實作 GARCH(1,1) 的遞迴與概似，並與 arch 套件對照
# ---------------------------------------------------------------------------
from scipy.optimize import minimize

def garch11_recursion(eps, omega, alpha, beta):
    \"\"\"給定參數，回傳條件變異數路徑 sigma^2_t。\"\"\"
    T = len(eps)
    sigma2 = np.empty(T)
    sigma2[0] = eps.var()                      # 以樣本無條件變異數初始化
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
    return sigma2

def garch11_neg_loglik(params, eps):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e10                            # 違反正性與定態條件
    s2 = garch11_recursion(eps, omega, alpha, beta)
    if np.any(s2 <= 0):
        return 1e10
    return 0.5 * np.sum(np.log(2*np.pi) + np.log(s2) + eps**2 / s2)

# 以 FANG 為例（先去除 AR(1) 均值以取得 eps）
r = panel["R_FANG"].dropna().values
phi = np.corrcoef(r[1:], r[:-1])[0, 1]
eps = r[1:] - r[1:].mean() - phi * (r[:-1] - r[:-1].mean())

res = minimize(garch11_neg_loglik, x0=[0.05, 0.08, 0.88], args=(eps,),
               method="L-BFGS-B", bounds=[(1e-8, None), (0, 1), (0, 1)])
w, a, b = res.x
print(f"手動 GARCH(1,1)：omega = {w:.5f}, alpha = {a:.4f}, beta = {b:.4f}")
print(f"  持續性 alpha+beta = {a+b:.4f}")
print(f"  長期變異數       = {w/(1-a-b):.4f}  →  長期日波動 = {np.sqrt(w/(1-a-b)):.4f}%")
print(f"  衝擊半衰期       = {np.log(0.5)/np.log(a+b):.1f} 個交易日")
"""),

code("""
# ---------------------------------------------------------------------------
# 對照 arch 套件（研究實際使用的實作）
# ---------------------------------------------------------------------------
from arch import arch_model

am = arch_model(panel["R_FANG"].dropna(), mean="AR", lags=1,
                vol="GARCH", p=1, q=1, dist="normal", rescale=False)
fit = am.fit(disp="off")
p = fit.params
print(f"arch 套件      ：omega = {p['omega']:.5f}, alpha = {p['alpha[1]']:.4f}, beta = {p['beta[1]']:.4f}")
print(f"  持續性        = {p['alpha[1]'] + p['beta[1]']:.4f}")
print()
print("差異來源：手動版對 AR(1) 用簡易估計、且與變異數方程分開估計；")
print("arch 套件對均值與變異數參數做聯合最大概似估計。方向與量級一致即說明實作正確。")
"""),

code("""
# ---------------------------------------------------------------------------
# 視覺化：條件波動路徑 vs 絕對報酬
# ---------------------------------------------------------------------------
r_series = panel["R_FANG"].dropna()
cond_vol = fit.conditional_volatility

fig, ax = plt.subplots(figsize=(9.5, 3.4))
ax.plot(r_series.index, r_series.abs(), lw=0.4, color="#c9c9c9", label="$|r_t|$（實際）")
ax.plot(cond_vol.index, cond_vol, lw=1.3, color="#bc4749", label=r"$\\sigma_t$（GARCH 條件波動）")
ax.set_title("GARCH(1,1) 條件波動：把叢聚結構萃取成一條平滑路徑")
ax.set_ylabel("% / 日")
ax.legend()
fig.tight_layout()
plt.show()
"""),

unit_header("單元 8", "GJR-GARCH：槓桿效果與門檻項",
            "門檻 GARCH；非對稱波動反應",
            "04_garch_dcc.py（階段一）",
            ["Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. Journal of Finance, 48(5), 1779–1801."]),

md("""
### 原理：GARCH(1,1) 漏掉了什麼

GARCH 的變異數方程只用到 $\\varepsilon_{t-1}^2$ —— **平方項對稱**，
意味著 +3% 與 −3% 的衝擊對明日波動有完全相同的影響。

但股票市場有一個穩健的實證規律：**下跌比同幅度的上漲更能提高波動**。
這稱為**槓桿效果（leverage effect）**，傳統解釋是股價下跌提高了公司的負債權益比，
使股權風險上升。

### 公式

GJR-GARCH(1,1,1) 加入一個**指示函數門檻項**：

$$
\\sigma_t^2 = \\omega + \\alpha \\varepsilon_{t-1}^2
+ \\gamma \\varepsilon_{t-1}^2 \\mathbb{I}_{[\\varepsilon_{t-1} < 0]}
+ \\beta \\sigma_{t-1}^2
$$

其中

其中指示函數 $\\mathbb{I}_{[\\varepsilon_{t-1} < 0]}$ 在 $\\varepsilon_{t-1} < 0$ 時取值 1，否則取值 0。

### 解讀 $\\gamma$

| 衝擊方向 | 對 $\\sigma_t^2$ 的邊際效果 |
|---|---|
| 正向（$\\varepsilon_{t-1} \\ge 0$） | $\\alpha$ |
| 負向（$\\varepsilon_{t-1} < 0$） | $\\alpha + \\gamma$ |

- $\\gamma > 0$：**槓桿效果**，負向衝擊放大波動（股票的典型型態）
- $\\gamma = 0$：退化為標準 GARCH
- $\\gamma < 0$：反向的非對稱 —— 正向衝擊反而提高波動

### 持續性的修正

因為門檻項平均只有一半的時間啟動（假設衝擊對稱分布），持續性變成

$$
\\text{persistence} = \\alpha + \\frac{\\gamma}{2} + \\beta
$$

程式碼裡正是這樣寫的：

```python
if best_name == "GJR-GARCH":
    persistence = alpha + (gamma if not np.isnan(gamma) else 0) / 2 + beta
```

> **為什麼 $\\gamma < 0$ 對本研究是有意義的發現**
>
> 代幣化國債的 TVL 序列若出現 $\\gamma < 0$，代表**流入（正向衝擊）比流出更能提高波動**。
> 這與股票的槓桿效果方向相反，但符合「大額機構申購造成 TVL 跳躍」的機制 ——
> 波動由資金流驅動，而非由價格發現驅動。單元 10 會看到這個型態的實際估計結果。
"""),

code("""
# ---------------------------------------------------------------------------
# 新聞衝擊曲線（News Impact Curve）：對稱 vs 非對稱的視覺化
# ---------------------------------------------------------------------------
def nic_garch(eps_range, omega, alpha, beta, sigma2_bar):
    return omega + alpha * eps_range**2 + beta * sigma2_bar

def nic_gjr(eps_range, omega, alpha, gamma, beta, sigma2_bar):
    ind = (eps_range < 0).astype(float)
    return omega + alpha * eps_range**2 + gamma * eps_range**2 * ind + beta * sigma2_bar

e = np.linspace(-5, 5, 400)
s2bar = 2.0
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(e, nic_garch(e, 0.05, 0.10, 0.85, s2bar), lw=1.8, color="#1b4965",
        label=r"GARCH：對稱（$\\gamma=0$）")
ax.plot(e, nic_gjr(e, 0.05, 0.05, 0.12, 0.85, s2bar), lw=1.8, color="#bc4749",
        label=r"GJR：$\\gamma=+0.12$（槓桿效果，股票型態）")
ax.plot(e, nic_gjr(e, 0.05, 0.16, -0.12, 0.85, s2bar), lw=1.8, color="#2a9d8f", ls="--",
        label=r"GJR：$\\gamma=-0.12$（反向，資金流驅動型態）")
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel(r"前期衝擊 $\\varepsilon_{t-1}$")
ax.set_ylabel(r"$\\sigma_t^2$")
ax.set_title("新聞衝擊曲線：$\\gamma$ 的符號決定曲線往哪邊傾斜")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
"""),

unit_header("單元 9", "EGARCH：對數變異數與無約束參數空間",
            "指數 GARCH；對數轉換避免正性約束",
            "04_garch_dcc.py（階段一）",
            ["Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. Econometrica, 59(2), 347–370."]),

md("""
### 原理：GARCH 與 GJR 的兩個實務痛點

1. **正性約束**：必須強制 $\\omega > 0, \\alpha \\ge 0, \\beta \\ge 0$，否則 $\\sigma_t^2$ 可能為負。
   這讓最佳化器在有界空間裡工作，容易卡在邊界。
2. **非對稱只能靠門檻**：GJR 用指示函數，是分段線性的粗糙處理。

EGARCH 用一個巧妙的轉換同時解決兩者：**對 $\\ln \\sigma_t^2$ 建模**。
因為指數函數恆正，$\\sigma_t^2 = \\exp(\\cdot)$ 自動滿足正性，參數可以自由取任意實數。

### 公式

$$
\\ln \\sigma_t^2 = \\omega
+ \\alpha \\left( |z_{t-1}| - \\mathbb{E}|z_{t-1}| \\right)
+ \\gamma z_{t-1}
+ \\beta \\ln \\sigma_{t-1}^2
$$

其中 $z_t = \\varepsilon_t / \\sigma_t$ 是**標準化**殘差。注意兩個關鍵設計：

**（一）用標準化殘差 $z$ 而非原始殘差 $\\varepsilon$**

這讓衝擊的影響與當時的波動水準無關 —— 高波動時期的 3% 變動與低波動時期的 3% 變動，
在標準化後有不同的意義，這正是我們想要的。

**（二）$\\alpha$ 與 $\\gamma$ 分工明確**

| 項 | 捕捉什麼 |
|---|---|
| $\\alpha(\\lvert z_{t-1}\\rvert - \\mathbb{E}\\lvert z_{t-1}\\rvert)$ | **幅度效果**：衝擊有多大（不分方向） |
| $\\gamma z_{t-1}$ | **符號效果**：衝擊往哪邊（線性，不是門檻） |

減去 $\\mathbb{E}|z_{t-1}|$ 是為了讓幅度項的期望值為零，使 $\\omega$ 保持「長期水準」的解讀。
在標準常態下 $\\mathbb{E}|z| = \\sqrt{2/\\pi} \\approx 0.7979$。

### $\\gamma$ 的符號意義 —— 與 GJR 相反

這是最容易出錯的地方：

| 模型 | 槓桿效果對應的符號 |
|---|---|
| GJR-GARCH | $\\gamma > 0$ |
| **EGARCH** | $\\gamma < 0$ |

因為 EGARCH 的 $\\gamma z_{t-1}$ 是**直接乘上帶符號的 $z$**：
若 $\\gamma < 0$，則 $z_{t-1} < 0$（壞消息）會使 $\\gamma z_{t-1} > 0$，推高對數變異數。

> 本研究的表 3 報告的 `gamma_asym` 欄位混合了兩種模型的 $\\gamma$，
> 因此表格註解必須說明「在本表所用的 EGARCH 參數化下，**負值**對應傳統股票槓桿效果」，
> 否則讀者會把符號讀反。

### 持續性

因為模型作用在對數尺度上，持續性由 $\\beta$ **單獨**決定：

$$
\\text{persistence} = \\beta
$$

程式碼：

```python
else:  # EGARCH: persistence of log-variance is beta alone
    persistence = beta
```

定態條件為 $|\\beta| < 1$，比 GARCH 的 $\\alpha + \\beta < 1$ 寬鬆。
"""),

code("""
# ---------------------------------------------------------------------------
# 三個模型並排配適同一序列，比較 gamma 的符號與量級
# ---------------------------------------------------------------------------
series = panel["R_FANG"].dropna()
specs = [("GARCH", dict(vol="GARCH", p=1, o=0, q=1)),
         ("GJR-GARCH", dict(vol="GARCH", p=1, o=1, q=1)),
         ("EGARCH", dict(vol="EGARCH", p=1, o=1, q=1))]

rows = []
for nm, kw in specs:
    f = arch_model(series, mean="AR", lags=1, dist="skewt", rescale=False, **kw).fit(disp="off")
    pr = f.params
    al, be = pr.get("alpha[1]", np.nan), pr.get("beta[1]", np.nan)
    ga = pr.get("gamma[1]", np.nan)
    if nm == "GJR-GARCH":
        pers = al + (0 if np.isnan(ga) else ga) / 2 + be
    elif nm == "GARCH":
        pers = al + be
    else:
        pers = be
    rows.append({"模型": nm, "alpha": al, "gamma": ga, "beta": be,
                 "持續性": pers, "AIC": f.aic, "BIC": f.bic})

cmp_df = pd.DataFrame(rows).set_index("模型")
display(cmp_df.round(4))

print("R_FANG 的 gamma 符號解讀：")
print(f"  GJR   gamma = {cmp_df.loc['GJR-GARCH','gamma']:+.4f}  → 正值 = 槓桿效果")
print(f"  EGARCH gamma = {cmp_df.loc['EGARCH','gamma']:+.4f}  → 負值 = 槓桿效果（符號慣例相反！）")
print()
print(f"AIC 最小者：{cmp_df['AIC'].idxmin()}")
"""),

unit_header("單元 10", "偏態 t 分配、AIC 模型選擇與邊界解偵測",
            "偏態 t 概似；AIC；箱型約束邊界解的診斷",
            "04_garch_dcc.py（階段一）",
            ["Hansen, B. E. (1994). Autoregressive conditional density estimation. International Economic Review, 35(3), 705–730.",
             "Akaike, H. (1974)"]),

md("""
### 為什麼用偏態 $t$ 而非常態

單元 4 已顯示報酬的峰態遠高於 3、偏態顯著不為零。GARCH 的條件分配若設為常態：

- 參數估計仍有一致性（QMLE 的優點）
- 但**標準誤被低估**，$t$ 統計量灌水
- **尾部量（VaR、CVaR）系統性偏低**
- 概似比檢定（單元 13）的分配扭曲

Hansen (1994) 的偏態 $t$ 有兩個形狀參數：自由度 $\\nu$ 控制尾部厚度，
偏態參數 $\\lambda \\in (-1, 1)$ 控制不對稱。密度為

左尾（$z < -a/b$）：

$$
f(z \\mid \\nu, \\lambda) = bc\\left(1 + \\dfrac{1}{\\nu-2}
\\left(\\dfrac{bz+a}{1-\\lambda}\\right)^2\\right)^{-(\\nu+1)/2}
$$

右尾（$z \\ge -a/b$）：

$$
f(z \\mid \\nu, \\lambda) = bc\\left(1 + \\dfrac{1}{\\nu-2}
\\left(\\dfrac{bz+a}{1+\\lambda}\\right)^2\\right)^{-(\\nu+1)/2}
$$

兩式只差在分母的 $1 \\mp \\lambda$：$\\lambda > 0$ 時右尾被拉長，$\\lambda < 0$ 時左尾被拉長。

其中 $a, b, c$ 是使 $\\mathbb{E}[z]=0, \\mathrm{Var}(z)=1$ 的標準化常數。

**自由度的解讀**：$\\nu \\to \\infty$ 收斂到常態；$\\nu \\le 4$ 時四階動差不存在（峰態無限大）。
本研究中代幣序列的 $\\nu$ 估計值多在 $2.05$ 附近 —— 這已經**逼近 $\\nu > 2$ 的下界**，
意味著變異數勉強存在、峰態不存在。這本身就是一個要報告的發現。

### AIC 模型選擇

$$
\\mathrm{AIC} = -2\\ell(\\hat{\\theta}) + 2k
$$

其中 $k$ 是參數個數。AIC 在「配適度」與「複雜度」之間權衡，選最小者。

（相對地，$\\mathrm{BIC} = -2\\ell + k\\ln T$ 對複雜度懲罰更重，在大樣本下偏好更簡約的模型。
本研究用 AIC，因為目標是**預測性的波動路徑**而非真模型識別。）

### 本研究的關鍵設計：邊界解偵測

這是單元 10 最重要的部分，也是實務上最常被跳過的一步。

最佳化器在箱型約束（box constraint）內搜尋。當估計值**恰好落在約束邊界上**，
代表最佳化器「想往外走但被擋住了」。這時的估計值**不是內點極值**，
標準的漸近理論（依賴一階條件 $\\partial \\ell / \\partial \\theta = 0$）**不成立**。

程式碼用一個明確的容忍度檢查：

```python
def is_boundary_pinned(res):
    p = res.params
    tol = 1e-6
    for key, lo, hi in [("alpha[1]", 0.0, 1.0), ("beta[1]", 0.0, 1.0), ("gamma[1]", -1.0, 2.0)]:
        if key in p.index:
            v = p[key]
            if abs(v - lo) < tol or abs(v - hi) < tol:
                return True
    return False
```

### 選模的三層後備策略

```python
converged_clean = [(v, r) for v, r in fitted if r.convergence_flag == 0 and not is_boundary_pinned(r)]
pool = converged_clean if converged_clean else [(v, r) for v, r in fitted if r.convergence_flag == 0]
pool = pool if pool else fitted
best_name, best_res = min(pool, key=lambda x: x[1].aic)
```

三層優先序：

1. **收斂且未卡邊界** → 最理想，用 AIC 在其中選
2. 若第一層是空的 → 退而求其次，只要求收斂
3. 若第二層也空 → 全部候選一起比

這個設計的價值在於它**把降級這件事變得可見**。如果直接用 `min(fitted, key=aic)`，
一個卡在邊界的模型可能因為 AIC 最小而被選中，而下游沒有任何人會知道。
"""),

code("""
# ---------------------------------------------------------------------------
# 完整的模型選擇表：哪些配適收斂、哪些卡在邊界
# ---------------------------------------------------------------------------
sel = pd.read_csv(PROC / "garch_model_selection.csv")
pivot = sel.pivot_table(index="series", columns="model",
                        values="AIC", aggfunc="first")
flags = sel[sel["selected"]].set_index("series")[["model", "boundary_pinned", "convergence_flag"]]
out = pivot.join(flags.rename(columns={"model": "選中", "boundary_pinned": "卡邊界",
                                        "convergence_flag": "收斂碼"}))
display(out.round(2))

print(f"卡在邊界的配適共 {int(sel['boundary_pinned'].sum())} 個（占全部 {len(sel)} 次配適）")
print()
print("被選中的模型中卡邊界者：")
bad = flags[flags["boundary_pinned"]]
print(bad.to_string() if len(bad) else "  （無）")
"""),

code("""
# ---------------------------------------------------------------------------
# 最終選定的邊際模型：持續性與非對稱參數
# ---------------------------------------------------------------------------
g = pd.read_csv(PROC / "garch_params.csv", index_col=0)
show = g[["model", "n_obs", "persistence", "gamma_asym", "nu_df", "AIC"]].round(4)
display(show)

print("三個關鍵觀察：")
print()
print("1. 傳統資產持續性高（0.95–1.00），代幣低且異質：")
for s in ["FANG", "SOXX", "IEF", "BUIDL", "OUSG", "sUSDS"]:
    print(f"     {s:6} {g.loc[s,'model']:10} persistence = {g.loc[s,'persistence']:.4f}")
print()
print("2. 自由度 nu 逼近下界（2.05）的序列 —— 峰態在理論上不存在：")
low_nu = g[g["nu_df"] < 2.5]
for s, row in low_nu.iterrows():
    print(f"     {s:6} nu = {row['nu_df']:.3f}")
print()
print("3. EGARCH 的 gamma 符號（負 = 傳統槓桿效果）：")
eg = g[g["model"] == "EGARCH"]
for s, row in eg.iterrows():
    tag = "槓桿效果（股票型態）" if row["gamma_asym"] < 0 else "反向（資金流驅動型態）"
    print(f"     {s:6} gamma = {row['gamma_asym']:+.4f}  {tag}")
"""),

md("""
### 這一節的結果如何影響論文的寫法

上面第 3 點的輸出值得停下來看。三個代幣都選了 EGARCH，但 $\\gamma$ 的符號**不一致**：

- BUIDL $\\gamma \\approx +0.51$、OUSG $\\gamma \\approx +0.12$ → 與股票**相反**，符合資金流驅動
- sUSDS $\\gamma \\approx -0.60$ → 與股票**相同**方向

論文初稿曾寫成「三個代幣的不對稱方向都與股票的槓桿效果相反」——
這是一個**被表格自己推翻的宣稱**。審計時修正為分別陳述，並明說「本樣本不支持三個代幣有共同的
不對稱特徵」。

> **教學重點**：當你要寫「所有 X 都具有性質 P」時，回頭把表格的每一列都對一遍。
> 「大部分」與「全部」在審稿人眼中是完全不同的兩個宣稱，而後者只要有一個反例就整句失效。
"""),

]
