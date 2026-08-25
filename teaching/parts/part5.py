# -*- coding: utf-8 -*-
"""第 5 篇：波動溢出與連結性（單元 14–17），對應 05_spillover.py 與 10_r_tvpvar_frequency.R。"""

from ._helpers import md, code, unit_header

CELLS = [

md("""
---
---

# 第 5 篇　波動溢出與連結性

DCC 回答「兩個資產的相關性如何演變」，但它一次只看一對。
當系統裡有 12 個資產時，我們想問的是**系統層級**的問題：

- 整個系統的關聯程度有多高？（總連結性）
- 誰在傳遞衝擊、誰在接收？（方向性）
- 這些傳遞發生在幾天的時間尺度上？（頻域）

Diebold–Yilmaz 的連結性框架把這些問題全部化約成**預測誤差變異數分解**的運算。
"""),

unit_header("單元 14", "VAR 與廣義預測誤差變異數分解（GFEVD）",
            "向量自迴歸；移動平均表示；廣義（順序無關）變異數分解",
            "05_spillover.py",
            ["Koop, G., Pesaran, M. H., & Potter, S. M. (1996). Impulse response analysis in nonlinear multivariate models. Journal of Econometrics, 74(1), 119–147.",
             "Pesaran, M. H., & Shin, Y. (1998). Generalized impulse response analysis in linear multivariate models. Economics Letters, 58(1), 17–29."]),

md("""
### 步驟一：VAR 與其移動平均表示

$N$ 變量的 VAR($p$)：

$$
x_t = \\sum_{i=1}^{p} \\Phi_i x_{t-i} + \\epsilon_t, \\qquad \\epsilon_t \\sim (0, \\Sigma)
$$

若定態，可反轉成無限階移動平均（Wold 表示）：

$$
x_t = \\sum_{h=0}^{\\infty} A_h \\epsilon_{t-h}
$$

其中 $A_h$ 由遞迴式生成：$A_0 = I_N$，$A_h = \\sum_{i=1}^{\\min(h,p)} \\Phi_i A_{h-i}$。

$A_h$ 的第 $(i,j)$ 元素就是「第 $j$ 個變數的一單位衝擊，$h$ 期後對第 $i$ 個變數的影響」。

### 步驟二：為什麼要用「廣義」分解

傳統的 FEVD 用 Cholesky 分解正交化衝擊。問題是 **Cholesky 分解依賴變數的排列順序** ——
排在前面的變數被假設「同期影響後面的變數，但不受後面影響」。

對本研究來說這是致命的：把 BUIDL 排在 FANG 前面還是後面，會給出不同的溢出結論，
而**沒有任何經濟理論能決定這個順序**。

Koop–Pesaran–Potter / Pesaran–Shin 的**廣義**分解不做正交化，
而是問「若第 $j$ 個變數受到一單位衝擊（其他變數依歷史相關結構同時反應），
對第 $i$ 個變數的 $H$ 期預測誤差變異數貢獻多少」：

$$
\\theta_{ij}(H) = \\frac{\\sigma_{jj}^{-1} \\sum_{h=0}^{H-1}\\left(e_i' A_h \\Sigma e_j\\right)^2}
{\\sum_{h=0}^{H-1} e_i' A_h \\Sigma A_h' e_i}
$$

其中 $e_i$ 是第 $i$ 個位置為 1 的單位向量，$\\sigma_{jj}$ 是 $\\Sigma$ 的第 $j$ 個對角元素。

| 部分 | 意義 |
|---|---|
| 分子 $\\left(e_i' A_h \\Sigma e_j\\right)^2$ | $j$ 的衝擊經 $h$ 期傳到 $i$ 的效果平方 |
| $\\sigma_{jj}^{-1}$ | 把衝擊標準化成一單位 |
| 分母 | $i$ 的 $H$ 期總預測誤差變異數 |

### 步驟三：列標準化

因為不正交化，$\\sum_j \\theta_{ij}(H) \\ne 1$。因此需要**列標準化**：

$$
\\tilde{\\theta}_{ij}(H) = \\frac{\\theta_{ij}(H)}{\\sum_{k=1}^{N}\\theta_{ik}(H)}
$$

標準化後 $\\sum_j \\tilde{\\theta}_{ij} = 1$，可以解讀為「$i$ 的預測誤差變異數中，
有多少比例來自 $j$」。

> **代價**：標準化讓數字可比較、可加總，但也讓「廣義」分解不再有嚴格的變異數分解解釋
> （因為原始的 $\\theta_{ij}$ 加總不為 1 反映的是衝擊間的相關性，標準化把這個資訊抹掉了）。
> 這是文獻的標準做法，但值得知道自己在交換什麼。
"""),

code("""
# ---------------------------------------------------------------------------
# 從零實作 GFEVD（與 05_spillover.py 相同的邏輯，逐步展開）
# ---------------------------------------------------------------------------
from statsmodels.tsa.api import VAR

def gfevd(var_res, H):
    names = var_res.names
    N = len(names)
    Sigma = var_res.sigma_u.values
    A = var_res.ma_rep(maxn=H - 1)          # A[0], ..., A[H-1]
    theta = np.zeros((N, N))
    for i in range(N):
        e_i = np.zeros(N); e_i[i] = 1.0
        # 分母：i 的 H 期總預測誤差變異數
        denom = sum(e_i @ A[h] @ Sigma @ A[h].T @ e_i for h in range(H))
        for j in range(N):
            e_j = np.zeros(N); e_j[j] = 1.0
            num = sum((e_i @ A[h] @ Sigma @ e_j) ** 2 for h in range(H)) / Sigma[j, j]
            theta[i, j] = num / denom if denom > 0 else np.nan
    theta_tilde = theta / theta.sum(axis=1, keepdims=True)   # 列標準化
    return theta, theta_tilde, names

# 用論文的 12 資產系統
SYS = {"FANG": "R_FANG", "SOXX": "R_SOXX", "SMH": "R_SMH", "SHV": "R_SHV", "IEF": "R_IEF",
       "BUIDL": "R_BUIDL_mkt", "OUSG": "R_OUSG_mkt",
       "BTC": "R_BTC", "ETH": "R_ETH", "UNI": "R_UNI", "AAVE": "R_AAVE", "GLD": "R_GLD"}
data = panel[list(SYS.values())].dropna()
data.columns = list(SYS.keys())
print(f"系統樣本：{len(data)} 個交易日 × {data.shape[1]} 個資產")

model = VAR(data)
p_opt = max(1, model.select_order(maxlags=5).aic)
res_var = model.fit(p_opt)
print(f"AIC 選定落後階數 p = {p_opt}")

theta_raw, theta_norm, names = gfevd(res_var, H=10)
print(f"\\n標準化前每列加總（前 5 個）：{theta_raw.sum(axis=1)[:5].round(3)}")
print(f"標準化後每列加總（前 5 個）：{theta_norm.sum(axis=1)[:5].round(3)}")
print("→ 標準化前不為 1，正是廣義分解不正交化的直接後果")
"""),

unit_header("單元 15", "Diebold–Yilmaz 連結性指標體系",
            "總連結性指數；方向性溢出；淨成對方向連結性",
            "05_spillover.py",
            ["Diebold, F. X., & Yilmaz, K. (2012). Better to give than to receive. International Journal of Forecasting, 28(1), 57–66.",
             "Diebold, F. X., & Yilmaz, K. (2014). On the network topology of variance decompositions. Journal of Econometrics, 182(1), 119–134."]),

md("""
### 從分解矩陣到四個指標

有了 $\\tilde{\\theta}_{ij}$，Diebold–Yilmaz 定義四個指標。全部都是這個矩陣的簡單彙總。

**（一）FROM：$i$ 從別人那裡接收的**

$$
\\mathrm{FROM}_i = 100 \\times \\sum_{j \\ne i} \\tilde{\\theta}_{ij}
= 100 \\times (1 - \\tilde{\\theta}_{ii})
$$

**（二）TO：$j$ 傳給別人的**

$$
\\mathrm{TO}_j = 100 \\times \\sum_{i \\ne j} \\tilde{\\theta}_{ij}
$$

注意 FROM 是**列**加總、TO 是**行**加總。TO 沒有 1 的上界（一個資產可以影響很多其他資產）。

**（三）NET：淨溢出**

$$
\\mathrm{NET}_i = \\mathrm{TO}_i - \\mathrm{FROM}_i
$$

- $\\mathrm{NET}_i > 0$：**淨傳遞者**，衝擊的來源
- $\\mathrm{NET}_i < 0$：**淨接收者**，衝擊的承受方

> 本研究的核心實證發現之一：代幣化國債的 NET 接近零（BUIDL $-0.80$、OUSG $+0.16$），
> 屬於**淨接收者一側**。這對金融穩定監理是個令人安心的訊號 ——
> 在目前的規模下，它們不是系統性風險的來源。

**（四）TCI：總連結性指數**

$$
\\mathrm{TCI} = \\frac{100}{N}\\sum_{i \\ne j}\\tilde{\\theta}_{ij}
= \\frac{100}{N}\\left(\\sum_{i,j}\\tilde{\\theta}_{ij} - \\mathrm{tr}(\\tilde{\\Theta})\\right)
$$

值域 $[0, 100]$。$\\mathrm{TCI} = 0$ 代表各資產完全獨立；接近 100 代表系統高度一體化。

**（五）NPDC：淨成對方向連結性**

$$
\\mathrm{NPDC}_{ij} = \\frac{100}{N}\\left(\\tilde{\\theta}_{ji} - \\tilde{\\theta}_{ij}\\right)
$$

這是網路圖的邊權重：$\\mathrm{NPDC}_{ij} > 0$ 代表 $i$ 淨傳遞給 $j$。
"""),

md("""
### 跨方法交叉驗證：本研究的方法論貢獻

本研究刻意用**兩個完全獨立的實作**估計同一個 12 資產系統：

| | Python 靜態估計 | R 的 TVP-VAR |
|---|---|---|
| 語言 | Python | R |
| 套件 | 自行實作 + statsmodels | `ConnectednessApproach` |
| 參數 | 靜態，AIC 選階 | 時變，Kalman 濾波 |
| 總連結性 | 43.3% | 45.4% |

兩者相差不到 2 個百分點，且 12 個變數中有 **10 個**的 NET 符號一致。

> **必須誠實報告的部分**：有 **2 個不一致** —— IEF（$-4.31$ vs $+0.09$）與
> OUSG（$+0.16$ vs $-2.06$）。論文初稿寫的是「**每一個**變數的淨溢出符號都一致」，
> 審計時發現這句話被自己的表格推翻。
>
> 修正後的寫法是：兩者在 12 個變數中有 10 個一致，不一致的兩個都**接近零**，
> 因此分歧是「一個近乎中性的資產微幅偏向哪一側」，而非「它在系統中的角色」。
> 對 OUSG 這個焦點資產，安全的解讀是**在兩個估計量下都近乎中性**。
>
> 這個修正讓宣稱變弱，但也讓它站得住腳。交叉驗證的價值恰恰在於它會**抓到不一致** ——
> 如果只是拿來確認「我們都對」，那就沒有做的必要。
"""),

code("""
# ---------------------------------------------------------------------------
# 從分解矩陣計算四個指標
# ---------------------------------------------------------------------------
def connectedness_table(theta_tilde, names):
    N = len(names)
    diag = np.diag(theta_tilde)
    TO = 100 * (theta_tilde.sum(axis=0) - diag)      # 行加總
    FROM = 100 * (theta_tilde.sum(axis=1) - diag)    # 列加總
    NET = TO - FROM
    TCI = 100 * (theta_tilde.sum() - np.trace(theta_tilde)) / N
    npdc = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            npdc.loc[ni, nj] = 100 * (theta_tilde[j, i] - theta_tilde[i, j]) / N
    return pd.DataFrame({"TO": TO, "FROM": FROM, "NET": NET}, index=names), TCI, npdc

tbl, tci, npdc = connectedness_table(theta_norm, names)
display(tbl.sort_values("NET", ascending=False).round(2))
print(f"總連結性指數 TCI = {tci:.2f}%")
print()
print("淨傳遞者（NET > 0）：", ", ".join(tbl[tbl.NET > 0].sort_values('NET', ascending=False).index))
print("淨接收者（NET < 0）：", ", ".join(tbl[tbl.NET < 0].sort_values('NET').index))
"""),

code("""
# ---------------------------------------------------------------------------
# 兩個獨立實作的交叉驗證：逐變數比對符號
# ---------------------------------------------------------------------------
tvp = pd.read_csv(PROC / "r_tvpvar_static_table.csv").set_index("variable")
sta = pd.read_csv(PROC / "spillover_static_C.csv", index_col=0)

cmp = pd.DataFrame({
    "TVP-VAR (R)": tvp["NET"],
    "靜態 GFEVD (Python)": sta["NET"],
})
cmp["符號一致"] = np.sign(cmp["TVP-VAR (R)"]) == np.sign(cmp["靜態 GFEVD (Python)"])
display(cmp.round(3))

agree = int(cmp["符號一致"].sum())
print(f"符號一致：{agree} / {len(cmp)}")
print()
print("不一致的變數：")
for v, row in cmp[~cmp["符號一致"]].iterrows():
    print(f"    {v:6} TVP-VAR = {row['TVP-VAR (R)']:+7.2f}, 靜態 = {row['靜態 GFEVD (Python)']:+7.2f}"
          f"   （兩者皆接近零，分歧在於偏向哪一側）")
"""),

unit_header("單元 16", "TVP-VAR：Kalman 濾波與遺忘因子",
            "時變參數 VAR；Kalman 濾波；遺忘因子（forgetting factor）",
            "10_r_tvpvar_frequency.R",
            ["Antonakakis, N., Chatziantoniou, I., & Gabauer, D. (2020). Refined measures of dynamic connectedness based on time-varying parameter vector autoregressions. Journal of Risk and Financial Management, 13(4), 84."]),

md("""
### 原理：滾動窗口的三個缺陷

要得到「隨時間變動的連結性」，最直覺的做法是**滾動窗口**：每 100 天估一次 VAR。
本研究的 Python 端就是這樣做的（`05_spillover.py`）。但這個做法有三個問題：

1. **窗口長度是任意的**：100 天？250 天？結果會不同，而沒有原則能決定
2. **觀測值權重是階梯函數**：窗口內的第 1 天與第 100 天權重相同，窗口外權重驟降為 0
3. **樣本浪費**：前 100 天完全沒有估計值

TVP-VAR 用狀態空間模型解決全部三個。

### 狀態空間表示

**觀測方程**（VAR 本身，但係數帶時間下標）：

$$
x_t = Z_t \\beta_t + \\epsilon_t, \\qquad \\epsilon_t \\sim N(0, S_t)
$$

**狀態方程**（係數自己隨機漫步）：

$$
\\beta_t = \\beta_{t-1} + \\nu_t, \\qquad \\nu_t \\sim N(0, R_t)
$$

其中 $Z_t$ 由落後的 $x$ 構成，$\\beta_t$ 是拉平的 VAR 係數向量。

### Kalman 濾波遞迴

**預測步**

$$
\\beta_{t|t-1} = \\beta_{t-1|t-1}, \\qquad
\\Sigma_{t|t-1} = \\Sigma_{t-1|t-1} + R_t
$$

**更新步**

$$
K_t = \\Sigma_{t|t-1} Z_t' \\left(Z_t \\Sigma_{t|t-1} Z_t' + S_t\\right)^{-1}
$$

$$
\\beta_{t|t} = \\beta_{t|t-1} + K_t\\left(x_t - Z_t \\beta_{t|t-1}\\right)
$$

$$
\\Sigma_{t|t} = \\Sigma_{t|t-1} - K_t Z_t \\Sigma_{t|t-1}
$$

$K_t$ 是 **Kalman 增益**：它決定「新觀測的預測誤差要修正多少係數」。
若觀測噪音 $S_t$ 大，$K_t$ 小，係數變動保守；反之則靈敏。

### 遺忘因子：本方法的關鍵

直接估計 $R_t$（狀態噪音共變異數）需要大量參數。Antonakakis 等人採用
Koop–Korobilis 的**遺忘因子**技巧，用一個純量 $\\kappa_1$ 取代整個矩陣：

$$
R_t = \\left(\\frac{1}{\\kappa_1} - 1\\right)\\Sigma_{t-1|t-1}
\\quad\\Longrightarrow\\quad
\\Sigma_{t|t-1} = \\frac{1}{\\kappa_1}\\Sigma_{t-1|t-1}
$$

誤差共變異數則用 EWMA 更新：

$$
S_t = \\kappa_2 S_{t-1} + (1 - \\kappa_2)\\hat{\\epsilon}_t \\hat{\\epsilon}_t'
$$

### 遺忘因子的直觀：等效窗口長度

$\\kappa$ 越接近 1，過去的資訊衰減越慢。其**等效記憶長度**約為

$$
\\text{有效觀測數} \\approx \\frac{1}{1 - \\kappa}
$$

| $\\kappa$ | 有效窗口 | 特性 |
|---|---|---|
| 0.99 | ~100 天 | 平滑，適合偵測結構性趨勢 |
| 0.96 | ~25 天 | 靈敏，適合偵測短期震盪 |
| 0.94 | ~17 天 | 非常靈敏，噪音較多 |

本研究主設定為 $\\kappa_1 = \\kappa_2 = 0.99$，並用 $(0.96, 0.94)$ 做**敏感度檢查**。

> **敏感度檢查的結果本身就是發現**：
> TCI 從 45.4%（0.99/0.99）上升到 54.6%（0.96/0.94）。
> 這不是「哪個對」的問題 —— 更靈敏的設定會捕捉到更多短期同步，
> 因此測到更高的連結性。報告兩者，並說明差異的來源，比只報一個數字誠實得多。
"""),

code("""
# ---------------------------------------------------------------------------
# 遺忘因子敏感度：TCI 對 kappa 的反應
# ---------------------------------------------------------------------------
kap = pd.read_csv(PROC / "r_tvpvar_kappa_sensitivity.csv")
display(kap.round(3))

eff = lambda k: 1 / (1 - k)
print(f"kappa = 0.99 → 有效窗口 ≈ {eff(0.99):.0f} 個交易日")
print(f"kappa = 0.96 → 有效窗口 ≈ {eff(0.96):.0f} 個交易日")
print(f"kappa = 0.94 → 有效窗口 ≈ {eff(0.94):.0f} 個交易日")
print()
print(f"TCI 差異 = {kap['mean_TCI'].iloc[1] - kap['mean_TCI'].iloc[0]:.1f} 個百分點")
print("解讀：更短的有效窗口捕捉更多短期同步 → 更高的測得連結性。")
print("      兩個數字都是對的，只是在回答略微不同的問題。")
"""),

code("""
# ---------------------------------------------------------------------------
# TVP-VAR 的時變 TCI 路徑
# ---------------------------------------------------------------------------
tci_ts = pd.read_csv(PROC / "r_tvpvar_tci_series.csv", parse_dates=["date"]).set_index("date")

fig, ax = plt.subplots(figsize=(9.5, 3.2))
ax.plot(tci_ts.index, tci_ts["TCI"], lw=1.1, color="#1b4965")
ax.axhline(tci_ts["TCI"].mean(), color="#bc4749", ls="--", lw=1.2,
           label=f"平均 = {tci_ts['TCI'].mean():.1f}%")
ax.set_title("TVP-VAR 總連結性指數：系統一體化程度隨時間變動")
ax.set_ylabel("TCI (%)")
ax.legend()
fig.tight_layout()
plt.show()

print(f"TCI 區間：[{tci_ts['TCI'].min():.1f}%, {tci_ts['TCI'].max():.1f}%]")
print("滾動窗口做不到的事：每一天都有估計值，且權重平滑衰減而非階梯跳動。")
"""),

unit_header("單元 17", "Baruník–Křehlík 頻域分解",
            "頻譜表示；廣義因果頻譜；頻帶連結性",
            "10_r_tvpvar_frequency.R",
            ["Baruník, J., & Křehlík, T. (2018). Measuring the frequency dynamics of financial connectedness and systemic risk. Journal of Financial Econometrics, 16(2), 271–296."]),

md("""
### 原理：時域指標混淆了兩種不同的溢出

單元 15 的 TCI 是一個數字，但它把兩種本質不同的現象加在一起：

- **短期溢出**：流動性衝擊、造市商調整部位、當日套利 → 幾天內完成
- **長期溢出**：基本面重新定價、資產配置的結構性移轉 → 數月才顯現

一個「TCI = 45%」的系統，可能是「短期高度同步但長期無關」，
也可能是「長期深度整合但日常獨立」。這兩者對金融穩定的含意完全不同。

Baruník–Křehlík 把變異數分解搬到**頻域**，讓我們能分開來看。

### 頻譜表示

VAR 的移動平均係數 $A_h$ 的傅立葉轉換：

$$
\\Psi(e^{-i\\omega}) = \\sum_{h=0}^{\\infty} e^{-i\\omega h} A_h
$$

**廣義因果頻譜**（單元 14 的 GFEVD 在頻率 $\\omega$ 上的版本）：

$$
\\left(f(\\omega)\\right)_{ij} =
\\frac{\\sigma_{jj}^{-1}\\left|\\left(\\Psi(e^{-i\\omega})\\Sigma\\right)_{ij}\\right|^2}
{\\left(\\Psi(e^{-i\\omega})\\Sigma\\Psi'(e^{+i\\omega})\\right)_{ii}}
$$

**頻帶內加總**：對頻帶 $d = (\\underline{\\omega}, \\overline{\\omega})$，

$$
\\tilde{\\theta}_{ij}(d) = \\frac{1}{2\\pi}\\int_{d} \\Gamma_i(\\omega)\\left(f(\\omega)\\right)_{ij}  d\\omega
$$

其中 $\\Gamma_i(\\omega)$ 是把頻譜密度標準化的權重函數。

### 關鍵性質：頻帶可加

$$
\\tilde{\\theta}_{ij} = \\sum_{d} \\tilde{\\theta}_{ij}(d)
$$

各頻帶的連結性**加總回時域的連結性**。這讓「45.4 點的總連結性中，有多少來自短期」
成為一個有明確答案的問題。

### 頻帶的設定

程式碼把 $[0, \\pi]$ 分割成三段：

```r
partition = c(pi, pi/5, pi/20, 0)
```

轉換成週期：頻率 $\\omega$ 對應週期 $2\\pi/\\omega$ 天。

| 分割 | 頻率區間 | 週期 | 標籤 |
|---|---|---|---|
| $(\\pi/5, \\pi)$ | 高頻 | 2–10 天 | `1-5` |
| $(\\pi/20, \\pi/5)$ | 中頻 | 10–40 天 | `5-20` |
| $(0, \\pi/20)$ | 低頻 | 40 天以上 | `20-Inf` |

### 本研究的結果與一個被修正的錯誤

實際的頻帶分解為：

$$
\\underbrace{45.4}_{\\text{總}} = \\underbrace{37.6}_{\\text{1-5 天}} + \\underbrace{5.8}_{\\text{5-20 天}} + \\underbrace{2.0}_{\\text{20 天以上}}
$$

即 **83% 的連結性在一週內完成**。這個發現對監理有直接含意：
代幣化國債若有溢出，是**幾天的事，不是幾個月的事**。

> **論文初稿寫的是「約 41 點來自 1-5 天、3 點來自 5-20 天、不到 1 點來自 20 天以上」**。
> 三個數字全錯。定性結論（短期主導）成立，但數字對不上表格。
>
> 這類錯誤特別危險，因為它**通過所有的合理性檢查** —— 41 + 3 + 1 ≈ 45，
> 加總大致對得起來，方向也對，只有真的去比對來源檔案才會發現。
> 教訓是：正文裡的每一個數字都必須有一條可追溯到輸出檔案的路徑（見單元 26）。
"""),

code("""
# ---------------------------------------------------------------------------
# 頻帶分解：驗證可加性與各頻帶佔比
# ---------------------------------------------------------------------------
freq_tci = pd.read_csv(PROC / "r_frequency_tci_series.csv").groupby("band")["TCI"].mean()
order = ["1-5", "5-20", "20-Inf", "Total"]
freq_tci = freq_tci.reindex([b for b in order if b in freq_tci.index])

bands = freq_tci.drop("Total")
tbl = pd.DataFrame({
    "TCI": bands,
    "佔總連結性": bands / freq_tci["Total"],
})
tbl.index = ["1–5 天（短期）", "5–20 天（中期）", "20 天以上（長期）"]
display(tbl.round(3))

print(f"三個頻帶加總 = {bands.sum():.2f}")
print(f"時域總連結性 = {freq_tci['Total']:.2f}")
print(f"可加性檢查   = {abs(bands.sum() - freq_tci['Total']):.4f}  （應接近 0）")
print()
print(f"→ {bands.iloc[0]/freq_tci['Total']:.0%} 的連結性在一週內完成")
"""),

code("""
# ---------------------------------------------------------------------------
# 逐資產的頻帶淨溢出：長期溢出幾乎消失
# ---------------------------------------------------------------------------
fn = pd.read_csv(PROC / "r_frequency_static_table.csv")
pivot = fn.pivot(index="variable", columns="band", values="NET")
pivot = pivot[[c for c in ["1-5", "5-20", "20-Inf", "Total"] if c in pivot.columns]]
display(pivot.sort_values("Total", ascending=False).round(2))

fig, ax = plt.subplots(figsize=(9.5, 3.6))
sub = pivot.drop(columns="Total").sort_values("1-5")
x = np.arange(len(sub))
w = 0.27
for k, (col, c) in enumerate(zip(sub.columns, ["#1b4965", "#f4a261", "#2a9d8f"])):
    ax.bar(x + (k-1)*w, sub[col], width=w, label=f"{col} 天", color=c)
ax.set_xticks(x); ax.set_xticklabels(sub.index, rotation=45, ha="right")
ax.axhline(0, color="grey", lw=0.8)
ax.set_ylabel("淨溢出")
ax.set_title("頻帶淨溢出：短期主導，長期幾乎歸零")
ax.legend()
fig.tight_layout()
plt.show()

print("BUIDL 的淨溢出從總計 -0.80 縮到 20 天以上頻帶的 "
      f"{pivot.loc['BUIDL','20-Inf']:+.2f} —— 本已微小的溢出在長期尺度上更小。")
"""),

]
