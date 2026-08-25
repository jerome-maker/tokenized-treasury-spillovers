# -*- coding: utf-8 -*-
"""第 4 篇：動態條件相關（單元 11–13），對應 04_garch_dcc.py 階段二。"""

from ._helpers import md, code, unit_header

CELLS = [

md("""
---
---

# 第 4 篇　動態條件相關

這一篇是整個研究的核心。研究問題是「代幣化國債與風險資產的相關性會不會在關鍵時刻飆升」，
而 DCC 正是把「相關性隨時間變動」形式化的工具。

三個單元的邏輯鏈：

1. **單元 11**：DCC 如何讓相關矩陣隨時間演化，以及兩階段估計為何可行
2. **單元 12**：ADCC 如何檢定「共同的負向衝擊是否比正向衝擊更能提高相關」
3. **單元 13**：如何檢定「相關性真的在變動」，以及這個檢定的**陷阱**

單元 13 的陷阱部分請務必讀完 —— 本研究 34 組配對中有 18 組的最佳化落在參數邊界上，
若不處理，會得出完全錯誤的結論。
"""),

unit_header("單元 11", "DCC：兩階段 QMLE 與 $Q_t$ 遞迴",
            "動態條件相關模型；兩階段準最大概似估計",
            "04_garch_dcc.py（階段二）",
            ["Engle, R. F. (2002). Dynamic conditional correlation: A simple class of multivariate GARCH models. Journal of Business and Economic Statistics, 20(3), 339–350."]),

md("""
### 原理：從常數相關到動態相關

多變量 GARCH 的核心是把條件共變異數矩陣分解：

$$
H_t = D_t R_t D_t
$$

其中 $D_t = \\mathrm{diag}(\\sigma_{1,t}, \\ldots, \\sigma_{N,t})$ 是條件標準差的對角矩陣
（由第 3 篇的單變量模型提供），$R_t$ 是**條件相關矩陣**。

- Bollerslev 的 **CCC** 模型假設 $R_t = R$ 為常數
- Engle 的 **DCC** 讓 $R_t$ 隨時間演化

這個分解的價值在於：**$D_t$ 與 $R_t$ 可以分開估計**，這正是兩階段法可行的原因。

### 公式：$Q_t$ 遞迴

DCC 不直接對 $R_t$ 建模（因為要保證它是合法的相關矩陣：正定、對角為 1，很難直接約束），
而是先建模一個**輔助矩陣** $Q_t$，再標準化成相關矩陣。

令 $u_t = D_t^{-1}\\varepsilon_t$ 為標準化殘差（來自第 3 篇）：

$$
Q_t = (1 - a - b)\\bar{Q} + a  u_{t-1} u_{t-1}' + b  Q_{t-1}
$$

$$
R_t = \\mathrm{diag}(Q_t)^{-1/2}  Q_t  \\mathrm{diag}(Q_t)^{-1/2}
$$

其中 $\\bar{Q} = \\mathbb{E}[u_t u_t']$ 是標準化殘差的無條件共變異數（用樣本估計）。

### 為什麼這樣設計

| 設計 | 用意 |
|---|---|
| 對 $Q_t$ 而非 $R_t$ 遞迴 | $Q_t$ 只需正定，不需對角為 1；正定性由「正定矩陣的正係數凸組合仍正定」保證 |
| $(1-a-b)\\bar{Q}$ 這一項 | 讓 $\\mathbb{E}[Q_t] = \\bar{Q}$，即**均值回歸到無條件相關** |
| 最後的標準化 | 把 $Q_t$ 轉成合法的相關矩陣，對角自動為 1 |

參數意義與 GARCH 平行：

- $a$ = **新聞參數**：昨日殘差的外積對今日相關的影響
- $b$ = **持續性參數**：相關結構的記憶長度
- 定態條件：$a + b < 1$，且 $a, b \\ge 0$

### 兩階段估計的理論基礎

完整的對數概似可以分解為

$$
\\ell(\\theta) = \\underbrace{\\ell_V(\\theta_1)}_{\\text{變異數部分}} + \\underbrace{\\ell_C(\\theta_1, \\theta_2)}_{\\text{相關部分}}
$$

**階段一**：對每個序列各自最大化其單變量概似 → 得 $\\hat{\\theta}_1$（第 3 篇做的事）

**階段二**：固定 $\\hat{\\theta}_1$，只對 $\\theta_2 = (a, b)$ 最大化相關部分

$$
\\ell_C(a, b) = -\\frac{1}{2}\\sum_{t=1}^{T}
\\left[ \\ln|R_t| + u_t' R_t^{-1} u_t - u_t' u_t \\right]
$$

最後那個 $-u_t' u_t$ 項是**與參數無關的常數**，加它是為了讓 $\\ell_C$ 在
$R_t = I$ 時等於零，方便數值比較。它不影響最佳化的解。

> **代價**：兩階段估計有一致性但**沒有效率**，且標準誤需要修正
> （因為階段二把 $\\hat{\\theta}_1$ 當成已知）。換來的是計算可行性 ——
> $N$ 個序列的聯合估計參數量是 $O(N^2)$，兩階段則是 $O(N) + O(1)$。
"""),

code("""
# ---------------------------------------------------------------------------
# 從零實作 DCC：Q_t 遞迴與相關部分的概似
# ---------------------------------------------------------------------------
def build_Q_path(U, a, b, g=0.0, asymmetric=False):
    \"\"\"回傳 T x N x N 的 Q_t 路徑。U 為 T x N 的標準化殘差矩陣。\"\"\"
    T, N = U.shape
    Qbar = np.cov(U.T, bias=True)                 # Q-bar：無條件共變異數
    if asymmetric:
        Neg = np.where(U < 0, U, 0.0)             # n_t = min(u_t, 0)
        Nbar = (Neg.T @ Neg) / T
    Qt = Qbar.copy()
    Q_path = np.zeros((T, N, N))
    Q_path[0] = Qt
    for t in range(1, T):
        u_prev = U[t-1, :].reshape(-1, 1)
        if asymmetric:
            n_prev = np.where(u_prev < 0, u_prev, 0.0)
            Qt = (Qbar - a*Qbar - b*Qbar - g*Nbar) \\
                 + a * (u_prev @ u_prev.T) + g * (n_prev @ n_prev.T) + b * Qt
        else:
            Qt = (1 - a - b) * Qbar + a * (u_prev @ u_prev.T) + b * Qt
        Q_path[t] = Qt
    return Q_path


def corr_from_Q(Q_path):
    \"\"\"把 Q_t 標準化成相關矩陣，回傳 rho_12 序列（雙變量情形）。\"\"\"
    rhos = []
    for Qt in Q_path:
        d = np.sqrt(np.diag(Qt))
        Rt = Qt / np.outer(d, d)
        rhos.append(Rt[0, 1])
    return np.array(rhos)


def neg_corr_loglik(params, U, asymmetric=False):
    a, b = params[0], params[1]
    g = params[2] if asymmetric else 0.0
    if a < 0 or b < 0 or g < 0 or (a + b + g) >= 0.999:
        return 1e10
    Q_path = build_Q_path(U, a, b, g, asymmetric)
    ll = 0.0
    for t in range(U.shape[0]):
        Qt = Q_path[t]
        d = np.sqrt(np.diag(Qt))
        if np.any(d <= 0):
            return 1e10
        Rt = Qt / np.outer(d, d)
        sign, logdet = np.linalg.slogdet(Rt)
        if sign <= 0:
            return 1e10
        u_t = U[t, :]
        ll += logdet + u_t @ np.linalg.inv(Rt) @ u_t - (u_t @ u_t)
    return 0.5 * ll

print("DCC 核心函數已定義：build_Q_path / corr_from_Q / neg_corr_loglik")
print("注意 build_Q_path 的 asymmetric 分支即為單元 12 的 ADCC。")
"""),

code("""
# ---------------------------------------------------------------------------
# 用真實資料跑一次：IEF vs GLD（本研究中訊號最強的一組配對）
# ---------------------------------------------------------------------------
from arch import arch_model

def std_resid(series):
    f = arch_model(series, mean="AR", lags=1, vol="GARCH", p=1, q=1,
                   dist="skewt", rescale=False).fit(disp="off")
    return f.resid / f.conditional_volatility

z_ief = std_resid(panel["R_IEF"].dropna())
z_gld = std_resid(panel["R_GLD"].dropna())
Z = pd.concat([z_ief, z_gld], axis=1).dropna()
Z.columns = ["IEF", "GLD"]
U = Z.values
print(f"重疊觀測數：{len(U)}")

res_dcc = minimize(neg_corr_loglik, [0.03, 0.90], args=(U, False),
                   method="SLSQP", bounds=[(1e-6, 0.3), (1e-6, 0.998)],
                   constraints=[{"type": "ineq", "fun": lambda x: 0.998 - (x[0] + x[1])}])
a_hat, b_hat = res_dcc.x
rho_path = corr_from_Q(build_Q_path(U, a_hat, b_hat))

print(f"a = {a_hat:.4f}（新聞）, b = {b_hat:.4f}（持續性）, a+b = {a_hat+b_hat:.4f}")
print(f"rho 平均 = {rho_path.mean():.4f}, 標準差 = {rho_path.std():.4f}, "
      f"區間 = [{rho_path.min():.3f}, {rho_path.max():.3f}]")
print(f"（管線報告值：mean rho = 0.3160，此處因均值方程與樣本對齊細節略有差異）")
"""),

code("""
# ---------------------------------------------------------------------------
# 視覺化：動態相關路徑 vs 常數相關
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 3.4))
ax.plot(Z.index, rho_path, lw=1.1, color="#1b4965", label=r"DCC $\\rho_t$")
ax.axhline(np.corrcoef(U[:, 0], U[:, 1])[0, 1], color="#bc4749", ls="--", lw=1.4,
           label="CCC 常數相關")
ax.axhline(0, color="grey", lw=0.7)
ax.set_title("IEF vs GLD：動態相關在 0.1 到 0.6 之間擺盪，常數相關完全掩蓋了這個結構")
ax.set_ylabel(r"$\\rho_t$")
ax.legend()
fig.tight_layout()
plt.show()
"""),

unit_header("單元 12", "ADCC：非對稱衝擊下的相關動態",
            "非對稱動態條件相關；共同負向衝擊效果",
            "04_garch_dcc.py（階段二）",
            ["Cappiello, L., Engle, R. F., & Sheppard, K. (2006). Asymmetric dynamics in the correlations of global equity and bond returns. Journal of Financial Econometrics, 4(4), 537–572."]),

md("""
### 原理：DCC 遺漏的非對稱

單元 8 的 GJR 處理了**單一資產**的波動非對稱。ADCC 把同樣的想法推廣到**相關性**：

> 當兩個資產**同時**下跌時，它們的相關性是否比同時上漲時更高？

這個問題對本研究至關重要。假說 B（鏈上流動性通道主導）預測的正是：
代幣與風險資產的相關性會**在共同下跌時飆升**——也就是避險最需要的時候失效。
若 ADCC 的非對稱參數顯著為正，就是支持假說 B 的直接證據。

### 公式

定義**負向指示殘差**（逐元素取負部）：

$$
n_t = \\min(u_t, 0) \\quad \\text{（逐元素）}
$$

則 ADCC 的遞迴為

$$
Q_t = \\left(\\bar{Q} - a\\bar{Q} - b\\bar{Q} - g\\bar{N}\\right)
+ a  u_{t-1}u_{t-1}'
+ g  n_{t-1}n_{t-1}'
+ b  Q_{t-1}
$$

其中 $\\bar{N} = \\mathbb{E}[n_t n_t']$。

### 關鍵：$n_{t-1}n_{t-1}'$ 這一項只在「兩者同時為負」時才貢獻非零的非對角元素

這是理解 ADCC 的核心。考慮雙變量情形，$n_{t-1} = (n_1, n_2)'$：

外積 $n_{t-1}n_{t-1}'$ 的四個元素為：

| | 第 1 欄 | 第 2 欄 |
|---|---|---|
| **第 1 列** | $n_1^2$ | $n_1 n_2$ |
| **第 2 列** | $n_1 n_2$ | $n_2^2$ |

非對角元素 $n_1 n_2$ 只有在 $u_1 < 0$ **且** $u_2 < 0$ 時才不為零。
因此 $g > 0$ 代表：**共同的壞消息**特別能推高相關性。

| 情境 | $n_1$ | $n_2$ | $n_1 n_2$ | 對相關的貢獻 |
|---|---|---|---|---|
| 同時下跌 | $<0$ | $<0$ | $>0$ | **推高相關** |
| 一漲一跌 | $0$ 或 $<0$ | $<0$ 或 $0$ | $0$ | 無 |
| 同時上漲 | $0$ | $0$ | $0$ | 無 |

### 正定性條件

嚴格的條件是

$$
a + b + \\delta g < 1, \\qquad \\delta = \\lambda_{\\max}\\left(\\bar{Q}^{-1/2}\\bar{N}\\bar{Q}^{-1/2}\\right)
$$

本研究的實作用了一個**較簡單的充分條件**近似：

$$
a + b + g < 1
$$

程式碼註解明確記錄了這個簡化：

```python
# The ADCC positive-definiteness condition a+b+delta*g<1 (delta = max eigenvalue
# of Nbar) is approximated by the simpler sufficient-in-practice constraint
# a+b+g<1 (documented in the supplementary appendix).
```

> **為什麼這樣可以接受**：因為 $\\delta \\le 1$ 在實務上幾乎總是成立
> （$\\bar{N}$ 只累積負部，其特徵值不會超過 $\\bar{Q}$ 的），所以 $a+b+g<1$ 蘊含原條件。
> 這是**更嚴格**的約束，代價是可能排除少數合法的參數組合，但不會產生非法的 $Q_t$。
>
> **教學重點**：做這種簡化本身沒有問題 —— 有問題的是**不揭露**。
> 論文與程式碼都必須寫清楚哪裡用了近似、為什麼可以接受、代價是什麼。
"""),

code("""
# ---------------------------------------------------------------------------
# ADCC vs DCC：同一組配對的兩條路徑
# ---------------------------------------------------------------------------
res_adcc = minimize(neg_corr_loglik, [0.03, 0.90, 0.02], args=(U, True),
                    method="SLSQP",
                    bounds=[(1e-6, 0.3), (1e-6, 0.998), (1e-6, 0.3)],
                    constraints=[{"type": "ineq",
                                  "fun": lambda x: 0.998 - (x[0] + x[1] + x[2])}])
a_A, b_A, g_A = res_adcc.x
rho_adcc = corr_from_Q(build_Q_path(U, a_A, b_A, g_A, asymmetric=True))

print(f"DCC  : a = {a_hat:.4f}, b = {b_hat:.4f}")
print(f"ADCC : a = {a_A:.4f}, b = {b_A:.4f}, g = {g_A:.4f}")
print()
print(f"對數概似 DCC  = {-res_dcc.fun:10.4f}")
print(f"對數概似 ADCC = {-res_adcc.fun:10.4f}")
print(f"改善        = {-res_adcc.fun + res_dcc.fun:.4f}  （g 多了 1 個參數）")

# 同時為負的日子佔比
both_neg = ((U[:, 0] < 0) & (U[:, 1] < 0)).mean()
print(f"\\n兩者同時為負的日子佔 {both_neg:.1%} —— 這些日子才是 g 起作用的地方")
"""),

code("""
# ---------------------------------------------------------------------------
# 展示 g 的作用：只在共同負向衝擊日推高相關
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9.5, 5), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})

axes[0].plot(Z.index, rho_path, lw=1.0, color="#1b4965", label=r"DCC $\\rho_t$")
axes[0].plot(Z.index, rho_adcc, lw=1.0, color="#bc4749", label=r"ADCC $\\rho_t$")
axes[0].set_ylabel(r"$\\rho_t$")
axes[0].legend()
axes[0].set_title("DCC vs ADCC 相關路徑")

both_neg_mask = (U[:, 0] < 0) & (U[:, 1] < 0)
axes[1].fill_between(Z.index, 0, both_neg_mask.astype(int),
                     step="mid", color="#6d6875", alpha=0.6)
axes[1].set_ylabel("同時為負")
axes[1].set_yticks([0, 1])
fig.tight_layout()
plt.show()

diff = rho_adcc - rho_path
print(f"兩條路徑的平均差異     ：{diff.mean():+.5f}")
print(f"共同負向衝擊日的平均差異：{diff[both_neg_mask].mean():+.5f}")
print(f"其他日子的平均差異     ：{diff[~both_neg_mask].mean():+.5f}")
"""),

unit_header("單元 13", "CCC 對照組與概似比檢定",
            "常數條件相關；概似比檢定；邊界解與檢定失效",
            "04_garch_dcc.py（階段二）",
            ["Bollerslev, T. (1990)", "Engle, R. F., & Sheppard, K. (2001). Theoretical and empirical properties of dynamic conditional correlation multivariate GARCH. NBER Working Paper 8554."]),

md("""
### 檢定的邏輯

估出一條看起來會動的 $\\rho_t$ 路徑，**不代表相關性真的在動** ——
那條路徑也可能只是雜訊。需要一個正式的檢定。

**虛無假設**：相關為常數，即 $H_0: a = b = 0$。此時 $Q_t = \\bar{Q}$ 對所有 $t$ 成立，
$R_t$ 退化成常數矩陣，模型即 Bollerslev (1990) 的 **CCC**。

**概似比統計量**：

$$
\\mathrm{LR} = 2\\left(\\ell_{\\mathrm{DCC}} - \\ell_{\\mathrm{CCC}}\\right)
\\xrightarrow{d}  \\chi^2(2)
$$

自由度 2，因為限制了兩個參數。

程式碼直接利用了「$a=b=0$ 即 CCC」這個性質，不必另外寫一個模型：

```python
loglik_ccc = -neg_corr_loglik([0.0, 0.0], U, False)
lr_stat = max(0.0, 2 * (loglik_dcc - loglik_ccc))
lr_pvalue = 1 - chi2.cdf(lr_stat, df=2)
```

`max(0.0, ...)` 是數值保護：理論上 DCC 的概似不可能低於 CCC（後者是前者的受限版本），
但最佳化器若未完全收斂可能給出微小的負值。

### 陷阱一：邊界上的虛無假設

這是本單元最重要的內容。

$H_0: a = b = 0$ 落在參數空間 $\\{a \\ge 0, b \\ge 0\\}$ 的**邊界**上。
標準的 $\\chi^2$ 漸近理論要求虛無假設在參數空間的**內點**，這裡並不滿足。

正確的漸近分配是 $\\chi^2$ 分配的**混合**（Andrews 2001 型的結果），
其臨界值比標準 $\\chi^2(2)$ **保守**。因此：

> 用 $\\chi^2(2)$ 算出的 $p$ 值**偏小**，會**過度拒絕**虛無假設。

本研究的處理方式是誠實揭露：程式碼註解寫明這是
「Engle–Sheppard-style test ... see supplementary appendix for the exact-score-test caveat」。
而且因為本研究的實證結果是**幾乎都不拒絕**，這個偏誤的方向反而讓結論更保守 ——
真正的臨界值更高，會拒絕得更少。

### 陷阱二：邊界解讓檢定完全失去意義

比陷阱一更嚴重的問題。當最佳化器把 $a$ 或 $b$ 推到邊界（通常是 $a \\to 0$），會發生什麼？

$$
a \\to 0,   b \\to 0  \\Longrightarrow  Q_t \\equiv \\bar{Q}  \\Longrightarrow  \\rho_t \\equiv \\text{常數}
$$

此時 DCC 與 CCC 的概似**完全相同**，$\\mathrm{LR} = 0$，$p = 1$。

**這個 $p = 1$ 不代表「相關性確實是常數」，只代表「最佳化器走到了角落」。**

兩者的差別是決定性的：

| 情境 | $p$ 值 | 正確解讀 |
|---|---|---|
| 內點解，$a, b$ 有意義的正值，LR 不顯著 | 大 | 有證據支持相關為常數 |
| **邊界解，$a = b = 0$** | $= 1$ | **無資訊**，不能當作支持常數的證據 |

### 本研究的實際狀況

34 組配對中：
- **18 組**至少有一個參數落在邊界
- **8 組** $a = b = 0$（相關路徑在建構上就是常數）
- **11 組**的 $\\rho_t$ 標準差低於 $0.001$（數值噪音等級）

因此論文的寫法從「無法拒絕常數相關」改為「**相關性小且未顯示清楚的時變性**」，
並在表 5、表 11、表 12 中逐列標記退化配對。這個措辭差異不是修辭 ——
「接受虛無假設」在統計上本來就是不合法的推論，在有邊界解時更是完全站不住腳。
"""),

code("""
# ---------------------------------------------------------------------------
# LR 檢定結果全覽：拒絕的是哪些配對？
# ---------------------------------------------------------------------------
summ = pd.read_csv(PROC / "dcc_adcc_summary.csv", index_col=0)
show = summ[["n_obs", "a_DCC", "b_DCC", "g_ADCC", "mean_rho_DCC", "sd_rho_DCC",
             "LR_stat_vs_CCC", "LR_pvalue_vs_CCC"]].round(4)
display(show)

rej = summ[summ["LR_pvalue_vs_CCC"] < 0.05]
print(f"在 5% 水準下拒絕常數相關的配對：{len(rej)} / {len(summ)}")
for pair, row in rej.iterrows():
    print(f"    {pair:14} LR = {row['LR_stat_vs_CCC']:7.2f}, p = {row['LR_pvalue_vs_CCC']:.4f}, "
          f"mean rho = {row['mean_rho_DCC']:+.4f}")
print()
print("→ 全部三組都屬於傳統 ETF 對照組，沒有任何一組代幣配對拒絕。")
"""),

code("""
# ---------------------------------------------------------------------------
# 邊界解診斷：把「無資訊的 p=1」與「真正不顯著」分開
# ---------------------------------------------------------------------------
TOL = 1e-6
summ["a_邊界"] = summ["a_DCC"].abs() < TOL
summ["b_邊界"] = summ["b_DCC"].abs() < TOL
summ["任一邊界"] = summ["a_邊界"] | summ["b_邊界"]
summ["兩者皆零"] = summ["a_邊界"] & summ["b_邊界"]
summ["路徑退化"] = summ["sd_rho_DCC"] < 1e-3

print(f"至少一個參數在邊界 : {int(summ['任一邊界'].sum()):2d} / {len(summ)}")
print(f"a 與 b 皆為零      : {int(summ['兩者皆零'].sum()):2d} / {len(summ)}")
print(f"rho 路徑退化       : {int(summ['路徑退化'].sum()):2d} / {len(summ)}   (sd < 0.001)")
print()

def classify(row):
    if row["兩者皆零"]:
        return "① 邊界解 → LR 無資訊"
    if row["路徑退化"]:
        return "② 路徑近乎常數 → 弱證據"
    if row["LR_pvalue_vs_CCC"] < 0.05:
        return "③ 拒絕常數相關"
    return "④ 內點解但不顯著"

summ["診斷"] = summ.apply(classify, axis=1)
print(summ["診斷"].value_counts().sort_index().to_string())
print()
print("『④ 內點解但不顯著』才是唯一可以說『有證據支持相關為常數』的類別。")
print("把 ① 和 ② 混進來當作證據，是本研究審計時修正的一個實質錯誤。")
"""),

code("""
# ---------------------------------------------------------------------------
# 視覺對照：退化路徑 vs 真實動態路徑
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

d_deg = pd.read_csv(PROC / "dcc_BUIDL_SOXX.csv", parse_dates=["date"]).set_index("date")
axes[0].plot(d_deg.index, d_deg["rho_dcc"], lw=1.2, color="#6d6875")
axes[0].set_title(f"BUIDL–SOXX（退化）\\nsd = {d_deg['rho_dcc'].std():.2e}, LR p = "
                  f"{summ.loc['BUIDL-SOXX','LR_pvalue_vs_CCC']:.3f}")
axes[0].set_ylabel(r"$\\rho_t$")

d_real = pd.read_csv(PROC / "dcc_IEF_GLD.csv", parse_dates=["date"]).set_index("date")
axes[1].plot(d_real.index, d_real["rho_dcc"], lw=1.0, color="#1b4965")
axes[1].set_title(f"IEF–GLD（真實動態）\\nsd = {d_real['rho_dcc'].std():.3f}, LR p = "
                  f"{summ.loc['IEF-GLD','LR_pvalue_vs_CCC']:.4f}")

fig.tight_layout()
plt.show()

print("左圖的 y 軸尺度是 1e-6 等級 —— 那是浮點噪音，不是經濟訊號。")
print("兩張圖若不看 y 軸刻度會誤以為左圖也在『變動』，這正是退化配對的危險之處。")
"""),

]
