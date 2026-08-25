# -*- coding: utf-8 -*-
"""第 7 篇：結構變化與不確定性量化（單元 21–23）。"""

from ._helpers import md, code, unit_header

CELLS = [

md("""
---
---

# 第 7 篇　結構變化與不確定性量化

前面幾篇估出了一條 $\\rho_t$ 路徑。但估出來的路徑是**點估計**，兩個問題還沒回答：

1. 這條路徑上的起伏是**真實的結構變化**，還是估計噪音？（單元 21）
2. 平均相關係數的估計有多不確定？（單元 22）
3. 路徑的變動是**連續漂移**還是**離散的機制切換**？（單元 23）

三個單元用三種互補的角度回答同一件事：**這條路徑有多可信**。
"""),

unit_header("單元 21", "Bai–Perron 多重結構斷點",
            "多重結構變化的最小平方估計；動態規劃；supF 檢定；BIC 斷點數選擇",
            "11_r_baiperron_bootstrap.R",
            ["Bai, J., & Perron, P. (1998). Estimating and testing linear models with multiple structural changes. Econometrica, 66(1), 47–78.",
             "Zeileis, A., Leisch, F., Hornik, K., & Kleiber, C. (2002). strucchange: An R package for testing for structural change in linear regression models."]),

md("""
### 原理：斷點位置未知的困難

若已知斷點在 $t^*$，檢定「$t^*$ 前後係數是否相同」就是標準的 Chow 檢定。
但實務上**斷點位置未知**，而且可能不只一個。

天真的做法是「對每個可能的 $t$ 都做一次 Chow 檢定，取最大的」——
但這樣得到的統計量**不服從 $F$ 分配**（因為你取了最大值，等於做了 $T$ 次檢定）。
Bai–Perron 的貢獻是把估計與檢定都做對。

### 模型設定

$m$ 個斷點把樣本切成 $m+1$ 個區間：

$$
y_t = x_t'\\beta_j + u_t, \\qquad t = T_{j-1}+1, \\ldots, T_j, \\qquad j = 1, \\ldots, m+1
$$

本研究的應用中 $y_t = \\rho_t$（DCC 相關路徑），$x_t = 1$，
所以每個區間就是一個**均值水準**。

### 估計：全域最小化與動態規劃

斷點估計量為

$$
(\\hat{T}_1, \\ldots, \\hat{T}_m) = \\arg\\min_{T_1, \\ldots, T_m}
\\sum_{j=1}^{m+1}\\sum_{t=T_{j-1}+1}^{T_j}\\left(y_t - \\bar{y}_{[T_{j-1}+1, T_j]}\\right)^2
$$

窮舉所有分割的複雜度是 $O(T^m)$ —— 對 $T = 900$、$m = 5$ 完全不可行。
Bai–Perron 用**動態規劃**把它降到 $O(T^2)$：先算出所有區間 $[i, j]$ 的殘差平方和，
再用遞迴求最佳分割。

### 修剪參數 $h$

```r
bp <- breakpoints(ts_rho ~ 1, h = 0.15, breaks = 5)
```

`h = 0.15` 要求每個區間至少包含 15% 的觀測值。這不是任意的：

- 太小 → 區間內觀測太少，區間均值估計不穩，會抓到雜訊
- 太大 → 無法偵測靠近端點的真實斷點

0.15 是文獻的常用值，對應 $T = 900$ 時每個區間至少 135 天，
也意味著**最多只能有 5 個斷點**（因為 $6 \\times 0.15 = 0.90 < 1$）。
程式中的 `breaks = 5` 與此一致。

### 斷點個數的選擇：BIC

$$
\\mathrm{BIC}(m) = \\ln \\hat{\\sigma}^2(m) + \\frac{p^*(m)\\ln T}{T}
$$

其中 $p^*(m)$ 是總參數量（含斷點位置本身）。取 BIC 最小的 $m$。

```r
bic_tab <- summary(bp)$RSS["BIC", ]
opt_m <- which.min(bic_tab) - 1   # summary 含 m=0..5，索引 1 對應 0 個斷點
```

### supF 檢定

檢定 $H_0$：無斷點，對 $H_1$：有斷點（位置未知）。

$$
\\sup F = \\sup_{\\lambda \\in [h, 1-h]} F(\\lambda)
$$

其中 $F(\\lambda)$ 是在 $T_1 = \\lfloor \\lambda T\\rfloor$ 處的 Chow $F$ 統計量。
臨界值由 Andrews (1993) / Bai–Perron 的模擬表提供，**遠高於**標準 $F$ 分配的臨界值。
"""),

md("""
### 本研究的關鍵陷阱：退化序列上的 supF

這是本單元最重要的實務教訓。

$F$ 統計量的形式是「組間變異 / 組內變異」。當**組內變異趨近於零**時，
即使組間差異在經濟上完全微不足道，$F$ 也會爆炸。

單元 13 已經看到：34 組配對中有 11 組的 $\\rho_t$ 標準差低於 $0.001$ ——
那是浮點運算的噪音等級（$\\sim 10^{-6}$），不是經濟訊號。
對這種序列做 supF 檢定，會得到**形式上顯著但完全沒有意義**的結果。

R 腳本因此在檢定前先標記：

```r
is_degenerate <- sd(rho, na.rm = TRUE) < 1e-3
```

註解寫得很清楚：

> A supF break test on such a series can report a formally "significant" break
> purely because residual variance is near-zero (huge F-stat from economically
> meaningless numerical noise), not a real structural change.

**處理方式**：這些配對**仍然報告在表中**（不是偷偷刪掉），但標記為 degenerate 並且
**不納入解讀**。34 組全部呈現，11 組標記，剩 23 組作為有效樣本。

> **為什麼不直接刪掉？** 因為刪掉會讓讀者無從判斷「這 11 組去哪了」。
> 標記並保留，讓讀者能自行驗證退化的判準是否合理。這是可稽核性的基本要求。
"""),

code("""
# ---------------------------------------------------------------------------
# Bai-Perron 結果：退化標記與有效樣本
# ---------------------------------------------------------------------------
bp = pd.read_csv(PROC / "r_baiperron_results.csv")
print(f"總配對數：{len(bp)}，退化標記：{int(bp['degenerate'].sum())}，"
      f"有效樣本：{int((~bp['degenerate']).sum())}")
print()

nd = bp[~bp["degenerate"]].sort_values("opt_num_breaks", ascending=False)
display(nd[["pair", "n_obs", "sd_rho", "opt_num_breaks", "supF_stat", "supF_pvalue"]].round(4).head(12))

print("斷點最多的配對全部屬於傳統工具對照組或 OUSG 對加密資產：")
for _, r in nd.head(6).iterrows():
    print(f"    {r['pair']:12} {int(r['opt_num_breaks'])} 個斷點, supF = {r['supF_stat']:7.1f}")
print()
zero = (nd["opt_num_breaks"] == 0).sum()
print(f"有效樣本中完全沒有偵測到斷點的：{zero} / {len(nd)}")
"""),

code("""
# ---------------------------------------------------------------------------
# 退化序列的危險：組內變異趨零如何讓 F 統計量失去意義
# ---------------------------------------------------------------------------
deg = bp[bp["degenerate"]][["pair", "sd_rho", "opt_num_breaks", "supF_stat", "supF_pvalue"]]
display(deg.round(6))

sig_deg = deg[deg["supF_pvalue"] < 0.05]
print(f"退化配對中『形式上顯著』(p < 0.05) 者：{len(sig_deg)}")
for _, r in sig_deg.iterrows():
    print(f"    {r['pair']:12} sd(rho) = {r['sd_rho']:.2e}, supF = {r['supF_stat']:.2f}, "
          f"p = {r['supF_pvalue']:.4f}")
print()
print("→ sd(rho) 在 1e-6 等級時，任何『組間差異』都會被除以近乎零的組內變異。")
print("  若不標記，這些配對會被誤讀為『相關結構發生變化』。")
"""),

code("""
# ---------------------------------------------------------------------------
# 從零實作：動態規劃求最佳分割（單一斷點的簡化版）
# ---------------------------------------------------------------------------
def best_single_break(y, trim=0.15):
    \"\"\"窮舉單一斷點位置，回傳使 SSR 最小者與對應的 F 統計量。\"\"\"
    T = len(y)
    lo, hi = int(trim * T), int((1 - trim) * T)
    ssr_full = ((y - y.mean()) ** 2).sum()
    best = (None, np.inf)
    Fs = []
    for k in range(lo, hi):
        s1, s2 = y[:k], y[k:]
        ssr = ((s1 - s1.mean()) ** 2).sum() + ((s2 - s2.mean()) ** 2).sum()
        F = (ssr_full - ssr) / (ssr / (T - 2))
        Fs.append((k, F))
        if ssr < best[1]:
            best = (k, ssr)
    ks, Fv = zip(*Fs)
    return best[0], np.array(ks), np.array(Fv)

d = pd.read_csv(PROC / "dcc_IEF_GLD.csv", parse_dates=["date"]).set_index("date")
rho = d["rho_dcc"].values
k_star, ks, Fv = best_single_break(rho)

fig, axes = plt.subplots(2, 1, figsize=(9.5, 5), sharex=False)
axes[0].plot(d.index, rho, lw=0.9, color="#1b4965")
axes[0].axvline(d.index[k_star], color="#bc4749", ls="--", lw=1.4,
                label=f"最佳單一斷點：{d.index[k_star].date()}")
axes[0].set_ylabel(r"$\\rho_t$"); axes[0].legend(); axes[0].set_title("IEF–GLD 相關路徑")

axes[1].plot(d.index[ks], Fv, lw=1.1, color="#2a9d8f")
axes[1].axvline(d.index[k_star], color="#bc4749", ls="--", lw=1.2)
axes[1].set_ylabel("F 統計量"); axes[1].set_title(r"$F(\\lambda)$ 曲線：supF 取的是這條線的最大值")
fig.tight_layout()
plt.show()

print(f"supF = {Fv.max():.2f}（此為單斷點簡化版；R 的 strucchange 允許至多 5 個斷點，"
      f"報告值 {bp.loc[bp['pair']=='IEF_GLD','supF_stat'].iloc[0]:.1f}）")
"""),

unit_header("單元 22", "移動區塊拔靴法",
            "區塊拔靴（block bootstrap）；相依資料的重抽樣；百分位信賴區間",
            "11_r_baiperron_bootstrap.R",
            ["Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. Annals of Statistics, 17(3), 1217–1241.",
             "Politis, D. N., & Romano, J. P. (1994)."]),

md("""
### 原理：為什麼不能用普通拔靴法

標準拔靴法逐點有放回抽樣，隱含假設**觀測值 iid**。
但 $\\rho_t$ 是高度序列相關的（$b_{\\mathrm{DCC}}$ 常在 0.9 以上）。

逐點重抽會**摧毀序列相關結構**，讓重抽樣本看起來像白噪音。
後果是變異數被嚴重低估，信賴區間過窄 —— 也就是**過度自信**。

### 移動區塊拔靴

把序列切成長度 $\\ell$ 的重疊區塊：

$$
B_i = (y_i, y_{i+1}, \\ldots, y_{i+\\ell-1}), \\qquad i = 1, \\ldots, T - \\ell + 1
$$

有放回地抽取 $\\lceil T/\\ell \\rceil$ 個區塊，串接成長度 $T$ 的重抽序列。

**關鍵**：區塊**內部**的相關結構被完整保留，只有區塊**之間**的關聯被打斷。

### 區塊長度的取捨

| $\\ell$ | 後果 |
|---|---|
| 太短 | 保留不了長程相依，退化回 iid 拔靴，區間過窄 |
| 太長 | 有效的獨立區塊數 $T/\\ell$ 太少，估計本身噪音大 |

理論上的最適階數為 $\\ell \\propto T^{1/3}$，但常數依賴序列的相依程度。
本研究用一個**資料驅動**的規則：

```r
acf1 <- acf(rho, plot = FALSE, lag.max = 1)$acf[2]
block_len <- max(5, min(round(n^(1/3) / max(1 - abs(acf1), 0.05)), floor(n/10)))
```

拆解這一行：

| 部分 | 作用 |
|---|---|
| $n^{1/3}$ | 理論階數 |
| $\\div (1 - \\lvert\\hat{\\rho}_1\\rvert)$ | 相依越強（$\\hat{\\rho}_1 \\to 1$），區塊越長 |
| $\\max(\\cdot, 0.05)$ | 防止除以零（近單根序列） |
| $\\max(5, \\cdot)$ | 下界 |
| $\\min(\\cdot, n/10)$ | 上界：至少要有 10 個區塊 |

### 實際的區塊長度差異

這個規則在本研究產生了很大的差異：

- IEF–GLD（$\\hat{\\rho}_1$ 極高）：$\\ell = 90$
- 退化配對：$\\ell = 8$–$11$

同一份程式碼對不同序列自動給出差 10 倍的區塊長度，正是資料驅動規則的價值。

### 百分位信賴區間

對 $R = 1000$ 次重抽各計算統計量 $\\hat{\\theta}^{(r)}$，
95% 區間取經驗分配的 2.5 與 97.5 百分位：

$$
\\mathrm{CI}_{95} = \\left[\\hat{\\theta}^{(0.025)},  \\hat{\\theta}^{(0.975)}\\right]
$$
"""),

code("""
# ---------------------------------------------------------------------------
# 從零實作移動區塊拔靴，並與 R 的結果對照
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20260818)

def moving_block_bootstrap(y, R=1000, block_len=None):
    y = np.asarray(y); n = len(y)
    if block_len is None:
        acf1 = np.corrcoef(y[1:], y[:-1])[0, 1]
        block_len = max(5, min(int(round(n**(1/3) / max(1 - abs(acf1), 0.05))), n // 10))
    n_blocks = int(np.ceil(n / block_len))
    starts_max = n - block_len
    stats = np.empty(R)
    for r in range(R):
        idx = rng.integers(0, starts_max + 1, size=n_blocks)
        samp = np.concatenate([y[s:s + block_len] for s in idx])[:n]
        stats[r] = samp.mean()
    return stats, block_len

rho_ief = pd.read_csv(PROC / "dcc_IEF_GLD.csv")["rho_dcc"].values
boot_stats, L = moving_block_bootstrap(rho_ief)
ci = np.percentile(boot_stats, [2.5, 97.5])

r_res = pd.read_csv(PROC / "r_blockbootstrap_results.csv")
r_row = r_res[r_res["pair"] == "IEF_GLD"].iloc[0]

print(f"Python 實作：區塊長度 = {L}, 平均 = {rho_ief.mean():.4f}, "
      f"95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"R (boot)   ：區塊長度 = {int(r_row['block_length'])}, 平均 = {r_row['point_mean']:.4f}, "
      f"95% CI = [{r_row['ci95_lo']:.4f}, {r_row['ci95_hi']:.4f}]")
print()
print("兩者的區塊長度規則相同，微小差異來自 RNG 與 tsboot 的 sim='fixed' 細節。")
"""),

code("""
# ---------------------------------------------------------------------------
# 為什麼不能用 iid 拔靴：區間寬度的比較
# ---------------------------------------------------------------------------
iid_stats = np.array([rng.choice(rho_ief, size=len(rho_ief), replace=True).mean()
                      for _ in range(1000)])
ci_iid = np.percentile(iid_stats, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(8.5, 3))
ax.hist(iid_stats, bins=50, alpha=0.65, color="#bc4749", label=f"iid 拔靴（寬度 {ci_iid[1]-ci_iid[0]:.4f}）")
ax.hist(boot_stats, bins=50, alpha=0.65, color="#1b4965", label=f"區塊拔靴（寬度 {ci[1]-ci[0]:.4f}）")
ax.axvline(rho_ief.mean(), color="black", lw=1.2, ls="--", label="點估計")
ax.set_xlabel(r"重抽樣本的 $\\bar{\\rho}$"); ax.set_ylabel("次數")
ax.set_title("忽略序列相關會讓信賴區間嚴重過窄")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

print(f"區塊拔靴區間寬度是 iid 拔靴的 {(ci[1]-ci[0])/(ci_iid[1]-ci_iid[0]):.1f} 倍。")
print("若用 iid 拔靴，會宣稱一個遠比實際精確的估計 —— 典型的過度自信。")
"""),

code("""
# ---------------------------------------------------------------------------
# 全部配對的信賴區間森林圖
# ---------------------------------------------------------------------------
bo = pd.read_csv(PROC / "r_blockbootstrap_results.csv")
nd_bo = bo[~bo["degenerate"]].sort_values("point_mean")

fig, ax = plt.subplots(figsize=(7.5, 6.5))
y = np.arange(len(nd_bo))
ax.errorbar(nd_bo["point_mean"], y,
            xerr=[nd_bo["point_mean"] - nd_bo["ci95_lo"],
                  nd_bo["ci95_hi"] - nd_bo["point_mean"]],
            fmt="o", color="#1b4965", ecolor="#6d6875", capsize=3, markersize=4)
ax.axvline(0, color="grey", ls="--", lw=0.9)
ax.set_yticks(y); ax.set_yticklabels(nd_bo["pair"].str.replace("_", " vs. "), fontsize=7)
ax.set_xlabel("平均 DCC 相關（區塊拔靴 95% CI）")
ax.set_title("只有 IEF–GLD 的區間明顯遠離零")
fig.tight_layout()
plt.show()

excl_zero = nd_bo[(nd_bo["ci95_lo"] > 0) | (nd_bo["ci95_hi"] < 0)]
print(f"區間不含零的配對：{len(excl_zero)} / {len(nd_bo)}")
print(f"其中平均相關 > 0.1 的：{(excl_zero['point_mean'].abs() > 0.1).sum()}")
"""),

unit_header("單元 23", "Markov 轉換模型",
            "隱藏馬可夫鏈；Hamilton 濾波；預期持續期間",
            "07_regime.py",
            ["Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, 57(2), 357–384."]),

md("""
### 原理：斷點模型與機制模型的差別

單元 21 的 Bai–Perron 假設結構變化是**一次性、不可逆**的：
斷點之後進入新狀態，不會回到舊狀態。

但金融市場的許多變化是**可逆的**：危機期與平靜期會反覆交替。
Markov 轉換模型正是為此設計。

### 模型設定

令 $S_t \\in \\{1, 2\\}$ 為**不可觀測**的機制指標。觀測方程：

$$
\\rho_t = \\mu_{S_t} + \\varepsilon_t, \\qquad \\varepsilon_t \\sim N(0, \\sigma^2_{S_t})
$$

本研究讓**均值與變異數都隨機制切換**（`switching_variance=True`）。

機制的演化服從一階馬可夫鏈：

$$
P = \\begin{pmatrix} p_{11} & p_{12} \\\\ p_{21} & p_{22} \\end{pmatrix},
\\qquad p_{ij} = \\Pr(S_t = j \\mid S_{t-1} = i)
$$

其中 $p_{i1} + p_{i2} = 1$。

### 預期持續期間

若目前在機制 $i$，停留的期數服從幾何分配，期望值為

$$
\\mathbb{E}[\\text{持續期間}_i] = \\frac{1}{1 - p_{ii}}
$$

程式碼：

```python
duration_hi = 1 / (1 - P[hi_idx, hi_idx]) if P[hi_idx, hi_idx] < 1 else np.inf
```

| $p_{ii}$ | 預期持續 |
|---|---|
| 0.90 | 10 天 |
| 0.96 | 25 天 |
| 0.98 | 50 天 |

**這個數字是判斷機制是否「真實」的關鍵診斷**。若 $p_{ii} \\approx 0.4$，
預期持續只有 1.7 天 —— 那不是「機制」，只是模型在對雜訊做分類。

### Hamilton 濾波與 Kim 平滑

因為 $S_t$ 不可觀測，估計需要對所有可能路徑積分。Hamilton 濾波遞迴計算：

$$
\\Pr(S_t = j \\mid \\mathcal{F}_t) =
\\frac{f(\\rho_t \\mid S_t = j)\\sum_{i} p_{ij}\\Pr(S_{t-1} = i \\mid \\mathcal{F}_{t-1})}
{\\sum_{k} f(\\rho_t \\mid S_t = k)\\sum_{i} p_{ik}\\Pr(S_{t-1} = i \\mid \\mathcal{F}_{t-1})}
$$

這是**濾波**機率（只用到 $t$ 以前的資訊）。Kim (1994) 的平滑演算法再往回走一遍，
得到**平滑**機率 $\\Pr(S_t = j \\mid \\mathcal{F}_T)$ —— 用上全樣本資訊，
這才是報告與繪圖該用的。

### 一個實務細節：機制標籤不可識別

最大概似無法區分「機制 1 是高相關」與「機制 2 是高相關」——
兩者概似完全相同（label switching 問題）。

程式碼因此**事後依均值排序**，強制低相關機制為第一個：

```python
means = res.params[["const[0]", "const[1]"]].values
lo_idx, hi_idx = np.argsort(means)   # 低均值在前
```

沒有這一步，不同配對的「機制 1」會指涉不同的東西，表格完全無法比較。
"""),

code("""
# ---------------------------------------------------------------------------
# 機制估計結果：用持續期間判斷哪些機制是真實的
# ---------------------------------------------------------------------------
reg = pd.read_csv(PROC / "regime_summary.csv", index_col=0)
show = reg[["n_obs", "mean_rho_low_regime", "mean_rho_high_regime",
            "P_stay_high", "expected_duration_high_days", "frac_time_high_regime"]]
display(show.round(4))

print("診斷：預期持續期間 < 3 天者，實質上不是『機制』而是對雜訊的分類")
weak = reg[reg["expected_duration_high_days"] < 3]
strong = reg[reg["expected_duration_high_days"] >= 20]
print(f"  持續 < 3 天  ：{len(weak)} 組 → {', '.join(weak.index)}")
print(f"  持續 >= 20 天：{len(strong)} 組 → {', '.join(strong.index)}")
print()
print("最乾淨的機制結構：")
for p, r in strong.iterrows():
    print(f"    {p:14} 低 = {r['mean_rho_low_regime']:+.3f}, 高 = {r['mean_rho_high_regime']:+.3f}, "
          f"P(留在高) = {r['P_stay_high']:.3f}, 預期持續 = {r['expected_duration_high_days']:.0f} 天")
"""),

code("""
# ---------------------------------------------------------------------------
# 平滑機率路徑：真實機制 vs 雜訊分類
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.2), sharex=False)

for ax, pair, note in [(axes[0], "IEF_GLD", "真實機制：長期持續、切換清晰"),
                       (axes[1], "OUSG_FANG", "雜訊分類：頻繁跳動、無持續性")]:
    sp = pd.read_csv(PROC / f"regime_probs_{pair}.csv", index_col=0, parse_dates=True)
    hi_col = sp.columns[int(np.argmax([reg.loc[pair, "mean_rho_high_regime"] >= 0]))]
    ax.fill_between(sp.index, 0, sp.iloc[:, -1], color="#1b4965", alpha=0.75, step="mid")
    ax.set_ylim(0, 1); ax.set_ylabel("P(高相關機制)")
    ax.set_title(f"{pair.replace('_', ' vs. ')}　—　{note}"
                 f"（預期持續 {reg.loc[pair,'expected_duration_high_days']:.1f} 天）")

fig.tight_layout()
plt.show()

print("上圖有大塊連續的高機率區間；下圖幾乎是隨機的細碎跳動。")
print("同樣是『兩機制模型』，但只有上圖的機制在經濟上有意義。")
"""),

]
