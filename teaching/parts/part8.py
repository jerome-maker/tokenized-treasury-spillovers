# -*- coding: utf-8 -*-
"""第 8 篇：應用 — 避險（單元 24–25），對應 08_hedge.py。"""

from ._helpers import md, code, unit_header

CELLS = [

md("""
---
---

# 第 8 篇　應用：避險

前七篇都在估計「關聯有多強」。這一篇問一個資產配置者真正關心的問題：

> 這些關聯結構，在實際的避險操作上值多少錢？

這是把統計結果轉換成經濟量值的一步，也是最容易出錯的一步 ——
單元 25 會拆解一個真實的、三重複合的計算錯誤。
"""),

unit_header("單元 24", "Kroner–Sultan 最適避險比率",
            "變異數最小化避險；時變避險比率；前瞻偏誤的避免",
            "08_hedge.py",
            ["Kroner, K. F., & Sultan, J. (1993). Time-varying distributions and dynamic hedging with foreign currency futures. Journal of Financial and Quantitative Analysis, 28(4), 535–551."]),

md("""
### 推導

持有 1 單位待避險資產 $i$，賣空 $\\beta$ 單位的避險工具 $j$。組合報酬為

$$
r_{H,t} = r_{i,t} - \\beta  r_{j,t}
$$

組合變異數：

$$
\\mathrm{Var}(r_H) = h_{ii} - 2\\beta h_{ij} + \\beta^2 h_{jj}
$$

對 $\\beta$ 微分並令其為零：

$$
\\frac{\\partial \\mathrm{Var}(r_H)}{\\partial \\beta} = -2h_{ij} + 2\\beta h_{jj} = 0
$$

$$
\\boxed{ \\beta^* = \\frac{h_{ij}}{h_{jj}} }
$$

二階條件 $2h_{jj} > 0$ 恆成立，故為最小值。

### 時變版本

Kroner–Sultan 的貢獻是把 $h_{ij}, h_{jj}$ 換成**條件**動差，由 DCC 提供：

$$
\\beta^*_t = \\frac{h_{ij,t}}{h_{jj,t}} = \\frac{\\rho_t \\sigma_{i,t}\\sigma_{j,t}}{\\sigma_{j,t}^2}
= \\rho_t \\frac{\\sigma_{i,t}}{\\sigma_{j,t}}
$$

最後這個形式很有啟發性：**避險比率 = 相關係數 × 波動比**。

程式碼：

```python
h_ij = d["rho_dcc"] * d["sigma_tok"] * d["sigma_eq"]
h_jj = d["sigma_tok"] ** 2
beta_star = (h_ij / h_jj).clip(-5, 5)
```

### 兩個實作細節，都很重要

**（一）一日落後：避免前瞻偏誤**

```python
beta_lag = beta_star.shift(1)
```

$\\beta^*_t$ 用到了第 $t$ 期的 $\\rho_t, \\sigma_t$，而這些在 $t$ 期開盤時**還不知道**。
若直接用 $\\beta^*_t$ 建構 $t$ 期的避險組合，等於偷看了未來。

實務上可執行的策略是用 $t-1$ 期的資訊決定 $t$ 期的部位：

$$
r_{H,t} = r_{i,t} - \\beta^*_{t-1}  r_{j,t}
$$

> 這是回測中最常見的錯誤來源之一。忘記 shift 會系統性地高估避險效果，
> 而且高估的幅度往往大到足以把「沒用」變成「有效」。

**（二）截斷極端比率**

`.clip(-5, 5)` 的必要性來自 $\\beta^* = \\rho \\cdot \\sigma_i/\\sigma_j$ 的分母。
當避險工具本身的波動 $\\sigma_j$ 趨近於零時，$\\beta^*$ 會爆炸。

這正是**本研究把 SHV 排除在表 14 之外**的原因：SHV 的日波動只有 0.016%，
比股票低兩個數量級，會讓變異數比值極不穩定。表格註解明確說明了這一點。
"""),

code("""
# ---------------------------------------------------------------------------
# 從 DCC 輸出重建避險比率，並展示前瞻偏誤的量級
# ---------------------------------------------------------------------------
d = pd.read_csv(PROC / "dcc_IEF_FANG.csv", parse_dates=["date"]).set_index("date")
h_ij = d["rho_dcc"] * d["sigma_tok"] * d["sigma_eq"]
h_jj = d["sigma_tok"] ** 2
beta_star = (h_ij / h_jj).clip(-5, 5)

r_eq = panel["R_FANG"].reindex(d.index)
r_hedge = panel["R_IEF"].reindex(d.index)

# 正確做法：用 t-1 期資訊
r_H_correct = (r_eq - beta_star.shift(1) * r_hedge).dropna()
# 錯誤做法：偷看當期
r_H_lookahead = (r_eq - beta_star * r_hedge).dropna()

var_un = r_eq.reindex(r_H_correct.index).var()
HE_correct = 1 - r_H_correct.var() / var_un
HE_look = 1 - r_H_lookahead.reindex(r_H_correct.index).var() / var_un

print(f"平均避險比率 beta* = {beta_star.mean():.4f}")
print()
print(f"避險效能（正確，用 t-1 期 beta）  : {HE_correct:+.4f}")
print(f"避險效能（錯誤，偷看當期 beta）  : {HE_look:+.4f}")
print(f"前瞻偏誤誇大了                    : {HE_look - HE_correct:+.4f}"
      f"  （相對放大 {HE_look/HE_correct:.1f} 倍）")
"""),

code("""
# ---------------------------------------------------------------------------
# 避險比率的組成：相關係數 × 波動比
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(9.5, 6), sharex=True)
axes[0].plot(d.index, d["rho_dcc"], lw=0.9, color="#1b4965")
axes[0].set_ylabel(r"$\\rho_t$"); axes[0].set_title("(1) 條件相關")
axes[1].plot(d.index, d["sigma_eq"] / d["sigma_tok"], lw=0.9, color="#f4a261")
axes[1].set_ylabel(r"$\\sigma_i/\\sigma_j$"); axes[1].set_title("(2) 波動比")
axes[2].plot(d.index, beta_star, lw=0.9, color="#bc4749")
axes[2].axhline(0, color="grey", lw=0.7)
axes[2].set_ylabel(r"$\\beta^*_t$"); axes[2].set_title(r"(3) 避險比率 = (1) $\\times$ (2)")
fig.tight_layout()
plt.show()

print("代幣的 beta* 之所以幾乎為零，主因是 (1) 相關係數近零，而非 (2) 波動比異常。")
"""),

unit_header("單元 25", "Ederington 避險效能與交易成本",
            "變異數縮減比；週轉率成本；變異數指標與水準指標的區分",
            "08_hedge.py",
            ["Ederington, L. H. (1979). The hedging performance of the new futures markets. Journal of Finance, 34(1), 157–170."]),

md("""
### Ederington 避險效能

$$
\\mathrm{HE} = 1 - \\frac{\\mathrm{Var}(r_H)}{\\mathrm{Var}(r_i)}
$$

解讀為「避險消除了待避險資產多少比例的報酬變異數」。

| HE | 意義 |
|---|---|
| $1$ | 完全避險 |
| $0$ | 毫無作用 |
| $< 0$ | **反效果** —— 避險後波動反而變大 |

$\\mathrm{HE} < 0$ 在本研究中頻繁出現。這不是錯誤：當 $\\rho$ 幾乎為零時，
加入避險部位只是引入額外的波動來源，沒有抵銷任何東西。

### 交易成本：本單元的核心教學

一個時變的避險比率必須**持續再平衡**，而再平衡要花錢。週轉率為

$$
\\text{turnover}_t = \\left|\\beta^*_t - \\beta^*_{t-1}\\right| = \\left|\\Delta\\beta_t\\right|
$$

若單邊成本為 $c$，則當期成本為

$$
\\text{cost}_t = c \\cdot \\left|\\Delta\\beta_t\\right|
$$

因為 $\\beta$ 是「每單位待避險資產所持有的避險工具名目部位」，
$|\\Delta\\beta|$ 就是當期交易的名目量，成本與報酬序列同單位。

### 關鍵觀念：成本是水準效應，HE 是變異數指標

這是本單元最重要的一句話，也是本專案審計時修正的核心錯誤：

> **交易成本影響的是報酬的「平均水準」，Ederington HE 衡量的是報酬的「變異數」。
> 把成本折進變異數比值，得到的數字既不是變異數縮減、也不是成本，兩者都不是。**

正確做法是**分開報告**：

$$
\\mathrm{HE} = 1 - \\frac{\\mathrm{Var}(r_H)}{\\mathrm{Var}(r_i)}
\\qquad\\text{（純變異數縮減）}
$$

$$
\\text{年化成本拖累（bp）} = 100 \\times \\overline{\\text{cost}_t} \\times 252
\\qquad\\text{（純水準效應）}
$$

修正後的程式碼：

```python
turnover = combined["beta_lag"].diff().abs().fillna(0.0)
for bps in COST_BPS:
    c = bps / 100.0          # 基點 -> 百分點，對應 *100 的對數報酬
    cost_t = c * turnover
    cost_results[f"cost_bp_ann_{bps}bp"] = 100.0 * cost_t.mean() * 252.0
```

### 修正後才浮現的實證發現

這個修正不只是形式上的整潔 —— 它改變了論文的結論措辭。

| 避險工具 | HE（變異數縮減） | 年化再平衡成本 @10bp |
|---|---|---|
| IEF vs FANG+ | $+0.0201$ | **415 bp** |
| IEF vs ETH | $+0.0040$ | **545 bp** |
| 三個代幣（15 格） | ≈ 0 | 4–132 bp |

原本的結論是「沒有任何代幣勝過傳統國債 ETF，**不論成本前後**」。
修正後才看見：IEF 確實在變異數縮減上勝出，但它的避險比率變動大得多，
**再平衡成本是代幣的一個數量級以上**。

論文因此改寫為：IEF 仍然勝出，但**這個基準本身很弱** ——
它最好的一格只消除 2.0% 的變異數，卻要付幾百個基點的年化成本。
兩類工具在本樣本上**都不是有用的變異數最小化避險工具**，
代幣只是沒能改善一個原本就很差的基準。
"""),

code("""
# ---------------------------------------------------------------------------
# 修正後的避險結果：變異數縮減與成本分開呈現
# ---------------------------------------------------------------------------
h = pd.read_csv(PROC / "hedge_summary.csv")
h = h[h["hedge_instrument"] != "SHV"]          # SHV 因近零自身變異數而排除
cols = ["hedged_equity", "hedge_instrument", "mean_beta_star", "mean_turnover",
        "HE", "cost_bp_ann_10bp"]
display(h[cols].round(4).to_string(index=False))

tok = h[h["hedge_instrument"].isin(["BUIDL", "OUSG", "sUSDS"])]
ief = h[h["hedge_instrument"] == "IEF"]
print()
print(f"代幣（{len(tok)} 格）：週轉率 {tok['mean_turnover'].min():.4f}–{tok['mean_turnover'].max():.4f}，"
      f"年化成本 {tok['cost_bp_ann_10bp'].min():.0f}–{tok['cost_bp_ann_10bp'].max():.0f} bp")
print(f"IEF （{len(ief)} 格）：週轉率 {ief['mean_turnover'].min():.4f}–{ief['mean_turnover'].max():.4f}，"
      f"年化成本 {ief['cost_bp_ann_10bp'].min():.0f}–{ief['cost_bp_ann_10bp'].max():.0f} bp")
print()
print(f"IEF 最佳一格：HE = {ief['HE'].max():.4f}（消除 {ief['HE'].max()*100:.1f}% 變異數），"
      f"成本 {ief.loc[ief['HE'].idxmax(),'cost_bp_ann_10bp']:.0f} bp/年")
"""),

code("""
# ---------------------------------------------------------------------------
# 變異數縮減 vs 成本：兩個維度一起看才看得出全貌
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.2))
colors = {"BUIDL": "#1b4965", "OUSG": "#2a9d8f", "sUSDS": "#f4a261", "IEF": "#bc4749"}
for inst, grp in h.groupby("hedge_instrument"):
    ax.scatter(grp["cost_bp_ann_10bp"], grp["HE"] * 100, s=55,
               color=colors.get(inst, "grey"), label=inst, alpha=0.85, edgecolor="white")
    for _, r in grp.iterrows():
        ax.annotate(r["hedged_equity"], (r["cost_bp_ann_10bp"], r["HE"] * 100),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("年化再平衡成本（bp，10bp 單邊假設）")
ax.set_ylabel("變異數縮減 HE（%）")
ax.set_title("右上角才是有用的避險；本樣本沒有任何一格接近右上角")
ax.legend(title="避險工具", fontsize=8)
fig.tight_layout()
plt.show()

print("IEF 的點都在右側（成本高），代幣的點都擠在左側零軸附近（沒效果但也沒成本）。")
print("兩者都不在『高變異數縮減 + 低成本』的理想區域。")
"""),

md("""
### 這一節的方法論結論

把成本與效能分開報告，比折進單一數字更有資訊量。原因是兩者對投資人的意義不同：

- **變異數縮減**：風險管理的目標
- **成本拖累**：確定會發生的、對報酬的直接侵蝕

一個 HE = 0.02 但成本 415 bp 的避險，與一個 HE = 0.02 但成本 8 bp 的避險，
在單一「成本調整後 HE」指標下可能看起來差不多，但實際上是完全不同的兩件事。

> **給做實證的同學**：當你想把兩個不同單位、不同性質的量「調整」成一個數字時，
> 先問自己「調整後的數字還能被解讀嗎」。如果答案是「不太能」，
> 那就分開報告，讓讀者自己權衡。
"""),

]
