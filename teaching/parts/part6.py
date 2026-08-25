# -*- coding: utf-8 -*-
"""第 6 篇：時頻分析（單元 18–20），對應 06_wavelet.py 與 12_r_partial_wavelet.R。"""

from ._helpers import md, code, unit_header

CELLS = [

md("""
---
---

# 第 6 篇　時頻分析

第 5 篇的頻域分解告訴我們溢出**發生在哪個時間尺度**，但它給的是全樣本平均。
小波分析同時保留**時間**與**頻率**兩個維度，能回答更細的問題：

> 這兩個資產是在 2024 年的短週期上共動，還是在 2025 年的長週期上共動？

這對本研究特別重要，因為代幣化國債市場在樣本期間內規模成長了數倍 ——
早期與晚期的行為未必相同。
"""),

unit_header("單元 18", "連續小波轉換與 Morlet 母小波",
            "連續小波轉換（CWT）；Morlet 小波；時頻不確定性原理",
            "06_wavelet.py",
            ["Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet analysis. Bulletin of the American Meteorological Society, 79(1), 61–78.",
             "Torrence, C., & Webster, P. J. (1999). Interdecadal changes in the ENSO-monsoon system. Journal of Climate, 12(8), 2679–2690."]),

md("""
### 原理：傅立葉轉換丟掉了時間

標準的傅立葉轉換把序列分解成不同頻率的正弦波：

$$
\\hat{x}(\\omega) = \\int_{-\\infty}^{\\infty} x(t)  e^{-i\\omega t}  dt
$$

問題是基底函數 $e^{-i\\omega t}$ **在整個時間軸上都存在**，所以 $\\hat{x}(\\omega)$
告訴你「整段期間有多少 $\\omega$ 頻率的成分」，但完全不知道它發生在什麼時候。

對金融序列來說這是致命的：2020 年 3 月的高頻共動與 2023 年的高頻共動，
在傅立葉頻譜上長得一模一樣。

### 小波的解法：有限支撐的基底

小波用一個**局部化**的基底函數，同時沿時間平移與尺度伸縮：

$$
W_x(s, \\tau) = \\frac{1}{\\sqrt{s}}\\sum_{t} x_t  \\psi^*\\left(\\frac{t - \\tau}{s}\\right)
$$

| 參數 | 作用 |
|---|---|
| $\\tau$ | **平移**：小波中心落在哪個時點 |
| $s$ | **尺度**：小波拉多寬（大 $s$ = 低頻 = 長週期） |
| $1/\\sqrt{s}$ | 能量標準化，讓不同尺度的係數可比較 |

### Morlet 母小波

$$
\\psi_0(\\eta) = \\pi^{-1/4}  e^{i\\omega_0 \\eta}  e^{-\\eta^2/2}
$$

它是一個**被高斯窗包絡的複數正弦波**：

- $e^{i\\omega_0\\eta}$：振盪部分，提供頻率解析
- $e^{-\\eta^2/2}$：高斯窗，提供時間局部化
- $\\pi^{-1/4}$：能量標準化

**為什麼選 $\\omega_0 = 6$**（本研究與絕大多數文獻的預設）：

1. 滿足**可容許性條件**（小波均值為零）到數值精度
2. 在 $\\omega_0 = 6$ 時，尺度 $s$ 與傅立葉週期幾乎相等：$\\text{週期} \\approx 1.03  s$，
   讓「尺度」可以直接讀成「天數」
3. 在時間解析與頻率解析之間取得公認的良好平衡

**為什麼要用複數小波**：複數形式讓我們同時得到**振幅**與**相位**。
相位資訊是單元 19 判斷「誰領先誰」的基礎。

### 時頻不確定性原理

小波不是免費的午餐。Heisenberg–Gabor 極限規定：

$$
\\Delta t \\cdot \\Delta \\omega \\ge \\frac{1}{2}
$$

你不可能同時無限精確地知道「什麼時候」與「什麼頻率」。小波的取捨是：

- **高頻（小 $s$）**：時間解析好，頻率解析差
- **低頻（大 $s$）**：頻率解析好，時間解析差

這個取捨直接導致單元 19 的影響錐問題。
"""),

code("""
# ---------------------------------------------------------------------------
# 視覺化 Morlet 小波：不同尺度下的形狀
# ---------------------------------------------------------------------------
def morlet(eta, w0=6.0):
    return np.pi**-0.25 * np.exp(1j * w0 * eta) * np.exp(-eta**2 / 2)

fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
for ax, s in zip(axes, [4, 16, 64]):
    t = np.arange(-3*s, 3*s + 1)
    psi = morlet(t / s) / np.sqrt(s)
    ax.plot(t, psi.real, lw=1.2, color="#1b4965", label="實部")
    ax.plot(t, psi.imag, lw=1.0, color="#bc4749", ls="--", label="虛部")
    ax.plot(t, np.abs(psi), lw=1.0, color="#2a9d8f", label="包絡")
    ax.set_title(f"尺度 s = {s}（≈ {1.03*s:.0f} 天週期）")
    ax.set_xlabel("交易日")
axes[0].legend(fontsize=7)
fig.tight_layout()
plt.show()

print("觀察：s 越大，小波越寬 → 頻率解析越好，但時間定位越模糊。")
print("      這就是時頻不確定性原理在圖上的樣子。")
"""),

unit_header("單元 19", "小波同調、影響錐與蒙地卡羅顯著性",
            "交叉小波；小波同調（WTC）；影響錐（COI）；AR(1) 替代資料檢定",
            "06_wavelet.py",
            ["Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Application of the cross wavelet transform and wavelet coherence to geophysical time series. Nonlinear Processes in Geophysics, 11(5/6), 561–566."]),

md("""
### 從單序列到兩序列

**交叉小波轉換**：

$$
W_{xy}(s, \\tau) = W_x(s, \\tau)  W_y^*(s, \\tau)
$$

它的振幅代表「兩序列在該時點該尺度上共同的能量」，相位代表「兩者的相位差」。

### 小波同調（Wavelet Coherence, WTC）

交叉小波的振幅會被兩序列各自的波動放大，因此需要標準化。
小波同調是**時頻空間裡的局部相關係數平方**：

$$
R^2(s, \\tau) =
\\frac{\\left| S\\left(s^{-1} W_{xy}(s,\\tau)\\right)\\right|^2}
{S\\left(s^{-1}\\left|W_x(s,\\tau)\\right|^2\\right)\\cdot S\\left(s^{-1}\\left|W_y(s,\\tau)\\right|^2\\right)}
$$

其中 $S(\\cdot)$ 是**平滑運算子**（同時在時間與尺度方向平滑）。

**為什麼一定要平滑**：不平滑的話，由 Cauchy–Schwarz 不等式，
$R^2 \\equiv 1$ 處處成立 —— 兩個複數的「相關」在單點上總是完美的。
平滑引入了「鄰域」的概念，才讓 $R^2 < 1$ 有意義。

$R^2 \\in [0, 1]$：1 代表該時點該尺度上完全共動，0 代表無關。

### 影響錐（Cone of Influence, COI）

小波轉換是卷積運算，在序列的**兩端**，小波會延伸到資料之外。
實作上用補零或鏡射處理，但這會讓邊緣的係數被人為壓低。

影響錐標記出「受邊緣效應污染」的區域。對 Morlet 小波，e-折疊時間為 $\\sqrt{2}s$，
因此在距離端點 $\\sqrt{2}s$ 以內的區域不可信。

**關鍵後果**：尺度越大，受污染的區域越大。對一個 $T = 528$ 的序列，
128 天尺度的可用區間可能只剩中間一小段。

程式碼因此**只在影響錐內做頻帶平均**：

```python
for ti in range(WCT.shape[1]):
    valid_scales = mask_period & (periods < coi[ti])   # 只取錐內
    if valid_scales.any():
        vals.append(WCT[valid_scales, ti].mean())
```

> **這一行對本研究的一個結論起了決定性作用**。BUIDL–FANG 在 32–128 天頻帶的同調度是 0.350，
> 高於 IEF–GLD 的 0.272。但 BUIDL 只有 528 個觀測，扣掉影響錐後，
> 32–128 天尺度上可用的時間跨度非常窄。
>
> 論文因此不把這一格解讀成「低頻真實關聯」，而明說樣本長度限制了可信度。
> 初稿曾寫「IEF–GLD 是唯一在最長尺度仍高於 0.27 的配對」—— 被自己的表格推翻，
> 審計時修正為分別陳述並加上樣本長度的但書。

### 蒙地卡羅顯著性

$R^2$ 的分配沒有解析解。標準做法是**替代資料檢定（surrogate data test）**：

1. 估計兩序列各自的 AR(1) 參數
2. 生成 $M$ 組具有相同 AR(1) 結構的隨機序列
3. 對每組計算 WTC
4. 取逐點的 95 百分位作為顯著門檻

虛無假設是「兩序列都只是 AR(1) 紅噪音，彼此獨立」。

本研究用 `mc_count=60`（pycwt 預設為 300），程式碼註解說明了理由與代價：

```python
# mc_count reduced from the pycwt default of 300 to 60 Monte Carlo AR(1)
# surrogates to keep runtime tractable; the resulting 95% significance
# contour is a coarser but still valid approximation
```

> 60 次替代對 95 百分位的估計確實較粗糙（標準誤約 $\\sqrt{0.05 \\times 0.95 / 60} \\approx 0.028$，即 2.8 個百分點）。
> 這是可接受的取捨，因為**本研究報告的頻帶平均同調度不使用顯著性等高線** ——
> 只用同調度的量值本身。R 端的 `12_r_partial_wavelet.R` 甚至只用 `nrands = 10`，
> 註解也明確說明了同樣的理由。
"""),

code("""
# ---------------------------------------------------------------------------
# 頻帶平均同調度：讀取結果並解讀影響錐的影響
# ---------------------------------------------------------------------------
wav = pd.read_csv(PROC / "wavelet_band_summary.csv", index_col=0)
display(wav.round(3))

print("觀察：")
print(f"  1. 同調度全數落在 {wav.iloc[:, :3].values.min():.2f}–{wav.iloc[:, :3].values.max():.2f}，"
      "屬中等程度，沒有任何配對呈現壓倒性共動")
print(f"  2. OUSG 的配對在三個頻帶上異常平坦（約 0.20–0.26），無論對 FANG 還是 BTC")
print(f"  3. IEF–GLD 在短、中期最高（{wav.loc['IEF-GLD'].iloc[0]:.3f}, {wav.loc['IEF-GLD'].iloc[1]:.3f}），"
      f"長期降至 {wav.loc['IEF-GLD'].iloc[2]:.3f}")
print(f"  4. 但 BUIDL–FANG 在長期是 {wav.loc['BUIDL-FANG'].iloc[2]:.3f}，高於 IEF–GLD ——")
print(f"     BUIDL 樣本僅 {int(wav.loc['BUIDL-FANG','n_obs'])} 天，扣除影響錐後長尺度可用區間極窄，")
print("     故不解讀為真實的低頻關聯。")
"""),

code("""
# ---------------------------------------------------------------------------
# 影響錐的幾何：為什麼短樣本在長尺度上不可信
# ---------------------------------------------------------------------------
def coi_morlet(T, dt=1.0):
    \"\"\"Morlet 小波的影響錐：距端點 sqrt(2)*s 內不可信。\"\"\"
    t = np.arange(T)
    edge = np.minimum(t, T - 1 - t) * dt
    return edge / np.sqrt(2)          # 該時點可信的最大尺度

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
for ax, (T, label) in zip(axes, [(528, "BUIDL（528 天）"), (906, "IEF（906 天）")]):
    c = coi_morlet(T)
    ax.fill_between(np.arange(T), c, 128, color="#d9d9d9", label="影響錐外（不可信）")
    ax.plot(np.arange(T), c, color="#bc4749", lw=1.2)
    ax.axhspan(32, 128, color="#1b4965", alpha=0.12, label="32–128 天頻帶")
    ax.set_ylim(2, 128); ax.set_yscale("log")
    ax.set_title(label); ax.set_xlabel("交易日")
    usable = (c > 32).mean()
    ax.text(0.5, 0.06, f"32 天以上尺度可用比例：{usable:.0%}",
            transform=ax.transAxes, ha="center", fontsize=8,
            bbox=dict(fc="white", ec="grey", alpha=0.85))
axes[0].set_ylabel("尺度（天）"); axes[0].legend(fontsize=7, loc="upper right")
fig.tight_layout()
plt.show()
"""),

unit_header("單元 20", "偏小波同調：控制共同因子",
            "偏小波同調（PWC）；混淆因子的頻域控制",
            "12_r_partial_wavelet.R",
            ["Mihanović, H., Orlić, M., & Pasarić, Z. (2009). Diurnal thermocline oscillations driven by tidal flow around an island in the Middle Adriatic. Journal of Marine Systems, 78, S157–S168.",
             "Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004)."]),

md("""
### 原理：原始同調度無法區分兩種情況

假設我們觀察到代幣化國債與科技股之間有中等的小波同調度。這可能來自兩種完全不同的機制：

**機制一（直接關聯）**：代幣被當作 DeFi 抵押品，其清算壓力直接傳導到風險資產

**機制二（共同因子）**：兩者都對**同一個利率消息**反應
- 代幣化國債的收益率幾乎機械式地跟隨聯邦資金利率
- 成長股的估值透過折現率對利率高度敏感

機制二下，兩者之間**沒有任何直接通道**，但同調度依然會很高。
原始同調度無法區分兩者 —— 這是本研究必須處理的核心識別問題。

### 偏小波同調的公式

偏小波同調把控制變數 $x_2$（本研究用 3 個月期國庫券殖利率變動）的影響移除：

$$
RP^2(y, x_1, x_2) =
\\frac{\\left| R(y, x_1) - R(y, x_2) \\cdot R(x_1, x_2)^* \\right|^2}
{\\left[1 - R(y, x_2)^2\\right]\\left[1 - R(x_1, x_2)^2\\right]}
$$

其中 $R(\\cdot, \\cdot)$ 是複數的小波同調。

**與偏相關係數的類比**：普通偏相關的公式為

$$
\\rho_{y x_1 \\cdot x_2} = \\frac{\\rho_{y x_1} - \\rho_{y x_2}\\rho_{x_1 x_2}}
{\\sqrt{(1-\\rho_{y x_2}^2)(1-\\rho_{x_1 x_2}^2)}}
$$

PWC 就是把它逐點搬到時頻平面上，並處理複數。

### 衰減量的解讀

定義**衰減量（attenuation）**：

$$
\\text{attenuation} = \\mathrm{WTC} - \\mathrm{PWC}
$$

- **衰減大** → 原始同調度多半是共同因子造成的假象
- **衰減小** → 存在直接或流動性驅動的關聯，不是利率的副產品

### 本研究的結果與一個漂亮的內部驗證

實證結果有三層：

1. **控制利率後，每個配對、每個頻帶的同調度都下降**（衰減 0.03–0.13）
   → 部分共動確實是共同因子假象

2. **但同調度沒有消失**（PWC 仍有 0.14–0.40）
   → 存在超越利率因子的直接關聯

3. **衰減最大的是 IEF–GLD（0.127），不是任何代幣配對**

第 3 點是一個**內部一致性檢驗**，值得停下來欣賞。IEF–GLD 在其他所有分析裡
（DCC、結構斷點、機制切換）都被判定為「真實的、利率驅動的資金避風港關係」。
如果 PWC 的邏輯正確，移除利率因子就應該**對它衰減最多**。結果確實如此。

這種「方法在已知答案的案例上給出預期結果」的驗證，比任何模型配適統計量都有說服力。

> **一個被修正的過度宣稱**：初稿寫「整張表中最大的衰減屬於 IEF–GLD（0.10–0.13）」。
> 實際上 sUSDS–FANG 在兩個頻帶的衰減（0.113、0.104）超過 IEF–GLD 的 0.102、0.110。
> 只有**單一最大值**（0.127）確實屬於 IEF–GLD。
> 修正後改為：單一最大值屬 IEF–GLD，且它是唯一三個頻帶衰減皆超過 0.10 的配對。
"""),

code("""
# ---------------------------------------------------------------------------
# WTC vs PWC：衰減量與內部一致性檢驗
# ---------------------------------------------------------------------------
pw = pd.read_csv(PROC / "r_partial_wavelet.csv")
display(pw.round(4))

print("三層結果：")
print(f"  1. 衰減量全為正，範圍 {pw['attenuation'].min():.3f}–{pw['attenuation'].max():.3f}"
      "  → 部分共動確為共同因子假象")
print(f"  2. PWC 仍在 {pw['PWC'].min():.3f}–{pw['PWC'].max():.3f}  → 直接關聯未消失")
top = pw.loc[pw["attenuation"].idxmax()]
print(f"  3. 單一最大衰減：{top['pair']} 於 {top['band']}，衰減 = {top['attenuation']:.4f}")
print()
allband = pw.groupby("pair")["attenuation"].min().sort_values(ascending=False)
print("各配對的『最小頻帶衰減』（判斷哪個配對三頻帶皆強衰減）：")
print(allband.round(4).to_string())
print(f"\\n→ 只有 {allband.index[0]} 的三個頻帶衰減皆 > 0.10，符合『利率驅動』的預期。")
"""),

code("""
# ---------------------------------------------------------------------------
# 視覺化：控制利率因子前後的同調度
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 3.6))
pairs = pw["pair"].unique()
x = np.arange(len(pairs) * 3)
labels, wtc_v, pwc_v = [], [], []
for p in pairs:
    sub = pw[pw["pair"] == p]
    for _, r in sub.iterrows():
        labels.append(f"{p}\\n{r['band'].split('(')[0].strip()}")
        wtc_v.append(r["WTC"]); pwc_v.append(r["PWC"])

ax.bar(x - 0.2, wtc_v, width=0.4, label="WTC（原始）", color="#1b4965")
ax.bar(x + 0.2, pwc_v, width=0.4, label="PWC（控制 3M 利率）", color="#f4a261")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=6)
ax.set_ylabel("同調度")
ax.set_title("控制短率因子後同調度普遍下降，但未歸零")
ax.legend()
fig.tight_layout()
plt.show()
"""),

]
