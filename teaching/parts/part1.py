# -*- coding: utf-8 -*-
"""第 1 篇：資料工程（單元 1–3），對應 01_fetch_data.py 與 02_build_returns.py。"""

from ._helpers import md, code, unit_header

CELLS = [

md("""
---
---

# 第 1 篇　資料工程

實證論文的可信度上限，在資料建構階段就被決定了。這一篇的三個單元處理三個常被草率帶過、
但足以讓整篇結論失效的問題：**資料從哪裡來且能不能重現**、**不同交易日曆的序列如何對齊**、
**極端值該不該處理以及怎麼處理才不算操縱**。
"""),

unit_header("單元 1", "資料抓取與可重現性原則",
            "無（資料工程），但決定了後續所有統計推論的效力",
            "01_fetch_data.py",
            ["Yahoo Finance / FRED (fredgraph.csv) / DefiLlama 公開 API"]),

md("""
### 原理

計量方法課很少花時間在這裡，但**資料層的錯誤無法被下游任何統計方法修正**。本管線遵循三條原則：

**原則一：原始拉取一律逐字存檔**

每一次 API 回應都原封不動寫進 `data/raw/`，包含 DefiLlama 的原始 JSON。
理由是**可稽核性**：當三個月後某個數字看起來不對，你需要能回答「是抓取時就這樣，還是我後續處理弄壞的」。
如果只保留處理後的結果，這個問題永遠無解。

**原則二：失敗就記錄並留白，絕不回填**

```python
except Exception as e:
    log(f"[yfinance] {name} ({ticker}): ERROR {e}")
```

抓不到的序列會被記錄在 `_fetch_log.txt` 然後留空。**不做插補、不用鄰近值回填、不編造**。
留白會讓下游的樣本數變小（誠實地反映資訊不足），回填則會製造出不存在的統計檢定力。

**原則三：每一個代號都要親眼確認，不能望文生義**

這條原則在本專案救回了一個會毀掉整篇論文的錯誤。
""")
,

md("""
### 一個真實的識別碼陷阱

Yahoo Finance 上的 `UNI-USD` **不是** Uniswap。

| 代號 | 實際標的 | 價格量級 | 成交量 |
|---|---|---|---|
| `UNI-USD` | 某個 2025 年 4 月已下架的無關代幣 | ~\\$0.0001 | 接近零 |
| `UNI7083-USD` | Uniswap 治理代幣（CoinMarketCap ID 消歧義） | ~\\$3–6 | 全樣本連續 |

如果直接用 `UNI-USD`，論文裡「DeFi 治理代幣」這整條分析線會建立在一個死掉的殭屍代幣上，
而且**所有統計檢定都會正常執行、正常輸出結果、看起來完全合理**。
沒有任何一個下游的診斷統計量會警告你資料抓錯了。

程式碼裡因此留下了這段註解 —— 這是研究筆記，不是廢話：

```python
"UNI": "UNI7083-USD",  # Uniswap governance token, CMC-ID-disambiguated ticker: the
                       # plain "UNI-USD" symbol on Yahoo Finance resolves to an
                       # unrelated, near-worthless, delisted-since-2025-04 token
                       # (price ~$0.0001, near-zero volume) -- confirmed by manual
                       # inspection before use, not assumed from the ticker name
```

> **教學重點**：對每一個新的資料來源，做一次「這個序列的量級、起訖日、成交量是否符合我對這個標的的先驗認知」
> 的檢查。這個動作花五分鐘，但它防的是那種**不會報錯、只會給你錯誤答案**的失敗。
"""),

md("""
### 程式邏輯：三種資料來源的不同處理

| 來源 | 取得方式 | 頻率 | 對齊時的注意事項 |
|---|---|---|---|
| Yahoo Finance | `yfinance.download` | 交易日 | 可能回傳 MultiIndex 欄位，需攤平 |
| FRED | `fredgraph.csv?id=...` 公開端點，免金鑰 | 交易日，但**發布間為階梯函數** | 必須用 `ffill`，不可內插 |
| DefiLlama | `api.llama.fi` / `stablecoins.llama.fi` / `yields.llama.fi` | 每日，含盤中快照 | 同一日可能多筆，取當日最後一筆 |

FRED 的利率序列為什麼只能 `ffill` 而不能線性內插？因為**利率在兩次公布之間確實維持不變**，
它是階梯函數而非連續變數的離散取樣。線性內插會製造出不存在的每日微小變動，
進而讓後續的 `diff()` 產生假的利率變動序列。
"""),

code("""
# ---------------------------------------------------------------------------
# 檢視抓取紀錄：可重現性的第一道證據
# ---------------------------------------------------------------------------
log_path = RAW / "_fetch_log.txt"
if log_path.exists():
    lines = log_path.read_text(encoding="utf-8").splitlines()
    print(f"抓取紀錄共 {len(lines)} 行，前 12 行：\\n")
    for l in lines[:12]:
        print("  ", l)
    errs = [l for l in lines if "ERROR" in l or "EMPTY" in l]
    print(f"\\n失敗/空值紀錄：{len(errs)} 筆")
    for e in errs:
        print("  ", e)
else:
    print("找不到抓取紀錄（需先執行 01_fetch_data.py）")
"""),

code("""
# ---------------------------------------------------------------------------
# 示範原則三：檢查 UNI 序列的量級是否符合先驗
# ---------------------------------------------------------------------------
uni = pd.read_csv(RAW / "yf_UNI.csv", parse_dates=["Date"]).set_index("Date")["UNI"]

print("UNI（UNI7083-USD）價格檢查")
print(f"  期間      : {uni.index.min().date()} → {uni.index.max().date()}")
print(f"  觀測數    : {len(uni)}")
print(f"  價格區間  : {uni.min():.4f} – {uni.max():.4f}")
print(f"  最新價格  : {uni.iloc[-1]:.4f}")
print()
print("先驗判斷：Uniswap 治理代幣在 2023–2026 應在個位數美元量級、且全樣本連續。")
print("結論：", "✓ 符合" if 1 < uni.median() < 20 and len(uni) > 800 else "✗ 不符，須重新確認代號")
"""),

unit_header("單元 2", "交易日對齊、對數報酬與雙軌報酬建構",
            "對數報酬轉換；不同交易日曆的序列對齊；代理變數設計",
            "02_build_returns.py",
            ["標準時間序列金融計量實務"]),

md("""
### 原理一：為什麼用對數報酬

給定價格 $P_t$，兩種報酬定義：

$$
R_t^{\\text{簡單}} = \\frac{P_t - P_{t-1}}{P_{t-1}}, \\qquad
R_t^{\\text{對數}} = \\ln\\frac{P_t}{P_{t-1}}
$$

選對數報酬有三個理由，其中第三個對本研究特別關鍵：

1. **時間可加性**：$k$ 期的累積對數報酬等於各期對數報酬之和，
   $\\ln(P_{t+k}/P_t) = \\sum_{j=1}^{k} r_{t+j}$。簡單報酬需要連乘。
2. **對稱性**：+50% 之後 −50% 不會回到原點（簡單報酬），但 $+\\ln 1.5$ 與 $-\\ln 1.5$ 會相消。
3. **與 GARCH 族的分配假設相容**：GARCH 假設條件分配（本研究用偏態 $t$）定義在實數線上。
   簡單報酬有 −100% 的下界，對數報酬沒有。

程式中乘上 100 轉成百分點：

```python
def log_ret(series, mult=100):
    return mult * np.log(series / series.shift(1))
```

> **這個 ×100 在單元 25 會變成一個真實錯誤的根源** —— 交易成本用基點計算時，
> 若忘記報酬已經是百分點單位，除數會差 100 倍。先記住這裡。
"""),

md("""
### 原理二：主交易日曆與對齊策略

本研究混合了三種交易日曆：

- **美股**：週一至週五，扣除美國假日 ≈ 252 天/年
- **加密貨幣**：365 天/年，全年無休
- **鏈上 TVL**：365 天/年，但 DefiLlama 的快照時點不固定

設計選擇：**以美股交易日為主日曆**（取自 NYFANG），其他序列 reindex 到這個日曆上。

```python
calendar = nyfang.index.sort_values()
prices[label] = s.reindex(calendar).ffill(limit=2)
```

`limit=2` 是刻意的：允許最多補兩天（涵蓋一般假日），但不允許把長期缺漏靜靜補滿。
超過兩天的缺口會留成 `NaN`，讓下游的 `dropna()` 誠實地縮減樣本。

**為什麼不用加密日曆？** 因為研究問題是「代幣化國債對**股票**的關聯」，
而股票在週末沒有價格。若採 365 天日曆，週末的股票報酬會是人造的零，
這會系統性地把相關係數往零的方向拉 —— 剛好是我們想檢定的方向，構成偏誤。
"""),

md("""
### 原理三：雙軌報酬 —— 本研究最重要的設計決策

代幣化國債沒有免費、每日、無金鑰的 NAV 報價。本研究因此把每個代幣的報酬拆成兩軌：

**基本面軌（NAV track）** —— 現金流通道

$$
r^{\\text{nav}}_{t} = \\frac{y_t / 100}{365} \\times 100
$$

其中 $y_t$ 是年化收益率（百分點）。對 BUIDL 與 OUSG，$y_t$ 用 3 個月期美國國庫券殖利率
（FRED `DGS3MO`）代理；對 sUSDS，$y_t$ 是**實際回報的 Sky Savings Rate**。

**市值軌（market-cap track）** —— 流動性與資金流通道

$$
r^{\\text{mkt}}_{t} = 100 \\times \\ln\\frac{\\mathrm{TVL}_t}{\\mathrm{TVL}_{t-1}}
$$

**流動性溢價代理**

$$
\\mathrm{LIQ}_t = r^{\\text{mkt}}_{t} - r^{\\text{nav}}_{t}
$$

### 這個設計的限制必須誠實說明

TVL 的變動同時混合了兩件事：既有持有部位的**估值變動**，以及**淨申購／贖回的資金流**。
交易層級資料（Dune、Etherscan）可以逐筆分離兩者，但需要付費 API。
本研究因此用兩軌之差作為較粗糙的流動性代理，並在論文中明確揭露這個限制，
而不是把它包裝成比實際更精確的東西。

> **一個必須揭露的後果**：BUIDL 與 OUSG 的 NAV 軌都用同一個 3 個月期國庫券殖利率代理，
> 因此 `R_BUIDL_nav` 與 `R_OUSG_nav` **在建構上是完全相同的序列**。
> 描述統計表中這兩列的數字會一模一樣。任何基於 NAV 軌的結果都**無法區分這兩個產品**。
> 這一點必須寫進表格註解，否則審稿人會以為是複製貼上的錯誤。
"""),

code("""
# ---------------------------------------------------------------------------
# 驗證雙軌報酬的建構，以及 NAV 軌的同一性
# ---------------------------------------------------------------------------
tracks = panel[["R_BUIDL_nav", "R_BUIDL_mkt", "LIQ_BUIDL",
                "R_OUSG_nav", "R_OUSG_mkt", "LIQ_OUSG",
                "R_sUSDS_nav", "R_sUSDS_mkt"]].describe().T[["count", "mean", "std", "min", "max"]]
display(tracks)

same = (panel["R_BUIDL_nav"].dropna() - panel["R_OUSG_nav"].dropna()).abs().max()
print(f"\\nBUIDL 與 OUSG 的 NAV 軌最大絕對差異：{same:.2e}")
print("→ 兩軌完全相同（同一個 DGS3MO 代理），必須在表格註解揭露")
print()
print("對照：sUSDS 的 NAV 軌用真實 Sky Savings Rate，標準差 = "
      f"{panel['R_sUSDS_nav'].std():.6f}，與 BUIDL 的 {panel['R_BUIDL_nav'].std():.6f} 不同")
"""),

code("""
# ---------------------------------------------------------------------------
# 視覺化：基本面軌 vs 市值軌的量級差異
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

sub = panel.loc[panel["R_OUSG_mkt"].notna()]
axes[0].plot(sub.index, sub["R_OUSG_nav"], lw=0.9, color="#1b4965")
axes[0].set_title("OUSG 基本面軌（3M 國庫券日累積）— 幾乎是平滑的確定性序列")
axes[0].set_ylabel("% / 日")

axes[1].plot(sub.index, sub["R_OUSG_mkt"], lw=0.6, color="#bc4749")
axes[1].set_title("OUSG 市值軌（TVL 對數變動）— 大幅離散跳動，反映機構申贖")
axes[1].set_ylabel("% / 日")

fig.tight_layout()
plt.show()

print(f"基本面軌標準差：{sub['R_OUSG_nav'].std():.4f}")
print(f"市值軌標準差  ：{sub['R_OUSG_mkt'].std():.4f}  （約 {sub['R_OUSG_mkt'].std()/sub['R_OUSG_nav'].std():.0f} 倍）")
"""),

unit_header("單元 3", "Winsorization 與極端值處理",
            "尾部截尾（winsorization）；資料清理規則的事前設定",
            "02_build_returns.py",
            ["Mafrur (2025), Tokenize everything, but can you sell it? (arXiv:2508.11651)"]),

md("""
### 原理

Winsorization 把超過某個分位數的觀測值**壓縮到該分位數**，而非刪除：

$$
\\tilde{r}_t = \\begin{cases}
q_{\\alpha} & \\text{if } r_t < q_{\\alpha} \\\\
r_t & \\text{if } q_{\\alpha} \\le r_t \\le q_{1-\\alpha} \\\\
q_{1-\\alpha} & \\text{if } r_t > q_{1-\\alpha}
\\end{cases}
$$

與**截斷（truncation，直接刪除）**的關鍵差別：winsorization **保留樣本數與時間序列的連續性**。
對 GARCH 這類需要連續遞迴的模型，中間挖洞是災難性的 —— 條件變異數的遞迴會斷掉。

### 為什麼需要它

GARCH 族的概似函數對極端值極度敏感。單一個 15 個標準差的觀測值可以：
- 把 $\\omega$（常數項）推高到吸收掉整個尾部
- 讓 $\\alpha + \\beta$ 逼近 1（假的高持續性）
- 或反過來把參數推到邊界解

### 本研究的兩級門檻設計

| 序列類型 | 門檻 | 理由 |
|---|---|---|
| 傳統資產報酬 | 0.5% / 99.5% | 標準實務 |
| 代幣市值軌、LIQ | 1% / 99% | **更緊** |

代幣市值軌用更緊的門檻，理由不是「讓結果好看」，而是有明確的經濟學依據：
DefiLlama 的 TVL 序列存在大幅離散跳動，對應**機構的整筆申購／贖回**。
單一機構一次投入數億美元，會在 TVL 序列上製造一個純粹反映單筆交易時點的巨大「報酬」，
這不是市場定價資訊。Mafrur (2025) 記錄了代幣化 RWA 次級市場即使在總量龐大時仍然交投稀薄，
正是這種跳動的來源。

> **研究倫理的界線在哪裡**
>
> Winsorization 很容易變成 p-hacking 的工具：試幾組門檻，選出最顯著的那組。
> 本研究把這條界線畫在三個地方：
> 1. 門檻**事前設定**，寫在資料建構腳本裡，不在看到結果後調整
> 2. 兩級門檻的差異有**獨立於結果的經濟學理由**（機構申贖 vs 市場定價）
> 3. 在論文中**明確揭露**，並說明這是資料品質控制而非製造顯著性的手段
>
> 一個實用的自我檢查：如果你調整了門檻並發現結論改變了，正確的做法是**報告兩組結果**，
> 而不是選一組。
"""),

md("""
### 額外的清理規則：OUSG 孵化期截斷

OUSG 於 2023-01-27 上線，但最初約一週半的 TVL 低於 1,000 萬美元。
在那個窗口內，TVL 從 135 美元成長到超過 1,000 萬美元 —— 對數變動會產生數百個百分點的「報酬」。

這**不是報酬，是小分母假象**。程式因此丟棄 TVL 首次超過 1,000 萬美元之前的觀測：

```python
INCUBATION_CUTOFF = ousg_tvl_c[ousg_tvl_c > 10_000_000].index.min()
for c in ["R_OUSG_mkt", "LIQ_OUSG"]:
    returns.loc[returns.index < INCUBATION_CUTOFF, c] = np.nan
```

門檻（1,000 萬美元）與規則（首次跨越）都是事前設定的、可驗證的，並在論文中揭露為資料清理規則。
"""),

code("""
# ---------------------------------------------------------------------------
# Winsorization 的效果：從零實作，對照兩級門檻
# ---------------------------------------------------------------------------
def winsorize(s, lo_q, hi_q):
    lo, hi = s.quantile(lo_q), s.quantile(hi_q)
    return s.clip(lo, hi)

# 用尚未 winsorize 的原始 TVL 重建一次，以展示差異
ousg_tvl = pd.read_csv(RAW / "defillama_OUSG_tvl.csv", parse_dates=["date"]).set_index("date")["OUSG_TVL"]
ousg_tvl = ousg_tvl.groupby(ousg_tvl.index.normalize()).last()
raw_ret = 100 * np.log(ousg_tvl / ousg_tvl.shift(1))
raw_ret = raw_ret[raw_ret.index >= "2023-02-06"].dropna()   # 已過孵化期

comp = pd.DataFrame({
    "原始": raw_ret.describe(),
    "0.5%/99.5%": winsorize(raw_ret, 0.005, 0.995).describe(),
    "1%/99%（本研究採用）": winsorize(raw_ret, 0.01, 0.99).describe(),
}).loc[["std", "min", "max"]]
comp.loc["峰態"] = [raw_ret.kurtosis(),
                    winsorize(raw_ret, 0.005, 0.995).kurtosis(),
                    winsorize(raw_ret, 0.01, 0.99).kurtosis()]
display(comp)

print("觀察：更緊的門檻把峰態從極端值壓回可用範圍，但樣本數完全不變 ——")
print("      這正是 winsorization 相對於截斷的優勢，GARCH 的遞迴不會被打斷。")
"""),

]
