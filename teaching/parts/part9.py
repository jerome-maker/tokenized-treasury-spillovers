# -*- coding: utf-8 -*-
"""第 9 篇：研究誠信（單元 26）—— 本專案投稿前審計實際發現的錯誤。"""

from ._helpers import md, code, unit_header

CELLS = [

md("""
---
---

# 第 9 篇　研究誠信

計量方法課通常只教「怎麼做對」。這一篇教的是另外一半：**做錯了長什麼樣、要怎麼抓出來**。

以下全部是這個專案**投稿前審計時真實發現**的問題，不是為了教學而編造的例子。
每一個都通過了作者本人的檢查、程式沒有報錯、輸出看起來完全合理。
"""),

unit_header("單元 26", "四類錯誤的解剖",
            "計算錯誤、覆蓋率缺口、陳述與資料不符、產出過時",
            "全部管線",
            ["本專案 CHANGELOG.md v1.1.0"]),

md("""
## 錯誤一：三重複合的量綱錯誤（交易成本）

### 症狀

修正前的每一列，成本調整後的避險效能都**高於**未調整值：

```
BUIDL->FANG : HE = -0.0014569,  HE_cost_10bp = -0.0014544   ← 成本讓效能變好？
OUSG->FANG  : HE = -0.0047869,  HE_cost_10bp = -0.0047837   ← 同樣
sUSDS->FANG : HE = +0.0003922,  HE_cost_10bp = +0.0003956   ← 同樣
```

**加上成本之後結果變好，在經濟上不可能。** 這是最明確的警訊。

### 原始程式碼

```python
c = bps / 10000.0
r_hedged_cost = r_hedged - c * turnover * combined["r_hedge"].abs()
var_hedged_cost = r_hedged_cost.var()
HE_cost = 1 - var_hedged_cost / var_unhedged
```

### 三個獨立的錯誤疊在一起

| # | 錯誤 | 後果 |
|---|---|---|
| 1 | `bps / 10000` 把基點轉成**小數**，但報酬序列是 `*100` 的**百分點** | 成本小了 **100 倍** |
| 2 | 多乘一個 `|r_hedge|` | 量綱錯誤；且 $\\lvert r\\rvert \\approx 2$–3 使量級再偏 |
| 3 | 把成本折進**變異數比值** | 成本是水準效應，不是變異數效應（見單元 25） |

錯誤 1 與 2 讓成本小到在小數點後四位看不見；
錯誤 3 讓那一點點殘留的效果**方向相反** ——
從報酬中減去一個與報酬正相關的小量，可能反而**降低**變異數。

### 為什麼沒被發現

- 程式**沒有報錯**
- 數字**看起來合理**（都在 $10^{-3}$ 量級）
- 論文的結論（「代幣沒有勝過 IEF」）**碰巧不受影響**
- 表格只報告 10bp 一欄，5bp 與 20bp 沒印出來，看不出成本幾乎沒作用

### 抓出來的方法

**單調性檢查**：成本越高，效能必須越差。這是一個不需要知道正確答案就能做的檢查。
""")
,

code("""
# ---------------------------------------------------------------------------
# 單調性檢查：一個不需要知道正確答案就能做的診斷
# ---------------------------------------------------------------------------
h = pd.read_csv(PROC / "hedge_summary.csv")

print("修正後：成本隨 bp 假設單調遞增（必要條件）")
cost_cols = ["cost_bp_ann_5bp", "cost_bp_ann_10bp", "cost_bp_ann_20bp"]
sub = h[["hedged_equity", "hedge_instrument"] + cost_cols].head(6)
display(sub.round(2))

ok = ((h[cost_cols[0]] <= h[cost_cols[1]]) & (h[cost_cols[1]] <= h[cost_cols[2]])).all()
print(f"所有列皆滿足 cost(5bp) <= cost(10bp) <= cost(20bp)：{ok}")
print()

print("淨報酬檢查：成本必須侵蝕平均報酬")
net_ok = (h["mean_ret_net_10bp"] <= h["mean_ret_hedged"] + 1e-12).all()
print(f"所有列皆滿足 mean_ret_net_10bp <= mean_ret_hedged：{net_ok}")
print()
print("修正前若跑這兩個檢查，第二項會立刻失敗 —— 但當時根本沒有 mean_ret_* 欄位可查。")
print("教訓：把中間量也寫進輸出，診斷才有材料可用。")
"""),

md("""
## 錯誤二：靜默的覆蓋率缺口（Bai–Perron 少了 3 組配對）

### 症狀

表 11、表 12 只有 31 列，但 DCC 階段產生了 **34** 組配對。

### 原因

R 腳本用萬用字元掃描檔案：

```r
dcc_files <- list.files(proc_dir, pattern = "^dcc_.*\\\\.csv$", full.names = TRUE)
```

它被執行的時間點，早於 `dcc_IEF_ETH.csv`、`dcc_SHV_BTC.csv`、`dcc_SHV_ETH.csv`
這三個檔案被寫出的時間。腳本**只處理了當時存在的 31 個檔案，正常結束，沒有任何警告**。

### 後果

漏掉的 IEF–ETH 有 **3 個結構斷點、supF = 28.7、$p = 3\\times10^{-6}$** ——
這是一個完全遺失的實質結果，而且屬於論文論述倚重的傳統工具對照組。

### 修正：覆蓋率守衛

```r
expected <- sub("^dcc_", "", tools::file_path_sans_ext(basename(dcc_files)))
missing_bp <- setdiff(expected, bp_df$pair)
if (length(missing_bp) || length(missing_bo)) {
  stop(sprintf("coverage guard: %d/%d pairs ... Missing: %s ...", ...))
}
```

**設計原則：寧可大聲失敗，也不要安靜地寫出不完整的表格。**

> 守衛上線的第一次執行就抓到了另一個潛在問題：`dcc_adcc_summary.csv` 符合萬用字元
> 但不是配對檔。它原本會被當成配對嘗試處理（因無 `rho_dcc` 欄而被跳過），
> 現在被明確排除。一個好的斷言常常會抓到你原本沒在找的東西。
"""),

md("""
## 錯誤三：陳述與自己的資料不符

這一類最難抓，因為**沒有任何程式會報錯** —— 錯的是論文正文裡的句子。
審計時共發現六處：

| # | 論文初稿的陳述 | 資料實際顯示 |
|---|---|---|
| 1 | 「兩個 ETF 基準都明顯拒絕常數相關」 | SHV–FANG $p = 1.00$、SHV–SOXX $p = 0.31$，**只有 IEF 拒絕** |
| 2 | 「OUSG 是唯一與治理代幣相關性達顯著的代幣」 | $p = 0.342$ 與 $0.595$，**兩個都不顯著** |
| 3 | 「兩個估計量在**每一個**變數的溢出符號上都一致」 | 12 個中 **10 個**一致，IEF 與 OUSG 相反 |
| 4 | 頻域「約 41 / 3 / 不到 1」 | 實為 **37.6 / 5.8 / 2.0** |
| 5 | 「IEF–GLD 在**每個**頻帶都最高」 | BUIDL–FANG 在最長頻帶 0.350 > 0.272 |
| 6 | 「三個代幣的不對稱**都**與股票相反」 | sUSDS 的 $\\gamma = -0.60$，與股票**同向** |

### 共同的型態

六處全部都是**全稱量詞**：「每一個」、「都」、「唯一」、「全部」。

> **一條實用的寫作規則**：每當你寫下「所有 X 都具有性質 P」，
> 回頭把表格的每一列對過一遍。「大部分」與「全部」在審稿人眼中是完全不同的宣稱，
> 而後者只要一個反例就整句失效。

### 抓出來的方法：機器可驗證的宣稱

把論文中的每一個數字寫成可執行的斷言，對照來源檔案跑一遍。
""")
,

code("""
# ---------------------------------------------------------------------------
# 把論文宣稱寫成可執行的斷言：稿件數字的自動驗證
# ---------------------------------------------------------------------------
import json

T = ROOT.parent / "iaj_submission" / "tables_data"
def J(name):
    if not (T / f"{name}.json").exists():
        return None
    return {r[0]: r for r in json.loads((T / f"{name}.json").read_text(encoding="utf-8"))["rows"]}

dcc_t, garch_t = J("table_dcc_full"), J("table_garch")
tv_t, st_t = J("table_tvpvar"), J("table_spillover_C")

checks = []
if dcc_t:
    rej = [k for k, v in dcc_t.items() if float(v[8]) < 0.05]
    checks.append(("拒絕常數相關的配對全屬 IEF", rej == ["IEF-FANG", "IEF-SOXX", "IEF-GLD"], rej))
    checks.append(("SHV 對科技股皆不拒絕",
                   float(dcc_t["SHV-FANG"][8]) > 0.05 and float(dcc_t["SHV-SOXX"][8]) > 0.05,
                   f"p = {dcc_t['SHV-FANG'][8]}, {dcc_t['SHV-SOXX'][8]}"))
    checks.append(("OUSG 對治理代幣皆不顯著",
                   float(dcc_t["OUSG-UNI"][8]) > 0.05 and float(dcc_t["OUSG-AAVE"][8]) > 0.05,
                   f"p = {dcc_t['OUSG-UNI'][8]}, {dcc_t['OUSG-AAVE'][8]}"))
if tv_t and st_t:
    agree = sum(1 for k in tv_t if (float(tv_t[k][3]) > 0) == (float(st_t[k][3]) > 0))
    checks.append(("兩估計量符號一致數 = 10（非 12）", agree == 10, f"{agree}/12"))
if garch_t:
    checks.append(("sUSDS 的 EGARCH gamma 為負（與股票同向）",
                   float(garch_t["sUSDS"][4]) < 0, garch_t["sUSDS"][4]))

fr = pd.read_csv(PROC / "r_frequency_tci_series.csv").groupby("band")["TCI"].mean().round(1)
checks.append(("頻帶分解為 37.6 / 5.8 / 2.0",
               (fr["1-5"], fr["5-20"], fr["20-Inf"]) == (37.6, 5.8, 2.0),
               f"{fr['1-5']} / {fr['5-20']} / {fr['20-Inf']}"))

for name, ok, detail in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38} {detail}")
print()
print("這種斷言腳本應該在投稿前跑一次，並在每次重跑管線後再跑一次。")
"""),

md("""
## 錯誤四：產出檔案過時

### 症狀

`tables/table_descriptive.tex` 顯示 $n = 907$，但 `returns_panel.csv` 只支援 $n = 906$。

### 原因

`03_pretests.py` 的輸出是用**更早一版**的報酬面板產生的，之後面板重跑過，
但前置檢定沒有跟著重跑。GARCH 以後的所有階段都用了新面板，
只有前置檢定的表格停留在舊版。

### 為什麼危險

論文的表 1（描述統計）與表 3（GARCH 參數）會出現**互相矛盾的樣本數**，
而這正是審稿人最容易注意到的那種不一致。

### 結構性的成因與修正

真正的問題是**管線沒有相依關係管理**。修正包含三部分：

1. 修正 `parse_tables.py` 與 `build_excel.py` 中指向已不存在目錄的硬編碼路徑
2. 在 README 明確標示 Python 階段必須在 R 階段之前執行
3. 加入覆蓋率守衛（錯誤二）

> **更根本的解法**是把管線改成 Make / Snakemake / DVC 這類有相依圖的工具，
> 讓「輸入變了，輸出自動失效」由工具保證，而不是靠人記得。
> 對研究生的專案而言，這個投資通常在第二次重跑資料時就回本了。
"""),

md("""
## 總結：四類錯誤與對應的防禦

| 錯誤類型 | 為何難以察覺 | 防禦手段 |
|---|---|---|
| **量綱／符號錯誤** | 程式不報錯，數字量級看似合理 | **單調性與符號的必要條件檢查**；把中間量寫進輸出 |
| **覆蓋率缺口** | 迴圈正常結束，只是少跑了 | **斷言式守衛**：預期 $N$ 筆，少一筆就中止 |
| **陳述與資料不符** | 錯的是散文，不是程式 | **機器可驗證的宣稱**：每個數字寫成斷言 |
| **產出過時** | 每個檔案單獨看都正確 | **相依圖管理**（Make/DVC）；或至少交叉檢查樣本數 |

### 三條可以帶走的原則

**一、寧可大聲失敗**

安靜的降級（silent degradation）是實證研究最大的敵人。
一個中止執行的腳本會逼你面對問題；一個少跑三組配對卻正常結束的腳本不會。

**二、寫下不需要知道正確答案的檢查**

「成本越高，效能越差」、「三個頻帶加總等於總計」、「每列機率加總為 1」——
這類**必要條件**不需要你事先知道答案，卻能抓到大部分的計算錯誤。

**三、正文的每個數字都要有一條可追溯的路徑**

如果你無法在三十秒內指出論文某個數字來自哪個檔案的哪一欄，
那個數字就有風險。手動轉錄是錯誤的溫床 ——
本專案的表格全部由 `tex` 檔自動解析成 JSON 再寫進 Word，
正是為了消除這條路徑上的人為介入。
"""),

md("""
---

## 結語

這份教材涵蓋了從資料抓取到避險評估的完整流程，26 個單元對應 26 個統計技術。
但若只能記住一件事，我希望是最後這一篇的內容。

模型可以查書、公式可以推導、程式可以複製。
**判斷自己的結果是否可信，是唯一沒有捷徑的能力**，
而它只能透過反覆地、有系統地懷疑自己的輸出來培養。

那四個錯誤全部通過了作者本人的檢查。它們被抓出來，
不是因為有人更聰明，而是因為有人**列了一張清單，逐條對過**。

---

*本教材對應論文投稿於 Investment Analysts Journal。*
*完整管線、資料與可重現腳本見本 repository；錯誤修正紀錄見 `CHANGELOG.md`。*
"""),

]
