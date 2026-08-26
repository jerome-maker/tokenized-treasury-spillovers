# 研究方法完整教材

**代幣化國債的跨資產波動關聯：從資料抓取到避險評估的計量流程拆解**

這份教材把本 repository 的完整實證管線，依照**研究流程**與**流程中所應用的統計技術**
拆解成 26 個教學單元，供研究所計量財務、時間序列分析課程使用。

📓 **[research_methods.ipynb](research_methods.ipynb)** — 主教材（含已執行的輸出與 19 張圖）

---

## 每個單元的四個層次

| 層次 | 內容 |
|---|---|
| **原理** | 這個技術要解決什麼問題？為什麼前一步不夠用？ |
| **公式** | 完整數學定義，逐一說明每個符號 |
| **演算法** | 從公式到可計算步驟，含數值實作的關鍵細節 |
| **程式邏輯** | 對應的實際程式碼，以及「為什麼這樣寫」 |

程式碼儲存格分兩類：**示範型**（讀取 `data/processed/` 的既有結果並解讀）與
**從零實作型**（把核心演算法用最少的程式碼重寫，讓公式到程式的對應無所遁形，
刻意以可讀性優先而非效能）。

---

## 單元一覽

| 篇 | 單元 | 統計技術 | 對應腳本 |
|---|---|---|---|
| **0 導論** | 0.1–0.3 | 研究設計、資料架構、環境設定 | — |
| **1 資料工程** | 1 | 資料抓取與可重現性原則 | `01_fetch_data.py` |
| | 2 | 交易日對齊、對數報酬、雙軌報酬 | `02_build_returns.py` |
| | 3 | Winsorization 與極端值處理 | `02_build_returns.py` |
| **2 前置檢定** | 4 | 描述統計與 Jarque–Bera | `03_pretests.py` |
| | 5 | 單根與定態：ADF、Phillips–Perron、KPSS | `03_pretests.py` |
| | 6 | Ljung–Box 與 ARCH-LM | `03_pretests.py` |
| **3 邊際波動** | 7 | GARCH(1,1) | `04_garch_dcc.py` |
| | 8 | GJR-GARCH 與槓桿效果 | `04_garch_dcc.py` |
| | 9 | EGARCH 與對數變異數 | `04_garch_dcc.py` |
| | 10 | 偏態 t、AIC 選模、邊界解偵測 | `04_garch_dcc.py` |
| **4 動態相關** | 11 | DCC：兩階段 QMLE 與 Q_t 遞迴 | `04_garch_dcc.py` |
| | 12 | ADCC：非對稱衝擊 | `04_garch_dcc.py` |
| | 13 | CCC 對照與概似比檢定 | `04_garch_dcc.py` |
| **5 溢出** | 14 | VAR 與廣義預測誤差變異數分解 | `05_spillover.py` |
| | 15 | Diebold–Yilmaz 連結性指標 | `05_spillover.py` |
| | 16 | TVP-VAR：Kalman 濾波與遺忘因子 | `10_r_tvpvar_frequency.R` |
| | 17 | Baruník–Křehlík 頻域分解 | `10_r_tvpvar_frequency.R` |
| **6 時頻分析** | 18 | 連續小波轉換與 Morlet 母小波 | `06_wavelet.py` |
| | 19 | 小波同調、影響錐、蒙地卡羅顯著性 | `06_wavelet.py` |
| | 20 | 偏小波同調：控制共同因子 | `12_r_partial_wavelet.R` |
| **7 結構變化** | 21 | Bai–Perron 多重結構斷點 | `11_r_baiperron_bootstrap.R` |
| | 22 | 移動區塊拔靴法 | `11_r_baiperron_bootstrap.R` |
| | 23 | Markov 轉換模型 | `07_regime.py` |
| **8 應用** | 24 | Kroner–Sultan 最適避險比率 | `08_hedge.py` |
| | 25 | Ederington 效能與交易成本 | `08_hedge.py` |
| **9 研究誠信** | 26 | 四類真實錯誤的解剖 | 全部管線 |

---

## 關於第 9 篇

計量方法課通常只教「怎麼做對」。單元 26 教的是另外一半：**做錯了長什麼樣、要怎麼抓出來**。

裡面的四類錯誤全部是本專案**投稿前審計時真實發現**的，不是為了教學編造的例子。
每一個都通過了作者本人的檢查、程式沒有報錯、輸出看起來完全合理：

1. **三重複合的量綱錯誤**（交易成本）—— 基點與百分點單位不符（100 倍）、
   多乘一個量綱錯誤的因子、以及把水準效應折進變異數比值。
   結果是「加上交易成本後避險效能反而變好」，在經濟上不可能，卻在每一列都發生。
2. **靜默的覆蓋率缺口** —— R 腳本用萬用字元掃檔案，執行時有三個檔案還沒生成，
   於是只處理了 34 組中的 31 組，正常結束，沒有任何警告。漏掉的那組有 `p = 3e-6` 的實質結果。
3. **陳述與資料不符** —— 六處全稱量詞（「每一個」「都」「唯一」）被自己的表格推翻。
4. **產出檔案過時** —— 前置檢定的表格停留在更早一版的報酬面板，樣本數與其他表格不一致。

對應的防禦手段（單調性檢查、斷言式守衛、機器可驗證的宣稱、相依圖管理）在單元中逐一說明。
修正紀錄見 [`../CHANGELOG.md`](../CHANGELOG.md)。

---

## 執行方式

教材已含執行輸出，**可直接在 GitHub 上閱讀，不需執行**。若要自行執行：

```bash
pip install -r ../requirements.txt
pip install jupyter nbformat nbclient
jupyter lab research_methods.ipynb
```

筆記本會自動偵測路徑（從 `teaching/` 往上找 `data/processed/`）。
單元 16、17、20、21、22 涉及 R 套件（`ConnectednessApproach`、`strucchange`、
`boot`、`biwavelet`），這些單元以說明與結果解讀為主，不在筆記本內執行 R。

---

## 修改教材

筆記本由 `parts/` 下的模組組裝，不要直接編輯 `.ipynb`：

```
teaching/
├── build_notebook.py      # 組裝器
├── parts/
│   ├── _helpers.py        # 儲存格建構函式 + 數學排版慣例
│   ├── part0.py           # 導論
│   ├── part1.py ~ part9.py
└── research_methods.ipynb # 產出（含輸出）
```

```bash
python build_notebook.py --check   # 只驗證，不寫檔
python build_notebook.py           # 產生 research_methods.ipynb
```

### 圖形文字一律用英文

matplotlib 的預設字型（DejaVu Sans）沒有 CJK 字符，中文標題與軸標籤會**靜靜地**
渲染成一排豆腐方塊 —— matplotlib 與筆記本都不會發出任何警告。
因此圖形內的文字（標題、軸標籤、圖例、標註）一律使用英文，敘述與 `print()` 輸出
則維持中文（後者走 HTML 層，CJK 正常）。

`build_notebook.py` 內建守衛：任何 matplotlib 文字參數含非拉丁文字就中止建置。
涵蓋 CJK、假名、諺文、希伯來文、阿拉伯文、泰文 —— 範圍刻意大於本專案所需，
因為把守衛縮到剛好等於「目前踩過的坑」，正是下一個變體漏過去的方式。

```bash
python build_notebook.py
# figures : text checked, no CJK (would render as tofu)
```

想直接驗證渲染結果，可攔截 matplotlib 的缺字警告 —— 最近一次實測為 0：

```bash
python -W always -c "
import warnings, matplotlib; matplotlib.use('Agg')
import nbformat
nb = nbformat.read('research_methods.ipynb', as_version=4)
ns, miss = {}, []
for c in [x.source for x in nb.cells if x.cell_type=='code']:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        exec(c.replace('display(', 'print('), ns)
        miss += [str(x.message) for x in w if 'missing from font' in str(x.message)]
print('missing glyphs:', len(miss))
"
```

### 數學排版慣例

教材會在三個渲染器下閱讀：Jupyter/VS Code、nbviewer/Colab、GitHub 的 .ipynb 檢視器。
只有 `$...$` 與 `$$...$$` 三者皆支援，因此採用這組分隔符
（GitHub 專用的保護寫法在 Jupyter 中會顯示成程式碼）。

代價是 GitHub 會在擷取數學式之前先跑一次 CommonMark 反斜線轉義，把某些序列吃掉。
下列全部在 `parts/` 中避開，實測皆為零：

| 序列 | 用途 | 本教材的替代做法 |
|---|---|---|
| 反斜線加逗號／分號／冒號／驚嘆號 | 間距微調 | 用一般空格，或省略 |
| 反斜線加百分號 | 百分號 | 把百分號移到數學式外 |
| 連續兩個反斜線 | 列分隔 | **完全不使用多列數學區塊**：`cases` 改寫為 min/max 等價式或表格，`pmatrix` 改為表格，分段密度改為兩條分開的式子 |
| `\operatorname` | 運算子 | 改用 `\mathrm`（GitHub 的 KaTeX 拒絕前者） |
| 表格儲存格內的豎線 | 絕對值、範數、行內程式碼 | 改用 `\lvert` 與 `\rvert`，或改寫文字避開豎線 |

最後一項在實測中抓到一個真實錯誤：一列表格裡的行內程式碼含未跳脫的豎線，
把欄位切壞，連帶讓同一列後面的數學式整段消失 —— 而頁面看起來完全正常。

### 驗證方式

不要相信渲染出來的頁面，要驗證**渲染器實際收到什麼**。
把所有 markdown 儲存格送進 GitHub 的 markdown API，比對 LaTeX 指令的存活數：

```bash
python check_notebook_render.py research_methods.ipynb
```

這支腳本跑四項檢查：被 CommonMark 吃掉的轉義序列、表格儲存格內的豎線、
繪圖呼叫中的非拉丁文字，以及最關鍵的一項 —— 把每個 markdown 儲存格送進
GitHub 自己的渲染器，比對 LaTeX 指令的存活數。exit code 非零，可當 pre-commit hook。

最近一次實測：**82 種指令、975 次出現，全數存活；渲染失敗 0**。

腳本刻意 vendored 在 repo 內而非從共用位置引用：任何人 clone 下來都要能重現，
而一個會去讀作者家目錄的建置流程做不到這件事。

---

## 授權

與本 repository 相同（MIT）。教材中的公式與方法均註明原始文獻出處，
引用時請一併引用該方法的原始論文。
