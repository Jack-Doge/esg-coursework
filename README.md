# ESG REITs 投資組合分析

本專案旨在分析與比較 ESG (環境、社會、公司治理) 型的房地產投資信託 (REITs) 與傳統型 REITs 的表現。我們透過持有部位資料建構「綠色 (Green)」與「傳統 (Traditional)」兩種投資組合，並進行績效評估與迴歸分析。

## 分析流程

腳本 `analysis.py` 包含完整的分析流程，從資料前處理、投資組合建構、績效計算到模型分析。

### 1. 資料前處理 (`preprocessing`)

- 讀取 `data/raw_data.csv` 的原始持股資料。
- 篩選出兩檔代表性的 REITs ETF：VNQ (傳統型) 與 NURE (ESG 型)。
- 為了確保比較基礎的一致性，僅保留在整個分析期間內持續存在的成分股。
- 計算每支成分股的價格 (`price`) 與週報酬率 (`ret`)。
- 將處理完成的資料儲存至 `data/processed_data.csv`。

### 2. 投資組合建構 (`form_portfolio` & `reit_portfolio_venn`)

- **區分成分股**：
  - **Green Portfolio**: 僅存在於 NURE 但不存在於 VNQ 的成分股。
  - **Traditional Portfolio**: 僅存在於 VNQ 但不存在於 NURE 的成分股。
  - **Shared Portfolio**: 兩者皆有的成分股 (不納入後續分析)。
- **視覺化**：產生一個文氏圖 (`figures/reit_portfolio_venn.png`) 來展示兩類投資組合的重疊情況。
- **標記資料**：將區分好的 Green 與 Traditional 成分股資料合併，並加上 `label` 欄位以供後續分析，儲存至 `data/labeled_portfolio.csv`。

### 3. 投資組合績效分析

- **計算累積報酬 (`plot_portfolio_performance`)**:
  - 計算 Green 與 Traditional 投資組合的每日等權重報酬率。
  - 處理報酬率為零的特殊情況，避免影響累積報酬計算。
  - 計算並繪製兩個投資組合的累積報酬曲線圖 (`figures/portfolio_performance.png`)。
  - 將包含績效指標的最終資料儲存至 `data/final_data.csv`。
- **計算年化績效指標 (`calculate_portfolio_performance`)**:
  - 基於週報酬資料，計算各投資組合的年化報酬率、年化波動度與夏普比率。
  - 將結果儲存至 `data/portfolio_performance.csv`。
- **績效視覺化 (`plot_performance_comparison`)**:
  - 繪製橫條圖 (`figures/portfolio_performance_bar_annualized.png`)，直觀比較兩個投資組合的各項年化績效指標。

### 4. 資金流數據整合與迴歸分析

- **整合資金流資料 (`add_fundflow_data`)**:
  - 將 `data/fundflow_data.csv` 中的資金流數據合併至績效資料中。
  - 儲存為 `data/portfolio_with_fundflow.csv`。
- **迴歸模型 (`model1`, `model2`, `model3`)**:
  - **模型一**: 分析當期的 `NAV` (淨值)、`cum_fundflow` (累積資金流) 對投資組合報酬 (`portfolio_ret`) 的影響。
  - **模型二**: 使用落後一期的 `NAV` 與 `cum_fundflow` 作為解釋變數，分析其對當期報酬的影響。
  - **模型三**: 加入市場狀態變數 (`period_return`)，探討在不同市場環境下，`NAV` 與 `cum_fundflow` 對報酬的影響。

## 如何執行

1.  **安裝相依套件**:
    ```bash
    pip install pandas numpy loguru matplotlib matplotlib-venn seaborn statsmodels scikit-learn
    ```
2.  **執行分析腳本**:
    ```bash
    python analysis.py
    ```
    腳本會自動執行上述所有分析流程，並產生對應的數據檔案與圖表。

## 產出檔案

- **數據檔案 (`data/`)**:
  - `processed_data.csv`: 清理過的成分股資料。
  - `labeled_portfolio.csv`: 標記為 Green/Traditional 的投資組合資料。
  - `final_data.csv`: 包含每日報酬與累積報酬的最終資料。
  - `portfolio_performance.csv`: 年化績效指標。
  - `portfolio_with_fundflow.csv`: 整合資金流後的資料。
- **圖表 (`figures/`)**:
  - `reit_portfolio_venn.png`: VNQ 與 NURE 成分股重疊情況的文氏圖。
  - `portfolio_performance.png`: Green vs. Traditional 投資組合累積報酬曲線圖。
  - `portfolio_performance_bar_annualized.png`: 年化績效指標比較長條圖。
