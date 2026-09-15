# Macro-Quant Trading Pipeline: Triple Barrier Labeling & Expected Value Optimization

This repository implements an **end-to-end quantitative trading machine learning pipeline** to predict stock market movements. The central methodological contribution of this thesis is the shift from traditional classification metrics (Accuracy) to financial business metrics (**Expected Value - EV**), combined with the translation of macroeconomic events into dimensionless physical features (Deltas and Accelerations) to prevent temporal overfitting.

---

## Project Overview

The primary objective is to demonstrate that financial market prediction models must be evaluated by their mathematical expectation of profit, rather than their raw predictive accuracy. The pipeline evaluates 10 equities (5 US, 5 BR) using a custom Triple Barrier Labeling method, Technical Indicators, and Macroeconomic data, ultimately deploying **Specialist XGBoost Models** optimized via Recursive Feature Elimination (RFE).

### Datasets

| Dataset / Feature | Source / Ticker | Description |
|---|---|---|
| **Equities (10 Stocks)** | Yahoo Finance (`yfinance`) | 5 US Stocks and 5 Brazilian (B3) Stocks |
| **Fed Funds Rate** | US Federal Reserve | Complete historical series (2015-2024) |
| **Selic Rate** | Banco Central do Brasil | Brazilian base interest rate |
| **Crude Oil** | `BZ=F` & `CL=F` | Brent (Global/BR proxy) and WTI (US proxy) |
| **Elections (US & BR)** | Derived Feature | Countdown (in days) to presidential election results |

---

## Pipeline Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  Phase 0  │  Setup: Folder structure + yfinance Data Fetching   │
└────────────────────────────┬────────────────────────────────────┘
                             │
          ┌──────────────────┴───────────────────┐
          ▼                                      ▼
┌─────────────────────┐                ┌───────────────────────┐
│  Phase 1            │                │  Phase 2              │
│  Feature Eng.       │                │  Labeling Method      │
│  (transforming.py)  │                │                       │
│                     │                │  Triple Barrier       │
│  Volatility (ATR)   │                │  Upper (Take Profit)  │
│  SMA 20/50 & Slopes │                │  Lower (Stop Loss)    │
│  RSI, MACD (Signal) │                │  Vertical (Horizon)   │
│  Bollinger Bands    │                │                       │
│  Volume & Deltas    │                │  Target: 10%          │
│                     │                │  Horizon: 180 days    │
└──────────┬──────────┘                └──────────┬────────────┘
           │                                      │
           └──────────────┬───────────────────────┘
                          ▼
               ┌───────────────────────┐
               │  Phase 3              │
               │  Macro Engineering    │
               │                       │
               │  Absolute Rates → ❌   │
               │  Deltas (5,21,63d) → ✅│
               │  Accelerations     → ✅│
               │  Election Clocks   → ✅│
               └──────────┬────────────┘
                          │
          ┌───────────────┴──────────────┐
          ▼                              ▼
┌─────────────────────┐      ┌───────────────────────┐
│  Phase 4            │      │  Phase 5              │
│  Model Selection    │      │  Optimization         │
│                     │      │                       │
│  Baselines:         │      │  RFE applied          │
│  MLP, RF, SVM       │      │  Testing 15, 12, 9, 6,│
│  XGBoost (Winner)   │      │  and 3 features.      │
│                     │      │                       │
│  Transition from    │      │  Max EV achieved      │
│  Generalist to      │      │  with Top 3 Features  │
│  Specialist Models  │      │                       │
└──────────┬──────────┘      └──────────┬────────────┘
           │                            │
           └──────────────┬─────────────┘
                          ▼
               ┌───────────────────────┐
               │  Phase 6              │
               │  Business Validation  │
               │                       │
               │  EV-Long Calculation  │
               │  EV = +4.14% / 180d   │
               │  Annualized: 8.45%    │
               │                       │
               │  Opportunity Cost     │
               │  US: Alpha generated  │
               │  BR: Selic wins       │
               └───────────────────────┘


```
## Methodological Concepts and Nomenclature

| Concept | Formal Definition | Role in Pipeline |
|---|---|---|
| **Triple Barrier Labeling** | Labels data as `1` (Hits upper bound), `-1` (Hits lower bound), or `0` (Hits vertical time limit). | Replaces fixed-horizon arbitrary labeling, accounting for volatility and path-dependency. |
| **EV (Expected Value)** | `(P_win × Target) - (P_loss × Stop)` | The ultimate business metric. Measures the mathematical expectation of profit per trade. |
| **Macro Deltas** | `Rate(t) - Rate(t-n)` | Prevents the XGBoost model from using absolute interest rates as a temporal "barcode" to memorize years (Overfitting). |
| **RFE** | Recursive Feature Elimination | Algorithm that recursively prunes the least important features, validating the Principle of Parsimony in financial data. |
| **Opportunity Cost** | The Risk-Free Rate (Selic for BR, Fed Funds for US) | The baseline hurdle rate the model must beat to justify the risk of equity deployment. |

---

## Detailed Methodology & Evolution

### 1. Labeling and Experimental Horizons
The project initially explored an "Average Barrier" concept (labeling with triple barrier, predicting with double barrier to treat noise). After academic consultation (Advisor: Danilo), this was discarded in favor of a pure Triple Barrier formulation. A massive **Sensitivity Analysis** was conducted across $n$ threshold values (3% to 50%) and $m$ time horizons (15 days to 5 years). Given the inherent bullish bias of equities due to inflation, the final framework was stabilized at a **10% Target with a 180-day Horizon**.

### 2. Model Selection: Generalist vs. Specialist
Initial baselines included Multilayer Perceptron (MLP), Random Forest, SVM, and XGBoost. **XGBoost** was selected due to its superior handling of non-linear financial boundaries and speed. The architecture was then upgraded from a single "Generalist" model to **Specialist Models** (one isolated XGBoost instance per stock), vastly improving localized feature importance and localized Expected Value (EV).

### 3. Macroeconomic Feature Engineering (The "Graveyard" & The "Goldmine")
Adding exogeneous variables required strict physical translation to avoid data leakage and temporal memorization:

*   **Interest Rates (Selic/Fed):** Initial tests using absolute values destroyed out-of-sample performance (temporal overfitting). Converting them to *Deltas* and *Accelerations* restored accuracy and allowed the model to trade the "interest rate momentum".
*   **Crude Oil (Brent/WTI):** Implemented via Deltas and Lags. Improved EV-Long marginally, with RFE successfully capturing the 61-day WTI lag in the top 12 features.
*   **Elections Countdowns:** A countdown integer to US and BR elections successfully improved performance, being captured by the RFE down to the 6-feature baseline.

**The Graveyard (Discarded via RFE and EV drops):**
1.  **VIX (Fear Index):** Degraded model performance and was entirely ignored by the RFE algorithm.
2.  **Gold (GC=F):** Captured by RFE (Top 9), but caused a severe **~25% drop in EV-Long**. Removed.
3.  **Bitcoin (BTC-USD):** RFE captured its 21-day volatility, but failed to improve the EV. Removed.

### 4. Recursive Feature Elimination (RFE)
RFE was applied to filter out noise from over 60+ engineered features. The pipeline tested subsets of 15, 12, 9, 6, and 3 features. 
**Key Finding:** The absolute highest Expected Value for the Long position (EV-Compra) was achieved using strictly the **Top 3 features**, proving that in quantitative finance, signal-to-noise ratio matters more than feature quantity.

---

## Financial Validation Results (Phase 6)

The model's business viability was tested by segregating Expected Value (EV) by direction (Long vs. Short). 

*   **EV-Long (180 days):** `+0.0414` (4.14% Expected Return per signaled trade).
*   **Annualized EV (Compounded, 2 trades/yr):** `8.45%`

### The Opportunity Cost Assessment

| Market | Risk-Free Rate | Model Annualized EV | Verdict |
|---|---|---|---|
| **Brazil** | Selic (~14.75%) | 8.45% | ❌ **Negative Risk Premium.** The model loses to local government bonds. Not viable for deployment in BRL. |
| **USA** | Fed Funds (~3.63%) | 8.45% | ✅ **Alpha Generation.** The model generates ~4.8% pure Alpha over US Treasuries. Highly viable for USD deployment. |

**Conclusion:** The pipeline is statistically and financially validated. However, its commercial deployment is geographically constrained by local monetary policy. It operates as a highly efficient Alpha-generator for the US market.

---

## Repository Structure

```text
project/
├── 0-raw-data/                 ← yfinance extractions (Equities, Rates, Oil)
├── 1-processed-data/           ← Cleaned datasets post-transforming.py
├── 2-features/                 ← CSVs with all generated technical/macro features
├── 3-labels/                   ← Triple Barrier outputs
├── 4-modelos-generalistas/     ← Deprecated baselines (MLP, RF, SVM)
├── 5-modelos-especialistas/    ← Core Pipeline: XGBoost per ticker
│   ├── xgboost/
│   │   ├── rfe/                ← RFE feature selection logs & charts
│   │   └── ev_analysis/        ← Expected Value calculations
├── src/
│   ├── transforming.py         ← Technical indicators and Lags module
│   └── data_acquisition.py     ← API calls and Macro feature engineering
└── README.md
```


## Dependencies

```bash
pip install pandas numpy xgboost scikit-learn yfinance matplotlib seaborn
```

| Library | Usage |
|---|---|
| `yfinance` | Data acquisition for stocks, rates, and commodities |
| `pandas` / `numpy` | Feature engineering, Deltas, Accelerations |
| `scikit-learn` | RFE (Feature Selection), compute_sample_weight |
| `xgboost` | Core classification algorithm (`objective='multi:softprob'`) |
| `matplotlib` / `seaborn` | Confusion matrices, EV distribution plots |

---

## Academic Context
Developed as an Undergraduate Thesis in Computer Science at **FCT UNESP** (Faculdade de Ciências e Tecnologia da Universidade Estadual Paulista), Presidente Prudente - SP, Brazil. 
**Advisor:** Prof. Danilo.
