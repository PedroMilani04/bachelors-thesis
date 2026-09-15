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
