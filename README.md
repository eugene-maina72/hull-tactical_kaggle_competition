# 🧠 Predicting Market Excess Returns with Machine Learning

## Overview
This project applies **machine learning** methods to predict **market forward excess returns**, aiming to design a **volatility-controlled investment strategy** that can potentially outperform the S&P 500.

The project follows the **CRISP-DM** methodology — from **data understanding** to **deployment-ready evaluation** — and compares two main modeling approaches:  
a **Neural Network (PyTorch)** and **XGBoost (tree-based gradient boosting)**.

---

## 🎯 Objectives
- Predict **market excess returns** using structured financial data.  
- Design a **systematic strategy** that:
  - Maximizes excess return over the benchmark.  
  - Keeps volatility ≤ **120% of the market** (vol-cap).  
- Evaluate and compare the performance of different model architectures.

---

## 📊 Data Summary
The dataset contains daily financial indicators grouped by domain:

| Prefix | Category | Description |
|:-------|:----------|:-------------|
| `M*` | Market Dynamics | Technical / market features |
| `E*` | Economic | Macro indicators |
| `V*` | Volatility | Volatility measures |
| `I*` | Interest Rates | Yield curve and rates |
| `P*` | Pricing | Price and valuation ratios |
| `S*` | Sentiment | Investor sentiment features |
| `D*` | Dummy | Binary / categorical encodings |

**Target variable:** `market_forward_excess_returns` — the forward return in excess of the S&P 500 benchmark.  
**Benchmark variable:** `forward_returns` — proxy for the market (S&P 500).

---

## 🧹 Data Preparation
- Imputed missing values **by feature group**:
  - `M*`, `P*` → median  
  - `V*`, `E*` → mean  
  - `I*` → forward/backward fill  
  - `S*` → set to neutral (0)  
  - `D*` → mode or 0  
- Added `_isna` binary flags for missingness (informative feature).  
- Scaled continuous variables using **StandardScaler**.  
- Split chronologically (80/20) to avoid lookahead bias.  
- Excluded all forward-looking columns except the benchmark `forward_returns`.

---

## 🤖 Models

### 1️⃣ Neural Network (PyTorch)
**Architecture:**  
Two hidden layers (128 → 64 → 1), ReLU activation, Dropout (0.2 / 0.1), L2 regularization.  
**Loss:** Huber (robust to outliers).  
**Training:** Early stopping by **validation IC (Spearman)**.

**Performance:**

| Metric | Value |
|:--------|:------|
| RMSE | 0.0111 |
| MAE | 0.0079 |
| R² | 0.0025 |
| IC (Spearman) | 0.0526 |
| Hit Rate | 52.9% |
| Ann. Return | 2.43% |
| Ann. Vol | 15.6% |
| Sharpe (Vol-Capped @120%) | 0.16 |

**Observations:**
- The network detected a weak but statistically valid signal (IC ≈ 0.05).
- Regression accuracy near noise level (expected for daily excess returns).
- Volatility-capped returns show modest but positive performance.

---

### 2️⃣ XGBoost (Tree Ensemble)
**Parameters:**  
`n_estimators=800`, `max_depth=4`, `learning_rate=0.03`,  
`subsample=0.7`, `colsample_bytree=0.7`, `reg_lambda=5`.

**Evaluation:** Spearman IC.  
**Position sizing:** Z-scored predictions (clipped ±3), vol-capped at 120%.

**Performance (Pre-Tuning):**

| Metric | Value |
|:--------|:------|
| Ann. Return | 3.21% |
| Ann. Vol | 14.3% |
| Sharpe (Vol-Capped) | 0.22 |

**Observations:**
- Outperformed the neural network in IC and Sharpe ratio.
- Smoother cumulative return curve and better out-of-sample stability.
- Confirms **tree-based models** are well-suited for structured, noisy market data.

---

## 📈 Strategy Construction
- Trading signals converted into positions:  
  `position = zscore(prediction)` (clipped ±3).  
- Volatility targeting enforced via 20-day rolling std (shifted 1 day to avoid lookahead).  
- Strategy returns calculated as:
- Turnover cost applied (default 5 bps per unit change in position).

---

## ⚙️ Next Steps
- 🔧 Tune XGBoost hyperparameters (depth, learning rate, λ).  
- 📅 Test multi-day forward horizons (5- and 10-day labels).  
- 💸 Include transaction costs and report **net Sharpe**.  
- 🧩 Ensemble models (NN + XGB) for stability.  
- 🧭 Implement walk-forward validation for live-like testing.

---

## 🧾 Key Insights
- Predictability of daily excess returns exists (IC ~0.05–0.07) but is **small and noisy**.  
- Proper **risk management** (vol targeting, cost control) converts small signals into tradable edges.  
- **Tree models** outperform deep nets for tabular financial data.  
- Framework is modular — extendable to cross-asset, multi-horizon, or regime detection models.

---

## 🧩 Tech Stack
- **Language:** Python 3.12  
- **Libraries:** pandas, numpy, scikit-learn, torch, xgboost, matplotlib, seaborn  
- **Framework:** CRISP-DM  
- **Outputs:** Reproducible notebook (`hull-notebook.ipynb`) and visual analytics

---

## 📦 Repository Structure

```markdown
├── data/
│ ├── train.csv
│ ├── test.csv
│── hull-notebook.ipynb 
├── models/
│ ├── neural_net.py
│ └── xgb_model.py
├── images/
│ ├── correlation_top_20.png
│ ├── cumulative_return_nn.png
│ └── cumulative_return_xgb.png
└── README.md
```
## 📚 References
- López de Prado, M. *Advances in Financial Machine Learning*  
- Bailey et al., *Pseudo-Mathematics and Financial Charlatanism*  
- Jensen’s Alpha and Information Coefficient theory  

---

## Contact & Ownership


Eugene Maina |
Data Scientist | RPA Developer

* [LinkedIn](https://www.linkedin.com/in/eugene-maina-4a8b9a128/) | [GitHub](https://github.com/eugene-maina72) | [Email](mailto:eugenemaina72@gmail.com)

