# ⚽ Football Match Outcome Prediction — ELO Ratings + Feature Engineering + SHAP

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-yellow?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-European_Soccer_DB-20BEFF?style=for-the-badge&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Predicts football match outcomes across five major European leagues using ELO ratings, rolling form features, betting market probabilities, and head-to-head statistics — with SHAP explainability and a three-layer anti-leakage architecture.**

### 🚀 [Try the Live Demo →](https://huggingface.co/spaces/bk1210/Football-Match-Outcome-Prediction)

[![Open in HuggingFace Spaces](https://img.shields.io/badge/🤗-Live%20Demo%20on%20HuggingFace%20Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces/bk1210/Football-Match-Outcome-Prediction)

[Features](#-features) • [How It Works](#-how-it-works) • [Results](#-results) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Tech Stack](#-tech-stack) • [Contact](#-contact)

</div>

---

## 📌 Overview

Most football match prediction studies train and test on a single league — making models that can't generalise. This project fixes that by building a **multi-league framework** trained across five major European leagues over eight seasons (2008/09–2015/16), with a strict temporal train-test split and no data leakage.

Four complementary feature signals are combined:
- 📈 **ELO Ratings** — long-term team strength, computed chronologically and used as ML features (not just a baseline)
- 🔄 **Rolling Form** — dual 3-match and 5-match windows for goals, goal difference, wins, and points
- 🎰 **Betting Market Probabilities** — normalized implied probabilities from Bet365, Betwin, and Interwetten
- ⚔️ **Head-to-Head Win Rate** — O(n) vectorized groupby computation replacing O(n²) iterative approach

---

## ✨ Features

### 📊 33-Feature Engineering Pipeline
- **Rolling window form** (3 and 5 matches): goals for/against, goal difference, wins, points — computed with `shift(1)` to prevent look-ahead leakage
- **Differential features**: `diff_gd_last3`, `diff_wins_last3`, `diffPtsLast` (home minus away)
- **Season cumulative points** for long-term momentum
- **ELO features**: `elo_home`, `elo_away`, `elo_diff`, `elo_home_win_prob` (K=20, home advantage +100, starting rating 1500)
- **H2H win rate**: vectorized O(n) groupby — dramatically faster than iterative O(n²)
- **Betting odds**: 1/odds converted to implied probabilities, averaged across 3 bookmakers, normalized to remove overround → `odds_home_prob`, `odds_draw_prob`, `odds_away_prob`

### 🤖 Three ML Models
- **Logistic Regression** — multinomial, C=0.5, balanced class weights, lbfgs solver
- **Random Forest** — 400 trees, max_depth=8, min_samples_leaf=10
- **Gradient Boosting** — 300 trees, max_depth=4, lr=0.05, subsample=0.8 *(best model)*

### 🛡️ Three-Layer Anti-Leakage Architecture
- `shift(1)` on all rolling statistics — features only use past matches
- ELO ratings saved pre-match — never updated with post-match result before feature join
- scikit-learn `Pipeline` + `TimeSeriesSplit` (5 folds) — prevents future data leaking into past training folds

### 🔍 SHAP Explainability
- `PermutationExplainer` on 100 background samples, 300 test samples
- Output tensor shape: (300, 33, 3) — samples × features × classes (Win/Draw/Loss)
- Global importance bar charts, per-class beeswarm plots, match-level waterfall plots
- Top signals: `odds_home_prob`, `elo_diff`, `diff_gd_last3`

### 📏 Expected Points MAE
- Custom calibration metric: E[pts] = 3×P(Win) + 1×P(Draw)
- Measures how well predicted probabilities match actual points earned

---

## 🖥️ Demo

### 🔴 Live App
> **[https://huggingface.co/spaces/bk1210/Football-Match-Outcome-Prediction](https://huggingface.co/spaces/bk1210/Football-Match-Outcome-Prediction)**
> Select any two teams from 259 available clubs and get instant match outcome prediction with win probabilities, ELO ratings, and expected points.

### Example Predictions

| Match | Predicted Winner | Home Win % | Draw % | Away Win % |
|---|---|---|---|---|
| Real Madrid CF vs Bournemouth | 🏆 Real Madrid CF | 73.6% | 11.9% | 14.5% |
| Manchester City vs Arsenal | 🏆 Manchester City | ~65% | ~18% | ~17% |
| Barcelona vs Atletico Madrid | 🏆 Barcelona | ~58% | ~22% | ~20% |

### Model vs ELO Baseline
```
Gradient Boosting → Accuracy: 0.55 | Draw Recall: 0.25 | Exp. Pts MAE: 0.91
ELO Baseline      → Accuracy: 0.50 | Draw Recall: 0.04 | Exp. Pts MAE: 1.05
```

### SHAP Top Features
```
#1 → odds_home_prob   (betting market encodes injuries, tactics, context)
#2 → odds_away_prob
#3 → elo_diff         (long-term team strength gap)
#4 → elo_home_win_prob
#5 → away_cum_pts     (season momentum)
```

---

## 🚀 Installation

### Prerequisites
- Python 3.12
- European Soccer Database (`database.sqlite`) from Kaggle

### Step 1 — Clone the Repository
```bash
git clone https://github.com/bk1210/Football-Match-Outcome-Prediction-.git
cd Football-Match-Outcome-Prediction-
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download the Dataset
Get the dataset from Kaggle:
👉 [European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer)

Place `database.sqlite` in the project root.

### Step 4 — Run the Notebook
```bash
jupyter notebook Football_match_prediction.ipynb
```

---

## 📖 Usage

### Running the Full Pipeline

Open `Football_match_prediction.ipynb` and run all cells — the notebook handles:

1. Load `database.sqlite` via SQLite — Match, Team, League, Player tables
2. EDA — league distribution, match outcomes, goals per match, player ratings
3. Chronological ELO rating computation (K=20, home advantage +100)
4. Rolling window feature engineering (3 and 5 match windows, shift(1))
5. H2H win rate computation (O(n) vectorized groupby)
6. Betting odds → normalized implied probabilities
7. Training LR, RF, Gradient Boosting in anti-leakage pipelines
8. Temporal 80/20 split + 5-fold TimeSeriesSplit cross-validation
9. SHAP PermutationExplainer — global + per-class + waterfall plots
10. Expected Points MAE evaluation per season and league

---

## 🏗️ Architecture

### Full Pipeline

```
European Soccer Database (SQLite)
[25,979 matches | 5 leagues | 8 seasons | 2008/09–2015/16]
    │
    ▼
Chronological Sort + Preprocessing
[Encode outcomes: Win=2, Draw=1, Loss=0 | Impute missing odds]
    │
    ├──► ELO Rating Engine
    │    [K=20 | home +100 | pre-match ratings saved → elo_home, elo_away, elo_diff]
    │
    ├──► Rolling Form Features (shift(1) — no leakage)
    │    [3-match + 5-match windows | gf, ga, gd, wins, pts | differential features]
    │
    ├──► H2H Win Rate
    │    [O(n) vectorized groupby | cumulative sum + shift(1)]
    │
    └──► Betting Implied Probabilities
         [1/odds | avg across Bet365/Betwin/Interwetten | normalize overround]
    │
    ▼
33-Feature Matrix
    │
    ├──► Logistic Regression (Pipeline: StandardScaler + LR)
    ├──► Random Forest       (Pipeline: StandardScaler + RF)
    └──► Gradient Boosting   (Pipeline: StandardScaler + GB) ← Best
    │
    ▼
Temporal 80/20 Split + TimeSeriesSplit (5 folds)
    │
    ▼
SHAP PermutationExplainer
[Global importance | Win/Draw beeswarms | Match waterfall plots]
```

### Project Structure

```
Football-Match-Outcome-Prediction-/
│
├── Football_match_prediction.ipynb    # Full pipeline
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---
## 🔄 Pipeline

![Pipeline](pipeline.png)
## 📊 Results

### Model Comparison — Temporal 20% Test Set

| Model | Accuracy | Draw Recall | Exp. Pts MAE |
|---|---|---|---|
| Logistic Regression | 0.52 | 0.18 | 0.97 |
| Random Forest | 0.54 | 0.22 | 0.94 |
| **Gradient Boosting** | **0.55** | **0.25** | **0.91** |
| ELO Baseline | 0.50 | 0.04 | 1.05 |

### TimeSeriesSplit Cross-Validation (Gradient Boosting)

| Metric | Value |
|---|---|
| Mean accuracy | 0.54 ± 0.02 |
| Fold range | 0.51 – 0.57 |

### SHAP Top Features

| Rank | Feature | Meaning |
|---|---|---|
| 1 | `odds_home_prob` | Betting market home win probability |
| 2 | `odds_away_prob` | Betting market away win probability |
| 3 | `elo_diff` | ELO rating gap between teams |
| 4 | `elo_home_win_prob` | ELO-derived win probability |
| 5 | `away_cum_pts` | Away team season cumulative points |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Core language |
| scikit-learn | LR, RF, GB models, Pipeline, TimeSeriesSplit |
| SQLite / sqlite3 | European Soccer Database loading |
| SHAP | PermutationExplainer, global + local attribution |
| Pandas / NumPy | Feature engineering, rolling windows, ELO |
| Matplotlib / Seaborn | EDA, SHAP plots, confusion matrix |
| Streamlit | Live demo web app |

---

## 📦 Dependencies

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
shap>=0.42.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

- [ ] Add player-level stats, FIFA ratings, injury reports, tactical formations
- [ ] xG (expected goals) as additional feature
- [ ] Probability calibration — Platt scaling or isotonic regression
- [ ] Draw-specific features — defence stability scores, xG balance indicators
- [ ] Real-time in-play prediction system
- [ ] Extend to more leagues and recent seasons

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

| Name | GitHub |
|---|---|
| Bharath Kesav R | [@bk1210](https://github.com/bk1210) |
| Goutham Divakaran S Menon | [@goutham](https://github.com/) |

Supervised by **Kirubavathi G** — Department of Mathematics, Amrita Vishwa Vidyapeetham, Coimbatore

---

## 👤 Contact

**Bharath Kesav R**
- 📧 Email: bharathkesav1275@gmail.com
- 🐙 GitHub: [@bk1210](https://github.com/bk1210)
- 🎓 Institution: Amrita Vishwa Vidyapeetham, Coimbatore

---

## 🙏 Acknowledgements

- [Hugo Mathien](https://www.kaggle.com/datasets/hugomathien/soccer) — European Soccer Database on Kaggle
- [Hvattum & Arntzen (2010)](https://doi.org/10.1016/j.ijforecast.2009.10.002) — ELO ratings for football prediction
- [Lundberg & Lee (2017)](https://proceedings.neurips.cc/paper/2017) — SHAP framework

---

<div align="center">

**⭐ If you found this project useful, please give it a star on GitHub! ⭐**

[![Open in HuggingFace Spaces](https://img.shields.io/badge/🤗-Try%20Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/bk1210/Football-Match-Outcome-Prediction)

*Built with ❤️ for multi-league football analytics*

</div>
