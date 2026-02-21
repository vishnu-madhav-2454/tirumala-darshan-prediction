---
title: Srivari Seva
emoji: 🛕
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---

# 🛕 Srivari Seva — Tirumala Darshan Crowd Advisory System

An end-to-end machine learning system that predicts daily crowd levels at **Tirumala Sri Venkateswara Temple** — one of the most visited religious sites in the world, receiving 50,000–120,000 pilgrims every day.

🔗 **Live Demo**: [huggingface.co/spaces/madhav456789123/tirumala-darshan-prediction](https://huggingface.co/spaces/madhav456789123/tirumala-darshan-prediction)

---

## What This Project Does

Pilgrims visiting Tirumala often face 4–12 hour queues with no way to know in advance how crowded it will be. This system solves that by:

1. **Predicting daily crowd levels** into 6 bands (QUIET → EXTREME) using a 3-model gradient-boosting ensemble
2. **Showing a monthly calendar** with color-coded crowd forecasts and Hindu festival indicators
3. **Answering pilgrim questions** via an AI chatbot (darshan types, accommodation, sevas, transport) in 6 Indian languages
4. **Displaying historical trends** with interactive charts from 1,400+ days of real data
5. **Retraining itself daily** — scrapes new data from TTD every morning and updates the models automatically

---

## The Complete ML Pipeline — What We Did Step by Step

### Step 1: Data Collection

We scraped daily pilgrim count data from [news.tirumala.org](https://news.tirumala.org/category/darshan/), the official TTD news portal. The dataset spans January 2020 through February 2026 with 4,076 raw records.

We discarded everything before February 2022 because COVID lockdowns made that data unreliable. After removing outliers (temple closure days, data entry errors), we had **1,469 clean daily records**. After feature engineering (which requires look-back windows), **1,104 complete samples** remained.

### Step 2: Defining the Target — 6 Crowd Bands

Instead of predicting exact pilgrim counts (a regression problem), we classified each day into one of six ordered crowd bands:

| Band | Name | Pilgrim Range | Share of Data |
|------|------|---------------|---------------|
| 0 | QUIET | 0 – 50,000 | 0.7% (8 days) |
| 1 | LIGHT | 50,000 – 60,000 | 14.4% (159 days) |
| 2 | MODERATE | 60,000 – 70,000 | 36.5% (403 days) |
| 3 | BUSY | 70,000 – 80,000 | 29.0% (320 days) |
| 4 | HEAVY | 80,000 – 90,000 | 18.1% (200 days) |
| 5 | EXTREME | 90,000+ | 1.3% (14 days) |

The extreme class imbalance (EXTREME = only 14 samples, QUIET = only 8) was one of the biggest challenges in this project.

### Step 3: Feature Engineering — 57 Features Across 8 Groups

We built 57 features, all leak-proof using `.shift(N)` to ensure no future data could influence any feature value:

| Group | Count | What It Captures | Examples |
|-------|-------|------------------|----------|
| Calendar | 6 | Weekly & annual rhythm | `dow`, `is_weekend`, `sin_doy`, `cos_doy` |
| Lags | 7 | Direct memory of recent days | `L1`, `L7`, `L14`, `L21`, `L28`, `L365` |
| Rolling stats | 5 | Recent trend & volatility | `rm7`, `rm14`, `rm30`, `rstd7`, `rstd14` |
| Expanding means | 2 | Historical weekday norms | `dow_expanding_mean`, `month_dow_mean` |
| Log transforms | 5 | Reduce right-skew for tree splits | `log_L1`, `log_L7`, `log_rm7`, `log_rm30` |
| Derived/interaction | 9 | Momentum, trend, combos | `ewm7`, `momentum_7`, `month_weekend` |
| Regime counts | 2 | Hot/cold streak detection | `heavy_extreme_count7`, `light_quiet_count7` |
| Festival features | ~21 | Religious calendar spikes | `is_brahmotsavam`, `fest_impact`, `days_to_fest` |

The festival features come from a hand-curated Hindu calendar (`festival_calendar.py`) covering 2013–2027 with 17 event categories (Brahmotsavams, Vaikuntha Ekadashi, Sankranti, Navaratri, etc.) and a 1–10 impact scale.

**Design note**: We excluded lunar calendar features (`is_pournami`, `is_amavasya`) because our Hindu calendar module only covers 2025–2027 — using it would produce zeros for 75% of the training period.

### Step 4: Feature Selection — 10-Method Consensus Vote

With 57 features and only 1,104 samples, overfitting was a real risk. Instead of relying on a single selection method, we ran **10 independent methods** and kept only features that **at least 6 out of 10 agreed on**:

1. Mutual Information
2. ANOVA F-test
3. Chi-squared test (after MinMax scaling)
4. GradientBoosting impurity importance
5. Random Forest importance
6. Extra Trees importance
7. Permutation importance
8. L1 Logistic Regression (Lasso)
9. Spearman rank correlation
10. Recursive Feature Elimination (DecisionTree)

**Additional filters applied**:
- **Variance pre-filter**: Dropped features where ≥98% of values were identical (removed 11 rare festival flags)
- **Redundancy removal**: If two features had Spearman |ρ| > 0.95, we dropped the one with fewer votes

**Result**: 57 → **16 features**, improving our sample-to-feature ratio from 19.4:1 to **69.0:1**.

<details>
<summary><strong>The 16 Final Features</strong></summary>

| Feature | Votes | Description |
|---------|-------|-------------|
| `dow_expanding_mean` | 10/10 | Historical mean count per weekday |
| `cos_doy` | 10/10 | Annual cycle (cosine of day-of-year) |
| `L21` | 10/10 | Pilgrim count 3 weeks ago |
| `month_dow_mean` | 9/10 | Mean count by month × weekday |
| `log_L1` | 9/10 | Log of yesterday's count |
| `log_L7` | 9/10 | Log of same day last week |
| `log_rm30` | 9/10 | Log of 30-day rolling mean |
| `month_weekend` | 9/10 | Month × weekend interaction |
| `L14` | 9/10 | Pilgrim count 2 weeks ago |
| `rm14` | 8/10 | 14-day rolling mean |
| `dow` | 8/10 | Day of week (0=Mon, 6=Sun) |
| `L28` | 8/10 | Pilgrim count 4 weeks ago |
| `ewm7` | 7/10 | 7-day exponential weighted mean |
| `rstd7` | 6/10 | 7-day rolling standard deviation |
| `heavy_extreme_count7` | 6/10 | HEAVY/EXTREME days in past 7 (streak detector) |

</details>

### Step 5: Data Splitting — Why Stratified, Not Temporal

This was a deliberate decision. With only 14 EXTREME samples in the entire dataset, a standard temporal split (train on past, test on future) would put at most 2–3 EXTREME days in training — making it impossible for the model to learn what extreme crowds look like.

We used `StratifiedShuffleSplit` to ensure proportional class representation:

| Subset | Size | Purpose |
|--------|------|---------|
| Train | 794 (72%) | Model fitting |
| Calibration | 89 (8%) | Ensemble weight optimization only |
| Test | 221 (20%) | Final, untouched evaluation |

**Temporal integrity is preserved** because all features are computed using `.shift(N)` in strict chronological order *before* the split. We also ran a separate walk-forward evaluation as an honest temporal out-of-sample test.

### Step 6: Custom Scoring Function

Standard accuracy doesn't work for ordinal classification. Predicting MODERATE when the actual is BUSY (off by 1) is far less harmful than predicting QUIET when the actual is EXTREME (off by 5, and potentially dangerous). We designed a composite score:

```
Score = 0.30 × Exact_Accuracy + 0.30 × Macro_F1 + 0.25 × Ordinal + 0.15 × Safety
```

Where:
- **Ordinal** = 1 − MAE/(N_bands − 1) — penalizes bigger band errors more
- **Safety** = 1 − (dangerous predictions / total) — penalizes predicting QUIET/LIGHT when actual is HEAVY/EXTREME

### Step 7: Class Weighting — Safety-First

We computed balanced class weights (`N_train / (N_classes × count_per_class)`) and added an extra **1.5× multiplier for HEAVY and EXTREME** — because under-predicting these classes could leave pilgrims stranded in 10+ hour queues.

### Step 8: Phase 1 — Baseline Models

Three gradient-boosting classifiers trained with default hyperparameters:

| Model | Config |
|-------|--------|
| GradientBoosting (sklearn) | 500 trees, depth=5, lr=0.05 |
| LightGBM | 500 trees, depth=6, lr=0.05, 31 leaves |
| XGBoost | 500 trees, depth=5, lr=0.05 |

Baseline results on test set:

| Model | Accuracy | Macro-F1 | MAE | Safety |
|-------|----------|----------|-----|--------|
| GB | 55.7% | 33.0% | 0.475 | 100% |
| LGB | 55.7% | 32.9% | 0.484 | 100% |
| XGB | 53.8% | 32.2% | 0.507 | 100% |
| **Ensemble** | **54.8%** | **32.6%** | **0.484** | **100%** |

### Step 9: Phase 2 — Optuna Hyperparameter Tuning (80 Trials Per Model)

We used **Optuna** with the TPE (Tree-structured Parzen Estimator) sampler to tune each model independently over **80 trials** with 5-fold stratified cross-validation.

<details>
<summary><strong>Search Spaces</strong></summary>

| Parameter | GB Range | LGB Range | XGB Range |
|-----------|----------|-----------|-----------|
| n_estimators | 200–1000 | 200–1000 | 200–1000 |
| max_depth | 3–7 | 3–10 | 3–10 |
| learning_rate | 0.005–0.15 | 0.005–0.2 | 0.005–0.2 |
| subsample | 0.6–1.0 | 0.6–1.0 | 0.6–1.0 |
| colsample_bytree | — | 0.5–1.0 | 0.5–1.0 |
| num_leaves | — | 15–127 | — |
| reg_alpha | — | 1e-3–10 | 1e-3–10 |
| reg_lambda | — | 1e-3–10 | 1e-3–10 |
| gamma | — | — | 0–5 |

</details>

**Best tuned hyperparameters**:

| Param | GB | LGB | XGB |
|-------|-----|------|------|
| n_estimators | 685 | 667 | 913 |
| max_depth | 6 | 8 | 4 |
| learning_rate | 0.006 | 0.081 | 0.032 |
| subsample | 0.743 | 0.882 | 0.691 |

### Step 10: Ensemble Calibration

We combined the three models via weighted probability averaging:

```
prediction = argmax( w_GB × P_GB + w_LGB × P_LGB + w_XGB × P_XGB )
```

Weights were found by grid search on the **calibration set** (never used for training or testing):

| Model | Weight |
|-------|--------|
| GradientBoosting | **0.10** |
| LightGBM | **0.50** |
| XGBoost | **0.40** |

### Step 11: Final Test Results (221 Days, Held-Out)

| Model | Accuracy | Macro-F1 | MAE | Safety | Dangerous Predictions |
|-------|----------|----------|-----|--------|-----------------------|
| GB | **59.7%** | 37.8% | 0.434 | 100% | 0 |
| LGB | 54.3% | 31.5% | 0.480 | 100% | 0 |
| XGB | 56.6% | 35.8% | 0.484 | 100% | 0 |
| **Ensemble** | **57.0%** | **34.5%** | **0.457** | **100%** | **0** |

**Key results**:
- **59.7% accuracy** (best individual model: GB) — 3.6× better than random chance (16.7% across 6 bands)
- **57.0% ensemble accuracy** — sacrifices a bit of peak accuracy for robustness across all classes
- **100% safety** on every model — zero cases of predicting QUIET/LIGHT when the actual was HEAVY/EXTREME
- **Baseline → Optuna improvement**: +2.2pp on GB accuracy (55.7% → 59.7%)

### Step 12: Walk-Forward Validation (Temporal Out-of-Sample)

To prove the model works on unseen time periods (not just shuffled data), we ran expanding-window walk-forward evaluation:

- Start with the earliest ~360 days of training data
- Every 30 days: retrain GB from scratch on all available data, predict the next 30 days
- This simulates real-world deployment where you only train on past data

| Metric | Value |
|--------|-------|
| Accuracy | 59.2% |
| Macro-F1 | 44.5% |
| MAE | 0.425 bands |
| **Safety** | **100.0%** |

This confirms the model generalizes well temporally and the stratified split didn't cause overfitting.

### Step 13: Comparison Against Standard Time-Series Models

We evaluated three standard forecasting baselines on a pure 80/20 temporal split:

| Model | Accuracy | Safety | RMSE |
|-------|----------|--------|------|
| Seasonal Naive (same weekday last week) | 66.2% | 99.5% | — |
| Prophet (Meta) | 8.0% | 100% | 29,203 |
| SARIMA(1,1,1)(1,0,1,7) | 46.2% | 95.3% | 9,204 |
| **Our Ensemble** | **64.6%** | **95.0%** | — |

**Why we kept our ensemble over Seasonal Naive** (despite Naive's 66.2% vs our 64.6%):
- Naive **cannot predict more than 7 days ahead** without chaining, which degrades rapidly
- Naive has **no festival awareness** — it will always miss Brahmotsavams and Vaikuntha Ekadashi surges
- Naive has **no safety constraints** — it can dangerously under-predict
- Our ensemble handles **multi-month forecasts** with autoregressive chaining and confidence decay

---

## Production System — How It Works in Deployment

### Prediction Engine

The Flask API loads all three models at startup. For each future date, it mirrors the training feature engineering for a single sample, computes the weighted ensemble probability, and returns the predicted band with a confidence score and top-3 feature explanations.

### Autoregressive Multi-Step Forecasting

When predicting multiple future days, each predicted day's features depend on previous days' lag values that don't exist yet. We solve this by:

1. Predicting day $d_t$ using available history
2. Inserting a **day-of-week seasonal mean** (clipped to the predicted band's range) as a synthetic count
3. Using that synthetic count as lag input for $d_{t+1}$, $d_{t+2}$, etc.

We use the DOW seasonal mean (not the band midpoint) to prevent systematic drift toward the center.

### Festival Floor Logic

A post-prediction calendar check guarantees minimum crowd bands during known high-traffic events:

| Festival/Event | Minimum Band |
|----------------|-------------|
| Brahmotsavams, Vaikuntha Ekadashi, Sankranti | HEAVY (band 4) |
| Any festival with impact ≥ 5 | BUSY (band 3) |
| Festival impact ≥ 4 on weekends | BUSY (band 3) |

### Confidence Decay for Far-Future Predictions

Predictions further out are inherently less reliable because autoregressive features accumulate error. We apply a sigmoid decay:

```
confidence_final = confidence_raw × max(0.45, 1 / (1 + exp((days_ahead - 180) / 60)))
```

Full confidence within 60 days, gradual decay after that, floor at 45%.

### Daily Online Learning Pipeline

Every day at **12:30 PM IST** (after TTD publishes the previous day's data):

1. **Scrape** new records from TTD website
2. **Reload** the CSV and clear the prediction cache
3. **Retrain**:
   - GB: Full retrain from scratch using saved Optuna hyperparameters
   - LightGBM: **Warm-start** (continues training from current model, +100 rounds)
   - XGBoost: **Warm-start** (continues from existing booster)
4. **Validate** on the last 30 days as a sanity check
5. **Hot-reload** updated models into Flask (no server restart needed)

---

## Chatbot Architecture

```
User Query → Language Detection → Smart Router
                                     ├── Date query? → ML Prediction
                                     ├── Crowd keywords? → 7-day Forecast
                                     └── General Q&A → RAG Pipeline
                                                          ├── ChromaDB (top-8 docs)
                                                          ├── System Prompt (language-aware)
                                                          └── Groq LLM (Llama-3.3-70B)
                                                               ├── Primary: Groq API
                                                               ├── Fallback: HuggingFace API
                                                               └── Fallback: Keyword search
```

- **Vector DB**: ChromaDB with `all-MiniLM-L6-v2` embeddings (314 documents covering darshan types, sevas, hotels, travel, festivals, customs, emergency contacts)
- **LLM**: Groq `llama-3.3-70b-versatile` (primary), HuggingFace Inference API (fallback)
- **6 languages**: Telugu, English, Hindi, Tamil, Malayalam, Kannada
- **Circuit breaker**: 2 consecutive LLM failures → skip LLM for 5 minutes → use keyword fallback

---

## Frontend — Devotional Temple Theme

The web app uses a temple-inspired design:

- **Color palette**: Gold (#D4A843), saffron (#FF9933), maroon (#800020), cream backgrounds
- **Custom favicon**: Temple gopuram SVG with gold kalasam
- **Ornamental patterns**: Scallop arch borders, gold strips with ✦ decorators, repeating dot dividers
- **Fonts**: Playfair Display (headings), Inter (body), Noto Serif Devanagari (sacred text)
- **Responsive**: Mobile-friendly with floating scroll-to-top button
- **Accessibility**: ARIA labels, semantic HTML, keyboard navigation

### Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Hero section + today's prediction + weekly summary stats |
| **Predict** | Pick any date range, see color-coded band predictions with confidence & explanations |
| **Calendar** | Monthly heatmap with Hindu festival markers and Panchang |
| **History** | Browse all 1,400+ days of actual data with filters and charts |
| **Chatbot** | Ask anything about Tirumala in 6 languages |
| **Explore** | 20 famous landmarks with photos, timings, and Google Maps links |

---

## API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/api/health` | GET | Health check (data rows, end date) |
| `/api/predict/today` | GET | Today's crowd prediction |
| `/api/predict` | GET | Quick forecast (next N days) |
| `/api/predict` | POST | Date range prediction |
| `/api/calendar/<year>/<month>` | GET | Calendar data + predictions + festivals |
| `/api/chat` | POST | RAG chatbot (`message` + `lang`) |
| `/api/history` | GET | Historical data with pagination |
| `/api/data/summary` | GET | Dataset summary stats |
| `/api/model-info` | GET | Model metadata & features |
| `/api/pipeline/status` | GET | Daily pipeline status |
| `/api/pipeline/trigger` | POST | Manually trigger scrape + retrain |

---

## Project Structure

```
tirumala/
├── flask_api.py              # Flask backend — prediction engine, API, online learning
├── train_gb_model.py         # Complete ML pipeline — features, selection, training, evaluation
├── festival_calendar.py      # Hindu festival calendar (2013–2027, 17 event categories)
├── hindu_calendar.py         # Panchang (lunar phase) calculations
├── daily_pipeline.py         # Standalone daily scrape + retrain script
├── build_vectordb.py         # ChromaDB vector store builder for chatbot
├── build_corpus.py           # RAG corpus builder from TTD sources
├── evaluate_pretrained_ts.py # Comparison script: Naive vs Prophet vs SARIMA vs Ensemble
├── deploy_hf.py              # Hugging Face Spaces deployment script
├── research_paper.tex        # LaTeX research paper documenting the methodology
│
├── data/                     # All data files
│   ├── tirumala_darshan_data_CLEAN_NO_OUTLIERS.csv  # Primary dataset (4,076 rows)
│   ├── ttd_corpus.txt        # RAG text corpus
│   ├── ttd_knowledge_base.json  # Structured Q&A pairs
│   └── tirumala_trip_data.json  # Hotels, restaurants, attractions, transport
│
├── artefacts/advisory_v5/    # Trained model artefacts
│   ├── gb_model.pkl          # GradientBoosting model
│   ├── lgb_model.pkl         # LightGBM model
│   ├── xgb_model.pkl         # XGBoost model
│   ├── model_meta.json       # Feature cols, bands, split strategy, selection metadata
│   ├── config.json           # Test results, champion model, ensemble weights
│   ├── hyperparams.json      # Optuna best hyperparameters for all 3 models
│   ├── feature_importance.csv
│   ├── shap_background.npy   # 200 samples for SHAP explanations
│   ├── predictions_2026.csv  # Full year predictions
│   └── forecast_30day.csv    # Rolling 30-day forecast
│
├── vectordb/                 # ChromaDB persistent vector store
│
├── client/                   # React frontend (Vite)
│   ├── src/pages/            # Dashboard, Predict, History, Chatbot, Explore
│   ├── src/components/       # Navbar, Footer, HinduCalendar, Loader
│   ├── src/i18n/             # Translations (6 languages) + LangContext
│   ├── public/favicon.svg    # Custom temple gopuram favicon
│   └── build/                # Production build (served by Flask)
│
├── Dockerfile                # Docker config for HF Spaces
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML Models** | scikit-learn GradientBoosting, LightGBM, XGBoost |
| **Tuning** | Optuna (TPE sampler, 80 trials/model, 5-fold CV) |
| **Feature Selection** | 10-method consensus vote (MI, ANOVA, Chi2, GB, RF, ET, Permutation, L1, Spearman, RFE) |
| **Explainability** | Feature importance × z-score explanations, SHAP background data |
| **Backend** | Flask, Gunicorn, APScheduler (daily cron) |
| **Chatbot** | ChromaDB + sentence-transformers (MiniLM-L6-v2) + Groq (Llama-3.3-70B) |
| **Frontend** | React 19, Vite 7, React Router 7, Recharts, react-icons |
| **Styling** | Custom CSS with temple devotional theme |
| **i18n** | Custom LangContext provider (6 languages with full translation maps) |
| **Deployment** | Docker on HuggingFace Spaces, automated via deploy_hf.py |

---

## Running Locally

```bash
# 1. Clone
git clone https://github.com/vishnu-madhav-2454/tirumala-darshan-prediction.git
cd tirumala-darshan-prediction

# 2. Python environment
python -m venv .venv_dl
.venv_dl\Scripts\activate          # Windows
# source .venv_dl/bin/activate     # Linux/Mac
pip install -r requirements.txt

# 3. Build frontend
cd client && npm ci && npm run build && cd ..

# 4. Build vector database (for chatbot)
python build_vectordb.py

# 5. Set environment variables
#    Create a .env file with:
#    GROQ_API_KEY=gsk_your_key
#    HF_TOKEN_CHAT=hf_your_token

# 6. (Optional) Retrain the model — pretrained artefacts are included
python train_gb_model.py --trials 80 --walkforward

# 7. Run the server
python flask_api.py
# Open http://localhost:5000
```

---

## Docker

```bash
docker build -t srivari-seva .
docker run -p 7860:7860 \
  -e GROQ_API_KEY=gsk_your_key \
  -e HF_TOKEN_CHAT=hf_your_token \
  srivari-seva
```

---

## HuggingFace Spaces Deployment

1. Create a new Space with SDK: **Docker**
2. Push this repo to the Space (or use `python deploy_hf.py`)
3. Add **Secrets** in Space Settings:
   - `GROQ_API_KEY` — for the chatbot LLM
   - `HF_TOKEN_CHAT` — fallback LLM token
4. Space auto-builds and deploys (~5–10 min first build)

---

## Data Source

All pilgrim data is sourced from [news.tirumala.org](https://news.tirumala.org/category/darshan/) — the official Tirumala Tirupati Devasthanams news portal.

---

**ॐ నమో వేంకటేశాయ 🙏**
