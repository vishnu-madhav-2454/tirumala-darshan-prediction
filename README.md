---
title: Srivari Seva
emoji: 🛕
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---

# 🛕 Srivari Seva — Tirumala Crowd Advisory

AI-powered crowd prediction & trip planning for **Tirumala Sri Venkateswara Temple**.

🔗 **Live Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/madhav456789123/tirumala-darshan-prediction)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Crowd Prediction** | 6-band ML forecast (QUIET → EXTREME) using a 3-model ensemble (GB + LightGBM + XGBoost) |
| 📅 **Hindu Calendar** | Monthly crowd heatmap with festival indicators & Panchang |
| 🤖 **AI Chatbot** | RAG-powered Q&A about TTD darshan, travel, sevas (Groq Llama-3.3-70B) |
| 🗺️ **Explore Places** | 20 famous landmarks around Tirumala & Tirupati with photos, timings & maps |
| 📈 **History** | Browse 1,400+ days of actual pilgrim data with filters & charts |
| 🌐 **Multilingual** | 6 languages — English, Telugu, Hindi, Tamil, Malayalam, Kannada |
| 📱 **Responsive** | Mobile-friendly design for all screen sizes |

---

## 📊 ML Pipeline — End to End

### 1. Data Collection & Cleaning

- **Source**: [news.tirumala.org](https://news.tirumala.org/category/darshan/) — official TTD news portal
- **Period**: Feb 2022 – Feb 2026 (post-COVID era only)
- **Raw records**: 1,469 daily pilgrim counts
- **After cleaning**: 1,104 rows (outliers removed, missing dates handled)
- **Automated daily pipeline**: Scrapes new data at 12:30 PM IST via APScheduler

### 2. Target Variable — 6 Crowd Bands

| Band ID | Name | Pilgrim Range | % of Data |
|---------|------|---------------|-----------|
| 0 | QUIET | 0 – 50,000 | 0.7% |
| 1 | LIGHT | 50,000 – 60,000 | 9.5% |
| 2 | MODERATE | 60,000 – 70,000 | 37.4% |
| 3 | BUSY | 70,000 – 80,000 | 36.1% |
| 4 | HEAVY | 80,000 – 90,000 | 15.0% |
| 5 | EXTREME | 90,000+ | 1.3% |

### 3. Feature Engineering — 57 Features (8 Groups)

| # | Feature Group | Count | Examples |
|---|---------------|-------|----------|
| 1 | **Calendar** | 5 | `dow`, `month`, `is_weekend`, `sin_doy`, `cos_doy` |
| 2 | **Lag Features** | 7 | `L1`, `L2`, `L7`, `L14`, `L21`, `L28`, `L365` |
| 3 | **Rolling Stats** | 5 | `rm7`, `rm14`, `rm30`, `rstd7`, `rstd14` |
| 4 | **Expanding Means** | 2 | `dow_expanding_mean`, `month_dow_mean` |
| 5 | **Log Transforms** | 5 | `log_L1`, `log_L7`, `log_rm7`, `log_rm30`, `log_L365` |
| 6 | **Derived / Interaction** | 10 | `momentum_7`, `ewm7`, `ewm14`, `trend_7_14`, `month_weekend` |
| 7 | **Regime Counts** | 2 | `heavy_extreme_count7`, `light_quiet_count7` |
| 8 | **Festival Features** | ~21 | `is_festival`, `fest_impact`, `is_brahmotsavam`, `is_vaikuntha_ekadashi`, etc. |

Festival features sourced from a hand-curated calendar (`festival_calendar.py`) spanning 2013–2027 with 17 event categories and 5-level impact scoring.

### 4. Feature Selection — 10-Method Consensus Voting

Instead of relying on a single method, we use **10 independent feature selection methods** and only keep features selected by **≥6 out of 10**:

| # | Method | Type |
|---|--------|------|
| 1 | Mutual Information | Non-linear dependency |
| 2 | ANOVA F-test | Linear class separability |
| 3 | Chi-squared test | Non-negative feature relevance |
| 4 | GradientBoosting importance | Tree-based impurity |
| 5 | Random Forest importance | Decorrelated trees |
| 6 | Extra Trees importance | Extremely randomised trees |
| 7 | Permutation importance | Accuracy drop on shuffle |
| 8 | L1 Logistic Regression | Non-zero Lasso coefficients |
| 9 | Spearman rank correlation | Monotonic target dependency |
| 10 | Recursive Feature Elimination | Backward elimination (DecisionTree) |

**Pre-filter**: Variance filter drops features where ≥98% of values are identical (removed 11 rare festival features).  
**Post-filter**: Redundancy removal — if two features have Spearman |ρ| > 0.95, drop the one with fewer votes (removed 6 redundant features).

**Result**: 57 → **16 features** (sample:feature ratio improved from 19.4:1 → **69.0:1**)

<details>
<summary><strong>Final 16 Selected Features (click to expand)</strong></summary>

| Feature | Votes (out of 10) | Description |
|---------|--------------------|-------------|
| `dow_expanding_mean` | 10 | Expanding mean pilgrim count per day-of-week |
| `rm7` | 10 | 7-day rolling mean (kept after redundancy check) |
| `cos_doy` | 10 | Cosine of day-of-year (seasonality) |
| `L21` | 10 | 21-day lag |
| `month_dow_mean` | 9 | Mean pilgrim count by month × day-of-week |
| `log_L1` | 9 | Log of 1-day lag |
| `log_L7` | 9 | Log of 7-day lag |
| `log_rm30` | 9 | Log of 30-day rolling mean |
| `month_weekend` | 9 | Month × weekend interaction |
| `L14` | 9 | 14-day lag |
| `rm14` | 8 | 14-day rolling mean |
| `dow` | 8 | Day of week |
| `L28` | 8 | 28-day lag |
| `ewm7` | 7 | 7-day exponentially weighted mean |
| `rstd7` | 6 | 7-day rolling standard deviation |
| `heavy_extreme_count7` | 6 | Number of HEAVY/EXTREME days in past 7 |

</details>

### 5. Stratified Data Split

Using `StratifiedShuffleSplit` to ensure all 6 classes are proportionally represented:

| Split | % | Samples | EXTREME class samples |
|-------|---|---------|----------------------|
| Train | 72% | 794 | 10 |
| Calibration | 8% | 89 | 1 |
| Test | 20% | 221 | 3 |

Calibration set is used **only** for ensemble weight tuning — prevents leakage into test evaluation.

### 6. Custom Scoring — Ordinal Score

Since crowd bands are **ordinal** (QUIET < LIGHT < ... < EXTREME), standard accuracy isn't enough. Our custom scoring function:

$$\text{ordinal\_score} = 0.30 \times \text{exact\_accuracy} + 0.30 \times \text{macro\_F1} + 0.25 \times \left(1 - \frac{\text{MAE}}{N_{\text{bands}} - 1}\right) + 0.15 \times \text{safety}$$

Where **safety** = fraction of predictions that aren't dangerously wrong (predicting QUIET/LIGHT when actual is HEAVY/EXTREME).

### 7. Phase 1 — Baseline Models (No Tuning)

Three gradient boosting models with sensible defaults:

| Model | Default Config |
|-------|---------------|
| **GradientBoosting** | n_est=500, depth=5, lr=0.05, subsample=0.8 |
| **LightGBM** | n_est=500, depth=6, lr=0.05, leaves=31, class_weight=balanced |
| **XGBoost** | n_est=500, depth=6, lr=0.05, subsample=0.8 |

**Baseline 5-fold CV**: GB=0.6573, LGB=0.6526, XGB=0.6669

| Model | Exact% | Adjacent% | Macro-F1% | MAE | Safety% |
|-------|--------|-----------|-----------|-----|---------|
| GB | 55.7 | 96.8 | 33.0 | 0.475 | 100.0 |
| LGB | 55.7 | 95.9 | 32.9 | 0.484 | 100.0 |
| XGB | 53.8 | 95.5 | 32.2 | 0.507 | 100.0 |
| **ENS** | **54.8** | **96.8** | **32.6** | **0.484** | **100.0** |

Baseline ensemble weights: GB=0.50, LGB=0.10, XGB=0.40

### 8. Phase 2 — Optuna Hyperparameter Tuning

**80 trials per model** using TPE sampler with 5-fold StratifiedKFold cross-validation.

<details>
<summary><strong>Optuna search spaces (click to expand)</strong></summary>

| Parameter | GB | LGB | XGB |
|-----------|----|----|-----|
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

**Best CV scores**: GB=0.6815, LGB=0.6602, XGB=0.6801

### 9. Ensemble Calibration

Grid search over weight triplets (GB, LGB, XGB) optimised on the held-out **calibration** split:

**Optuna ensemble weights**: GB=0.10, LGB=0.50, XGB=0.40

### 10. Final Test Results

| Model | Exact% | Adjacent% | Macro-F1% | MAE | Safety% |
|-------|--------|-----------|-----------|-----|---------|
| GB | 59.7 | 96.8 | 37.8 | 0.434 | 100.0 |
| LGB | 54.3 | **97.7** | 31.5 | 0.480 | 100.0 |
| XGB | 56.6 | 95.0 | 35.8 | 0.484 | 100.0 |
| **ENS** | **57.0** | **97.3** | **34.5** | **0.457** | **100.0** |

**Baseline → Optuna improvement**: +0.5pp adjacent accuracy (96.8% → 97.3%)  
**Champion model**: LGB (adj=97.7%)  
**All models achieved 100% safety** — zero dangerous under-predictions.

Per-band breakdown (Ensemble):

| Band | n | Exact% | Adjacent% |
|------|---|--------|-----------|
| QUIET | 1 | 0.0 | 100.0 |
| LIGHT | 21 | 33.3 | 95.2 |
| MODERATE | 83 | 66.3 | 97.6 |
| BUSY | 80 | 60.0 | 98.8 |
| HEAVY | 33 | 48.5 | 97.0 |
| EXTREME | 3 | 0.0 | 66.7 |

### 11. Walk-Forward Validation (Temporal Out-of-Sample)

Expanding-window re-training over 360 days with step_size=30 — the most rigorous test since it simulates real deployment:

| Metric | Value |
|--------|-------|
| Exact accuracy | 59.2% |
| **Adjacent accuracy** | **98.3%** |
| Macro-F1 | 44.5% |
| MAE | 0.425 bands |
| **Safety** | **100.0%** |

| Band | n | Exact% | Adjacent% |
|------|---|--------|-----------|
| LIGHT | 19 | 42.1 | 94.7 |
| MODERATE | 117 | 75.2 | 97.4 |
| BUSY | 143 | 57.3 | 99.3 |
| HEAVY | 69 | 50.7 | 98.6 |
| EXTREME | 12 | 0.0 | 100.0 |

### Top-15 Feature Importances (GradientBoosting)

```
month_dow_mean          9.81%  ###################
dow_expanding_mean      9.76%  ###################
L14                     8.23%  ################
ewm7                    8.00%  ################
rm14                    7.37%  ##############
L21                     7.30%  ##############
log_rm7                 7.12%  ##############
log_L7                  7.10%  ##############
log_L1                  6.82%  #############
log_rm30                6.23%  ############
rstd7                   5.98%  ###########
L28                     5.18%  ##########
cos_doy                 4.74%  #########
dow                     2.65%  #####
month_weekend           2.57%  #####
```

---

## 🤖 Chatbot Architecture

```
User Query → Language Detection → Smart Router
                                     ├── Date query? → ML Prediction
                                     ├── Crowd keywords? → 7-day Forecast
                                     └── General Q&A → RAG Pipeline
                                                          ├── ChromaDB (top-8 docs)
                                                          ├── System Prompt (lang-aware)
                                                          └── Groq LLM (Llama-3.3-70B)
                                                               ├── Primary: Groq API
                                                               ├── Fallback: HuggingFace API
                                                               └── Fallback: Keyword search
```

- **Vector DB**: ChromaDB with `all-MiniLM-L6-v2` embeddings (~90MB, CPU)
- **Corpus**: TTD knowledge base + trip data + scraped content
- **LLM**: Groq `llama-3.3-70b-versatile` (primary), HuggingFace (fallback)
- **Circuit Breaker**: After 2 consecutive LLM failures, skips LLM for 5 minutes
- **6 languages**: The `lang` parameter is sent with every request; the system prompt forces response in the target language's native script

---

## 🌐 Supported Languages

| Code | Language | Script |
|------|----------|--------|
| `en` | English | English |
| `te` | Telugu | తెలుగు |
| `hi` | Hindi | हिन्दी |
| `ta` | Tamil | தமிழ் |
| `ml` | Malayalam | മലയാളം |
| `kn` | Kannada | ಕನ್ನಡ |

---

## 🔌 API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/api/health` | GET | Health check (data rows, end date) |
| `/api/predict/today` | GET | Today's crowd prediction |
| `/api/predict` | GET | Quick forecast (next N days, max 90) |
| `/api/predict` | POST | Date range prediction |
| `/api/calendar/<year>/<month>` | GET | Calendar data + predictions + festivals |
| `/api/chat` | POST | RAG chatbot (`message` + `lang`) |
| `/api/history` | GET | Historical data with pagination |
| `/api/data/summary` | GET | Dataset summary stats |
| `/api/model-info` | GET | Model metadata & features |
| `/api/pipeline/status` | GET | Daily pipeline status |
| `/api/pipeline/trigger` | POST | Manually trigger scrape + retrain |

---

## 🚀 Local Development

```bash
# 1. Clone & setup
git clone <repo-url> && cd tirumala
python -m venv .venv_dl && .venv_dl\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Build frontend
cd client && npm ci && npm run build && cd ..

# 3. Build vector database (for chatbot)
python build_vectordb.py

# 4. Set environment variables
cp .env.example .env  # Add your Groq / HuggingFace tokens

# 5. Train the model (optional — pre-trained artefacts included)
python train_gb_model.py --trials 80 --walkforward

# 6. Run
python flask_api.py
# Open http://localhost:5000
```

---

## 🐳 Docker

```bash
docker build -t srivari-seva .
docker run -p 7860:7860 \
  -e GROQ_API_KEY=gsk_your_key \
  -e HF_TOKEN_CHAT=hf_your_token \
  srivari-seva
```

---

## 🔧 HuggingFace Spaces Deployment

1. Create a new Space → SDK: **Docker**
2. Push this repo to the Space
3. Add **Secrets** in Space Settings:
   - `GROQ_API_KEY` — Groq API key (primary LLM)
   - `HF_TOKEN_CHAT` — HuggingFace token (fallback LLM)
4. The Space will auto-build and deploy

---

## 📁 Project Structure

```
tirumala/
├── flask_api.py              # Flask backend (API + static serving)
├── train_gb_model.py         # Full ML pipeline (features, selection, training)
├── festival_calendar.py      # Hindu festival calendar (2013–2027, 17 categories)
├── hindu_calendar.py         # Panchang calculations
├── daily_pipeline.py         # Automated daily scrape + retrain
├── build_vectordb.py         # ChromaDB vector store builder
├── build_corpus.py           # RAG corpus builder
├── ttd_corpus.txt            # RAG corpus for chatbot
├── ttd_knowledge_base.json   # Structured TTD knowledge
├── tirumala_trip_data.json   # Trip planning data
├── artefacts/advisory_v5/    # Trained models + metadata
│   ├── gb_model.pkl          # GradientBoosting model
│   ├── lgb_model.pkl         # LightGBM model
│   ├── xgb_model.pkl         # XGBoost model
│   ├── config.json           # Test results + ensemble weights
│   ├── model_meta.json       # Feature cols, split strategy, selection info
│   ├── hyperparams.json      # Optuna best hyperparameters
│   ├── feature_importance.csv
│   ├── predictions_2026.csv  # Full year predictions
│   └── forecast_30day.csv    # 30-day rolling forecast
├── vectordb/                 # ChromaDB persistent store
├── client/                   # React frontend (Vite)
│   ├── src/pages/            # Dashboard, Predict, History, Chatbot, Explore
│   ├── src/components/       # Navbar, Footer, HinduCalendar, Loader
│   ├── src/i18n/             # Translations (6 languages) + LangContext
│   └── build/                # Production build (served by Flask)
├── Dockerfile                # HF Spaces deployment
├── requirements.txt          # Python dependencies
└── .env.example              # Environment variable template
```

---

## 📦 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML Models** | scikit-learn GradientBoosting, LightGBM, XGBoost |
| **Tuning** | Optuna (TPE sampler, 80 trials/model) |
| **Feature Selection** | 10-method consensus voting |
| **Explainability** | SHAP (background saved for inference-time explanations) |
| **Backend** | Flask, Gunicorn, APScheduler |
| **Chatbot** | ChromaDB + sentence-transformers + Groq (Llama-3.3-70B) |
| **Frontend** | React 18, Vite, React Router, Recharts |
| **i18n** | Custom LangContext (6 languages with full translation sets) |
| **Deployment** | Docker, HuggingFace Spaces |

---

## 📌 Data Source

Pilgrim data sourced from [news.tirumala.org](https://news.tirumala.org/category/darshan/) — the official TTD news portal.

---

**ॐ నమో వేంకటేశాయ 🙏**
