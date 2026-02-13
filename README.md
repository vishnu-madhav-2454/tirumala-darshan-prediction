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

🔗 **Live Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Crowd Prediction** | 6-band ML forecast (QUIET → EXTREME) using LightGBM + XGBoost ensemble |
| 📅 **Hindu Calendar** | Monthly crowd heatmap with festival indicators |
| 🤖 **AI Chatbot** | RAG-powered Q&A about TTD darshan, travel, sevas (Llama-3.3-70B) |
| 🗺️ **Trip Planner** | AI-generated itineraries with budget estimates (Qwen2.5-72B) |
| 📈 **History** | Browse 1,400+ days of actual pilgrim data with filters & charts |
| 🌐 **Multilingual** | English, Telugu (తెలుగు), Hindi (हिंदी) |
| 📱 **Responsive** | Mobile-friendly design for all screen sizes |

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
cp .env.example .env  # Edit with your HF tokens

# 5. Run
python flask_api.py
# Open http://localhost:5000
```

---

## 🐳 Docker

```bash
docker build -t srivari-seva .
docker run -p 7860:7860 \
  -e HF_TOKEN_CHAT=hf_your_token \
  -e HF_TOKEN_TRIP=hf_your_token \
  srivari-seva
```

---

## 🔧 HuggingFace Spaces Deployment

1. Create a new Space → SDK: **Docker**
2. Push this repo to the Space
3. Add **Secrets** in Space Settings:
   - `HF_TOKEN_CHAT` — HuggingFace token for chatbot LLM
   - `HF_TOKEN_TRIP` — HuggingFace token for trip planner LLM
4. The Space will auto-build and deploy

---

## 📁 Project Structure

```
tirumala/
├── flask_api.py              # Flask backend (API + static serving)
├── crowd_advisory_v5.py      # ML pipeline (training & features)
├── festival_calendar.py      # Hindu festival calendar
├── hindu_calendar.py         # Panchang calculations
├── build_vectordb.py         # ChromaDB vector store builder
├── tirumala_trip_data.json    # Trip planner knowledge base
├── ttd_corpus.txt            # RAG corpus for chatbot
├── artefacts/advisory_v5/    # Trained ML models (LGB + XGB)
├── vectordb/                 # ChromaDB vector store
├── client/                   # React frontend (Vite)
│   ├── src/pages/            # Dashboard, Predict, History, Chatbot, TripPlanner
│   ├── src/components/       # Navbar, Footer, Calendar, Loader
│   └── build/                # Production build (served by Flask)
├── Dockerfile                # HF Spaces deployment
└── requirements.txt          # Python dependencies
```

---

## 📌 Data Source

Pilgrim data sourced from [news.tirumala.org](https://news.tirumala.org/category/darshan/) — the official TTD news portal.

---

**ॐ नमो वेंकटेशाय 🙏**
