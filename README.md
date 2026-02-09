# 🛕 Tirumala Darshan Prediction

A web application that predicts the **daily pilgrim count** at Tirumala Sri Venkateswara Swamy Temple.

Pick a date → see how many pilgrims are expected → plan your trip accordingly.

---

## 🚀 How to Run

```bash
# Activate the environment
.venv_dl\Scripts\activate

# Launch the website
streamlit run app/dashboard.py
```

Open **http://localhost:8501** → pick a date → get the prediction.

---

## ✨ Features

- 📅 **Date Picker** — select any date from 2023 to 90 days ahead
- 🔮 **Pilgrim Prediction** — AI-powered crowd forecast
- 📊 **Past Dates** — shows both actual count AND what was predicted (accuracy check)
- 🚦 **Crowd Level** — Low / Moderate / High / Very High
- 📈 **7-Day Trend** — visual bar chart of the upcoming week
- 📉 **Recent Footfall** — last 60 days of actual pilgrim data
- 🔄 **Auto-Updates** — data refreshes automatically from official TTD sources

---

## 📁 Project Structure

```
tirumala/
├── app/
│   ├── config.py        # Configuration
│   ├── features.py      # Feature engineering
│   ├── scraper.py       # Data scraper (news.tirumala.org)
│   ├── trainer.py       # Model training
│   ├── predictor.py     # Prediction engine
│   ├── dashboard.py     # Main website (Streamlit)
│   ├── server.py        # REST API (optional)
│   └── scheduler.py     # Pipeline orchestrator
├── artefacts/           # Saved models & scalers (auto-created)
├── tirumala_darshan_data_clean.csv
├── requirements.txt
└── .gitignore
```

---

## 📌 Data Source

All pilgrim data is sourced from [news.tirumala.org](https://news.tirumala.org/category/darshan/) — the official TTD news portal.

---

**ॐ नमो वेंकटेशाय 🙏**
