"""
🛕 Tirumala Darshan Prediction
================================
Pick a date → see how many pilgrims are expected.

Launch:   streamlit run app/dashboard.py
"""
import os, sys, json, threading, time, logging
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import DATA_CSV, LGB_MODEL_PATH

# ────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tirumala Darshan Prediction",
    page_icon="🛕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ────────────────────────────────────────────────────────────────────
#  CSS
# ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --gold: #C5A028; --gold-l: #E8D48B; --gold-d: #8B7118;
    --maroon: #800020; --maroon-d: #5C0018;
    --cream: #FFF8E7;
}

/* global */
.stApp { background: linear-gradient(180deg, var(--cream) 0%, #fff 50%, #F8F6F0 100%); }
#MainMenu, footer, header { visibility: hidden; }

/* header */
.hdr { background: linear-gradient(135deg, var(--maroon) 0%, var(--maroon-d) 60%, #3D0010 100%);
       padding: 2.2rem 2rem 1.8rem; border-radius: 18px; text-align: center;
       box-shadow: 0 8px 40px rgba(128,0,32,.12); margin-bottom: 1.5rem; position: relative; overflow: hidden; }
.hdr::before { content:''; position: absolute; inset: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="none" stroke="rgba(197,160,40,.07)" stroke-width=".5"/><circle cx="50" cy="50" r="25" fill="none" stroke="rgba(197,160,40,.05)" stroke-width=".5"/></svg>');
    background-size: 180px; opacity: .5; }
.hdr h1 { font-family: 'Playfair Display',serif; color: var(--gold); font-size: 2.4rem;
           font-weight: 700; margin: 0; text-shadow: 0 2px 8px rgba(0,0,0,.3); position: relative; }
.hdr p  { color: var(--gold-l); font-family: 'Inter',sans-serif; font-size: .95rem;
           margin: .4rem 0 0; font-weight: 300; position: relative; }
.hdr .div { width: 70px; height: 2px; background: var(--gold); margin: .7rem auto; position: relative; }

/* prediction card */
.pred { background: linear-gradient(135deg, var(--maroon), var(--maroon-d));
        border-radius: 22px; padding: 2.6rem 2rem; text-align: center;
        box-shadow: 0 10px 50px rgba(128,0,32,.15); position: relative; overflow: hidden; }
.pred::after { content:''; position: absolute; top: -50%; right: -50%;
    width: 100%; height: 100%;
    background: radial-gradient(circle, rgba(197,160,40,.08) 0%, transparent 70%); }
.pred .lbl { font-family: 'Inter',sans-serif; color: var(--gold-l);
             font-size: .85rem; text-transform: uppercase; letter-spacing: 2.5px;
             margin-bottom: .4rem; position: relative; }
.pred .num { font-family: 'Playfair Display',serif; color: var(--gold);
             font-size: 4rem; font-weight: 800; position: relative;
             text-shadow: 0 2px 12px rgba(197,160,40,.3); }
.pred .sub { font-family: 'Inter',sans-serif; color: rgba(255,248,231,.65);
             font-size: .85rem; margin-top: .3rem; position: relative; }

/* info card */
.icard { background: #fff; border: 1px solid var(--gold-l); border-radius: 16px;
         padding: 1.5rem; text-align: center;
         box-shadow: 0 4px 20px rgba(128,0,32,.07); }
.icard .il { font-family: 'Inter',sans-serif; color: var(--gold-d);
             font-size: .75rem; text-transform: uppercase; letter-spacing: 1.5px; }
.icard .iv { font-family: 'Playfair Display',serif; color: var(--maroon);
             font-size: 1.6rem; font-weight: 700; margin: .3rem 0; }
.icard .is { font-family: 'Inter',sans-serif; color: #aaa; font-size: .75rem; }

/* actual vs predicted (past date) */
.past-box { background: #fff; border-radius: 18px; padding: 2rem 1.5rem;
            box-shadow: 0 6px 30px rgba(128,0,32,.08);
            border-top: 4px solid var(--gold); text-align: center; }
.past-box .pb-label { font-family: 'Inter',sans-serif; color: #888;
                      font-size: .75rem; text-transform: uppercase; letter-spacing: 1.5px; }
.past-box .pb-val   { font-family: 'Playfair Display',serif; font-size: 2.6rem;
                      font-weight: 800; margin: .3rem 0; }
.past-box .pb-tag   { font-family: 'Inter',sans-serif; font-size: .75rem;
                      padding: .25rem .7rem; border-radius: 12px; display: inline-block; }

/* crowd badge */
.crowd { display: inline-block; padding: .45rem 1.2rem; border-radius: 25px;
         font-family: 'Inter',sans-serif; font-size: .9rem; font-weight: 600;
         letter-spacing: .5px; margin: .5rem 0; }

/* section heading */
.sec { font-family: 'Playfair Display',serif; color: var(--maroon);
       font-size: 1.5rem; font-weight: 600;
       border-bottom: 2px solid var(--gold-l); padding-bottom: .4rem;
       margin: 2rem 0 1rem; }

/* footer */
.ftr { text-align: center; padding: 1.8rem 0 .8rem; color: #bbb;
       font-family: 'Inter',sans-serif; font-size: .78rem;
       border-top: 1px solid var(--gold-l); margin-top: 2.5rem; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────
#  BACKGROUND AUTO-UPDATER (silent — user never sees this)
# ────────────────────────────────────────────────────────────────────
def _bg_auto_update():
    """Scrape new data + retrain models every 6 hours, silently."""
    import schedule as sched
    def _run():
        try:
            from app.scraper import scrape_incremental
            from app.trainer import retrain as do_retrain
            added = scrape_incremental(max_pages=5)
            if added > 0:
                do_retrain(force=False)
        except Exception:
            pass
    _run()
    sched.every(6).hours.do(_run)
    while True:
        sched.run_pending()
        time.sleep(120)

if "bg_started" not in st.session_state:
    st.session_state.bg_started = True
    threading.Thread(target=_bg_auto_update, daemon=True).start()

# ────────────────────────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    if not os.path.exists(DATA_CSV):
        return None
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    return df[["date", "total_pilgrims"]].dropna().sort_values("date").reset_index(drop=True)

def fmt(n):
    return f"{int(n):,}" if n is not None else "—"

def crowd(n):
    if n < 50000:  return "🟢 Low Crowd",    "#4CAF50", "#E8F5E9", "Comfortable darshan — shorter waiting times"
    if n < 65000:  return "🟡 Moderate",      "#F57F17", "#FFF8E1", "Normal crowd — standard waiting times"
    if n < 80000:  return "🟠 High Crowd",    "#E65100", "#FFF3E0", "Heavy crowd — longer waiting times likely"
    return            "🔴 Very High Crowd", "#C62828", "#FFEBEE", "Peak crowd — carry essentials, plan for long wait"

# ────────────────────────────────────────────────────────────────────
#  HEADER
# ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
    <h1>🛕 Tirumala Darshan Prediction</h1>
    <div class="div"></div>
    <p>Know the expected crowd before you plan your visit</p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────
#  DATE PICKER — front and centre
# ────────────────────────────────────────────────────────────────────
raw = load_data()
if raw is None or len(raw) == 0:
    st.info("⏳ Loading data… please refresh in a moment.")
    st.stop()
if not os.path.exists(LGB_MODEL_PATH):
    st.info("⏳ Models are getting ready… please refresh in a moment.")
    st.stop()

last_data = raw["date"].max().date()
_, centre, _ = st.columns([1, 2, 1])
with centre:
    selected = st.date_input(
        "**Select a date to check the expected crowd**",
        value=last_data + timedelta(days=1),
        min_value=date(2023, 1, 1),
        max_value=last_data + timedelta(days=90),
    )

# ────────────────────────────────────────────────────────────────────
#  PREDICTION
# ────────────────────────────────────────────────────────────────────
with st.spinner(""):
    from app.predictor import predict_single_date
    result = predict_single_date(selected)

is_past = result["is_actual"]

if is_past:
    # ── PAST DATE: show actual + predicted side-by-side ──
    actual  = result.get("actual")
    predicted = result.get("predicted")

    st.markdown(f'<div class="sec">{selected.strftime("%A, %d %B %Y")}</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        lbl_c, _, _ , _  = crowd(actual) if actual else ("—", "#888", "#fff", "")
        st.markdown(f"""
        <div class="past-box">
            <div class="pb-label">Actual Pilgrims</div>
            <div class="pb-val" style="color: #1B5E20;">{fmt(actual)}</div>
            <div class="pb-tag" style="background: #E8F5E9; color: #2E7D32;">✅ Recorded</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        if predicted is not None:
            err = abs(actual - predicted)
            err_pct = (err / actual * 100) if actual else 0
            acc_pct = 100 - err_pct
            st.markdown(f"""
            <div class="past-box">
                <div class="pb-label">AI Predicted</div>
                <div class="pb-val" style="color: var(--maroon);">{fmt(predicted)}</div>
                <div class="pb-tag" style="background: #FFF3E0; color: #E65100;">
                    🎯 {acc_pct:.1f}% accurate (off by {fmt(err)})
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="past-box">
                <div class="pb-label">AI Predicted</div>
                <div class="pb-val" style="color: #aaa;">—</div>
                <div class="pb-tag" style="background: #f5f5f5; color: #888;">
                    Not enough data to predict
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── FUTURE DATE: show prediction prominently ──
    pred_val = result["predicted"]
    lbl, clr, bg, advice = crowd(pred_val)
    days_ahead = result.get("days_ahead", 1)
    conf_lo = result.get("confidence_low", pred_val)
    conf_hi = result.get("confidence_high", pred_val)

    _, card_col, _ = st.columns([1, 2, 1])
    with card_col:
        st.markdown(f"""
        <div class="pred">
            <div class="lbl">Expected Pilgrims</div>
            <div class="num">{fmt(pred_val)}</div>
            <div class="sub">{selected.strftime('%A, %d %B %Y')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="icard" style="border-left: 4px solid {clr};">
            <div class="il">Crowd Level</div>
            <div class="iv" style="color:{clr}; font-size:1.3rem;">{lbl}</div>
            <div class="is">{advice}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="icard">
            <div class="il">Expected Range</div>
            <div class="iv">{fmt(conf_lo)} — {fmt(conf_hi)}</div>
            <div class="is">Possible range of pilgrims</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        day_name = selected.strftime("%A")
        st.markdown(f"""
        <div class="icard">
            <div class="il">Day</div>
            <div class="iv" style="font-size:1.3rem;">{day_name}</div>
            <div class="is">{days_ahead} day{'s' if days_ahead != 1 else ''} from now</div>
        </div>
        """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
#  7-DAY TREND GRAPH
# ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="sec">📈 7-Day Trend from {selected.strftime('%d %B %Y')}</div>
""", unsafe_allow_html=True)

with st.spinner(""):
    trend_data = []
    for i in range(7):
        d = selected + timedelta(days=i)
        r = predict_single_date(d)
        val = r["actual"] if r["is_actual"] and r["actual"] else r["predicted"]
        trend_data.append({
            "date": d,
            "day": d.strftime("%a\n%d %b"),
            "pilgrims": val if val else 0,
            "type": "Actual" if r["is_actual"] else "Predicted",
        })
    trend_df = pd.DataFrame(trend_data)

# Build the chart
colors = []
for _, row in trend_df.iterrows():
    if row["type"] == "Actual":
        colors.append("#2E7D32")
    else:
        _, c, _, _ = crowd(row["pilgrims"])
        colors.append(c)

fig = go.Figure()

# Actual bars
actual_mask = trend_df["type"] == "Actual"
if actual_mask.any():
    fig.add_trace(go.Bar(
        x=trend_df.loc[actual_mask, "day"],
        y=trend_df.loc[actual_mask, "pilgrims"],
        marker_color="#2E7D32",
        name="Actual",
        text=[fmt(v) for v in trend_df.loc[actual_mask, "pilgrims"]],
        textposition="outside",
        textfont=dict(family="Inter", size=11, color="#2C1810"),
        hovertemplate="<b>%{x}</b><br>Actual: %{y:,.0f} pilgrims<extra></extra>",
    ))

# Predicted bars
pred_mask = trend_df["type"] == "Predicted"
if pred_mask.any():
    pred_colors = [c for c, t in zip(colors, trend_df["type"]) if t == "Predicted"]
    fig.add_trace(go.Bar(
        x=trend_df.loc[pred_mask, "day"],
        y=trend_df.loc[pred_mask, "pilgrims"],
        marker_color=pred_colors,
        name="Predicted",
        text=[fmt(v) for v in trend_df.loc[pred_mask, "pilgrims"]],
        textposition="outside",
        textfont=dict(family="Inter", size=11, color="#2C1810"),
        hovertemplate="<b>%{x}</b><br>Predicted: %{y:,.0f} pilgrims<extra></extra>",
    ))

fig.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    height=430,
    margin=dict(l=50, r=30, t=10, b=50),
    yaxis=dict(title="Pilgrims", gridcolor="#f0e8d8",
               title_font=dict(family="Inter", size=12)),
    xaxis=dict(title_font=dict(family="Inter", size=12)),
    font=dict(family="Inter"),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(family="Inter", size=12)),
    bargap=0.25,
)
# Threshold lines
fig.add_hline(y=65000, line_dash="dot", line_color="#F57F17", line_width=1,
              annotation_text="Moderate", annotation_position="right",
              annotation_font=dict(size=10, color="#F57F17"))
fig.add_hline(y=80000, line_dash="dot", line_color="#E65100", line_width=1,
              annotation_text="High", annotation_position="right",
              annotation_font=dict(size=10, color="#E65100"))

st.plotly_chart(fig, width="stretch")


# ────────────────────────────────────────────────────────────────────
#  RECENT HISTORY (last 30 days line chart)
# ────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📊 Recent Pilgrim Footfall</div>', unsafe_allow_html=True)

recent = raw.tail(60).copy()
recent["ma7"] = recent["total_pilgrims"].rolling(7).mean()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=recent["date"], y=recent["total_pilgrims"],
    mode="lines+markers", name="Daily Count",
    line=dict(color="#C5A028", width=1.5),
    marker=dict(size=3),
    hovertemplate="%{x|%d %b %Y}<br>Pilgrims: %{y:,.0f}<extra></extra>",
))
fig2.add_trace(go.Scatter(
    x=recent["date"], y=recent["ma7"],
    mode="lines", name="7-Day Average",
    line=dict(color="#800020", width=2.5),
    hovertemplate="%{x|%d %b %Y}<br>7-day avg: %{y:,.0f}<extra></extra>",
))
fig2.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    height=380,
    margin=dict(l=50, r=30, t=10, b=40),
    xaxis=dict(gridcolor="#f5f0e0"),
    yaxis=dict(title="Pilgrims", gridcolor="#f5f0e0",
               title_font=dict(family="Inter", size=12)),
    hovermode="x unified",
    font=dict(family="Inter"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(family="Inter", size=12)),
)
st.plotly_chart(fig2, width="stretch")


# ────────────────────────────────────────────────────────────────────
#  FOOTER
# ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ftr">
    <strong>Sri Venkateswara Swamy Devasthanam, Tirumala</strong><br>
    Data updates automatically from official TTD sources<br>
    <span style="color:#C5A028;">ॐ नमो वेंकटेशाय 🙏</span>
</div>
""", unsafe_allow_html=True)
