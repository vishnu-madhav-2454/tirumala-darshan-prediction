import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { format } from "date-fns";
import { MdCalendarToday, MdTrendingUp, MdAutoGraph, MdPeople, MdInfo } from "react-icons/md";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, ReferenceLine,
} from "recharts";
import { predictDays, getDataSummary } from "../api";
import HinduCalendar from "../components/HinduCalendar";
import Loader from "../components/Loader";
import { useLang } from "../i18n/LangContext";

const BAND_EMOJI = {
  QUIET: "🔵", LIGHT: "🟢", MODERATE: "🟡",
  BUSY: "🟠", HEAVY: "🔴", EXTREME: "⛔",
};

export default function Dashboard() {
  const navigate = useNavigate();
  const { t } = useLang();
  const [todayData, setTodayData] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [tRes, s] = await Promise.all([predictDays(7), getDataSummary()]);
        const forecast = tRes.data.forecast || [];
        const todayStr = new Date().toISOString().slice(0, 10);
        let todayPrediction = forecast[0] || null;
        for (const f of forecast) {
          if (f.date === todayStr) { todayPrediction = f; break; }
        }
        setTodayData({ today: todayStr, today_prediction: todayPrediction, week_forecast: forecast });
        setSummary(s.data);
      } catch {
        setError(true);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <Loader text={t.loading} />;
  if (error) return (
    <div style={{ padding: "3rem", textAlign: "center" }}>
      <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🙏</div>
      <div style={{ fontSize: "1.1rem", color: "var(--maroon)", marginBottom: ".5rem" }}>{t.serviceError}</div>
      <p style={{ color: "var(--text-muted)", fontSize: ".9rem" }}>{t.serviceErrorSub}</p>
    </div>
  );

  const todayPred = todayData?.today_prediction;
  const weekForecast = todayData?.week_forecast || [];
  const stats = summary?.pilgrim_stats;
  const todayDate = new Date();
  const todayStr = todayDate.toISOString().slice(0, 10);

  /* Chart data */
  const chartData = weekForecast.map((f) => ({
    date: format(new Date(f.date + "T00:00:00"), "EEE, MMM d"),
    pilgrims: f.predicted_pilgrims || 0,
    band: f.predicted_band || f.band_name || "MODERATE",
    color: f.color || "#8BC34A",
    confidence: f.confidence,
    isToday: f.date === todayStr,
    isActual: f.is_actual || false,
  }));
  const avgPilgrims = stats?.mean || 0;

  const ChartTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <div className="dash-tooltip">
        <div className="dash-tooltip-date">{d.date}</div>
        <div className="dash-tooltip-val" style={{ color: d.color }}>
          {BAND_EMOJI[d.band]} {d.band}
          {d.isActual && <span style={{ marginLeft: 6, fontSize: ".7rem", color: "#388E3C" }}>✓ actual</span>}
        </div>
        <div className="dash-tooltip-pilgrims">
          <MdPeople size={14} /> ~{d.pilgrims.toLocaleString("en-IN")} pilgrims
        </div>
        {!d.isActual && (
          <div className="dash-tooltip-conf">{(d.confidence * 100).toFixed(0)}% confidence</div>
        )}
      </div>
    );
  };

  return (
    <div className="fade-in">
      {/* Hero */}
      <section className="hero">
        <div className="hero-content">
          <p className="om-text">{t.omText}</p>
          <h1>{t.heroTitle}</h1>
          <p style={{ fontSize: "1.05rem", marginBottom: ".25rem" }}>{t.heroSub}</p>
          <p style={{ opacity: 0.7, fontSize: ".9rem" }}>{t.heroPlan}</p>
          <div style={{ marginTop: "1.25rem", display: "inline-block", background: "rgba(197,160,40,.15)", padding: ".5rem 1.25rem", borderRadius: "9999px", border: "1px solid rgba(197,160,40,.3)" }}>
            <span style={{ color: "var(--gold-light)", fontWeight: 600, fontSize: ".95rem" }}>
              📅 {format(todayDate, "EEEE, MMMM d, yyyy")}
            </span>
          </div>
          <div className="quick-predict">
            <button className="btn btn-primary" onClick={() => navigate("/predict")}>
              <MdCalendarToday /> {t.btnPickDate}
            </button>
          </div>
        </div>
      </section>

      <div className="gold-strip" />

      <div className="main-content">
        {/* Today's prediction — FULL CARD */}
        {todayPred && (
          <div className="card today-prediction-card" style={{ borderLeft: `5px solid ${todayPred.color || "#FFC107"}` }}>
            <div className="card-header">
              <MdTrendingUp className="icon" />
              <h2>{t.todayPredTitle} — {format(todayDate, "EEE, MMM d, yyyy")}</h2>
            </div>
            <div className="card-body">
              <div className="today-grid">
                {/* Band + Pilgrims */}
                <div className="today-main">
                  <div className="today-band-badge" style={{
                    background: todayPred.bg || "#FFF8E1",
                    color: todayPred.color || "#FFC107",
                    border: `2px solid ${todayPred.color || "#FFC107"}`,
                  }}>
                    {BAND_EMOJI[todayPred.predicted_band]} {todayPred.predicted_band}
                    {todayPred.is_actual && (
                      <span style={{ marginLeft: 8, fontSize: ".7rem", color: "#388E3C", fontWeight: 700 }}>✓ actual</span>
                    )}
                  </div>
                  <div className="today-pilgrims">
                    <MdPeople size={20} />
                    <span>~{(todayPred.predicted_pilgrims || 0).toLocaleString("en-IN")}</span>
                    <span className="today-pilgrims-label">estimated pilgrims</span>
                  </div>
                  <div className="today-confidence">
                    {(todayPred.confidence * 100).toFixed(0)}% confidence
                  </div>
                </div>
                {/* Advice + Reason */}
                <div className="today-details">
                  {todayPred.advice && (
                    <div className="today-advice">
                      <MdInfo size={16} />
                      <span>{todayPred.advice}</span>
                    </div>
                  )}
                  {todayPred.reason && (
                    <div className="today-reason">
                      <strong>Why?</strong> {todayPred.reason.split(" | ").map((r, i) => (
                        <span key={i} className="reason-tag">{r}</span>
                      ))}
                    </div>
                  )}
                  {todayPred.is_weekend && (
                    <div className="today-weekend-tag">📅 Weekend — expect higher footfall</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Stats grid */}
        {stats && (
          <div className="stat-grid">
            <div className="stat-card">
              <div className="stat-icon">🛕</div>
              <div className="stat-value">{stats.mean?.toLocaleString("en-IN")}</div>
              <div className="stat-label">{t.avgDaily}</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">🔱</div>
              <div className="stat-value">{stats.max?.toLocaleString("en-IN")}</div>
              <div className="stat-label">{t.highestRecord}</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">📅</div>
              <div className="stat-value">{summary?.total_records?.toLocaleString("en-IN")}</div>
              <div className="stat-label">{t.totalDays}</div>
              <div className="stat-sub">{summary?.date_range?.start} {t.since}</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">🙏</div>
              <div className="stat-value">{stats.median?.toLocaleString("en-IN")}</div>
              <div className="stat-label">{t.medianDaily}</div>
            </div>
          </div>
        )}

        {/* Hindu Calendar */}
        <HinduCalendar />

        {/* 7-day forecast chart — BAR CHART with crowd colors */}
        {weekForecast.length > 0 && (
          <div className="card">
            <div className="card-header">
              <MdAutoGraph className="icon" />
              <h2>{t.next7Days}</h2>
            </div>
            <div className="card-body">
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={chartData} barCategoryGap="15%">
                    <CartesianGrid strokeDasharray="3 3" stroke="#F5EDDA" vertical={false} />
                    <XAxis dataKey="date" tick={{ fill: "#6B5B4E", fontSize: 12 }} />
                    <YAxis tick={{ fill: "#6B5B4E", fontSize: 11 }}
                           tickFormatter={(v) => (v / 1000).toFixed(0) + "K"} />
                    <Tooltip content={<ChartTooltip />} />
                    {avgPilgrims > 0 && (
                      <ReferenceLine y={avgPilgrims} stroke="#800020" strokeDasharray="6 4"
                        label={{ value: `Avg: ${(avgPilgrims / 1000).toFixed(0)}K`, fill: "#800020", fontSize: 11, position: "right" }} />
                    )}
                    <Bar dataKey="pilgrims" radius={[6, 6, 0, 0]}>
                      {chartData.map((entry, idx) => (
                        <Cell key={idx} fill={entry.color} fillOpacity={entry.isToday ? 1 : 0.75}
                              stroke={entry.isToday ? "#800020" : "none"} strokeWidth={entry.isToday ? 2 : 0} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* Week forecast cards */}
        {weekForecast.length > 0 && (
          <div className="forecast-grid">
            {weekForecast.map((f) => {
              const band = f.predicted_band || f.band_name || "MODERATE";
              const color = f.color || "#8BC34A";
              const isToday = f.date === todayStr;
              const isActual = f.is_actual || false;
              return (
                <div key={f.date} className={`forecast-card ${isToday ? "forecast-today" : ""}`}
                     style={{ borderLeft: `4px solid ${isActual ? "#388E3C" : color}` }}>
                  <div className="fc-date-row">
                    <div>
                      <div className="fc-date">{format(new Date(f.date + "T00:00:00"), "MMM d, yyyy")}</div>
                      <div className={`fc-day ${f.is_weekend ? "fc-weekend" : ""}`}>
                        {f.day} {f.is_weekend ? "🗓️" : ""}
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                      {isToday && <span className="fc-today-badge">TODAY</span>}
                      {isActual ? (
                        <span style={{ fontSize: ".65rem", color: "#388E3C", fontWeight: 700, background: "#E8F5E9", padding: "1px 6px", borderRadius: 10 }}>✓ actual</span>
                      ) : (
                        <span style={{ fontSize: ".6rem", color: "#999", background: "#F5F5F5", padding: "1px 6px", borderRadius: 10 }}>🔮 predicted</span>
                      )}
                    </div>
                  </div>
                  <div className="fc-band" style={{ color }}>
                    {BAND_EMOJI[band]} {band}
                  </div>
                  <div className="fc-pilgrims">
                    <MdPeople size={14} /> ~{(f.predicted_pilgrims || 0).toLocaleString("en-IN")}
                  </div>
                  {!isActual && (
                    <div className="fc-conf">{(f.confidence * 100).toFixed(0)}% confidence</div>
                  )}
                  {f.reason && (
                    <div className="fc-reason">{f.reason.split(" | ")[0]}</div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Slogan */}
        <div className="divider" style={{ margin: "2.5rem 0" }}>
          <span style={{ color: "var(--gold)", fontFamily: "Playfair Display, serif", fontSize: "1.1rem", fontWeight: 600, whiteSpace: "nowrap" }}>
            {t.slogan}
          </span>
        </div>
      </div>
    </div>
  );
}
