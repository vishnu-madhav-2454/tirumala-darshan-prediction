import { useState } from "react";
import { format } from "date-fns";
import { MdTimeline, MdAutoGraph } from "react-icons/md";
import { GiTempleDoor } from "react-icons/gi";
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { predictDays } from "../api";
import CrowdBadge from "../components/CrowdBadge";
import Loader from "../components/Loader";
import { useLang } from "../i18n/LangContext";

const CROWD_COLORS = {
  "Very Low": "#22C55E",
  Low: "#84CC16",
  Moderate: "#F59E0B",
  High: "#F97316",
  "Very High": "#EF4444",
};

export default function Forecast() {
  const { t } = useLang();
  const [days, setDays] = useState(7);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [view, setView] = useState("chart");

  async function handleForecast() {
    setLoading(true);
    setError(null);
    try {
      const res = await predictDays(days);
      setData(res.data);
    } catch (e) {
      setError(e.response?.data?.error || e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="main-content fade-in">
      <h2 className="section-title">
        <GiTempleDoor className="ornament" />
        {t.forecastTitle}
      </h2>

      {/* Controls */}
      <div className="card" style={{ maxWidth: 600 }}>
        <div className="card-header">
          <MdTimeline className="icon" />
          <h2>{t.forecastSettings}</h2>
        </div>
        <div className="card-body">
          <div style={{ display: "flex", gap: "1rem", alignItems: "flex-end", flexWrap: "wrap" }}>
            <div className="form-group" style={{ flex: "1 1 auto", minWidth: "120px", maxWidth: "200px", marginBottom: 0 }}>
              <label className="form-label">{t.daysAhead}</label>
              <input type="number" className="form-input" min={1} max={90} value={days} onChange={(e) => setDays(Math.max(1, Math.min(90, +e.target.value)))} />
            </div>
            <div style={{ display: "flex", gap: ".5rem", flexWrap: "wrap" }}>
              {[7, 14, 30].map((d) => (
                <button key={d} className={`btn ${days === d ? "btn-maroon" : "btn-secondary"}`} onClick={() => setDays(d)} style={{ minWidth: "60px" }}>
                  {d}d
                </button>
              ))}
            </div>
            <button className="btn btn-primary" onClick={handleForecast} disabled={loading} style={{ minWidth: "140px" }}>
              <MdAutoGraph /> {t.btnGenerate}
            </button>
          </div>
        </div>
      </div>

      {loading && <Loader text={`${days} ${t.forecastLoading}`} />}
      {error && <div className="error-message" style={{ marginTop: "1.5rem" }}>⚠️ {error}</div>}

      {data?.forecast && (
        <>
          {/* View toggle */}
          <div className="tab-bar" style={{ marginTop: "2rem" }}>
            <button className={`tab-btn ${view === "chart" ? "active" : ""}`} onClick={() => setView("chart")}>{t.chartView}</button>
            <button className={`tab-btn ${view === "cards" ? "active" : ""}`} onClick={() => setView("cards")}>{t.cardView}</button>
          </div>

          {/* Chart view */}
          {view === "chart" && (
            <div className="chart-container" style={{ marginBottom: "1.5rem" }}>
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={data.forecast}>
                  <defs>
                    <linearGradient id="forecastGold" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#C5A028" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#C5A028" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F5EDDA" />
                  <XAxis dataKey="date" tick={{ fill: "#6B5B4E", fontSize: 11 }} tickFormatter={(v) => format(new Date(v + "T00:00:00"), "MMM d")} interval={Math.max(0, Math.floor(data.forecast.length / 10))} />
                  <YAxis tick={{ fill: "#6B5B4E", fontSize: 12 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                  <Tooltip
                    contentStyle={{ background: "#FFF", border: "1px solid #C5A028", borderRadius: 8, fontSize: 13 }}
                    formatter={(v, name) => {
                      const label = name === "predicted_pilgrims" ? t.predicted : name === "confidence_low" ? t.minimum : t.maximum;
                      return [v.toLocaleString(), label];
                    }}
                    labelFormatter={(v) => format(new Date(v + "T00:00:00"), "EEE, MMM d yyyy")}
                  />
                  <Area type="monotone" dataKey="confidence_low" stroke="#E8D48B" strokeDasharray="4 4" fill="none" strokeWidth={1} />
                  <Area type="monotone" dataKey="confidence_high" stroke="#E8D48B" strokeDasharray="4 4" fill="none" strokeWidth={1} />
                  <Area type="monotone" dataKey="predicted_pilgrims" stroke="#C5A028" strokeWidth={3} fill="url(#forecastGold)" dot={{ fill: "#800020", stroke: "#C5A028", strokeWidth: 2, r: 4 }} activeDot={{ r: 6, fill: "#C5A028" }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Bar chart */}
          {view === "chart" && (
            <div className="chart-container">
              <h3 style={{ fontSize: "1rem", color: "var(--maroon)", marginBottom: "1rem" }}>{t.crowdByDay}</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={data.forecast}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F5EDDA" />
                  <XAxis dataKey="date" tick={{ fill: "#6B5B4E", fontSize: 11 }} tickFormatter={(v) => format(new Date(v + "T00:00:00"), "MMM d")} interval={Math.max(0, Math.floor(data.forecast.length / 10))} />
                  <YAxis tick={{ fill: "#6B5B4E", fontSize: 12 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                  <Tooltip contentStyle={{ background: "#FFF", border: "1px solid #C5A028", borderRadius: 8, fontSize: 13 }} formatter={(v) => [v.toLocaleString(), t.pilgrims]} labelFormatter={(v) => format(new Date(v + "T00:00:00"), "EEE, MMM d yyyy")} />
                  <Bar dataKey="predicted_pilgrims" radius={[4, 4, 0, 0]}>
                    {data.forecast.map((entry, i) => (
                      <Cell key={i} fill={CROWD_COLORS[entry.crowd_level] || "#C5A028"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ display: "flex", justifyContent: "center", gap: "1rem", marginTop: ".75rem", flexWrap: "wrap" }}>
                {Object.entries(CROWD_COLORS).map(([level, color]) => (
                  <div key={level} style={{ display: "flex", alignItems: "center", gap: ".3rem", fontSize: ".78rem" }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: color }} />
                    {level}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Cards view */}
          {view === "cards" && (
            <div className="forecast-grid">
              {data.forecast.map((f) => (
                <div key={f.date} className="forecast-card">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div>
                      <div className="fc-date">{format(new Date(f.date + "T00:00:00"), "MMM d, yyyy")}</div>
                      <div className="fc-day">{f.day}</div>
                    </div>
                    <CrowdBadge level={f.crowd_level} />
                  </div>
                  <div className="fc-value">{f.predicted_pilgrims.toLocaleString()}</div>
                  <div className="fc-range">{t.rangeLabel}: {f.confidence_low.toLocaleString()} — {f.confidence_high.toLocaleString()}</div>
                  <div style={{ fontSize: ".75rem", color: "var(--text-light)" }}>{f.days_ahead} {t.daysLater}</div>
                </div>
              ))}
            </div>
          )}

          {/* Summary stats */}
          <div className="stat-grid" style={{ marginTop: "2rem" }}>
            <div className="stat-card">
              <div className="stat-icon">📊</div>
              <div className="stat-value">{Math.round(data.forecast.reduce((s, f) => s + f.predicted_pilgrims, 0) / data.forecast.length).toLocaleString()}</div>
              <div className="stat-label">{t.avgDailyPilgrims}</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">📈</div>
              <div className="stat-value">{Math.max(...data.forecast.map((f) => f.predicted_pilgrims)).toLocaleString()}</div>
              <div className="stat-label">{t.peakDay}</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">📉</div>
              <div className="stat-value">{Math.min(...data.forecast.map((f) => f.predicted_pilgrims)).toLocaleString()}</div>
              <div className="stat-label">{t.quietDay}</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">🔮</div>
              <div className="stat-value">{data.forecast.reduce((s, f) => s + f.predicted_pilgrims, 0).toLocaleString()}</div>
              <div className="stat-label">{t.totalExpected}</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
