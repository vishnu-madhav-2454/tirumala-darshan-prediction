import { useState, useEffect, useMemo } from "react";
import { useLang } from "../i18n/LangContext";
import { predictDays, predictRange } from "../api";
import Loader from "../components/Loader";
import { GiTempleDoor } from "react-icons/gi";
import {
  MdAutoAwesome, MdCalendarMonth, MdTrendingUp, MdPeople,
  MdInfo, MdStar, MdBarChart,
} from "react-icons/md";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Cell, ReferenceLine,
} from "recharts";
import { format, parseISO, addDays } from "date-fns";

const BAND_EMOJI = {
  QUIET: "🔵", LIGHT: "🟢", MODERATE: "🟡",
  BUSY: "🟠", HEAVY: "🔴", EXTREME: "⛔",
};

export default function Predict() {
  const { t } = useLang();

  /* mode: "quick" or "range" */
  const [mode, setMode] = useState("quick");
  const [days, setDays] = useState(7);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [loading, setLoading] = useState(false);
  const [forecast, setForecast] = useState([]);
  const [summary, setSummary] = useState(null);
  const [bestDays, setBestDays] = useState([]);
  const [error, setError] = useState("");

  const todayStr = new Date().toISOString().slice(0, 10);
  const maxDate = format(addDays(new Date(), 90), "yyyy-MM-dd");

  /* auto-load 7-day forecast */
  useEffect(() => { fetchQuick(7); }, []);

  async function fetchQuick(n) {
    setLoading(true);
    setError("");
    setSummary(null);
    setBestDays([]);
    try {
      const res = await predictDays(n);
      const arr = res.data?.forecast || res.data || [];
      setForecast(Array.isArray(arr) ? arr : []);
    } catch { setError("Failed to fetch forecast"); }
    finally { setLoading(false); }
  }

  async function fetchRange() {
    if (!startDate || !endDate) { setError("Select both dates"); return; }
    if (startDate > endDate) { setError("Start date must be before end date"); return; }
    setLoading(true);
    setError("");
    try {
      const res = await predictRange(startDate, endDate);
      const arr = res.data?.predictions || res.data?.forecast || res.data || [];
      setForecast(Array.isArray(arr) ? arr : []);
      setSummary(res.data?.summary || null);
      setBestDays(res.data?.best_days || []);
    } catch { setError("Failed to fetch predictions"); }
    finally { setLoading(false); }
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (mode === "quick") fetchQuick(days);
    else fetchRange();
  }

  /* Chart data */
  const chartData = useMemo(() => forecast.map((f) => {
    const band = f.predicted_band || f.band_name || "MODERATE";
    return {
      date: f.date ? format(parseISO(f.date), "dd MMM") : "",
      fullDate: f.date ? format(parseISO(f.date), "EEE, dd MMM yyyy") : "",
      pilgrims: f.predicted_pilgrims || 0,
      band,
      color: f.color || "#8BC34A",
      confidence: f.confidence,
      isToday: f.date === todayStr,
      isWeekend: f.is_weekend,
    };
  }), [forecast, todayStr]);

  const avgPilgrims = useMemo(() => {
    if (chartData.length === 0) return 0;
    return Math.round(chartData.reduce((s, d) => s + d.pilgrims, 0) / chartData.length);
  }, [chartData]);

  /* Band distribution for summary */
  const bandDist = useMemo(() => {
    if (!summary) return null;
    return Object.entries(summary).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  }, [summary]);

  const PredictTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <div className="dash-tooltip">
        <div className="dash-tooltip-date">{d.fullDate}</div>
        <div className="dash-tooltip-val" style={{ color: d.color }}>
          {BAND_EMOJI[d.band]} {d.band}
        </div>
        <div className="dash-tooltip-pilgrims">
          <MdPeople size={14} /> ~{d.pilgrims.toLocaleString("en-IN")} pilgrims
        </div>
        <div className="dash-tooltip-conf">{(d.confidence * 100).toFixed(0)}% confidence</div>
      </div>
    );
  };

  return (
    <section className="page predict-page">
      <div className="page-header">
        <GiTempleDoor className="page-header-icon" />
        <h2>{t.predictTitle || "Crowd Prediction"}</h2>
        <p className="page-subtitle">
          <MdAutoAwesome style={{ verticalAlign: "middle", marginRight: 4, color: "#DAA520" }} />
          {t.predictSubtitle || "AI-powered pilgrim forecast for Tirumala"}
        </p>
      </div>

      {/* Mode toggle + form */}
      <div className="card predict-form-card">
        <div className="card-body">
          <div className="mode-toggle">
            <button className={`mode-btn ${mode === "quick" ? "active" : ""}`} onClick={() => setMode("quick")}>
              <MdTrendingUp /> {t.predictQuick || "Quick Forecast"}
            </button>
            <button className={`mode-btn ${mode === "range" ? "active" : ""}`} onClick={() => setMode("range")}>
              <MdCalendarMonth /> {t.predictRange || "Date Range"}
            </button>
          </div>
          <form className="predict-form" onSubmit={handleSubmit}>
            {mode === "quick" ? (
              <div className="form-group">
                <label className="form-label">{t.predictDays || "Forecast days"}</label>
                <select className="form-input" value={days} onChange={(e) => setDays(+e.target.value)}>
                  {[3, 5, 7, 10, 14, 30].map((d) => (
                    <option key={d} value={d}>Next {d} days</option>
                  ))}
                </select>
              </div>
            ) : (
              <div className="predict-range-inputs">
                <div className="form-group">
                  <label className="form-label">Start date</label>
                  <input type="date" className="form-input" value={startDate}
                         onChange={(e) => setStartDate(e.target.value)}
                         min={todayStr} max={maxDate} />
                </div>
                <div className="form-group">
                  <label className="form-label">End date</label>
                  <input type="date" className="form-input" value={endDate}
                         onChange={(e) => setEndDate(e.target.value)}
                         min={startDate || todayStr} max={maxDate} />
                </div>
              </div>
            )}
            <button type="submit" className="btn btn-primary" disabled={loading}>
              <MdAutoAwesome /> {loading ? "Loading..." : (t.predictBtn || "Predict")}
            </button>
          </form>
          {error && <div className="form-error">{error}</div>}
        </div>
      </div>

      {loading && <Loader text="Predicting crowd levels..." />}

      {/* Best Days + Summary (for date range mode) */}
      {bestDays.length > 0 && (
        <div className="card best-days-card">
          <div className="card-header"><MdStar /> Best Days to Visit</div>
          <div className="card-body">
            <div className="best-days-row">
              {bestDays.map((b, i) => (
                <div key={i} className="best-day-item" style={{ borderLeft: `4px solid ${b.color}` }}>
                  <div className="best-day-rank">#{i + 1}</div>
                  <div>
                    <div className="best-day-date">{format(parseISO(b.date), "EEE, dd MMM yyyy")}</div>
                    <div className="best-day-band" style={{ color: b.color }}>
                      {BAND_EMOJI[b.band_name]} {b.band_name} — {(b.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
            {bandDist && (
              <div className="band-dist-row">
                <MdBarChart size={16} />
                <span className="band-dist-label">Distribution:</span>
                {bandDist.map(([name, count]) => (
                  <span key={name} className="band-dist-tag">{name}: {count}d</span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Chart — crowd-colored bar chart */}
      {chartData.length > 0 && (
        <div className="card chart-card">
          <div className="card-header">
            <MdTrendingUp /> {t.predictChart || "Forecast Chart"} — {forecast.length} days
          </div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={chartData} barCategoryGap={forecast.length > 14 ? "5%" : "15%"}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E8D48B" vertical={false} />
                <XAxis dataKey="date" tick={{ fill: "#6B5B4E", fontSize: forecast.length > 14 ? 9 : 12 }}
                       interval={forecast.length > 20 ? Math.floor(forecast.length / 10) : 0}
                       angle={forecast.length > 14 ? -35 : 0}
                       textAnchor={forecast.length > 14 ? "end" : "middle"}
                       height={forecast.length > 14 ? 55 : 30} />
                <YAxis tick={{ fill: "#6B5B4E", fontSize: 11 }}
                       tickFormatter={(v) => (v / 1000).toFixed(0) + "K"} />
                <Tooltip content={<PredictTooltip />} />
                {avgPilgrims > 0 && (
                  <ReferenceLine y={avgPilgrims} stroke="#800020" strokeDasharray="6 4"
                    label={{ value: `Avg: ${(avgPilgrims / 1000).toFixed(0)}K`, fill: "#800020", fontSize: 11, position: "right" }} />
                )}
                <Bar dataKey="pilgrims" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, idx) => (
                    <Cell key={idx} fill={entry.color} fillOpacity={entry.isToday ? 1 : 0.8}
                          stroke={entry.isToday ? "#800020" : "none"} strokeWidth={entry.isToday ? 2 : 0} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Forecast Cards */}
      {forecast.length > 0 && (
        <div className="forecast-grid">
          {forecast.map((f, i) => {
            const band = f.predicted_band || f.band_name || "MODERATE";
            const color = f.color || "#8BC34A";
            const isToday = f.date === todayStr;
            return (
              <div key={i} className={`forecast-card ${isToday ? "forecast-today" : ""}`}
                   style={{ borderLeft: `4px solid ${color}` }}>
                <div className="fc-date-row">
                  <div>
                    <div className="forecast-date">
                      {f.date ? format(parseISO(f.date), "EEE, dd MMM") : `Day ${i + 1}`}
                    </div>
                    {f.is_weekend && <span className="fc-weekend-tag">Weekend</span>}
                  </div>
                  {isToday && <span className="fc-today-badge">TODAY</span>}
                </div>
                <div className="forecast-band" style={{ color }}>
                  {BAND_EMOJI[band]} {band}
                </div>
                <div className="forecast-pilgrims">
                  <MdPeople /> ~{(f.predicted_pilgrims || 0).toLocaleString("en-IN")}
                </div>
                {f.confidence && (
                  <div className="forecast-confidence">{Math.round(f.confidence * 100)}% confidence</div>
                )}
                {f.advice && (
                  <div className="fc-advice">
                    <MdInfo size={12} /> {f.advice}
                  </div>
                )}
                {f.reason && (
                  <div className="fc-reason">{f.reason.split(" | ")[0]}</div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
