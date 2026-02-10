import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { format } from "date-fns";
import {
  MdCalendarToday,
  MdTrendingUp,
  MdAutoGraph,
} from "react-icons/md";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell,
} from "recharts";
import { predictDays, getDataSummary } from "../api";
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

const TIP_KEYS = {
  "Very Low": "tipVeryLow",
  Low: "tipLow",
  Moderate: "tipModerate",
  High: "tipHigh",
  "Very High": "tipVeryHigh",
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
        const [tRes, s] = await Promise.all([
          predictDays(7),
          getDataSummary(),
        ]);
        const forecast = tRes.data.forecast || [];
        const todayStr = new Date().toISOString().slice(0, 10);
        let todayPrediction = forecast[0] || null;
        for (const f of forecast) {
          if (f.date === todayStr) { todayPrediction = f; break; }
        }
        setTodayData({
          today: todayStr,
          today_prediction: todayPrediction,
          week_forecast: forecast,
        });
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

  if (error)
    return (
      <div style={{ padding: "3rem", textAlign: "center" }}>
        <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🙏</div>
        <div style={{ fontSize: "1.1rem", color: "var(--maroon)", marginBottom: ".5rem" }}>
          {t.serviceError}
        </div>
        <p style={{ color: "var(--text-muted)", fontSize: ".9rem" }}>
          {t.serviceErrorSub}
        </p>
      </div>
    );

  const todayPred = todayData?.today_prediction;
  const weekForecast = todayData?.week_forecast || [];
  const stats = summary?.pilgrim_stats;
  const todayDate = new Date();

  return (
    <div className="fade-in">
      {/* Hero */}
      <section className="hero">
        <div className="hero-content">
          <p className="om-text">{t.omText}</p>
          <h1>{t.heroTitle}</h1>
          <p style={{ fontSize: "1.05rem", marginBottom: ".25rem" }}>{t.heroSub}</p>
          <p style={{ opacity: 0.7, fontSize: ".9rem" }}>{t.heroPlan}</p>

          {/* Today's date */}
          <div style={{ marginTop: "1.25rem", display: "inline-block", background: "rgba(197,160,40,.15)", padding: ".5rem 1.25rem", borderRadius: "9999px", border: "1px solid rgba(197,160,40,.3)" }}>
            <span style={{ color: "var(--gold-light)", fontWeight: 600, fontSize: ".95rem" }}>
              📅 {format(todayDate, "EEEE, MMMM d, yyyy")}
            </span>
          </div>

          {/* Quick actions */}
          <div className="quick-predict">
            <button className="btn btn-primary" onClick={() => navigate("/predict")}>
              <MdCalendarToday /> {t.btnPickDate}
            </button>
            <button
              className="btn btn-secondary"
              style={{ color: "var(--cream)", borderColor: "var(--gold-light)" }}
              onClick={() => navigate("/forecast")}
            >
              <MdAutoGraph /> {t.btnWeekForecast}
            </button>
          </div>
        </div>
      </section>

      <div className="gold-strip" />

      <div className="main-content">
        {/* Today's prediction — main card */}
        {todayPred && (
          <div className="card" style={{ marginBottom: "2rem" }}>
            <div className="card-header">
              <MdTrendingUp className="icon" />
              <h2>
                {t.todayPredTitle} — {format(todayDate, "MMM d, yyyy")} ({format(todayDate, "EEEE")})
              </h2>
            </div>
            <div className="card-body">
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-around", flexWrap: "wrap", gap: "2rem" }}>
                {/* Big number */}
                <div className="prediction-result" style={{ padding: "1rem" }}>
                  <div className="prediction-value">
                    {todayPred.predicted_pilgrims?.toLocaleString()}
                  </div>
                  <div className="prediction-label">{t.estimatedPilgrims}</div>
                  <div className="prediction-confidence">
                    {todayPred.confidence_low?.toLocaleString()} — {todayPred.confidence_high?.toLocaleString()} {t.range}
                  </div>
                </div>

                {/* Crowd level + tip */}
                <div style={{ textAlign: "center", maxWidth: 300 }}>
                  <CrowdBadge level={todayPred.crowd_level} />
                  <div style={{ marginTop: "1rem", padding: "1rem", background: "var(--off-white)", borderRadius: "var(--radius-sm)", border: "1px solid var(--cream-dark)" }}>
                    <div style={{ fontSize: ".9rem", color: "var(--maroon)", lineHeight: 1.6 }}>
                      {t[TIP_KEYS[todayPred.crowd_level]] || t.tipDefault}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Stats grid */}
        <div className="stat-grid">
          <div className="stat-card">
            <div className="stat-icon">🛕</div>
            <div className="stat-value">{stats?.mean?.toLocaleString()}</div>
            <div className="stat-label">{t.avgDaily}</div>
            {t.avgDailySub && <div className="stat-sub">{t.avgDailySub}</div>}
          </div>
          <div className="stat-card">
            <div className="stat-icon">🔱</div>
            <div className="stat-value">{stats?.max?.toLocaleString()}</div>
            <div className="stat-label">{t.highestRecord}</div>
            {t.highestRecordSub && <div className="stat-sub">{t.highestRecordSub}</div>}
          </div>
          <div className="stat-card">
            <div className="stat-icon">📅</div>
            <div className="stat-value">{summary?.total_records?.toLocaleString()}</div>
            <div className="stat-label">{t.totalDays}</div>
            <div className="stat-sub">{summary?.date_range?.start} {t.since}</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">🙏</div>
            <div className="stat-value">{stats?.median?.toLocaleString()}</div>
            <div className="stat-label">{t.medianDaily}</div>
            {t.medianDailySub && <div className="stat-sub">{t.medianDailySub}</div>}
          </div>
        </div>

        {/* 7-day chart */}
        {weekForecast.length > 0 && (
          <div className="card">
            <div className="card-header">
              <MdAutoGraph className="icon" />
              <h2>{t.next7Days}</h2>
            </div>
            <div className="card-body">
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={weekForecast}>
                    <defs>
                      <linearGradient id="goldGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#C5A028" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#C5A028" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F5EDDA" />
                    <XAxis dataKey="date" tick={{ fill: "#6B5B4E", fontSize: 12 }} tickFormatter={(v) => format(new Date(v + "T00:00:00"), "EEE, MMM d")} />
                    <YAxis tick={{ fill: "#6B5B4E", fontSize: 12 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                    <Tooltip
                      contentStyle={{ background: "#FFF", border: "1px solid #C5A028", borderRadius: 8, fontSize: 13 }}
                      formatter={(v) => [v.toLocaleString() + " " + t.pilgrims, t.estimate]}
                      labelFormatter={(v) => format(new Date(v + "T00:00:00"), "EEEE, MMM d yyyy")}
                    />
                    <Area type="monotone" dataKey="predicted_pilgrims" stroke="#C5A028" strokeWidth={3} fill="url(#goldGrad)" dot={{ fill: "#800020", stroke: "#C5A028", strokeWidth: 2, r: 5 }} activeDot={{ r: 7, fill: "#C5A028" }} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Crowd bars */}
              <div style={{ marginTop: "1.5rem" }}>
                <h3 style={{ fontSize: ".95rem", color: "var(--maroon)", marginBottom: ".75rem" }}>{t.crowdLevelDaily}</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={weekForecast}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F5EDDA" />
                    <XAxis dataKey="date" tick={{ fill: "#6B5B4E", fontSize: 11 }} tickFormatter={(v) => format(new Date(v + "T00:00:00"), "EEE")} />
                    <YAxis tick={{ fill: "#6B5B4E", fontSize: 12 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                    <Tooltip contentStyle={{ background: "#FFF", border: "1px solid #C5A028", borderRadius: 8, fontSize: 13 }} formatter={(v) => [v.toLocaleString(), t.pilgrims]} labelFormatter={(v) => format(new Date(v + "T00:00:00"), "EEEE, MMM d")} />
                    <Bar dataKey="predicted_pilgrims" radius={[4, 4, 0, 0]}>
                      {weekForecast.map((entry, i) => (
                        <Cell key={i} fill={CROWD_COLORS[entry.crowd_level] || "#C5A028"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div style={{ display: "flex", justifyContent: "center", gap: "1rem", marginTop: ".5rem", flexWrap: "wrap" }}>
                  {Object.entries(CROWD_COLORS).map(([level, color]) => (
                    <div key={level} style={{ display: "flex", alignItems: "center", gap: ".3rem", fontSize: ".75rem" }}>
                      <div style={{ width: 10, height: 10, borderRadius: 2, background: color }} />
                      {level}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Week at a glance */}
        {weekForecast.length > 0 && (
          <div style={{ marginTop: "2rem" }}>
            <h3 className="section-title" style={{ fontSize: "1.2rem" }}>
              <span className="ornament">🗓️</span>
              {t.weekStatus}
            </h3>
            <div className="forecast-grid">
              {weekForecast.map((f) => (
                <div key={f.date} className="forecast-card">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div>
                      <div className="fc-date">{format(new Date(f.date + "T00:00:00"), "MMM d, yyyy")}</div>
                      <div className="fc-day">{f.day}</div>
                    </div>
                    <CrowdBadge level={f.crowd_level} />
                  </div>
                  <div className="fc-value">{f.predicted_pilgrims.toLocaleString()}</div>
                  <div className="fc-range">
                    {f.confidence_low.toLocaleString()} — {f.confidence_high.toLocaleString()} {t.pilgrims}
                  </div>
                </div>
              ))}
            </div>
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
