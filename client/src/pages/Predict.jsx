import { useEffect, useState } from "react";
import { format } from "date-fns";
import { MdAutoGraph } from "react-icons/md";
import { GiTempleDoor } from "react-icons/gi";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";
import { predictDays } from "../api";
import Loader from "../components/Loader";
import { useLang } from "../i18n/LangContext";

export default function Predict() {
  const { t } = useLang();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await predictDays(7);
        setData(res.data.forecast || []);
      } catch (e) {
        setError(e.response?.data?.error || e.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="main-content fade-in">
      <h2 className="section-title">
        <GiTempleDoor className="ornament" />
        {t.predictTitle}
      </h2>
      <p style={{ textAlign: "center", color: "var(--text-muted)", marginBottom: "1.5rem", fontSize: ".95rem" }}>
        {t.predictSub || "Predicted pilgrim footfall for the coming 7 days"}
      </p>

      {loading && <Loader text={t.predicting} />}
      {error && <div className="error-message">⚠️ {error}</div>}

      {data && data.length > 0 && (
        <>
          {/* Summary stats */}
          <div className="stat-grid" style={{ marginBottom: "2rem" }}>
            <div className="stat-card">
              <div className="stat-icon">📊</div>
              <div className="stat-value">
                {Math.round(
                  data.reduce((s, f) => s + f.predicted_pilgrims, 0) / data.length
                ).toLocaleString()}
              </div>
              <div className="stat-label">{t.avgDailyPilgrims || "Avg Daily Pilgrims"}</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">📈</div>
              <div className="stat-value">
                {Math.max(...data.map((f) => f.predicted_pilgrims)).toLocaleString()}
              </div>
              <div className="stat-label">{t.peakDay || "Peak Day"}</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">📉</div>
              <div className="stat-value">
                {Math.min(...data.map((f) => f.predicted_pilgrims)).toLocaleString()}
              </div>
              <div className="stat-label">{t.quietDay || "Quietest Day"}</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">🔮</div>
              <div className="stat-value">
                {data.reduce((s, f) => s + f.predicted_pilgrims, 0).toLocaleString()}
              </div>
              <div className="stat-label">{t.totalExpected || "Total Expected"}</div>
            </div>
          </div>

          {/* 7-day Area Chart */}
          <div className="card">
            <div className="card-header">
              <MdAutoGraph className="icon" />
              <h2>{t.next7Days || "Next 7 Days Prediction"}</h2>
            </div>
            <div className="card-body">
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={data}>
                    <defs>
                      <linearGradient id="predictGold" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#C5A028" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="#C5A028" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F5EDDA" />
                    <XAxis
                      dataKey="date"
                      tick={{ fill: "#6B5B4E", fontSize: 12 }}
                      tickFormatter={(v) =>
                        format(new Date(v + "T00:00:00"), "EEE, MMM d")
                      }
                    />
                    <YAxis
                      tick={{ fill: "#6B5B4E", fontSize: 12 }}
                      tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#FFF",
                        border: "1px solid #C5A028",
                        borderRadius: 8,
                        fontSize: 13,
                      }}
                      formatter={(v) => [
                        v.toLocaleString() + " " + t.pilgrims,
                        t.estimate || "Estimate",
                      ]}
                      labelFormatter={(v) =>
                        format(new Date(v + "T00:00:00"), "EEEE, MMM d yyyy")
                      }
                    />
                    <Area
                      type="monotone"
                      dataKey="confidence_low"
                      stroke="#E8D48B"
                      strokeDasharray="4 4"
                      fill="none"
                      strokeWidth={1}
                      name={t.minimum || "Low"}
                    />
                    <Area
                      type="monotone"
                      dataKey="confidence_high"
                      stroke="#E8D48B"
                      strokeDasharray="4 4"
                      fill="none"
                      strokeWidth={1}
                      name={t.maximum || "High"}
                    />
                    <Area
                      type="monotone"
                      dataKey="predicted_pilgrims"
                      stroke="#C5A028"
                      strokeWidth={3}
                      fill="url(#predictGold)"
                      dot={{
                        fill: "#800020",
                        stroke: "#C5A028",
                        strokeWidth: 2,
                        r: 5,
                      }}
                      activeDot={{ r: 7, fill: "#C5A028" }}
                      name={t.predicted || "Predicted"}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Day cards */}
          <div style={{ marginTop: "2rem" }}>
            <h3 className="section-title" style={{ fontSize: "1.2rem" }}>
              <span className="ornament">🗓️</span>
              {t.weekStatus || "Week at a Glance"}
            </h3>
            <div className="forecast-grid">
              {data.map((f) => (
                <div key={f.date} className="forecast-card">
                  <div>
                    <div className="fc-date">
                      {format(new Date(f.date + "T00:00:00"), "MMM d, yyyy")}
                    </div>
                    <div className="fc-day">{f.day}</div>
                  </div>
                  <div className="fc-value">
                    {f.predicted_pilgrims.toLocaleString()}
                  </div>
                  <div className="fc-range">
                    {f.confidence_low.toLocaleString()} —{" "}
                    {f.confidence_high.toLocaleString()} {t.pilgrims}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
