import { useState } from "react";
import { format } from "date-fns";
import { MdFilterList, MdAutoGraph } from "react-icons/md";
import { GiTempleDoor } from "react-icons/gi";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";
import { getHistory } from "../api";
import Loader from "../components/Loader";
import { useLang } from "../i18n/LangContext";

export default function History() {
  const { t } = useLang();
  const today = new Date().toISOString().slice(0, 10);
  const thirtyDaysAgo = new Date(Date.now() - 30 * 86400000)
    .toISOString()
    .slice(0, 10);

  const [startDate, setStartDate] = useState(thirtyDaysAgo);
  const [endDate, setEndDate] = useState(today);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleFilter() {
    if (!startDate || !endDate) return;
    setLoading(true);
    setError(null);
    try {
      const res = await getHistory(1, 5000, null, null, startDate, endDate);
      setData(res.data);
    } catch (e) {
      setError(e.response?.data?.error || e.message);
    } finally {
      setLoading(false);
    }
  }

  const records = data?.data || [];
  const total = data?.total_records || 0;

  // Compute stats
  const avgPilgrims =
    records.length > 0
      ? Math.round(
          records.reduce((s, r) => s + (r.total_pilgrims || 0), 0) /
            records.length
        )
      : 0;
  const maxPilgrims =
    records.length > 0
      ? Math.max(...records.map((r) => r.total_pilgrims || 0))
      : 0;
  const minPilgrims =
    records.length > 0
      ? Math.min(...records.map((r) => r.total_pilgrims || 0))
      : 0;

  return (
    <div className="main-content fade-in">
      <h2 className="section-title">
        <GiTempleDoor className="ornament" />
        {t.historyTitle}
      </h2>

      {/* Date Range Picker */}
      <div className="card" style={{ marginBottom: "1.5rem" }}>
        <div className="card-header">
          <MdFilterList className="icon" />
          <h2>{t.filterData || "Select Date Range"}</h2>
        </div>
        <div className="card-body">
          <div
            style={{
              display: "flex",
              gap: "1rem",
              alignItems: "flex-end",
              flexWrap: "wrap",
            }}
          >
            <div className="form-group" style={{ minWidth: 160, marginBottom: 0 }}>
              <label className="form-label">{t.startDate || "Start Date"}</label>
              <input
                type="date"
                className="form-input"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                max={endDate}
              />
            </div>
            <div className="form-group" style={{ minWidth: 160, marginBottom: 0 }}>
              <label className="form-label">{t.endDate || "End Date"}</label>
              <input
                type="date"
                className="form-input"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                min={startDate}
                max={today}
              />
            </div>
            <button
              className="btn btn-primary"
              onClick={handleFilter}
              disabled={loading}
            >
              <MdFilterList /> {t.btnFilter || "Show Data"}
            </button>
          </div>
        </div>
      </div>

      {loading && <Loader text={t.histLoading} />}
      {error && <div className="error-message">⚠️ {error}</div>}

      {records.length > 0 && !loading && (
        <>
          {/* Info badge */}
          <div className="info-tag" style={{ marginBottom: "1rem" }}>
            📊 {total.toLocaleString()} {t.records || "records"} —{" "}
            {format(new Date(startDate + "T00:00:00"), "MMM d, yyyy")} to{" "}
            {format(new Date(endDate + "T00:00:00"), "MMM d, yyyy")}
          </div>

          {/* Stats */}
          <div className="stat-grid" style={{ marginBottom: "1.5rem" }}>
            <div className="stat-card">
              <div className="stat-icon">📊</div>
              <div className="stat-value">{avgPilgrims.toLocaleString()}</div>
              <div className="stat-label">{t.avgDailyPilgrims || "Avg Daily"}</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">📈</div>
              <div className="stat-value">{maxPilgrims.toLocaleString()}</div>
              <div className="stat-label">{t.peakDay || "Peak Day"}</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">📉</div>
              <div className="stat-value">{minPilgrims.toLocaleString()}</div>
              <div className="stat-label">{t.quietDay || "Quietest Day"}</div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">📅</div>
              <div className="stat-value">{total.toLocaleString()}</div>
              <div className="stat-label">{t.totalDays || "Total Days"}</div>
            </div>
          </div>

          {/* Chart */}
          <div className="card">
            <div className="card-header">
              <MdAutoGraph className="icon" />
              <h2>{t.historyChart || "Pilgrim Footfall Trend"}</h2>
            </div>
            <div className="card-body">
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={450}>
                  <AreaChart data={records}>
                    <defs>
                      <linearGradient id="histGold" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#C5A028" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="#C5A028" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F5EDDA" />
                    <XAxis
                      dataKey="date"
                      tick={{ fill: "#6B5B4E", fontSize: 11 }}
                      tickFormatter={(v) =>
                        format(new Date(v + "T00:00:00"), "MMM d")
                      }
                      interval={Math.max(
                        0,
                        Math.floor(records.length / 12)
                      )}
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
                        v?.toLocaleString(),
                        t.pilgrims || "Pilgrims",
                      ]}
                      labelFormatter={(v) =>
                        format(new Date(v + "T00:00:00"), "EEE, MMM d yyyy")
                      }
                    />
                    <Area
                      type="monotone"
                      dataKey="total_pilgrims"
                      stroke="#C5A028"
                      strokeWidth={2.5}
                      fill="url(#histGold)"
                      dot={false}
                      activeDot={{
                        r: 5,
                        fill: "#800020",
                        stroke: "#C5A028",
                      }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      )}

      {data && records.length === 0 && !loading && (
        <div
          style={{
            textAlign: "center",
            padding: "3rem",
            color: "var(--text-muted)",
          }}
        >
          📊 {t.noData || "No data available for the selected date range."}
        </div>
      )}

      {/* Prompt user to select a range if no data loaded yet */}
      {!data && !loading && (
        <div
          style={{
            textAlign: "center",
            padding: "3rem",
            color: "var(--text-muted)",
            fontSize: ".95rem",
          }}
        >
          📅 {t.selectRange || "Select a date range above and click Show Data to view the pilgrim footfall chart."}
        </div>
      )}
    </div>
  );
}
