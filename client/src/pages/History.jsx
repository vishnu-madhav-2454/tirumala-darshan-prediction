import { useEffect, useState } from "react";
import { format } from "date-fns";
import { MdChevronLeft, MdChevronRight, MdFilterList } from "react-icons/md";
import { GiTempleDoor } from "react-icons/gi";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";
import { getHistory } from "../api";
import Loader from "../components/Loader";
import { useLang } from "../i18n/LangContext";

export default function History() {
  const { t } = useLang();
  const [page, setPage] = useState(1);
  const [perPage] = useState(50);
  const [year, setYear] = useState("");
  const [month, setMonth] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [view, setView] = useState("table");

  async function load(p = page, y = year, m = month) {
    setLoading(true);
    setError(null);
    try {
      const res = await getHistory(p, perPage, y || undefined, m || undefined);
      setData(res.data);
    } catch (e) {
      setError(e.response?.data?.error || e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  function handleFilter() {
    setPage(1);
    load(1, year, month);
  }

  function goPage(p) {
    setPage(p);
    load(p);
  }

  const years = [];
  for (let y = 2013; y <= 2026; y++) years.push(y);
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  return (
    <div className="main-content fade-in">
      <h2 className="section-title">
        <GiTempleDoor className="ornament" />
        {t.historyTitle}
      </h2>

      {/* Filters */}
      <div className="card" style={{ marginBottom: "1.5rem" }}>
        <div className="card-header">
          <MdFilterList className="icon" />
          <h2>{t.filterData}</h2>
        </div>
        <div className="card-body">
          <div style={{ display: "flex", gap: "1rem", alignItems: "flex-end", flexWrap: "wrap" }}>
            <div className="form-group" style={{ minWidth: 120, marginBottom: 0 }}>
              <label className="form-label">{t.yearLabel}</label>
              <select className="form-select" value={year} onChange={(e) => setYear(e.target.value)}>
                <option value="">{t.allYears}</option>
                {years.map((y) => <option key={y} value={y}>{y}</option>)}
              </select>
            </div>
            <div className="form-group" style={{ minWidth: 120, marginBottom: 0 }}>
              <label className="form-label">{t.monthLabel}</label>
              <select className="form-select" value={month} onChange={(e) => setMonth(e.target.value)}>
                <option value="">{t.allMonths}</option>
                {months.map((m, i) => <option key={i} value={i + 1}>{m}</option>)}
              </select>
            </div>
            <button className="btn btn-primary" onClick={handleFilter}>
              <MdFilterList /> {t.btnFilter}
            </button>
            <button className="btn btn-secondary" onClick={() => { setYear(""); setMonth(""); setPage(1); load(1, "", ""); }}>
              {t.btnClear}
            </button>
          </div>
        </div>
      </div>

      {loading && <Loader text={t.histLoading} />}
      {error && <div className="error-message">⚠️ {error}</div>}

      {data && !loading && (
        <>
          {/* Summary */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem", flexWrap: "wrap", gap: ".5rem" }}>
            <div className="info-tag">
              📊 {data.total_records.toLocaleString()} {t.records} • {t.page} {data.page} {t.of} {data.total_pages}
            </div>
            <div className="tab-bar" style={{ marginBottom: 0, borderBottom: "none" }}>
              <button className={`tab-btn ${view === "table" ? "active" : ""}`} onClick={() => setView("table")}>{t.tableView}</button>
              <button className={`tab-btn ${view === "chart" ? "active" : ""}`} onClick={() => setView("chart")}>{t.chartViewHist}</button>
            </div>
          </div>

          {/* Chart view */}
          {view === "chart" && data.data.length > 0 && (
            <div className="chart-container" style={{ marginBottom: "1.5rem" }}>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={data.data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F5EDDA" />
                  <XAxis dataKey="date" tick={{ fill: "#6B5B4E", fontSize: 11 }} tickFormatter={(v) => format(new Date(v + "T00:00:00"), "MMM d")} interval={Math.max(0, Math.floor(data.data.length / 8))} />
                  <YAxis tick={{ fill: "#6B5B4E", fontSize: 12 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                  <Tooltip
                    contentStyle={{ background: "#FFF", border: "1px solid #C5A028", borderRadius: 8, fontSize: 13 }}
                    formatter={(v) => [v.toLocaleString(), t.pilgrims]}
                    labelFormatter={(v) => format(new Date(v + "T00:00:00"), "EEE, MMM d yyyy")}
                  />
                  <Line type="monotone" dataKey="total_pilgrims" stroke="#C5A028" strokeWidth={2} dot={false} activeDot={{ r: 5, fill: "#800020", stroke: "#C5A028" }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Table view */}
          {view === "table" && (
            <div className="card">
              <div className="card-body" style={{ overflowX: "auto", padding: 0 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>{t.thDate}</th>
                      <th>{t.thDay}</th>
                      <th style={{ textAlign: "right" }}>{t.thTotal}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.data.map((row, i) => {
                      const d = new Date(row.date + "T00:00:00");
                      return (
                        <tr key={row.date}>
                          <td style={{ color: "var(--text-light)", fontSize: ".82rem" }}>{(page - 1) * perPage + i + 1}</td>
                          <td>{format(d, "MMM d, yyyy")}</td>
                          <td style={{ color: "var(--text-muted)" }}>{format(d, "EEEE")}</td>
                          <td style={{ textAlign: "right", fontWeight: 600, fontFamily: "Playfair Display, serif", fontSize: "1.05rem", color: "var(--maroon)" }}>
                            {row.total_pilgrims?.toLocaleString()}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Pagination */}
          {data.total_pages > 1 && (
            <div className="pagination">
              <button onClick={() => goPage(page - 1)} disabled={page <= 1}><MdChevronLeft /></button>
              {Array.from({ length: Math.min(7, data.total_pages) }, (_, i) => {
                let p;
                if (data.total_pages <= 7) p = i + 1;
                else if (page <= 4) p = i + 1;
                else if (page >= data.total_pages - 3) p = data.total_pages - 6 + i;
                else p = page - 3 + i;
                return <button key={p} className={page === p ? "active" : ""} onClick={() => goPage(p)}>{p}</button>;
              })}
              <button onClick={() => goPage(page + 1)} disabled={page >= data.total_pages}><MdChevronRight /></button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
