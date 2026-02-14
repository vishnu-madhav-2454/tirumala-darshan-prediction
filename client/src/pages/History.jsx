import { useState, useEffect, useMemo } from "react";
import { useLang } from "../i18n/LangContext";
import { getHistory } from "../api";
import Loader from "../components/Loader";
import { GiTempleDoor } from "react-icons/gi";
import {
  MdHistory, MdPeople, MdCalendarMonth, MdFilterList,
  MdArrowUpward, MdArrowDownward, MdTrendingUp,
} from "react-icons/md";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine, BarChart, Bar, Cell,
} from "recharts";
import { format, parseISO } from "date-fns";

const PER_PAGE = 30;

export default function History() {
  const { t } = useLang();
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [summary, setSummary] = useState(null);
  const [bandInfo, setBandInfo] = useState({ names: [], colors: {}, bg: {} });

  /* ── Filters ── */
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [filterBand, setFilterBand] = useState("ALL");
  const [sortCol, setSortCol] = useState("date");
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => { fetchData(); }, [page, startDate, endDate]);

  async function fetchData() {
    setLoading(true);
    try {
      const res = await getHistory(page, PER_PAGE, startDate || undefined, endDate || undefined);
      const d = res.data;
      const rows = d?.data || [];
      setData(Array.isArray(rows) ? rows : []);
      setTotalPages(d?.total_pages || 1);
      setTotal(d?.total || rows.length);
      setSummary(d?.summary || null);
      setBandInfo({
        names: d?.band_names || [],
        colors: d?.band_colors || {},
        bg: d?.band_bg || {},
      });
    } catch {
      setData([]);
    } finally {
      setLoading(false);
    }
  }

  function applyFilters() {
    setPage(1);
    fetchData();
  }
  function clearFilters() {
    setStartDate("");
    setEndDate("");
    setFilterBand("ALL");
    setPage(1);
  }

  /* ── Sort + filter data for display ── */
  const displayData = useMemo(() => {
    let rows = [...data];
    if (filterBand !== "ALL") {
      rows = rows.filter((r) => r.band_name === filterBand);
    }
    rows.sort((a, b) => {
      let cmp = 0;
      if (sortCol === "date") cmp = a.date.localeCompare(b.date);
      else if (sortCol === "pilgrims") cmp = a.total_pilgrims - b.total_pilgrims;
      else if (sortCol === "band") cmp = a.band_index - b.band_index;
      return sortAsc ? cmp : -cmp;
    });
    return rows;
  }, [data, filterBand, sortCol, sortAsc]);

  /* ── Chart data — reverse so old dates are on left ── */
  const chartData = useMemo(() => {
    return [...data]
      .filter((r) => r.date && r.total_pilgrims != null)
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((r) => ({
        date: format(parseISO(r.date), "dd MMM ''yy"),
        fullDate: format(parseISO(r.date), "EEE, dd MMM yyyy"),
        pilgrims: r.total_pilgrims,
        band: r.band_name,
        color: r.band_color,
        day: r.day_of_week,
      }));
  }, [data]);

  const avgLine = summary ? summary.avg : 0;

  function toggleSort(col) {
    if (sortCol === col) setSortAsc(!sortAsc);
    else { setSortCol(col); setSortAsc(col === "date" ? false : true); }
  }
  const SortIcon = ({ col }) =>
    sortCol === col ? (sortAsc ? <MdArrowUpward size={14} /> : <MdArrowDownward size={14} />) : null;

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <div className="history-tooltip">
        <div className="history-tooltip-date">{d.fullDate}</div>
        <div className="history-tooltip-value">
          <span className="history-tooltip-dot" style={{ background: d.color }} />
          {d.pilgrims.toLocaleString("en-IN")} pilgrims
        </div>
        <div className="history-tooltip-band" style={{ color: d.color }}>{d.band}</div>
      </div>
    );
  };

  return (
    <section className="page history-page">
      <div className="page-header">
        <GiTempleDoor className="page-header-icon" />
        <h2>{t.historyTitle || "Historical Crowd Data"}</h2>
        <p className="page-subtitle">
          <MdHistory style={{ verticalAlign: "middle", marginRight: 4 }} />
          {t.historySubtitle || `Daily pilgrim footfall records from TTD${summary ? ` (${summary.date_start?.slice(0,4)} — ${summary.date_end?.slice(0,4)})` : ""}`}
        </p>
      </div>

      {/* ── Summary Stats ── */}
      {summary && (
        <div className="history-stats-row">
          <div className="history-stat-card">
            <div className="history-stat-label">Total Records</div>
            <div className="history-stat-value">{total.toLocaleString("en-IN")}</div>
            <div className="history-stat-sub">{summary.date_start} → {summary.date_end}</div>
          </div>
          <div className="history-stat-card">
            <div className="history-stat-label">Daily Average</div>
            <div className="history-stat-value">{summary.avg.toLocaleString("en-IN")}</div>
            <div className="history-stat-sub">Median: {summary.median.toLocaleString("en-IN")}</div>
          </div>
          <div className="history-stat-card history-stat-busy">
            <div className="history-stat-label">🔴 Busiest Day</div>
            <div className="history-stat-value">{summary.busiest_pilgrims.toLocaleString("en-IN")}</div>
            <div className="history-stat-sub">{format(parseISO(summary.busiest_date), "EEE, dd MMM yyyy")}</div>
          </div>
          <div className="history-stat-card history-stat-quiet">
            <div className="history-stat-label">🔵 Quietest Day</div>
            <div className="history-stat-value">{summary.quietest_pilgrims.toLocaleString("en-IN")}</div>
            <div className="history-stat-sub">{format(parseISO(summary.quietest_date), "EEE, dd MMM yyyy")}</div>
          </div>
        </div>
      )}

      {/* ── Filters ── */}
      <div className="card filter-card">
        <div className="card-header"><MdFilterList /> Filter Data</div>
        <div className="card-body filter-row">
          <div className="filter-group">
            <label>From</label>
            <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)}
                   min={summary?.date_start || "2022-02-01"} max={summary?.date_end || ""} />
          </div>
          <div className="filter-group">
            <label>To</label>
            <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)}
                   min={summary?.date_start || "2022-02-01"} max={summary?.date_end || ""} />
          </div>
          <div className="filter-group">
            <label>Crowd Level</label>
            <select value={filterBand} onChange={(e) => setFilterBand(e.target.value)}>
              <option value="ALL">All Levels</option>
              {bandInfo.names.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </div>
          <div className="filter-actions">
            <button className="btn btn-sm btn-gold" onClick={applyFilters}>Apply</button>
            <button className="btn btn-sm btn-outline" onClick={clearFilters}>Clear</button>
          </div>
        </div>
      </div>

      {loading && <Loader text="Loading historical data..." />}

      {/* ── Chart — Bar chart with crowd level colors ── */}
      {chartData.length > 0 && (
        <div className="card chart-card">
          <div className="card-header">
            <MdTrendingUp /> {t.historyChart || "Footfall Trend"} — {chartData.length} days
          </div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={chartData} barCategoryGap="1%">
                <CartesianGrid strokeDasharray="3 3" stroke="#E8D48B" vertical={false} />
                <XAxis dataKey="date" tick={{ fill: "#6B5B4E", fontSize: 10 }}
                       interval={Math.max(0, Math.floor(chartData.length / 8) - 1)}
                       angle={-35} textAnchor="end" height={60} />
                <YAxis tick={{ fill: "#6B5B4E", fontSize: 11 }}
                       tickFormatter={(v) => (v / 1000).toFixed(0) + "K"}
                       domain={["dataMin - 5000", "dataMax + 5000"]} />
                <Tooltip content={<CustomTooltip />} />
                {avgLine > 0 && (
                  <ReferenceLine y={avgLine} stroke="#800020" strokeDasharray="6 4"
                    label={{ value: `Avg: ${(avgLine / 1000).toFixed(0)}K`, fill: "#800020", fontSize: 11, position: "right" }} />
                )}
                <Bar dataKey="pilgrims" radius={[2, 2, 0, 0]}>
                  {chartData.map((entry, idx) => (
                    <Cell key={idx} fill={entry.color} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            {/* Band legend */}
            <div className="chart-legend">
              {bandInfo.names.map((name) => (
                <span key={name} className="legend-item">
                  <span className="legend-dot" style={{ background: bandInfo.colors[name] }} />
                  {name}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Table ── */}
      {displayData.length > 0 && (
        <div className="card history-table-card">
          <div className="card-header">
            <MdPeople /> {t.historyTable || "Records"} — {total.toLocaleString("en-IN")} total
          </div>
          <div className="card-body table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="sortable" onClick={() => toggleSort("date")}>
                    Date <SortIcon col="date" />
                  </th>
                  <th>Day</th>
                  <th className="sortable" onClick={() => toggleSort("pilgrims")}>
                    Total Pilgrims <SortIcon col="pilgrims" />
                  </th>
                  <th className="sortable" onClick={() => toggleSort("band")}>
                    Crowd Level <SortIcon col="band" />
                  </th>
                </tr>
              </thead>
              <tbody>
                {displayData.map((r, i) => (
                  <tr key={i} style={{ borderLeft: `4px solid ${r.band_color}` }}>
                    <td className="td-date">
                      {r.date ? format(parseISO(r.date), "dd MMM yyyy") : ""}
                    </td>
                    <td className={`td-day ${r.is_weekend ? "weekend" : ""}`}>
                      {r.day_of_week}
                    </td>
                    <td className="td-pilgrims">
                      {(r.total_pilgrims || 0).toLocaleString("en-IN")}
                    </td>
                    <td>
                      <span className="crowd-badge" style={{
                        background: r.band_bg,
                        color: r.band_color,
                        border: `1px solid ${r.band_color}`,
                      }}>
                        {r.band_name}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {/* Pagination */}
          <div className="pagination">
            <button className="btn btn-sm" disabled={page <= 1}
                    onClick={() => setPage(1)} title="First page">⏮</button>
            <button className="btn btn-sm" disabled={page <= 1}
                    onClick={() => setPage((p) => p - 1)}>← Prev</button>
            <span className="page-info">
              Page {page} of {totalPages}
            </span>
            <button className="btn btn-sm" disabled={page >= totalPages}
                    onClick={() => setPage((p) => p + 1)}>Next →</button>
            <button className="btn btn-sm" disabled={page >= totalPages}
                    onClick={() => setPage(totalPages)} title="Last page">⏭</button>
          </div>
        </div>
      )}
    </section>
  );
}
