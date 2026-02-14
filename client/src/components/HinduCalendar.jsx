import { useState, useEffect } from "react";
import { MdChevronLeft, MdChevronRight } from "react-icons/md";
import { useLang } from "../i18n/LangContext";

const API_BASE = import.meta.env.DEV ? "http://localhost:5000/api" : "/api";

/* 6 distinct crowd-level colors — synced with backend BAND_COLORS */
function crowdColor(band) {
  const map = {
    QUIET:    "#2196F3",
    LIGHT:    "#4CAF50",
    MODERATE: "#8BC34A",
    BUSY:     "#FFC107",
    HEAVY:    "#FF5722",
    EXTREME:  "#B71C1C",
  };
  return map[band] || "#888";
}

function crowdTextColor(band) {
  const map = {
    QUIET:    "#1565C0",
    LIGHT:    "#2E7D32",
    MODERATE: "#558B2F",
    BUSY:     "#F57F17",
    HEAVY:    "#BF360C",
    EXTREME:  "#B71C1C",
  };
  return map[band] || "#888";
}

function crowdBg(band) {
  const map = {
    QUIET:    "#E3F2FD",
    LIGHT:    "#E8F5E9",
    MODERATE: "#F1F8E9",
    BUSY:     "#FFF8E1",
    HEAVY:    "#FBE9E7",
    EXTREME:  "#FFEBEE",
  };
  return map[band] || "#F5F5F5";
}

function badgeTextColor(band) {
  return (band === "BUSY" || band === "MODERATE") ? "#333" : "#fff";
}

const IMPACT_STYLE = {
  extreme: { bg: "#D32F2F", color: "#FFF" },
  very_high: { bg: "#E65100", color: "#FFF" },
  high: { bg: "#C5A028", color: "#FFF" },
  moderate: { bg: "#66BB6A", color: "#FFF" },
  low: { bg: "#E0E0E0", color: "#555" },
};

const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export default function HinduCalendar() {
  const { t } = useLang();
  const now = new Date();
  const [year, setYear] = useState(now.getFullYear());
  const [month, setMonth] = useState(now.getMonth() + 1);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/calendar/${year}/${month}`)
      .then((r) => r.json())
      .then((d) => setData(d))
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [year, month]);

  function prevMonth() {
    if (month === 1) { setMonth(12); setYear(year - 1); }
    else setMonth(month - 1);
  }
  function nextMonth() {
    if (month === 12) { setMonth(1); setYear(year + 1); }
    else setMonth(month + 1);
  }
  function goToday() {
    setYear(now.getFullYear());
    setMonth(now.getMonth() + 1);
  }

  const days = data?.days || [];
  const offset = data?.first_weekday ?? 0;
  const blanks = Array.from({ length: offset }, (_, i) => (
    <div key={`b${i}`} className="hc-cell hc-blank" />
  ));

  return (
    <div className="card hc-wrapper">
      {/* Header */}
      <div className="hc-header">
        <button className="hc-nav-btn" onClick={prevMonth}><MdChevronLeft size={22} /></button>
        <div className="hc-title">
          <div className="hc-month">{data?.month_name || ""} {year}</div>
          {data?.hindu_month?.name_te && (
            <div className="hc-hindu-month">📿 {data.hindu_month.name_te}</div>
          )}
        </div>
        <button className="hc-nav-btn hc-today-btn" onClick={goToday}>
          {t.calToday || "Today"}
        </button>
        <button className="hc-nav-btn" onClick={nextMonth}><MdChevronRight size={22} /></button>
      </div>

      {/* Weekday headers */}
      <div className="hc-grid">
        {WEEKDAYS.map((d) => (
          <div key={d} className={`hc-weekday ${d === "Sun" || d === "Sat" ? "hc-weekend-hdr" : ""}`}>
            {d}
          </div>
        ))}
      </div>

      {/* Day grid */}
      {loading ? (
        <div className="hc-loading">🙏 {t.loading || "Loading..."}</div>
      ) : (
        <div className="hc-grid" onMouseLeave={() => setTooltip(null)}>
          {blanks}
          {days.map((d) => {
            const hasEvents = d.events && d.events.length > 0;
            const topEvent = hasEvents ? d.events[0] : null;
            const impactStyle = topEvent ? IMPACT_STYLE[topEvent.impact] : null;
            const isActual = d.is_actual;
            const cellClass = `hc-cell ${d.is_today ? "hc-today" : ""} ${isActual ? "hc-actual" : ""}`;
            return (
              <div key={d.day} className={cellClass}
                style={{
                  background: d.is_today
                    ? "rgba(197,160,40,.10)"
                    : crowdBg(d.band_name),
                  borderLeft: d.is_today
                    ? "3px solid var(--gold)"
                    : `3px solid ${crowdColor(d.band_name)}`,
                }}
                onMouseEnter={(e) => setTooltip({ day: d, x: e.clientX, y: e.clientY })}
                onClick={() => setTooltip(tooltip?.day?.day === d.day ? null : { day: d, x: 0, y: 0 })}
              >
                {/* Row 1: Day number + event icon */}
                <div className="hc-cell-top">
                  <span className={`hc-day-num ${d.is_today ? "hc-day-today" : ""}`}>{d.day}</span>
                  {topEvent && <span className="hc-event-icon" title={topEvent.name}>{topEvent.emoji || "📌"}</span>}
                </div>

                {/* Row 2: Band badge (+ actual count if available) */}
                {isActual && d.total_pilgrims > 0 ? (
                  <div className="hc-actual-badge"
                    style={{ background: crowdColor(d.band_name), color: badgeTextColor(d.band_name) }}>
                    {(d.total_pilgrims / 1000).toFixed(0)}K ✓
                  </div>
                ) : (
                  <div className="hc-band-badge"
                    style={{ background: crowdColor(d.band_name), color: badgeTextColor(d.band_name) }}>
                    {d.band_name}
                  </div>
                )}

                {/* Row 3: Event name */}
                {topEvent && topEvent.type !== "school_holiday" && topEvent.type !== "lunar" && (
                  <div className="hc-event-name">{topEvent.name}</div>
                )}

                {/* Impact dot */}
                {impactStyle && d.events?.[0]?.impact !== "low" && (
                  <div className="hc-impact-dot" style={{ background: impactStyle.bg }} />
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Tooltip */}
      {tooltip && (
        <div className="hc-tooltip" style={{
          left: Math.min(tooltip.x + 10, window.innerWidth - 280),
          top: Math.min(tooltip.y + 10, window.innerHeight - 200),
        }}>
          <div className="hc-tooltip-title">{data?.month_name} {tooltip.day.day}, {year}</div>
          {tooltip.day.is_actual ? (
            <div className="hc-tooltip-actual">
              <div className="hc-tooltip-val">{(tooltip.day.total_pilgrims || 0).toLocaleString("en-IN")} pilgrims</div>
              <div className="hc-tooltip-src">✓ Recorded data from TTD</div>
            </div>
          ) : (
            <div className="hc-tooltip-pred" style={{
              borderLeft: `3px solid ${crowdColor(tooltip.day.band_name)}`,
              background: crowdBg(tooltip.day.band_name),
            }}>
              <div style={{ color: crowdTextColor(tooltip.day.band_name), fontWeight: 700, fontSize: ".9rem" }}>
                {tooltip.day.band_name}
              </div>
              <div className="hc-tooltip-conf">
                {(tooltip.day.confidence * 100).toFixed(0)}% confidence · ML prediction
              </div>
            </div>
          )}
          {tooltip.day.events?.length > 0 && (
            <div className="hc-tooltip-events">
              {tooltip.day.events.map((e, i) => (
                <div key={i} className="hc-tooltip-event">
                  <span>{e.emoji || "📌"}</span>
                  <span style={{ fontWeight: 500 }}>{e.name}</span>
                  {e.impact && (
                    <span className="hc-tooltip-impact" style={{
                      background: (IMPACT_STYLE[e.impact] || {}).bg || "#EEE",
                      color: (IMPACT_STYLE[e.impact] || {}).color || "#555",
                    }}>{e.impact}</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Legend */}
      <div className="hc-legend">
        <div className="hc-legend-bands">
          {[
            { label: "Quiet",    color: "#2196F3",  text: "#fff" },
            { label: "Light",    color: "#4CAF50",  text: "#fff" },
            { label: "Moderate", color: "#8BC34A",  text: "#333" },
            { label: "Busy",     color: "#FFC107",  text: "#333" },
            { label: "Heavy",    color: "#FF5722",  text: "#fff" },
            { label: "Extreme",  color: "#B71C1C",  text: "#fff" },
          ].map(({ label, color, text }) => (
            <span key={label} className="hc-legend-pill" style={{ background: color, color: text }}>
              {label}
            </span>
          ))}
        </div>
        <div className="hc-legend-info">
          <span className="hc-legend-item">
            <span className="hc-legend-swatch hc-legend-actual">✓</span>
            Actual data
          </span>
          <span className="hc-legend-item">
            <span className="hc-legend-swatch hc-legend-pred">BAND</span>
            ML prediction
          </span>
          <span>🛕 {t.calFestival || "Festival"}</span>
          <span>🌕 {t.calPurnima || "Purnima"}</span>
          <span>🌑 {t.calAmavasya || "Amavasya"}</span>
          <span>📿 {t.calEkadashi || "Ekadashi"}</span>
        </div>
      </div>

      {/* Festivals */}
      {data?.festivals?.length > 0 && (
        <div className="hc-festivals">
          <div className="hc-festivals-title">🎉 Festivals this month</div>
          {data.festivals.map((f, i) => (
            <div key={i} className="hc-festival-item">
              <strong>{f.day}</strong> — {f.name} {f.name_te && `(${f.name_te})`}
              {f.impact && (
                <span className="hc-festival-impact" style={{
                  background: (IMPACT_STYLE[f.impact] || {}).bg || "#EEE",
                  color: (IMPACT_STYLE[f.impact] || {}).color || "#555",
                }}>{f.impact}</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
