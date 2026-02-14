import { useState, useEffect } from "react";
import { MdChevronLeft, MdChevronRight } from "react-icons/md";
import { useLang } from "../i18n/LangContext";

const API_BASE = import.meta.env.DEV ? "http://localhost:5000/api" : "/api";

function crowdColor(band) {
  const map = { QUIET: "#388E3C", LIGHT: "#388E3C", MODERATE: "#C5A028", BUSY: "#C5A028", HEAVY: "#E65100", EXTREME: "#D32F2F" };
  return map[band] || "#888";
}

function crowdBg(band) {
  const map = { QUIET: "#E8F5E9", LIGHT: "#E8F5E9", MODERATE: "#FFF8E1", BUSY: "#FFF3E0", HEAVY: "#FBE9E7", EXTREME: "#FFEBEE" };
  return map[band] || "#F5F5F5";
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
    <div key={`b${i}`} style={{ ...cellStyle, visibility: "hidden" }} />
  ));

  const gridStyle = { display: "grid", gridTemplateColumns: "repeat(7, 1fr)", padding: "0 2px" };
  const navBtn = {
    background: "rgba(255,255,255,.15)", border: "1px solid rgba(255,255,255,.3)",
    borderRadius: 6, color: "#FFF", cursor: "pointer", display: "flex", alignItems: "center",
    padding: "4px 8px", fontSize: "1rem", transition: "all .2s",
  };

  return (
    <div className="card" style={{ marginBottom: "2rem", overflow: "visible" }}>
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "1rem 1.25rem", background: "var(--maroon)",
        borderRadius: "var(--radius-md) var(--radius-md) 0 0",
        color: "#FFF", flexWrap: "wrap", gap: ".5rem",
      }}>
        <button onClick={prevMonth} style={navBtn}><MdChevronLeft size={22} /></button>
        <div style={{ textAlign: "center", flex: 1 }}>
          <div style={{ fontFamily: "'Playfair Display', serif", fontSize: "1.25rem", fontWeight: 700 }}>
            {data?.month_name || ""} {year}
          </div>
          {data?.hindu_month?.name_te && (
            <div style={{ fontSize: ".8rem", opacity: .85, marginTop: 2 }}>
              📿 {data.hindu_month.name_te}
            </div>
          )}
        </div>
        <button onClick={goToday} style={{ ...navBtn, fontSize: ".7rem", padding: "4px 10px", borderRadius: 20 }}>
          {t.calToday || "Today"}
        </button>
        <button onClick={nextMonth} style={navBtn}><MdChevronRight size={22} /></button>
      </div>

      {/* Weekday headers */}
      <div style={gridStyle}>
        {WEEKDAYS.map((d) => (
          <div key={d} style={{
            textAlign: "center", fontWeight: 600, fontSize: ".75rem",
            color: d === "Sun" || d === "Sat" ? "var(--maroon)" : "var(--text-muted)",
            padding: "8px 0", borderBottom: "2px solid var(--cream-dark)",
          }}>
            {d}
          </div>
        ))}
      </div>

      {/* Day grid */}
      {loading ? (
        <div style={{ padding: "3rem", textAlign: "center", color: "var(--text-muted)" }}>
          🙏 {t.loading || "Loading..."}
        </div>
      ) : (
        <div style={gridStyle} onMouseLeave={() => setTooltip(null)}>
          {blanks}
          {days.map((d) => {
            const hasEvents = d.events && d.events.length > 0;
            const topEvent = hasEvents ? d.events[0] : null;
            const impactStyle = topEvent ? IMPACT_STYLE[topEvent.impact] : null;
            const isActual = d.is_actual;
            return (
              <div key={d.day} style={{
                ...cellStyle,
                background: d.is_today ? "rgba(197,160,40,.08)" : crowdBg(d.band_name),
                borderLeft: d.is_today ? "3px solid var(--gold)" : "none",
                borderBottom: isActual
                  ? "2px solid #388E3C"
                  : "1px solid var(--cream-dark)",
              }}
                onMouseEnter={(e) => setTooltip({ day: d, x: e.clientX, y: e.clientY })}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <span style={{
                    fontWeight: 700, fontSize: ".85rem",
                    color: d.is_today ? "var(--gold-dark)" : "var(--text-dark)",
                  }}>
                    {d.day}
                  </span>
                  <div style={{ display: "flex", gap: 2, alignItems: "center" }}>
                    {isActual ? (
                      <span title="Actual data" style={{ fontSize: ".6rem", color: "#388E3C" }}>✓</span>
                    ) : (
                      <span title="ML prediction" style={{ fontSize: ".55rem", color: "#999" }}>🔮</span>
                    )}
                    {topEvent && <span style={{ fontSize: ".7rem" }} title={topEvent.name}>{topEvent.emoji || "📌"}</span>}
                  </div>
                </div>
                {isActual && d.total_pilgrims > 0 ? (
                  <div style={{
                    fontSize: ".6rem", fontWeight: 700, textAlign: "center",
                    color: "#388E3C", marginTop: 2,
                  }}>
                    {(d.total_pilgrims / 1000).toFixed(0)}K
                  </div>
                ) : (
                  <div style={{
                    fontSize: ".65rem", fontWeight: 600, textAlign: "center",
                    color: crowdColor(d.band_name), marginTop: 2,
                  }}>
                    {d.band_name}
                  </div>
                )}
                {topEvent && topEvent.type !== "school_holiday" && topEvent.type !== "lunar" && (
                  <div style={{
                    fontSize: ".55rem", lineHeight: 1.2, marginTop: 2,
                    color: "var(--maroon)", fontWeight: 500,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {topEvent.name}
                  </div>
                )}
                {impactStyle && d.events?.[0]?.impact !== "low" && (
                  <div style={{
                    position: "absolute", bottom: 3, right: 3, width: 6, height: 6,
                    borderRadius: "50%", background: impactStyle.bg,
                  }} />
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Tooltip */}
      {tooltip && (
        <div style={{
          position: "fixed", left: Math.min(tooltip.x + 10, window.innerWidth - 260),
          top: tooltip.y + 10, background: "#FFF", border: "1px solid var(--cream-dark)",
          borderRadius: 8, padding: ".75rem", boxShadow: "0 4px 12px rgba(0,0,0,.15)",
          zIndex: 1000, minWidth: 200, fontSize: ".82rem", pointerEvents: "none",
        }}>
          <div style={{ fontWeight: 700, color: "var(--maroon)", marginBottom: 4 }}>
            {data?.month_name} {tooltip.day.day}, {year}
          </div>
          {tooltip.day.is_actual ? (
            <div style={{ color: "#388E3C", fontWeight: 600 }}>
              ✓ Actual: {(tooltip.day.total_pilgrims || 0).toLocaleString("en-IN")} pilgrims
            </div>
          ) : (
            <div style={{ color: crowdColor(tooltip.day.band_name), fontWeight: 600 }}>
              🔮 {tooltip.day.band_name} ({(tooltip.day.confidence * 100).toFixed(0)}%)
            </div>
          )}
          <div style={{
            fontSize: ".7rem", color: tooltip.day.is_actual ? "#388E3C" : "#999",
            marginTop: 2, fontStyle: "italic",
          }}>
            {tooltip.day.is_actual ? "Recorded data from TTD" : "ML prediction"}
          </div>
          {tooltip.day.events?.length > 0 && (
            <div style={{ borderTop: "1px solid var(--cream-dark)", paddingTop: 4, marginTop: 4 }}>
              {tooltip.day.events.map((e, i) => (
                <div key={i} style={{ display: "flex", gap: 4, alignItems: "center", marginBottom: 2 }}>
                  <span>{e.emoji || "📌"}</span>
                  <span style={{ fontWeight: 500 }}>{e.name}</span>
                  {e.impact && (
                    <span style={{
                      fontSize: ".6rem", padding: "1px 5px", borderRadius: 10,
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
      <div style={{
        display: "flex", flexWrap: "wrap", gap: ".6rem", padding: ".75rem 1rem",
        fontSize: ".7rem", color: "var(--text-muted)", justifyContent: "center",
        borderTop: "1px solid var(--cream-dark)",
      }}>
        <span>🛕 {t.calFestival || "Festival"}</span>
        <span>🔱 {t.calBrahmotsavam || "Brahmotsavam"}</span>
        <span>🌕 {t.calPurnima || "Purnima"}</span>
        <span>🌑 {t.calAmavasya || "Amavasya"}</span>
        <span>📿 {t.calEkadashi || "Ekadashi"}</span>
        <span>🏛️ {t.calHoliday || "Holiday"}</span>
        <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
          <span style={{ color: "#388E3C", fontWeight: 700, fontSize: ".75rem" }}>✓</span> Actual
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
          <span style={{ fontSize: ".65rem" }}>🔮</span> Predicted
        </span>
        {[
          { label: "Quiet", bg: "#388E3C" }, { label: "Moderate", bg: "#C5A028" },
          { label: "Heavy", bg: "#E65100" }, { label: "Extreme", bg: "#D32F2F" },
        ].map(({ label, bg }) => (
          <span key={label} style={{ display: "flex", alignItems: "center", gap: 3 }}>
            <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: bg }} />
            {label}
          </span>
        ))}
      </div>

      {/* Festivals */}
      {data?.festivals?.length > 0 && (
        <div style={{ padding: ".75rem 1rem", borderTop: "1px solid var(--cream-dark)", background: "var(--off-white)" }}>
          <div style={{ fontWeight: 600, fontSize: ".85rem", color: "var(--maroon)", marginBottom: ".5rem" }}>
            🎉 Festivals this month
          </div>
          {data.festivals.map((f, i) => (
            <div key={i} style={{ fontSize: ".8rem", margin: "2px 0" }}>
              <strong>{f.day}</strong> — {f.name} {f.name_te && `(${f.name_te})`}
              {f.impact && (
                <span style={{
                  marginLeft: 6, padding: "1px 6px", borderRadius: 10, fontSize: ".65rem",
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

const cellStyle = {
  minHeight: 72, padding: "4px 5px", position: "relative",
  borderBottom: "1px solid var(--cream-dark)", borderRight: "1px solid rgba(0,0,0,.03)",
};
