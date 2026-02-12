import { useState, useEffect } from "react";
import { MdChevronLeft, MdChevronRight } from "react-icons/md";
import { useLang } from "../i18n/LangContext";

const API_BASE =
  import.meta.env.DEV ? "http://localhost:5000/api" : "/api";

/* ── colour coding by predicted pilgrim count ── */
function crowdColor(pilgrims) {
  if (!pilgrims) return "transparent";
  if (pilgrims >= 75000) return "#D32F2F";   // red
  if (pilgrims >= 55000) return "#E65100";   // deep orange
  if (pilgrims >= 40000) return "#C5A028";   // gold
  return "#388E3C";                           // green
}

function crowdBg(pilgrims) {
  if (!pilgrims) return "transparent";
  if (pilgrims >= 75000) return "rgba(211,47,47,.08)";
  if (pilgrims >= 55000) return "rgba(230,81,0,.06)";
  if (pilgrims >= 40000) return "rgba(197,160,40,.06)";
  return "rgba(56,142,60,.05)";
}

/* ── impact badge style ── */
const IMPACT_STYLE = {
  extreme:   { bg: "#D32F2F", color: "#FFF" },
  very_high: { bg: "#E65100", color: "#FFF" },
  high:      { bg: "#C5A028", color: "#FFF" },
  moderate:  { bg: "#66BB6A", color: "#FFF" },
  low:       { bg: "#E0E0E0", color: "#555" },
};

const WEEKDAYS_EN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export default function HinduCalendar() {
  const { t } = useLang();
  const now = new Date();
  const [year, setYear] = useState(now.getFullYear());
  const [month, setMonth] = useState(now.getMonth() + 1);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tooltip, setTooltip] = useState(null); // { day, x, y }

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/calendar?year=${year}&month=${month}`)
      .then((r) => r.json())
      .then((d) => setData(d))
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [year, month]);

  function prevMonth() {
    if (month === 1) { setYear(year - 1); setMonth(12); }
    else setMonth(month - 1);
  }
  function nextMonth() {
    if (month === 12) { setYear(year + 1); setMonth(1); }
    else setMonth(month + 1);
  }
  function goToday() {
    setYear(now.getFullYear());
    setMonth(now.getMonth() + 1);
  }

  /* Build grid cells: empty slots before day-1, then actual days */
  const days = data?.days || [];
  // first_day_weekday: 0=Mon … 6=Sun
  const offset = data?.first_day_weekday ?? 0;
  const blanks = Array.from({ length: offset }, (_, i) => (
    <div key={`b${i}`} style={cellStyle(true)} />
  ));

  return (
    <div className="card" style={{ marginBottom: "2rem", overflow: "visible" }}>
      {/* ── Header ── */}
      <div
        style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "1rem 1.25rem", background: "var(--maroon)",
          borderRadius: "var(--radius-md) var(--radius-md) 0 0",
          color: "#FFF", flexWrap: "wrap", gap: ".5rem",
        }}
      >
        <button onClick={prevMonth} style={navBtn}><MdChevronLeft size={22} /></button>
        <div style={{ textAlign: "center", flex: 1 }}>
          <div style={{ fontFamily: "'Playfair Display', serif", fontSize: "1.25rem", fontWeight: 700 }}>
            {data?.month_name || ""} {year}
          </div>
          {data?.hindu_month_te && (
            <div style={{ fontSize: ".8rem", opacity: .85, marginTop: 2 }}>
              📿 {data.hindu_month_te}
            </div>
          )}
        </div>
        <button onClick={goToday} style={{ ...navBtn, fontSize: ".7rem", padding: "4px 10px", borderRadius: 20 }}>
          {t.calToday || "Today"}
        </button>
        <button onClick={nextMonth} style={navBtn}><MdChevronRight size={22} /></button>
      </div>

      {/* ── Weekday headers ── */}
      <div style={gridStyle}>
        {WEEKDAYS_EN.map((d) => (
          <div
            key={d}
            style={{
              textAlign: "center", fontWeight: 600, fontSize: ".75rem",
              color: d === "Sun" || d === "Sat" ? "var(--maroon)" : "var(--text-muted)",
              padding: "8px 0", borderBottom: "2px solid var(--cream-dark)",
            }}
          >
            {d}
          </div>
        ))}
      </div>

      {/* ── Day grid ── */}
      {loading ? (
        <div style={{ padding: "3rem", textAlign: "center", color: "var(--text-muted)" }}>
          🙏 {t.loading || "Loading..."}
        </div>
      ) : (
        <div style={gridStyle} onMouseLeave={() => setTooltip(null)}>
          {blanks}
          {days.map((d) => {
            const pil = d.pilgrims;
            const hasEvents = d.events && d.events.length > 0;
            const topEvent = hasEvents ? d.events[0] : null;
            const impactStyle = d.max_impact ? IMPACT_STYLE[d.max_impact] : null;

            return (
              <div
                key={d.day}
                style={{
                  ...cellStyle(false),
                  background: d.is_today
                    ? "rgba(197,160,40,.15)"
                    : d.is_weekend
                    ? "rgba(128,0,32,.03)"
                    : crowdBg(pil),
                  border: d.is_today ? "2px solid var(--gold)" : "1px solid var(--cream-dark)",
                  cursor: hasEvents ? "pointer" : "default",
                  position: "relative",
                }}
                onMouseEnter={(e) => {
                  if (hasEvents || pil) {
                    const rect = e.currentTarget.getBoundingClientRect();
                    setTooltip({ day: d, x: rect.left, y: rect.bottom });
                  }
                }}
                onMouseLeave={() => setTooltip(null)}
              >
                {/* Day number */}
                <div style={{
                  display: "flex", justifyContent: "space-between", alignItems: "flex-start",
                }}>
                  <span style={{
                    fontWeight: 700, fontSize: ".85rem",
                    color: d.is_today ? "var(--gold-dark)" : d.is_weekend ? "var(--maroon)" : "var(--text-dark)",
                  }}>
                    {d.day}
                  </span>
                  {/* Event emoji */}
                  {topEvent && (
                    <span style={{ fontSize: ".7rem" }} title={topEvent.name}>
                      {topEvent.emoji || "📌"}
                    </span>
                  )}
                </div>

                {/* Prediction number */}
                {pil != null && (
                  <div style={{
                    fontSize: ".65rem", fontWeight: 600, textAlign: "center",
                    color: crowdColor(pil), marginTop: 2,
                  }}>
                    {pil >= 1000 ? `${(pil / 1000).toFixed(1)}k` : pil}
                  </div>
                )}

                {/* Event name (truncated) */}
                {topEvent && topEvent.type !== "school_holiday" && topEvent.type !== "lunar" && (
                  <div style={{
                    fontSize: ".55rem", lineHeight: 1.2, marginTop: 2,
                    color: "var(--maroon)", fontWeight: 500,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {topEvent.name_te || topEvent.name}
                  </div>
                )}

                {/* Lunar icon row */}
                {hasEvents && d.events.some((e) => e.type === "lunar") && (
                  <div style={{ fontSize: ".6rem", textAlign: "center", marginTop: 1 }}>
                    {d.events.filter((e) => e.type === "lunar").map((e) => e.emoji).join("")}
                  </div>
                )}

                {/* Impact dot */}
                {impactStyle && d.max_impact !== "low" && (
                  <div style={{
                    position: "absolute", bottom: 2, right: 3,
                    width: 6, height: 6, borderRadius: "50%",
                    background: impactStyle.bg,
                  }} />
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── Tooltip ── */}
      {tooltip && tooltip.day && (
        <div
          style={{
            position: "fixed",
            left: Math.min(tooltip.x, window.innerWidth - 280),
            top: tooltip.y + 4,
            zIndex: 999,
            background: "#FFF",
            border: "1px solid var(--gold)",
            borderRadius: "var(--radius-sm)",
            padding: ".75rem 1rem",
            boxShadow: "var(--shadow-md)",
            maxWidth: 280,
            fontSize: ".82rem",
          }}
        >
          <div style={{ fontWeight: 700, color: "var(--maroon)", marginBottom: 4 }}>
            {tooltip.day.day_name}, {tooltip.day.date}
          </div>
          {tooltip.day.pilgrims != null && (
            <div style={{ marginBottom: 4 }}>
              <span style={{ fontWeight: 600 }}>
                {tooltip.day.source === "actual" ? "📊" : "🔮"}{" "}
                {tooltip.day.pilgrims.toLocaleString()} {t.pilgrims || "pilgrims"}
              </span>
              <span style={{ color: "var(--text-muted)", fontSize: ".75rem", marginLeft: 6 }}>
                ({tooltip.day.source === "actual" ? t.calActual || "actual" : t.calPredicted || "predicted"})
              </span>
            </div>
          )}
          {tooltip.day.confidence_low && (
            <div style={{ fontSize: ".75rem", color: "var(--text-muted)", marginBottom: 4 }}>
              {t.confidenceRange || "Range"}: {tooltip.day.confidence_low.toLocaleString()} — {tooltip.day.confidence_high.toLocaleString()}
            </div>
          )}
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
                      marginLeft: "auto", fontWeight: 600,
                    }}>
                      {e.impact}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
          {tooltip.day.crowd_reason && (
            <div style={{
              fontSize: ".75rem", color: "var(--gold-dark)", marginTop: 4,
              fontStyle: "italic",
            }}>
              📈 {t.calWhyBusy || "Why busy"}: {tooltip.day.crowd_reason}
            </div>
          )}
        </div>
      )}

      {/* ── Legend ── */}
      <div style={{
        display: "flex", flexWrap: "wrap", gap: "1rem", padding: ".75rem 1.25rem",
        borderTop: "1px solid var(--cream-dark)", fontSize: ".72rem",
        color: "var(--text-muted)", alignItems: "center", justifyContent: "center",
      }}>
        <span>🛕 {t.calFestival || "Festival"}</span>
        <span>🔱 {t.calBrahmotsavam || "Brahmotsavam"}</span>
        <span>🌕 {t.calPurnima || "Purnima"}</span>
        <span>🌑 {t.calAmavasya || "Amavasya"}</span>
        <span>📿 {t.calEkadashi || "Ekadashi"}</span>
        <span>🏛️ {t.calHoliday || "Holiday"}</span>
        <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: "#388E3C" }} />
          {"< 40k"}
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: "#C5A028" }} />
          {"40-55k"}
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: "#E65100" }} />
          {"55-75k"}
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: "#D32F2F" }} />
          {"75k+"}
        </span>
      </div>
    </div>
  );
}

/* ── Style helpers ── */
const gridStyle = {
  display: "grid",
  gridTemplateColumns: "repeat(7, 1fr)",
  padding: "0 2px",
};

const cellStyle = (isBlank) => ({
  minHeight: 72,
  padding: "4px 5px",
  borderBottom: "1px solid var(--cream-dark)",
  borderRight: "1px solid rgba(0,0,0,.03)",
  visibility: isBlank ? "hidden" : "visible",
});

const navBtn = {
  background: "rgba(255,255,255,.15)",
  border: "1px solid rgba(255,255,255,.3)",
  borderRadius: 6,
  color: "#FFF",
  cursor: "pointer",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 4,
  transition: "background .2s",
};
