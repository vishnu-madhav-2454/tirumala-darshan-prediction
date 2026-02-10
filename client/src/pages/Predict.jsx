import { useState } from "react";
import { format } from "date-fns";
import { MdCalendarMonth, MdSearch } from "react-icons/md";
import { GiTempleDoor } from "react-icons/gi";
import { predictDate } from "../api";
import CrowdBadge from "../components/CrowdBadge";
import Loader from "../components/Loader";
import { useLang } from "../i18n/LangContext";

export default function Predict() {
  const { t } = useLang();
  const [date, setDate] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handlePredict(e) {
    e.preventDefault();
    if (!date) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predictDate(date);
      setResult(res.data.prediction);
    } catch (e) {
      setError(e.response?.data?.error || e.message);
    } finally {
      setLoading(false);
    }
  }

  const pred = result;

  return (
    <div className="main-content fade-in">
      <h2 className="section-title">
        <GiTempleDoor className="ornament" />
        {t.predictTitle}
      </h2>

      {/* Date picker */}
      <div className="card" style={{ maxWidth: 600 }}>
        <div className="card-header">
          <MdCalendarMonth className="icon" />
          <h2>{t.pickDate}</h2>
        </div>
        <div className="card-body">
          <form onSubmit={handlePredict} style={{ display: "flex", gap: "1rem", alignItems: "flex-end", flexWrap: "wrap" }}>
            <div className="form-group" style={{ flex: "1 1 auto", minWidth: "200px", marginBottom: 0 }}>
              <label className="form-label">{t.dateLabel}</label>
              <input type="date" className="form-input" value={date} onChange={(e) => setDate(e.target.value)} required />
            </div>
            <button className="btn btn-primary" type="submit" disabled={loading || !date} style={{ minWidth: "140px" }}>
              <MdSearch /> {t.btnPredict}
            </button>
          </form>
        </div>
      </div>

      {loading && <Loader text={t.predicting} />}
      {error && <div className="error-message" style={{ marginTop: "1.5rem" }}>⚠️ {error}</div>}

      {/* Result */}
      {pred && (
        <div className="card" style={{ marginTop: "2rem" }}>
          <div className="card-header">
            <MdSearch className="icon" />
            <h2>
              {t.predictionFor} — {format(new Date(pred.date + "T00:00:00"), "EEEE, MMMM d, yyyy")}
            </h2>
          </div>
          <div className="card-body">
            <div className="prediction-result">
              <div className="prediction-value">
                {(pred.predicted_pilgrims ?? pred.actual_pilgrims)?.toLocaleString()}
              </div>
              <div className="prediction-label">
                {pred.is_past ? t.actualPilgrims : t.estimatedPilgrims}
              </div>

              {pred.crowd_level && (
                <div style={{ marginTop: ".75rem" }}>
                  <CrowdBadge level={pred.crowd_level} />
                </div>
              )}

              {pred.confidence_low && (
                <div className="prediction-confidence">
                  {t.confidenceRange}: {pred.confidence_low.toLocaleString()} — {pred.confidence_high.toLocaleString()} {t.pilgrims}
                </div>
              )}
            </div>

            {/* Past date with actual & predicted */}
            {pred.is_past && pred.predicted_pilgrims && (
              <div style={{ textAlign: "center", marginTop: "1rem", padding: "1rem", background: "var(--off-white)", borderRadius: "var(--radius-sm)" }}>
                <div style={{ fontSize: ".85rem", color: "var(--text-muted)", marginBottom: ".25rem" }}>{t.asPerEstimate}</div>
                <div style={{ fontFamily: "Playfair Display, serif", fontSize: "1.5rem", fontWeight: 700, color: "var(--gold-dark)" }}>
                  {pred.predicted_pilgrims.toLocaleString()}
                </div>
                {pred.error != null && (
                  <div style={{ fontSize: ".82rem", color: "var(--text-light)", marginTop: ".25rem" }}>
                    {t.difference}: {pred.error.toLocaleString()} {t.pilgrims}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
