import { useState, useEffect, useRef } from "react";
import { useLang } from "../i18n/LangContext";
import { getTripPlan } from "../api";
import Loader from "../components/Loader";
import { GiTempleDoor } from "react-icons/gi";
import {
  MdTravelExplore, MdGroup, MdAttachMoney, MdAutoAwesome,
  MdHotel, MdDirectionsBus, MdRestaurant, MdTempleHindu,
  MdLocalActivity, MdBackpack, MdCheckCircle, MdPlace,
} from "react-icons/md";

const INTERESTS = [
  { id: "temples", label: "Temples & Darshan", emoji: "🛕" },
  { id: "nature", label: "Nature & Trekking", emoji: "🌿" },
  { id: "history", label: "History & Culture", emoji: "🏛️" },
  { id: "food", label: "Local Food", emoji: "🍛" },
  { id: "photography", label: "Photography", emoji: "📸" },
  { id: "shopping", label: "Shopping", emoji: "🛍️" },
  { id: "spiritual", label: "Spiritual Experiences", emoji: "🕉️" },
];

const BUDGET_OPTIONS = [
  { id: "budget", label: "Budget 💰", value: "budget" },
  { id: "standard", label: "Standard 💰💰", value: "standard" },
  { id: "premium", label: "Premium 💰💰💰", value: "premium" },
];

const COST_ICONS = {
  transport: <MdDirectionsBus />,
  accommodation: <MdHotel />,
  food: <MdRestaurant />,
  darshan: <MdTempleHindu />,
  activities: <MdLocalActivity />,
  attractions: <MdLocalActivity />,
  total: <MdAttachMoney />,
};

export default function TripPlanner() {
  const { t } = useLang();
  const [form, setForm] = useState({
    days: 2,
    group_size: 2,
    budget: "standard",
    interests: ["temples", "spiritual"],
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const resultRef = useRef(null);
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);

  function toggleInterest(id) {
    setForm((prev) => {
      const interests = prev.interests.includes(id)
        ? prev.interests.filter((x) => x !== id)
        : [...prev.interests, id];
      return { ...prev, interests };
    });
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (form.interests.length === 0) {
      setError(t.tripSelectInterest || "Please select at least one interest");
      return;
    }
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const res = await getTripPlan(form);
      setResult(res.data);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth" }), 200);
    } catch (err) {
      setError(err?.response?.data?.error || t.tripError || "Failed to generate trip plan. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  /* Leaflet map rendering */
  useEffect(() => {
    if (!result?.map_points?.length || !mapContainerRef.current) return;

    const loadLeaflet = async () => {
      if (!document.getElementById("leaflet-css")) {
        const link = document.createElement("link");
        link.id = "leaflet-css";
        link.rel = "stylesheet";
        link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
        document.head.appendChild(link);
      }

      if (!window.L) {
        await new Promise((resolve) => {
          if (document.getElementById("leaflet-js")) {
            const check = setInterval(() => {
              if (window.L) { clearInterval(check); resolve(); }
            }, 100);
            return;
          }
          const script = document.createElement("script");
          script.id = "leaflet-js";
          script.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
          script.onload = resolve;
          document.head.appendChild(script);
        });
      }

      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }

      const L = window.L;
      const map = L.map(mapContainerRef.current).setView([13.6833, 79.3474], 12);
      mapInstanceRef.current = map;

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap",
      }).addTo(map);

      const markerIcon = L.divIcon({
        className: "custom-marker",
        html: `<div style="background:#800020;color:#FFD700;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:bold;border:2px solid #FFD700;font-size:14px;">🛕</div>`,
        iconSize: [28, 28],
      });

      result.map_points.forEach((pt) => {
        L.marker([pt.lat, pt.lng], { icon: markerIcon })
          .addTo(map)
          .bindPopup(`<b>${pt.name}</b><br/>${pt.description || ""}`);
      });

      const bounds = L.latLngBounds(result.map_points.map((p) => [p.lat, p.lng]));
      map.fitBounds(bounds, { padding: [40, 40] });
    };

    loadLeaflet();

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [result]);

  return (
    <section className="page trip-page">
      <div className="page-header">
        <GiTempleDoor className="page-header-icon" />
        <h2>{t.tripTitle || "AI Trip Planner"}</h2>
        <p className="page-subtitle">
          <MdAutoAwesome style={{ verticalAlign: "middle", marginRight: 4, color: "#DAA520" }} />
          {t.tripSubtitle || "Plan your perfect Tirumala pilgrimage"}
        </p>
      </div>

      {/* Form */}
      <form className="trip-form card" onSubmit={handleSubmit}>
        <div className="trip-form-grid">
          <div className="form-group">
            <label className="form-label"><MdTravelExplore className="form-icon" /> {t.tripDays || "Days"}</label>
            <select className="form-input" value={form.days} onChange={(e) => setForm({ ...form, days: +e.target.value })}>
              {[1, 2, 3, 4, 5, 6, 7].map((d) => (
                <option key={d} value={d}>{d} {d === 1 ? "Day" : "Days"}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label"><MdGroup className="form-icon" /> {t.tripGroupSize || "Group Size"}</label>
            <select className="form-input" value={form.group_size} onChange={(e) => setForm({ ...form, group_size: +e.target.value })}>
              {Array.from({ length: 10 }, (_, i) => i + 1).map((n) => (
                <option key={n} value={n}>{n} {n === 1 ? "Person" : "People"}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label"><MdAttachMoney className="form-icon" /> {t.tripBudget || "Budget"}</label>
            <div className="budget-options">
              {BUDGET_OPTIONS.map((b) => (
                <button
                  key={b.id}
                  type="button"
                  className={`budget-btn ${form.budget === b.value ? "active" : ""}`}
                  onClick={() => setForm({ ...form, budget: b.value })}
                >
                  {b.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="form-group interests-group">
          <label className="form-label">{t.tripInterests || "Your Interests"}</label>
          <div className="interest-pills">
            {INTERESTS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`interest-pill ${form.interests.includes(item.id) ? "active" : ""}`}
                onClick={() => toggleInterest(item.id)}
              >
                {item.emoji} {item.label}
              </button>
            ))}
          </div>
        </div>

        {error && <div className="form-error">{error}</div>}

        <button type="submit" className="btn btn-primary btn-lg" disabled={loading}>
          <MdAutoAwesome /> {loading ? (t.tripGenerating || "Generating...") : (t.tripGenerate || "Generate Trip Plan")}
        </button>
      </form>

      {loading && <Loader text={t.tripLoading || "Creating your personalized trip plan..."} />}

      {/* Results */}
      {result && (
        <div className="trip-results" ref={resultRef}>
          <div className="gold-strip" />
          <h3 className="section-title">{result.title || "Your Tirumala Trip Plan"}</h3>
          {result.summary && <p className="trip-summary">{result.summary}</p>}

          {/* Map */}
          {result.map_points?.length > 0 && (
            <div className="card trip-map-card">
              <div className="card-header"><MdPlace /> {t.tripMap || "Trip Map"}</div>
              <div className="card-body">
                <div ref={mapContainerRef} className="trip-map" />
              </div>
            </div>
          )}

          {/* Cost Breakdown */}
          {result.cost_breakdown && (
            <div className="card cost-card">
              <div className="card-header"><MdAttachMoney /> {t.tripCost || "Cost Breakdown"}</div>
              <div className="card-body">
                <div className="stat-grid">
                  {Object.entries(result.cost_breakdown)
                    .filter(([k]) => !["per_person", "group_total", "total"].includes(k))
                    .map(([key, val]) => (
                      <div className="stat-card" key={key}>
                        <div className="stat-icon">{COST_ICONS[key] || <MdAttachMoney />}</div>
                        <div className="stat-value">₹{typeof val === "number" ? val.toLocaleString("en-IN") : val}</div>
                        <div className="stat-label">{key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, " ")}</div>
                      </div>
                    ))}
                </div>
                <div className="cost-totals">
                  {result.cost_breakdown.per_person != null && (
                    <div className="cost-total-item">
                      <span>{t.tripPerPerson || "Per Person"}</span>
                      <strong>₹{result.cost_breakdown.per_person.toLocaleString("en-IN")}</strong>
                    </div>
                  )}
                  {result.cost_breakdown.group_total != null && (
                    <div className="cost-total-item highlight">
                      <span>{t.tripGroupTotal || "Group Total"}</span>
                      <strong>₹{result.cost_breakdown.group_total.toLocaleString("en-IN")}</strong>
                    </div>
                  )}
                  {result.cost_breakdown.total != null && !result.cost_breakdown.group_total && (
                    <div className="cost-total-item highlight">
                      <span>{t.tripTotal || "Total"}</span>
                      <strong>₹{result.cost_breakdown.total.toLocaleString("en-IN")}</strong>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Itinerary */}
          {result.itinerary?.length > 0 && (
            <div className="card itinerary-card">
              <div className="card-header"><MdTravelExplore /> {t.tripItinerary || "Day-by-Day Itinerary"}</div>
              <div className="card-body">
                {result.itinerary.map((day, di) => (
                  <div key={di} className="itinerary-day">
                    <h4 className="day-title">{day.day || `Day ${di + 1}`}</h4>
                    {day.hotel && (
                      <div className="day-hotel">
                        <MdHotel className="icon" /> <strong>{t.tripHotel || "Hotel"}:</strong> {day.hotel}
                      </div>
                    )}
                    <div className="day-activities">
                      {day.activities?.map((act, ai) => (
                        <div key={ai} className="activity-item">
                          <div className="activity-time">{act.time || ""}</div>
                          <div className="activity-info">
                            <div className="activity-name">{act.name || act.activity || ""}</div>
                            {act.details && <div className="activity-details">{act.details}</div>}
                            {act.cost != null && <div className="activity-cost">💰 ₹{act.cost}</div>}
                            {act.tip && <div className="activity-tip">💡 {act.tip}</div>}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Packing Tips */}
          {result.packing_tips?.length > 0 && (
            <div className="card packing-card">
              <div className="card-header"><MdBackpack /> {t.tripPacking || "Packing Checklist"}</div>
              <div className="card-body">
                <div className="packing-grid">
                  {result.packing_tips.map((item, i) => (
                    <div key={i} className="packing-item">
                      <MdCheckCircle className="check-icon" /> {item}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
