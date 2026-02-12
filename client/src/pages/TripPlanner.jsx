import { useState, useEffect, useRef, useCallback } from "react";
import { MdExplore, MdMap, MdRestaurant, MdHotel, MdDirectionsBus } from "react-icons/md";
import { GiTempleDoor, GiBackpack } from "react-icons/gi";
import { getTripPlan } from "../api";
import Loader from "../components/Loader";
import { useLang } from "../i18n/LangContext";

const BUDGET_OPTIONS = ["Budget", "Moderate", "Premium"];
const INTEREST_OPTIONS = [
  "Temples & Darshan",
  "Nature & Trekking",
  "History & Culture",
  "Local Food",
  "Photography",
  "Shopping",
  "Spiritual Experiences",
];

export default function TripPlanner() {
  const { t } = useLang();
  const [form, setForm] = useState({
    days: 2,
    group_size: 2,
    budget: "Moderate",
    interests: ["Temples & Darshan"],
  });
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);

  function toggleInterest(interest) {
    setForm((prev) => {
      const has = prev.interests.includes(interest);
      return {
        ...prev,
        interests: has
          ? prev.interests.filter((i) => i !== interest)
          : [...prev.interests, interest],
      };
    });
  }

  async function handleGenerate() {
    setLoading(true);
    setError(null);
    setPlan(null);
    try {
      const res = await getTripPlan({
        days: form.days,
        group_size: form.group_size,
        budget: form.budget,
        interests: form.interests,
      });
      setPlan(res.data);
    } catch (e) {
      setError(e.response?.data?.error || t.tripError || "Failed to generate trip plan.");
    } finally {
      setLoading(false);
    }
  }

  /* ── Leaflet Map ── */
  const initMap = useCallback(() => {
    if (!plan?.plan?.map_points?.length || !mapContainerRef.current) return;
    if (!window.L) return;

    // Destroy old map
    if (mapInstanceRef.current) {
      mapInstanceRef.current.remove();
      mapInstanceRef.current = null;
    }

    const L = window.L;
    const points = plan.plan.map_points;

    const map = L.map(mapContainerRef.current, {
      scrollWheelZoom: false,
    }).setView([13.6833, 79.3474], 13);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
      maxZoom: 18,
    }).addTo(map);

    const templeIcon = L.divIcon({
      html: '<div style="background:#800020;color:#FFD700;width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:18px;border:2px solid #FFD700;box-shadow:0 2px 6px rgba(0,0,0,0.3)">🛕</div>',
      iconSize: [32, 32],
      className: "",
    });

    const defaultIcon = L.divIcon({
      html: '<div style="background:#C5A028;color:#FFF;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;border:2px solid #800020;box-shadow:0 2px 6px rgba(0,0,0,0.3)">📍</div>',
      iconSize: [28, 28],
      className: "",
    });

    const validPoints = points.filter(
      (p) =>
        p.lat && p.lng && !isNaN(parseFloat(p.lat)) && !isNaN(parseFloat(p.lng))
    );

    if (validPoints.length === 0) {
      // Default: Tirumala temple marker
      L.marker([13.6833, 79.3474], { icon: templeIcon })
        .addTo(map)
        .bindPopup("<b>Sri Venkateswara Temple</b><br>Tirumala");
      return;
    }

    const bounds = [];
    validPoints.forEach((pt, i) => {
      const latLng = [parseFloat(pt.lat), parseFloat(pt.lng)];
      bounds.push(latLng);
      const icon = pt.name?.toLowerCase().includes("temple") ? templeIcon : defaultIcon;
      L.marker(latLng, { icon })
        .addTo(map)
        .bindPopup(`<b>${i + 1}. ${pt.name || "Point"}</b>${pt.description ? "<br>" + pt.description : ""}`);
    });

    if (bounds.length > 1) {
      L.polyline(bounds, {
        color: "#800020",
        weight: 3,
        opacity: 0.7,
        dashArray: "8 4",
      }).addTo(map);
      map.fitBounds(bounds, { padding: [40, 40] });
    } else {
      map.setView(bounds[0], 15);
    }

    mapInstanceRef.current = map;

    // Fix tile rendering after container is visible
    setTimeout(() => map.invalidateSize(), 200);
  }, [plan]);

  useEffect(() => {
    if (!plan?.plan?.map_points?.length) return;

    // Load Leaflet CSS
    if (!document.getElementById("leaflet-css")) {
      const link = document.createElement("link");
      link.id = "leaflet-css";
      link.rel = "stylesheet";
      link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
      document.head.appendChild(link);
    }

    // Load Leaflet JS
    if (!window.L) {
      if (!document.getElementById("leaflet-js")) {
        const script = document.createElement("script");
        script.id = "leaflet-js";
        script.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
        script.onload = () => setTimeout(initMap, 100);
        document.head.appendChild(script);
      } else {
        const check = setInterval(() => {
          if (window.L) {
            clearInterval(check);
            initMap();
          }
        }, 100);
      }
    } else {
      setTimeout(initMap, 100);
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [plan, initMap]);

  const tripPlan = plan?.plan;
  const cost = tripPlan?.cost_breakdown;

  return (
    <div className="main-content fade-in">
      <h2 className="section-title">
        <GiTempleDoor className="ornament" />
        {t.tripTitle}
      </h2>
      <p
        style={{
          textAlign: "center",
          color: "var(--text-muted)",
          marginBottom: "1.5rem",
          fontSize: ".95rem",
        }}
      >
        {t.tripSubtitleNew ||
          "Already in Tirupati? Tell us your plans and we'll craft a perfect itinerary!"}
      </p>

      {/* ── Form ── */}
      <div className="card" style={{ marginBottom: "2rem" }}>
        <div className="card-header">
          <MdExplore className="icon" />
          <h2>{t.tripPlanYourStay || "Plan Your Stay"}</h2>
        </div>
        <div className="card-body">
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: "1.25rem",
            }}
          >
            {/* Days */}
            <div className="form-group">
              <label className="form-label">{t.tripDays}</label>
              <select
                className="form-input"
                value={form.days}
                onChange={(e) =>
                  setForm((f) => ({ ...f, days: +e.target.value }))
                }
              >
                {[1, 2, 3, 4, 5, 6, 7].map((d) => (
                  <option key={d} value={d}>
                    {d} {d === 1 ? "Day" : "Days"}
                  </option>
                ))}
              </select>
            </div>

            {/* Group Size */}
            <div className="form-group">
              <label className="form-label">{t.tripGroupSize}</label>
              <select
                className="form-input"
                value={form.group_size}
                onChange={(e) =>
                  setForm((f) => ({ ...f, group_size: +e.target.value }))
                }
              >
                {[1, 2, 3, 4, 5, 6, 8, 10].map((g) => (
                  <option key={g} value={g}>
                    {g} {g === 1 ? "Person" : "People"}
                  </option>
                ))}
              </select>
            </div>

            {/* Budget */}
            <div className="form-group">
              <label className="form-label">{t.tripBudget}</label>
              <select
                className="form-input"
                value={form.budget}
                onChange={(e) =>
                  setForm((f) => ({ ...f, budget: e.target.value }))
                }
              >
                {BUDGET_OPTIONS.map((b) => (
                  <option key={b} value={b}>
                    {b === "Budget" ? "💰 Budget" : b === "Moderate" ? "💰💰 Moderate" : "💰💰💰 Premium"}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Interests */}
          <div className="form-group" style={{ marginTop: "1.25rem" }}>
            <label className="form-label">{t.tripInterests}</label>
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: ".5rem",
                marginTop: ".5rem",
              }}
            >
              {INTEREST_OPTIONS.map((interest) => {
                const active = form.interests.includes(interest);
                return (
                  <button
                    key={interest}
                    onClick={() => toggleInterest(interest)}
                    style={{
                      padding: ".45rem .9rem",
                      borderRadius: "9999px",
                      fontSize: ".85rem",
                      fontWeight: 500,
                      border: active
                        ? "2px solid var(--maroon)"
                        : "1px solid var(--cream-dark)",
                      background: active ? "var(--maroon)" : "var(--off-white)",
                      color: active ? "#FFF" : "var(--text-dark)",
                      cursor: "pointer",
                      transition: "all .2s",
                    }}
                  >
                    {interest}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Generate */}
          <button
            className="btn btn-primary"
            style={{ marginTop: "1.5rem", width: "100%", padding: ".85rem" }}
            onClick={handleGenerate}
            disabled={loading}
          >
            <MdExplore /> {t.tripGenerate}
          </button>
        </div>
      </div>

      {loading && <Loader text={t.tripGenerating} />}
      {error && <div className="error-message">⚠️ {error}</div>}

      {/* ── Results ── */}
      {tripPlan && !loading && (
        <>
          {/* Map */}
          <div className="card" style={{ marginBottom: "2rem" }}>
            <div className="card-header">
              <MdMap className="icon" />
              <h2>{t.tripMap}</h2>
            </div>
            <div className="card-body" style={{ padding: 0 }}>
              <div
                ref={mapContainerRef}
                style={{
                  width: "100%",
                  height: "400px",
                  borderRadius: "0 0 var(--radius) var(--radius)",
                  zIndex: 1,
                }}
              />
            </div>
          </div>

          {/* Cost Breakdown */}
          {cost && (
            <div className="card" style={{ marginBottom: "2rem" }}>
              <div className="card-header">
                <span className="icon" style={{ fontSize: "1.3rem" }}>💰</span>
                <h2>{t.tripCostBreakdown}</h2>
              </div>
              <div className="card-body">
                <div className="stat-grid">
                  {cost.transport != null && (
                    <div className="stat-card">
                      <div className="stat-icon"><MdDirectionsBus /></div>
                      <div className="stat-value">₹{cost.transport?.toLocaleString()}</div>
                      <div className="stat-label">{t.tripLocalTransport || "Local Transport"}</div>
                    </div>
                  )}
                  {cost.accommodation != null && (
                    <div className="stat-card">
                      <div className="stat-icon"><MdHotel /></div>
                      <div className="stat-value">₹{cost.accommodation?.toLocaleString()}</div>
                      <div className="stat-label">{t.tripStay}</div>
                    </div>
                  )}
                  {cost.food != null && (
                    <div className="stat-card">
                      <div className="stat-icon"><MdRestaurant /></div>
                      <div className="stat-value">₹{cost.food?.toLocaleString()}</div>
                      <div className="stat-label">{t.tripFood}</div>
                    </div>
                  )}
                  {cost.darshan_sevas != null && (
                    <div className="stat-card">
                      <div className="stat-icon">🛕</div>
                      <div className="stat-value">₹{cost.darshan_sevas?.toLocaleString()}</div>
                      <div className="stat-label">{t.tripDarshanSevas}</div>
                    </div>
                  )}
                  {cost.attractions != null && (
                    <div className="stat-card">
                      <div className="stat-icon">🏛️</div>
                      <div className="stat-value">₹{cost.attractions?.toLocaleString()}</div>
                      <div className="stat-label">{t.tripAttractions}</div>
                    </div>
                  )}
                </div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "center",
                    gap: "2rem",
                    marginTop: "1.5rem",
                    flexWrap: "wrap",
                  }}
                >
                  {cost.per_person_total != null && (
                    <div
                      style={{
                        textAlign: "center",
                        padding: "1rem 2rem",
                        background: "var(--off-white)",
                        borderRadius: "var(--radius-sm)",
                        border: "1px solid var(--cream-dark)",
                      }}
                    >
                      <div style={{ fontSize: "1.6rem", fontWeight: 700, color: "var(--maroon)" }}>
                        ₹{cost.per_person_total?.toLocaleString()}
                      </div>
                      <div style={{ fontSize: ".85rem", color: "var(--text-muted)" }}>
                        {t.tripPerPerson}
                      </div>
                    </div>
                  )}
                  {cost.group_total != null && (
                    <div
                      style={{
                        textAlign: "center",
                        padding: "1rem 2rem",
                        background: "var(--off-white)",
                        borderRadius: "var(--radius-sm)",
                        border: "1px solid var(--cream-dark)",
                      }}
                    >
                      <div style={{ fontSize: "1.6rem", fontWeight: 700, color: "var(--gold)" }}>
                        ₹{cost.group_total?.toLocaleString()}
                      </div>
                      <div style={{ fontSize: ".85rem", color: "var(--text-muted)" }}>
                        {t.tripGroupTotal} ({form.group_size} {t.tripPeople})
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Itinerary */}
          {tripPlan.itinerary && (
            <div className="card" style={{ marginBottom: "2rem" }}>
              <div className="card-header">
                <MdExplore className="icon" />
                <h2>{t.tripItinerary}</h2>
              </div>
              <div className="card-body">
                {tripPlan.itinerary.map((day, di) => (
                  <div
                    key={di}
                    style={{
                      marginBottom: di < tripPlan.itinerary.length - 1 ? "1.5rem" : 0,
                      paddingBottom: di < tripPlan.itinerary.length - 1 ? "1.5rem" : 0,
                      borderBottom:
                        di < tripPlan.itinerary.length - 1
                          ? "1px solid var(--cream-dark)"
                          : "none",
                    }}
                  >
                    <h3
                      style={{
                        color: "var(--maroon)",
                        fontSize: "1.05rem",
                        marginBottom: ".75rem",
                      }}
                    >
                      🗓️ {day.day || `Day ${di + 1}`}
                      {day.theme && (
                        <span
                          style={{
                            marginLeft: ".75rem",
                            fontSize: ".85rem",
                            color: "var(--gold)",
                            fontWeight: 400,
                          }}
                        >
                          — {day.theme}
                        </span>
                      )}
                    </h3>
                    {day.activities?.map((act, ai) => (
                      <div
                        key={ai}
                        style={{
                          display: "flex",
                          gap: ".75rem",
                          marginBottom: ".6rem",
                          padding: ".6rem .75rem",
                          background: ai % 2 === 0 ? "var(--off-white)" : "transparent",
                          borderRadius: "var(--radius-sm)",
                        }}
                      >
                        {act.time && (
                          <span
                            style={{
                              minWidth: 80,
                              fontWeight: 600,
                              color: "var(--gold)",
                              fontSize: ".85rem",
                            }}
                          >
                            {act.time}
                          </span>
                        )}
                        <div>
                          <div style={{ fontWeight: 500, color: "var(--text-dark)" }}>
                            {act.activity || act.name}
                          </div>
                          {act.details && (
                            <div
                              style={{
                                fontSize: ".85rem",
                                color: "var(--text-muted)",
                                marginTop: ".2rem",
                              }}
                            >
                              {act.details}
                            </div>
                          )}
                          {act.cost && (
                            <div
                              style={{
                                fontSize: ".8rem",
                                color: "var(--maroon)",
                                marginTop: ".2rem",
                              }}
                            >
                              💰 {act.cost}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                    {day.hotel && (
                      <div
                        style={{
                          marginTop: ".75rem",
                          padding: ".6rem .75rem",
                          background: "var(--off-white)",
                          borderRadius: "var(--radius-sm)",
                          fontSize: ".9rem",
                        }}
                      >
                        🏨 <strong>{t.tripRecommendedHotel}:</strong> {day.hotel}
                        {day.hotel_cost && (
                          <span style={{ marginLeft: ".5rem", color: "var(--maroon)" }}>
                            (₹{day.hotel_cost}/{t.tripNight})
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Packing Tips */}
          {tripPlan.packing_tips?.length > 0 && (
            <div className="card">
              <div className="card-header">
                <GiBackpack className="icon" />
                <h2>{t.tripPackingTips}</h2>
              </div>
              <div className="card-body">
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
                    gap: ".5rem",
                  }}
                >
                  {tripPlan.packing_tips.map((tip, i) => (
                    <div
                      key={i}
                      style={{
                        padding: ".5rem .75rem",
                        background: "var(--off-white)",
                        borderRadius: "var(--radius-sm)",
                        fontSize: ".9rem",
                        display: "flex",
                        alignItems: "center",
                        gap: ".5rem",
                      }}
                    >
                      <span>✅</span> {tip}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
