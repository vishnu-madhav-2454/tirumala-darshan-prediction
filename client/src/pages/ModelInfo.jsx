import { useEffect, useState } from "react";
import { MdInfo, MdMemory, MdScience, MdAutoGraph } from "react-icons/md";
import { GiTempleDoor } from "react-icons/gi";
import { getModelInfo } from "../api";
import Loader from "../components/Loader";
import { useLang } from "../i18n/LangContext";

export default function ModelInfo() {
  const { t } = useLang();
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModelInfo()
      .then((r) => setInfo(r.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Loader text={t.infoLoading} />;

  const models = info?.models || [];

  return (
    <div className="main-content fade-in">
      <h2 className="section-title">
        <GiTempleDoor className="ornament" />
        {t.aboutTitle}
      </h2>

      {/* Champion banner */}
      <div className="card" style={{ marginBottom: "2rem" }}>
        <div className="card-header">
          <MdAutoGraph className="icon" />
          <h2>{t.ourMethod}</h2>
        </div>
        <div className="card-body">
          <div style={{ display: "flex", justifyContent: "center", gap: "2rem", flexWrap: "wrap", padding: "1rem 0" }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "Playfair Display, serif", fontSize: "2.5rem", fontWeight: 800, color: "var(--maroon)" }}>2,354</div>
              <div style={{ fontSize: ".85rem", color: "var(--text-muted)" }}>{t.avgDiff}</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "Playfair Display, serif", fontSize: "2.5rem", fontWeight: 800, color: "var(--gold-dark)" }}>0.7504</div>
              <div style={{ fontSize: ".85rem", color: "var(--text-muted)" }}>{t.accuracy}</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "Playfair Display, serif", fontSize: "2.5rem", fontWeight: 800, color: "var(--maroon)" }}>{t.speed}</div>
              <div style={{ fontSize: ".85rem", color: "var(--text-muted)" }}>{t.speedSub}</div>
            </div>
          </div>

          <div style={{ display: "flex", justifyContent: "center", gap: "1rem", flexWrap: "wrap", marginTop: ".5rem" }}>
            <div className="info-tag">📊 {t.dataYears}</div>
            <div className="info-tag">🧬 {t.features68}</div>
            <div className="info-tag">📅 {t.dailyUpdate}</div>
          </div>
        </div>
      </div>

      {/* Models grid */}
      <h3 className="section-title" style={{ fontSize: "1.2rem" }}>
        <MdScience className="ornament" />
        {t.methods}
      </h3>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(350px, 1fr))", gap: "1.25rem" }}>
        {models.map((m) => (
          <div key={m.name} className="model-card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <h3 style={{ fontSize: "1.15rem", color: "var(--maroon)", fontFamily: "Playfair Display, serif" }}>{m.name}</h3>
                <div className="info-tag" style={{ marginTop: ".35rem" }}>{m.type}</div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontFamily: "Playfair Display, serif", fontSize: "1.8rem", fontWeight: 800, color: "var(--gold-dark)" }}>{(m.weight * 100).toFixed(1)}%</div>
                <div style={{ fontSize: ".75rem", color: "var(--text-light)" }}>{t.weight}</div>
              </div>
            </div>
            <p style={{ fontSize: ".88rem", color: "var(--text-muted)", marginTop: ".75rem", lineHeight: 1.5 }}>{m.description}</p>
            {m.paper && <div style={{ marginTop: ".5rem", fontSize: ".8rem", color: "var(--text-light)", fontStyle: "italic" }}>📄 {m.paper}</div>}
            <div className="model-weight-bar"><div className="fill" style={{ width: `${m.weight * 100}%` }} /></div>
          </div>
        ))}
      </div>

      {/* Feature engineering */}
      <div className="divider"><GiTempleDoor className="icon" /></div>

      <div className="card">
        <div className="card-header">
          <MdMemory className="icon" />
          <h2>{t.analysisParams}</h2>
        </div>
        <div className="card-body">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))", gap: "1rem" }}>
            {[
              { cat: "Calendar", feats: "Day-of-week, month, quarter, year, day-of-year, week-of-year, is_weekend" },
              { cat: "Cyclical", feats: "Sine/cosine encodings for DOW, month, day-of-year (circular periodicity)" },
              { cat: "Fourier", feats: "Yearly, bi-annual, quarterly, monthly harmonics (sin+cos pairs)" },
              { cat: "Hindu Calendar", feats: "Purnima, Amavasya, Ekadashi, Pradosham, Sankranti, Vaikuntha Ekadashi" },
              { cat: "Tirumala Domain", feats: "Brahmotsavam, annual festivals, summer/vacation rush, Garuda Seva" },
              { cat: "Lag Features", feats: "1–7 day lags, 14d, 30d, same-day-last-week, same-day-last-year" },
              { cat: "Rolling Stats", feats: "3/7/14/30-day rolling mean, std, min, max" },
              { cat: "EWMA", feats: "Exponential weighted moving averages (3/7/14-day spans)" },
              { cat: "Momentum", feats: "Day-over-day change, 7-day momentum, acceleration" },
              { cat: "Historical Avg", feats: "Expanding DOW means, monthly means, seasonal means" },
            ].map(({ cat, feats }) => (
              <div key={cat} style={{ background: "var(--off-white)", borderRadius: "var(--radius-sm)", padding: "1rem", border: "1px solid var(--cream-dark)" }}>
                <div style={{ fontWeight: 700, color: "var(--gold-dark)", fontSize: ".82rem", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: ".35rem" }}>{cat}</div>
                <div style={{ fontSize: ".85rem", color: "var(--text-muted)", lineHeight: 1.5 }}>{feats}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Research papers */}
      <div className="card" style={{ marginTop: "1.5rem" }}>
        <div className="card-header">
          <MdInfo className="icon" />
          <h2>{t.researchBasis}</h2>
        </div>
        <div className="card-body">
          <div style={{ display: "grid", gap: ".75rem" }}>
            {[
              { title: "Chronos: Learning the Language of Time Series", authors: "Ansari et al., 2024", venue: "arXiv:2403.07815", desc: "Amazon's T5-based foundation model for zero-shot time series forecasting" },
              { title: "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting", authors: "Challu et al., 2023", venue: "AAAI 2023", desc: "Multi-rate signal sampling with hierarchical interpolation" },
              { title: "N-BEATS: Neural Basis Expansion Analysis", authors: "Oreshkin et al., 2020", venue: "ICLR 2020", desc: "Interpretable deep learning with backward–forward residual blocks" },
              { title: "XGBoost: A Scalable Tree Boosting System", authors: "Chen & Guestrin, 2016", venue: "KDD 2016", desc: "Gradient boosting with regularization and efficient split finding" },
              { title: "LightGBM: A Highly Efficient Gradient Boosting Framework", authors: "Ke et al., 2017", venue: "NeurIPS 2017", desc: "Gradient-based one-side sampling for faster training" },
            ].map((paper) => (
              <div key={paper.title} style={{ padding: "1rem", background: "var(--off-white)", borderRadius: "var(--radius-sm)", border: "1px solid var(--cream-dark)" }}>
                <div style={{ fontWeight: 700, color: "var(--maroon)", fontSize: ".95rem" }}>{paper.title}</div>
                <div style={{ fontSize: ".82rem", color: "var(--text-muted)", marginTop: ".2rem" }}>{paper.authors} — <em>{paper.venue}</em></div>
                <div style={{ fontSize: ".85rem", color: "var(--text-light)", marginTop: ".25rem" }}>{paper.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
