import { GiTempleDoor } from "react-icons/gi";
import { useLang } from "../i18n/LangContext";

export default function Footer() {
  const { t } = useLang();

  return (
    <footer className="footer">
      <div className="gold-strip" style={{ marginBottom: "1.5rem" }} />
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: ".6rem" }}>
        <GiTempleDoor style={{ color: "var(--gold)", fontSize: "2rem", marginBottom: ".25rem" }} />
        <span className="gold-text" style={{ fontFamily: "Playfair Display, serif", fontSize: "1.15rem", fontWeight: 700, letterSpacing: ".5px" }}>
          {t.footerSlogan}
        </span>
        <div style={{ maxWidth: 400, lineHeight: 1.5 }}>{t.footerDesc}</div>
        <div style={{ opacity: 0.5, fontSize: ".75rem", marginTop: ".5rem" }}>
          © {new Date().getFullYear()} {t.footerCopy}
        </div>
      </div>
    </footer>
  );
}
