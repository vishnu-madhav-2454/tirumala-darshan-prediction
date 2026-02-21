import { GiTempleDoor } from "react-icons/gi";
import { useLang } from "../i18n/LangContext";

export default function Footer() {
  const { t } = useLang();
  return (
    <footer className="footer">
      <div className="footer-brand">
        <GiTempleDoor />
        <span>{t.brand}</span>
      </div>
      <div className="footer-slogan">{t.footerSlogan}</div>
      <div className="footer-ornament">✦ ✦ ✦ ✦ ✦</div>
      <div className="footer-desc">{t.footerDesc}</div>
      <div className="footer-copy">
        © {new Date().getFullYear()} {t.footerCopy}
      </div>
    </footer>
  );
}
