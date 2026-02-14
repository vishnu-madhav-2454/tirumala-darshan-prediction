import { useState } from "react";
import { NavLink } from "react-router-dom";
import { GiTempleDoor } from "react-icons/gi";
import { MdDashboard, MdCalendarMonth, MdStorage, MdLanguage, MdMenu, MdClose, MdSmartToy, MdExplore } from "react-icons/md";
import { useLang } from "../i18n/LangContext";

const ICONS = [MdDashboard, MdCalendarMonth, MdStorage, MdSmartToy, MdExplore];
const PATHS = ["/", "/predict", "/history", "/chatbot", "/explore"];
const KEYS = ["navHome", "navPredict", "navHistory", "navChatbot", "navExplore"];

export default function Navbar() {
  const { t, lang, setLang, SUPPORTED, translations } = useLang();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  function closeMenu() { setMobileMenuOpen(false); }

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <NavLink to="/" className="nav-brand-row" onClick={closeMenu}>
          <GiTempleDoor className="temple-icon" />
          <div>
            <h1>{t.brand}</h1>
            <div className="subtitle">{t.brandSub}</div>
          </div>
        </NavLink>

        <button className="mobile-menu-toggle" onClick={() => setMobileMenuOpen(!mobileMenuOpen)} aria-label="Toggle menu">
          {mobileMenuOpen ? <MdClose /> : <MdMenu />}
        </button>

        <div className={`nav-links ${mobileMenuOpen ? "mobile-open" : ""}`}>
          {PATHS.map((path, i) => {
            const Icon = ICONS[i];
            return (
              <NavLink key={path} to={path} end={path === "/"} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`} onClick={closeMenu}>
                <Icon className="nav-icon" />
                <span>{t[KEYS[i]]}</span>
              </NavLink>
            );
          })}
          <div className="lang-selector">
            <MdLanguage className="lang-icon" />
            <select value={lang} onChange={(e) => setLang(e.target.value)}>
              {SUPPORTED.map((l) => (
                <option key={l} value={l}>
                  {translations[l]?.flag} {translations[l]?.lang}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </nav>
  );
}
