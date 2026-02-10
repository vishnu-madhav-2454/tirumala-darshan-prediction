import { useState } from "react";
import { NavLink } from "react-router-dom";
import { GiTempleDoor } from "react-icons/gi";
import {
  MdDashboard,
  MdCalendarMonth,
  MdTimeline,
  MdStorage,
  MdLanguage,
  MdMenu,
  MdClose,
} from "react-icons/md";
import { useLang } from "../i18n/LangContext";

const ICONS = [MdDashboard, MdCalendarMonth, MdTimeline, MdStorage];
const PATHS = ["/", "/predict", "/forecast", "/history"];
const KEYS  = ["navHome", "navPredict", "navForecast", "navHistory"];

export default function Navbar() {
  const { t, lang, setLang, SUPPORTED, translations } = useLang();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  function closeMenu() {
    setMobileMenuOpen(false);
  }

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        {/* Brand + hamburger row */}
        <div className="nav-brand-row">
          <NavLink to="/" className="navbar-brand" onClick={closeMenu}>
            <GiTempleDoor className="temple-icon" />
            <div>
              <h1>{t.brand}</h1>
              <div className="subtitle">{t.brandSub}</div>
            </div>
          </NavLink>

          {/* Mobile menu toggle */}
          <button
            className="mobile-menu-toggle"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <MdClose /> : <MdMenu />}
          </button>
        </div>

        {/* Nav links */}
        <div className={`nav-links ${mobileMenuOpen ? "mobile-open" : ""}`}>
          {PATHS.map((path, i) => {
            const Icon = ICONS[i];
            return (
              <NavLink
                key={path}
                to={path}
                end={path === "/"}
                className={({ isActive }) =>
                  `nav-link${isActive ? " active" : ""}`
                }
                onClick={closeMenu}
              >
                <Icon className="icon" />
                {t[KEYS[i]]}
              </NavLink>
            );
          })}

          {/* Language switcher */}
          <div className="lang-switcher">
            <MdLanguage className="icon" style={{ fontSize: "1.1rem" }} />
            {SUPPORTED.map((code) => (
              <button
                key={code}
                className={`lang-btn${lang === code ? " active" : ""}`}
                onClick={() => setLang(code)}
                title={translations[code].lang}
              >
                {translations[code].lang}
              </button>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
