/**
 * Language Context — provides multilingual support across the app.
 * Persists language choice in localStorage.
 */
import { createContext, useContext, useState, useCallback } from "react";
import translations from "./translations";

const LangContext = createContext();

const SUPPORTED = ["te", "en", "hi"];
const STORAGE_KEY = "srivari_lang";

function getSavedLang() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && SUPPORTED.includes(saved)) return saved;
  } catch {}
  return "te"; // default: Telugu
}

export function LangProvider({ children }) {
  const [lang, setLangState] = useState(getSavedLang);

  const setLang = useCallback((code) => {
    if (SUPPORTED.includes(code)) {
      setLangState(code);
      try { localStorage.setItem(STORAGE_KEY, code); } catch {}
    }
  }, []);

  const t = translations[lang] || translations.te;

  return (
    <LangContext.Provider value={{ lang, setLang, t, SUPPORTED, translations }}>
      {children}
    </LangContext.Provider>
  );
}

export function useLang() {
  const ctx = useContext(LangContext);
  if (!ctx) throw new Error("useLang must be used within LangProvider");
  return ctx;
}
