import { createContext, useContext, useState, useCallback } from "react";
import translations from "./translations";

const LangContext = createContext();
const SUPPORTED = ["te", "en", "hi", "ta", "ml", "kn"];
const STORAGE_KEY = "srivari_lang";

function getSavedLang() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && SUPPORTED.includes(saved)) return saved;
  } catch {}
  return "en";
}

export function LangProvider({ children }) {
  const [lang, setLangState] = useState(getSavedLang);

  const setLang = useCallback((l) => {
    if (SUPPORTED.includes(l)) {
      setLangState(l);
      try { localStorage.setItem(STORAGE_KEY, l); } catch {}
    }
  }, []);

  const t = translations[lang] || translations.en;

  return (
    <LangContext.Provider value={{ lang, setLang, t, SUPPORTED, translations }}>
      {children}
    </LangContext.Provider>
  );
}

export function useLang() {
  return useContext(LangContext);
}
