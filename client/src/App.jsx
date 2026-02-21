import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import { LangProvider } from "./i18n/LangContext";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Dashboard from "./pages/Dashboard";
import Predict from "./pages/Predict";
import History from "./pages/History";
import Chatbot from "./pages/Chatbot";
import Explore from "./pages/Explore";
import { MdKeyboardArrowUp } from "react-icons/md";

/* Scroll-to-top on route change */
function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [pathname]);
  return null;
}

/* Scroll-to-top floating button */
function ScrollTopBtn() {
  const [show, setShow] = useState(false);
  useEffect(() => {
    const onScroll = () => setShow(window.scrollY > 400);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  if (!show) return null;
  return (
    <button
      className="scroll-top-btn"
      onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
      aria-label="Scroll to top"
    >
      <MdKeyboardArrowUp />
    </button>
  );
}

export default function App() {
  return (
    <LangProvider>
      <BrowserRouter>
        <ScrollToTop />
        <div className="app-container">
          <Navbar />
          <div className="gold-strip" />
          <div className="main-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/history" element={<History />} />
              <Route path="/chatbot" element={<Chatbot />} />
              <Route path="/explore" element={<Explore />} />
            </Routes>
          </div>
          <Footer />
          <ScrollTopBtn />
        </div>
      </BrowserRouter>
    </LangProvider>
  );
}
