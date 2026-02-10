import { BrowserRouter, Routes, Route } from "react-router-dom";
import { LangProvider } from "./i18n/LangContext";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Dashboard from "./pages/Dashboard";
import Predict from "./pages/Predict";
import Forecast from "./pages/Forecast";
import History from "./pages/History";

export default function App() {
  return (
    <LangProvider>
      <BrowserRouter>
        <div className="app-container">
          <Navbar />
          <div className="gold-strip" />
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/forecast" element={<Forecast />} />
            <Route path="/history" element={<History />} />
          </Routes>
          <Footer />
        </div>
      </BrowserRouter>
    </LangProvider>
  );
}
