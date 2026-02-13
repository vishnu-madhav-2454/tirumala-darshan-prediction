import { BrowserRouter, Routes, Route } from "react-router-dom";
import { LangProvider } from "./i18n/LangContext";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Dashboard from "./pages/Dashboard";
import Predict from "./pages/Predict";
import History from "./pages/History";
import Chatbot from "./pages/Chatbot";
import TripPlanner from "./pages/TripPlanner";

export default function App() {
  return (
    <LangProvider>
      <BrowserRouter>
        <div className="app-container">
          <Navbar />
          <div className="gold-strip" />
          <div className="main-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/history" element={<History />} />
              <Route path="/chatbot" element={<Chatbot />} />
              <Route path="/trip-planner" element={<TripPlanner />} />
            </Routes>
          </div>
          <Footer />
        </div>
      </BrowserRouter>
    </LangProvider>
  );
}
