import { useState, useRef, useEffect } from "react";
import { useLang } from "../i18n/LangContext";
import { sendChatMessage } from "../api";
import { MdSend, MdSmartToy, MdPerson, MdInfoOutline } from "react-icons/md";
import { GiTempleDoor } from "react-icons/gi";

const QUICK_QUESTIONS = {
  te: [
    "దర్శన రకాలు ఏమిటి?",
    "టెంపుల్ డ్రెస్ కోడ్ ఏమిటి?",
    "తిరుమల ఎలా చేరుకోవాలి?",
    "లడ్డూ గురించి చెప్పండి",
    "ఆన్‌లైన్ బుకింగ్ ఎలా?",
    "బ్రహ్మోత్సవాలు ఎప్పుడు?",
  ],
  en: [
    "What are the darshan types?",
    "What is the dress code?",
    "How to reach Tirumala?",
    "Tell me about laddu",
    "How to book online?",
    "When is Brahmotsavam?",
  ],
  hi: [
    "दर्शन के प्रकार क्या हैं?",
    "ड्रेस कोड क्या है?",
    "तिरुमला कैसे पहुंचें?",
    "लड्डू के बारे में बताएं",
    "ऑनलाइन बुकिंग कैसे करें?",
    "ब्रह्मोत्सवम कब होता है?",
  ],
};

export default function Chatbot() {
  const { t, lang } = useLang();
  const [messages, setMessages] = useState([
    {
      role: "bot",
      text: t.chatWelcome || "🙏 Om Namo Venkatesaya! Welcome to the TTD Chatbot. Ask me anything about Tirumala Temple, darshan, sevas, accommodation, travel, and more!",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    // Update welcome message when language changes
    setMessages([{
      role: "bot",
      text: t.chatWelcome || "🙏 Om Namo Venkatesaya! Welcome to the TTD Chatbot. Ask me anything about Tirumala Temple, darshan, sevas, accommodation, travel, and more!",
    }]);
  }, [lang]);

  async function handleSend(text) {
    const msg = (text || input).trim();
    if (!msg || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: msg }]);
    setInput("");
    setLoading(true);

    try {
      const res = await sendChatMessage(msg);
      const reply = res.data?.reply || t.chatError || "Sorry, something went wrong.";
      setMessages((prev) => [...prev, { role: "bot", text: reply }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: t.chatError || "🙏 Sorry, I couldn't connect to the server. Please try again." },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const quickQs = QUICK_QUESTIONS[lang] || QUICK_QUESTIONS.en;

  return (
    <section className="page chatbot-page">
      <div className="page-header">
        <GiTempleDoor className="page-header-icon" />
        <h2>{t.chatTitle || "TTD Chatbot"}</h2>
        <p className="page-subtitle">{t.chatSubtitle || "Ask me anything about Tirumala Tirupati Devasthanams"}</p>
      </div>

      <div className="chatbot-container">
        {/* Quick Questions Sidebar */}
        <div className="chat-sidebar">
          <div className="sidebar-header">
            <MdInfoOutline className="icon" />
            <span>{t.chatQuickQ || "Quick Questions"}</span>
          </div>
          <div className="quick-questions">
            {quickQs.map((q, i) => (
              <button
                key={i}
                className="quick-q-btn"
                onClick={() => handleSend(q)}
                disabled={loading}
              >
                {q}
              </button>
            ))}
          </div>
          <div className="sidebar-topics">
            <h4>{t.chatTopics || "I can help with"}</h4>
            <ul>
              <li>🛕 {t.chatTopicDarshan || "Darshan types & timings"}</li>
              <li>🙏 {t.chatTopicSevas || "Sevas & rituals"}</li>
              <li>🏨 {t.chatTopicAccommodation || "Accommodation"}</li>
              <li>🚌 {t.chatTopicTravel || "How to reach"}</li>
              <li>🍬 {t.chatTopicPrasadam || "Prasadam & Laddu"}</li>
              <li>👔 {t.chatTopicDressCode || "Dress code & rules"}</li>
              <li>🎉 {t.chatTopicFestivals || "Festivals"}</li>
              <li>💰 {t.chatTopicDonations || "Hundi & Donations"}</li>
              <li>🌐 {t.chatTopicOnline || "Online services"}</li>
            </ul>
          </div>
        </div>

        {/* Chat Area */}
        <div className="chat-main">
          <div className="chat-messages">
            {messages.map((m, i) => (
              <div key={i} className={`chat-bubble ${m.role}`}>
                <div className="bubble-avatar">
                  {m.role === "bot" ? (
                    <MdSmartToy className="avatar-icon bot-avatar" />
                  ) : (
                    <MdPerson className="avatar-icon user-avatar" />
                  )}
                </div>
                <div className="bubble-content">
                  <div className="bubble-text">{m.text}</div>
                </div>
              </div>
            ))}
            {loading && (
              <div className="chat-bubble bot">
                <div className="bubble-avatar">
                  <MdSmartToy className="avatar-icon bot-avatar" />
                </div>
                <div className="bubble-content">
                  <div className="bubble-text typing-indicator">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="chat-input-area">
            <input
              ref={inputRef}
              type="text"
              className="chat-input"
              placeholder={t.chatPlaceholder || "Type your question about TTD..."}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              autoFocus
            />
            <button
              className="chat-send-btn"
              onClick={() => handleSend()}
              disabled={!input.trim() || loading}
              title={t.chatSend || "Send"}
            >
              <MdSend />
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
