import { useState, useRef, useEffect } from "react";
import { useLang } from "../i18n/LangContext";
import { sendChatMessage } from "../api";
import { MdSend, MdSmartToy, MdPerson, MdInfoOutline, MdAutoAwesome } from "react-icons/md";
import { GiTempleDoor } from "react-icons/gi";

const QUICK_QUESTIONS = {
  te: [
    "దర్శన రకాలు ఏమిటి?",
    "టెంపుల్ డ్రెస్ కోడ్ ఏమిటి?",
    "తిరుమల ఎలా చేరుకోవాలి?",
    "లడ్డూ గురించి చెప్పండి",
    "ఆన్‌లైన్ బుకింగ్ ఎలా?",
    "బ్రహ్మోత్సవాలు ఎప్పుడు?",
    "తిరుమల హోటల్ ధరలు ఎంత?",
    "చెన్నై నుండి తిరుపతి ఎలా?",
  ],
  en: [
    "What are the darshan types?",
    "What is the dress code?",
    "How to reach Tirumala?",
    "Tell me about laddu prasadam",
    "How to book darshan online?",
    "When is Brahmotsavam?",
    "What are hotel prices in Tirumala?",
    "How to travel from Chennai to Tirupati?",
  ],
  hi: [
    "दर्शन के प्रकार क्या हैं?",
    "ड्रेस कोड क्या है?",
    "तिरुमला कैसे पहुंचें?",
    "लड्डू प्रसादम के बारे में बताएं",
    "ऑनलाइन बुकिंग कैसे करें?",
    "ब्रह्मोत्सवम कब होता है?",
    "तिरुमला में होटल की कीमतें?",
    "चेन्नई से तिरुपति कैसे जाएं?",
  ],
};

/* Simple markdown-like rendering for bot responses */
function renderBotText(text) {
  if (!text) return null;
  const lines = text.split("\n");
  const elements = [];
  let listItems = [];

  const flushList = () => {
    if (listItems.length > 0) {
      elements.push(<ul key={`ul-${elements.length}`}>{listItems}</ul>);
      listItems = [];
    }
  };

  lines.forEach((line, i) => {
    const trimmed = line.trim();
    if (!trimmed) { flushList(); return; }
    if (trimmed.startsWith("### ")) {
      flushList();
      elements.push(<h4 key={i} className="bot-heading">{trimmed.slice(4)}</h4>);
    } else if (trimmed.startsWith("## ")) {
      flushList();
      elements.push(<h3 key={i} className="bot-heading">{trimmed.slice(3)}</h3>);
    } else if (trimmed.startsWith("# ")) {
      flushList();
      elements.push(<h3 key={i} className="bot-heading">{trimmed.slice(2)}</h3>);
    } else if (/^[-*•]\s/.test(trimmed)) {
      const content = trimmed.replace(/^[-*•]\s/, "");
      const boldParsed = content.split(/\*\*(.*?)\*\*/g).map((part, j) =>
        j % 2 === 1 ? <strong key={j}>{part}</strong> : part
      );
      listItems.push(<li key={i}>{boldParsed}</li>);
    } else if (/^\d+[.)]\s/.test(trimmed)) {
      const content = trimmed.replace(/^\d+[.)]\s/, "");
      const boldParsed = content.split(/\*\*(.*?)\*\*/g).map((part, j) =>
        j % 2 === 1 ? <strong key={j}>{part}</strong> : part
      );
      listItems.push(<li key={i}>{boldParsed}</li>);
    } else {
      flushList();
      const boldParsed = trimmed.split(/\*\*(.*?)\*\*/g).map((part, j) =>
        j % 2 === 1 ? <strong key={j}>{part}</strong> : part
      );
      elements.push(<p key={i}>{boldParsed}</p>);
    }
  });
  flushList();
  return elements.length > 0 ? elements : <p>{text}</p>;
}

export default function Chatbot() {
  const { t, lang } = useLang();
  const [messages, setMessages] = useState([
    {
      role: "bot",
      text: t.chatWelcome || "🙏 Om Namo Venkatesaya! Welcome to the TTD AI Chatbot. I'm powered by AI and can help with darshan, sevas, accommodation, travel, and trip planning!",
      source: "system",
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
    setMessages([{
      role: "bot",
      text: t.chatWelcome || "🙏 Om Namo Venkatesaya! Welcome to the TTD AI Chatbot.",
      source: "system",
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
      const source = res.data?.source || "unknown";
      setMessages((prev) => [...prev, { role: "bot", text: reply, source }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: t.chatError || "🙏 Sorry, I couldn't connect to the server.", source: "error" },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  }

  const quickQs = QUICK_QUESTIONS[lang] || QUICK_QUESTIONS.en;

  return (
    <section className="page chatbot-page">
      <div className="page-header">
        <GiTempleDoor className="page-header-icon" />
        <h2>{t.chatTitle || "TTD AI Chatbot"}</h2>
        <p className="page-subtitle">
          <MdAutoAwesome style={{ verticalAlign: "middle", marginRight: 4, color: "#DAA520" }} />
          {t.chatSubtitle || "AI-powered assistant for Tirumala Tirupati Devasthanams"}
        </p>
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
              <button key={i} className="quick-q-btn" onClick={() => handleSend(q)} disabled={loading}>
                {q}
              </button>
            ))}
          </div>
          <div className="sidebar-topics">
            <h4>{t.chatTopics || "I can help with"}</h4>
            <ul>
              <li>🛕 {t.chatTopicDarshan || "Darshan types & timings"}</li>
              <li>🙏 {t.chatTopicSevas || "Sevas & rituals"}</li>
              <li>🏨 {t.chatTopicAccommodation || "Accommodation & hotels"}</li>
              <li>🚌 {t.chatTopicTravel || "Travel & transport"}</li>
              <li>🍬 {t.chatTopicPrasadam || "Prasadam & Laddu"}</li>
              <li>👔 {t.chatTopicDressCode || "Dress code & rules"}</li>
              <li>🎉 {t.chatTopicFestivals || "Festivals & events"}</li>
              <li>💰 {t.chatTopicDonations || "Hundi & Donations"}</li>
              <li>🌐 {t.chatTopicOnline || "Online services"}</li>
              <li>🗺️ {t.chatTopicTrip || "Trip planning tips"}</li>
            </ul>
          </div>
          <div className="ai-badge">
            <MdAutoAwesome /> {t.chatAIBadge || "Powered by AI"}
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
                  <div className="bubble-text">
                    {m.role === "bot" ? renderBotText(m.text) : m.text}
                  </div>
                  {m.role === "bot" && (m.source === "rag" || m.source === "gemini") && (
                    <div className="ai-source-tag">
                      <MdAutoAwesome size={12} /> {m.source === "rag" ? "RAG + AI" : "AI"}
                    </div>
                  )}
                  {m.role === "bot" && m.source === "rag_direct" && (
                    <div className="ai-source-tag" style={{background: "var(--tirumala-maroon, #8B1A1A)"}}>
                      <MdAutoAwesome size={12} /> Vector Search
                    </div>
                  )}
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
              placeholder={t.chatPlaceholder || "Ask me anything about TTD..."}
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
