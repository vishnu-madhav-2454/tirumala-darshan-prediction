export default function Loader({ text = "Loading..." }) {
  return (
    <div className="loader-container">
      <div className="loader-spinner" />
      <div className="loader-text">{text}</div>
      <div style={{ fontSize: "0.7rem", color: "var(--gold-dark)", marginTop: 4, opacity: 0.7, fontStyle: "italic" }}>
        🙏 Om Namo Venkatesaya
      </div>
    </div>
  );
}
