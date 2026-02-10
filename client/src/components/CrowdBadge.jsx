/**
 * CrowdBadge — colored pill showing crowd level
 */
export default function CrowdBadge({ level }) {
  const cls = level.toLowerCase().replace(/\s+/g, "-");
  return (
    <span className={`crowd-badge ${cls}`}>
      <span className={`crowd-dot ${cls}`} />
      {level}
    </span>
  );
}
