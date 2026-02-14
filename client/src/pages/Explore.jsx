import { useState, useMemo } from "react";
import { MdPlace, MdAccessTime, MdInfoOutline, MdFilterList, MdSearch } from "react-icons/md";
import { useLang } from "../i18n/LangContext";

/* ─── Place categories ─── */
const CATEGORIES = [
  { id: "all", label: "All Places", emoji: "🗺️" },
  { id: "temple", label: "Temples", emoji: "🛕" },
  { id: "nature", label: "Nature", emoji: "🌿" },
  { id: "waterfall", label: "Waterfalls", emoji: "💧" },
  { id: "historic", label: "Historic", emoji: "🏛️" },
  { id: "spiritual", label: "Spiritual", emoji: "🕉️" },
  { id: "wildlife", label: "Wildlife", emoji: "🦌" },
  { id: "viewpoint", label: "Viewpoints", emoji: "🏔️" },
];

/* ─── Famous places data with Wikimedia Commons / public domain photos ─── */
const PLACES = [
  {
    id: 1,
    name: "Sri Venkateswara Swamy Temple",
    nameTE: "శ్రీ వేంకటేశ్వర స్వామి దేవాలయం",
    category: "temple",
    rating: 5.0,
    distance: "Main Temple",
    timing: "3:00 AM – 12:00 AM",
    entryFee: "Free (₹500 Special Entry)",
    description: "The main temple of Lord Venkateswara atop the seven hills of Tirumala. One of the richest and most visited Hindu temples in the world, receiving 50,000–100,000 pilgrims daily. The presiding deity is Lord Venkateswara, a form of Vishnu, also known as Balaji or Govinda.",
    highlights: ["Ancient Dravidian architecture", "Golden Vimana (tower)", "Ananda Nilayam — the abode of bliss", "Brahmotsavam — 9-day annual grand festival"],
    tips: "Arrive early morning for shorter queues. Sarva Darshan is free. Special Entry Darshan (₹500) offers faster access.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Tirumala_090615.jpg/1280px-Tirumala_090615.jpg",
    mapUrl: "https://maps.google.com/?q=13.6833,79.3472",
  },
  {
    id: 2,
    name: "Sri Padmavathi Ammavari Temple",
    nameTE: "శ్రీ పద్మావతి అమ్మవారి దేవాలయం",
    category: "temple",
    rating: 4.8,
    distance: "5 km from Tirupati",
    timing: "8:00 AM – 8:00 PM",
    entryFee: "Free",
    description: "Located in Tiruchanoor, this temple is dedicated to Goddess Padmavathi, the consort of Lord Venkateswara. Devotees traditionally visit this temple before or after the Tirumala darshan. The temple features beautiful Dravidian architecture with intricate carvings.",
    highlights: ["Consort of Lord Venkateswara", "Ancient Dravidian temple", "Special pooja services available", "Beautiful temple tank (Padma Sarovaram)"],
    tips: "Visit before heading to Tirumala for complete pilgrimage. Temple is less crowded in the afternoon.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/Thiruchanoor_temple_1.JPG/1280px-Thiruchanoor_temple_1.JPG",
    mapUrl: "https://maps.google.com/?q=13.6167,79.4167",
  },
  {
    id: 3,
    name: "Sri Govindarajaswami Temple",
    nameTE: "శ్రీ గోవిందరాజస్వామి దేవాలయం",
    category: "temple",
    rating: 4.7,
    distance: "Tirupati Town Center",
    timing: "6:30 AM – 9:00 PM",
    entryFee: "Free",
    description: "One of the oldest and largest temples in Tirupati, dedicated to Lord Govindaraja (a form of Vishnu in reclining posture). Built in the 12th century, this temple is known for its magnificent Dravidian architecture with towering gopurams and elaborately carved pillars.",
    highlights: ["12th century ancient temple", "Lord Vishnu in reclining posture", "Magnificent gopuram gateway tower", "Annual Brahmotsavam festival"],
    tips: "Located in the heart of Tirupati. Combine with shopping at nearby markets.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Sri_Govindaraja_Swamy_temple_pond.jpg/1280px-Sri_Govindaraja_Swamy_temple_pond.jpg",
    mapUrl: "https://maps.google.com/?q=13.631,79.4194",
  },
  {
    id: 4,
    name: "Sri Kapileswara Swamy Temple",
    nameTE: "శ్రీ కపిలేశ్వర స్వామి దేవాలయం",
    category: "temple",
    rating: 4.5,
    distance: "3 km from Tirupati",
    timing: "6:00 AM – 8:00 PM",
    entryFee: "Free",
    description: "An ancient temple dedicated to Lord Shiva, situated at the foot of the Tirumala Hills near Kapila Theertham waterfall. The temple is built around a natural spring and cave, making it a unique blend of natural beauty and spiritual significance.",
    highlights: ["Lord Shiva temple with natural cave", "Adjacent to Kapila Theertham waterfall", "Scenic location at hill base", "Ancient rock-cut architecture"],
    tips: "Visit during monsoon season (July-September) to see the waterfall in full flow.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Kapila_theertham.jpg/1280px-Kapila_theertham.jpg",
    mapUrl: "https://maps.google.com/?q=13.6527,79.3889",
  },
  {
    id: 5,
    name: "Silathoranam (Natural Rock Arch)",
    nameTE: "శిలాతోరణం",
    category: "nature",
    rating: 4.6,
    distance: "1 km from Tirumala Temple",
    timing: "6:00 AM – 6:00 PM",
    entryFee: "Free",
    description: "A remarkable natural rock formation arch, believed to be over 1.5 billion years old. This geological wonder is shaped like a hood of a serpent and resembles the form of Lord Venkateswara's Sudarshana Chakra. It is a rare natural rock arch formation and a geological marvel.",
    highlights: ["Ancient natural rock arch formation", "Geological marvel of Tirumala Hills", "Resembles Lord's Sudarshana Chakra", "Short trek from Tirumala temple"],
    tips: "A short 15-minute walk from the main temple. Best viewed in morning light for photography.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Silathoranam.jpg/1280px-Silathoranam.jpg",
    mapUrl: "https://maps.google.com/?q=13.6792,79.3528",
  },
  {
    id: 6,
    name: "Sri Venkateswara National Park",
    nameTE: "శ్రీ వేంకటేశ్వర జాతీయ ఉద్యానవనం",
    category: "wildlife",
    rating: 4.5,
    distance: "Near Tirumala",
    timing: "6:00 AM – 5:30 PM",
    entryFee: "₹25 per person",
    description: "Spread over 353 sq km in the Eastern Ghats, this national park surrounds the Tirumala Hills. Home to diverse flora and fauna including slender loris, Indian pangolin, golden gecko, and over 178 bird species. The park features deciduous forests, waterfalls, and medicinal plants.",
    highlights: ["353 sq km biodiversity hotspot", "178+ bird species", "Rare golden gecko habitat", "Waterfalls & trekking trails"],
    tips: "Carry binoculars for birdwatching. Best visited October-March. Hire a guide at the entrance.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Sri_Venkateswara_National_Park%2CTirupati.jpg/1280px-Sri_Venkateswara_National_Park%2CTirupati.jpg",
    mapUrl: "https://maps.google.com/?q=13.65,79.35",
  },
  {
    id: 7,
    name: "Talakona Waterfall",
    nameTE: "తాళకోన జలపాతం",
    category: "waterfall",
    rating: 4.7,
    distance: "49 km from Tirupati",
    timing: "8:00 AM – 5:00 PM",
    entryFee: "₹50 per person",
    description: "The highest waterfall in Andhra Pradesh at 270 feet, nestled inside the Sri Venkateswara National Park. The water is believed to have medicinal properties. Surrounded by dense forest, it offers spectacular trekking opportunities through lush greenery.",
    highlights: ["Highest waterfall in AP (270 ft)", "Medicinal mineral water", "Dense forest trekking trail", "Rich biodiversity area"],
    tips: "Best visited July-December. Wear sturdy shoes for the 2 km trek. Carry drinking water.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/A_view_of_Talakona_water_falls.JPG/800px-A_view_of_Talakona_water_falls.JPG",
    mapUrl: "https://maps.google.com/?q=13.7208,79.2669",
  },
  {
    id: 8,
    name: "Papavinasanam Dam & Waterfalls",
    nameTE: "పాపవినాశనం",
    category: "waterfall",
    rating: 4.4,
    distance: "5 km from Tirumala Temple",
    timing: "7:00 AM – 6:00 PM",
    entryFee: "Free",
    description: "Sacred waterfalls near Tirumala where Lord Venkateswara is believed to have washed away the sins of devotees. The dam built on the Papavinasanam stream provides drinking water to Tirumala. Devotees take a holy dip here before proceeding for darshan.",
    highlights: ["Sacred waterfall for holy dip", "Believed to wash away sins", "Scenic dam and reservoir", "Peaceful picnic spot"],
    tips: "Take a dip before darshan for spiritual significance. Water flow is best during monsoon.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Papavinasam_Dam.JPG/1280px-Papavinasam_Dam.JPG",
    mapUrl: "https://maps.google.com/?q=13.6766,79.3266",
  },
  {
    id: 9,
    name: "Sri Venkateswara Museum",
    nameTE: "శ్రీ వేంకటేశ్వర మ్యూజియం",
    category: "historic",
    rating: 4.3,
    distance: "Near Tirumala Bus Stand",
    timing: "8:00 AM – 8:00 PM",
    entryFee: "₹10",
    description: "A comprehensive museum showcasing the rich history and heritage of the Tirumala temple and TTD. Houses ancient artifacts, sculptures, inscriptions, coins, arms, and paintings depicting the glorious past spanning over 2,000 years of the temple's history.",
    highlights: ["2,000+ years of temple history", "Ancient sculptures & inscriptions", "Historical coins & arms collection", "Paintings of temple evolution"],
    tips: "Allow 1-2 hours for a thorough visit. English descriptions available for all exhibits.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/9/96/Museum_Tirumala_02.jpg",
    mapUrl: "https://maps.google.com/?q=13.6833,79.3500",
  },
  {
    id: 10,
    name: "ISKCON Temple, Tirupati",
    nameTE: "ఇస్కాన్ దేవాలయం",
    category: "temple",
    rating: 4.5,
    distance: "12 km from Tirupati",
    timing: "4:30 AM – 8:30 PM",
    entryFee: "Free",
    description: "A beautiful modern temple dedicated to Lord Krishna, built by the International Society for Krishna Consciousness. Features stunning marble architecture, lush gardens, and houses the Sri Sri Radha Govinda deities. Known for its serene atmosphere and prasadam distribution.",
    highlights: ["Stunning white marble architecture", "Beautiful temple gardens", "Free prasadam distribution", "Regular kirtans & cultural programs"],
    tips: "Visit during evening aarti for the best experience. Free prasadam served at lunch time.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/ISKCON_Temple_Tirupati.jpg/1280px-ISKCON_Temple_Tirupati.jpg",
    mapUrl: "https://maps.google.com/?q=13.5978,79.4397",
  },
  {
    id: 11,
    name: "Akasa Ganga Waterfalls",
    nameTE: "ఆకాశగంగ జలపాతం",
    category: "waterfall",
    rating: 4.3,
    distance: "3 km from Tirumala Temple",
    timing: "6:00 AM – 6:00 PM",
    entryFee: "Free",
    description: "A sacred waterfall in the Tirumala Hills, believed to be a tributary of the celestial river Ganga that flows from heaven. The water from this falls is used for the daily abhishekam (sacred bath) of Lord Venkateswara at the main temple.",
    highlights: ["Sacred water used for Lord's abhishekam", "Celestial origin mythology", "Scenic hill stream", "Peaceful meditation spot"],
    tips: "Short walk from Tirumala. Best during and after monsoon. Water may dry up in summer.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Akasa_Ganga_falls.JPG/800px-Akasa_Ganga_falls.JPG",
    mapUrl: "https://maps.google.com/?q=13.6853,79.3397",
  },
  {
    id: 12,
    name: "Deer Park (Sri Venkateswara Zoological Park)",
    nameTE: "జింకల పార్క్ (జంతు ప్రదర్శనశాల)",
    category: "wildlife",
    rating: 4.2,
    distance: "2 km from Tirumala Temple",
    timing: "8:00 AM – 5:00 PM (Closed Mondays)",
    entryFee: "₹20 adults / ₹10 children",
    description: "A zoological park and deer sanctuary in the lush Tirumala Hills. Home to spotted deer, peacocks, rabbits, and other animals in a natural habitat setting. The park offers a refreshing break from temple darshan with walking trails through dense forest.",
    highlights: ["Spotted deer in natural habitat", "Peacocks & native birds", "Shady forest walking trails", "Children's play area"],
    tips: "Closed on Mondays. Best visited in the morning when animals are most active.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Tirumala_Hills_11.jpg/1280px-Tirumala_Hills_11.jpg",
    mapUrl: "https://maps.google.com/?q=13.6889,79.3528",
  },
  {
    id: 13,
    name: "Chandragiri Fort",
    nameTE: "చంద్రగిరి కోట",
    category: "historic",
    rating: 4.4,
    distance: "15 km from Tirupati",
    timing: "10:00 AM – 5:00 PM (Closed Fridays)",
    entryFee: "₹15 Indians / ₹200 Foreigners",
    description: "A magnificent hilltop fort with origins dating to the 11th century, later serving as the capital of the Vijayanagara Empire in the 16th–17th century. The fort complex includes the Raja Mahal and Rani Mahal palaces, which now house an archaeological museum. Light and sound shows narrate the fort's glorious history.",
    highlights: ["Historic Vijayanagara-era capital", "Raja Mahal & Rani Mahal palaces", "Archaeological museum inside", "Sound & light show in evenings"],
    tips: "Plan 2-3 hours. Evening sound & light show is a must-see. Closed on Fridays.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Chandragiri_fort.jpg/1280px-Chandragiri_fort.jpg",
    mapUrl: "https://maps.google.com/?q=13.5833,79.3167",
  },
  {
    id: 14,
    name: "Srivari Mettu (Ancient Trekking Path)",
    nameTE: "శ్రీవారి మెట్టు",
    category: "nature",
    rating: 4.6,
    distance: "Alipiri to Tirumala",
    timing: "3:00 AM – 9:00 PM",
    entryFee: "Free",
    description: "The ancient sacred trekking path with 3,550 steps from Alipiri at the foothills to Tirumala temple at the top. This is the traditional route taken by devotees for centuries. The trek covers approximately 12 km and takes 3-4 hours, offering stunning views of the Eastern Ghats.",
    highlights: ["3,550 steps — ancient sacred path", "12 km trek through seven hills", "Stunning panoramic views", "Centuries-old pilgrim tradition"],
    tips: "Start early morning (4-5 AM) to avoid heat. Free buttermilk stalls along the way. Wear comfortable shoes.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/Alipiri_Steps_Tirumala.jpg/800px-Alipiri_Steps_Tirumala.jpg",
    mapUrl: "https://maps.google.com/?q=13.6530,79.3810",
  },
  {
    id: 15,
    name: "Japali Tirtham",
    nameTE: "జపాలి తీర్థం",
    category: "spiritual",
    rating: 4.3,
    distance: "4 km from Tirumala Temple",
    timing: "6:00 AM – 6:00 PM",
    entryFee: "Free",
    description: "A sacred cave and water spring where Sage Japali is believed to have performed intense penance. The cave houses ancient idols of Lord Shiva and Vishnu. Surrounded by dense forest, this serene spot offers a deeply spiritual and meditative atmosphere.",
    highlights: ["Ancient sage's meditation cave", "Sacred spring water", "Shiva & Vishnu idols in cave", "Dense forest meditation spot"],
    tips: "Requires a short trek. Carry a torch for the inner cave. Best combined with Papavinasanam visit.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Japali_Teertham_Tirumala_02.jpg/800px-Japali_Teertham_Tirumala_02.jpg",
    mapUrl: "https://maps.google.com/?q=13.670,79.330",
  },
  {
    id: 16,
    name: "Srivari Padalu (Lord's Footprints)",
    nameTE: "శ్రీవారి పాదాలు",
    category: "spiritual",
    rating: 4.4,
    distance: "2 km from Tirumala Temple",
    timing: "6:00 AM – 7:00 PM",
    entryFee: "Free",
    description: "A sacred spot on the Tirumala Hills where the footprints of Lord Venkateswara are imprinted on a rock. Devotees believe these are the divine footprints left by the Lord when he first set foot on the seven hills. The location also offers a panoramic viewpoint.",
    highlights: ["Sacred divine footprints on rock", "Panoramic hilltop viewpoint", "Spiritual significance", "Photography-worthy sunset views"],
    tips: "Visit during sunset for breathtaking views. A peaceful spot away from the main temple crowd.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/TVMalai_033.jpg/800px-TVMalai_033.jpg",
    mapUrl: "https://maps.google.com/?q=13.680,79.348",
  },
  {
    id: 17,
    name: "Sri Kalyana Venkateswara Swamy Temple, Srinivasa Mangapuram",
    nameTE: "శ్రీ కల్యాణ వేంకటేశ్వర స్వామి దేవాలయం, శ్రీనివాస మంగాపురం",
    category: "temple",
    rating: 4.5,
    distance: "12 km from Tirupati",
    timing: "6:00 AM – 8:30 PM",
    entryFee: "Free",
    description: "This ancient temple marks the spot where Lord Venkateswara married Goddess Padmavathi. The temple depicts the celestial wedding scene and is an essential pilgrimage stop. It is often called 'Kalyana Tirumala' and is administered by TTD.",
    highlights: ["Site of Lord's celestial wedding", "TTD-managed ancient temple", "Less crowded than Tirumala", "Beautiful Kalyana Mandapam"],
    tips: "Ideal to visit on the way to or from Tirumala. Combined ticket with other TTD temples available.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/Thiruchanoor_temple_1.JPG/1280px-Thiruchanoor_temple_1.JPG",
    mapUrl: "https://maps.google.com/?q=13.6285,79.3806",
  },
  {
    id: 18,
    name: "Srikalahasti Temple",
    nameTE: "శ్రీకాళహస్తి దేవాలయం",
    category: "temple",
    rating: 4.8,
    distance: "38 km from Tirupati",
    timing: "6:00 AM – 9:00 PM",
    entryFee: "Free (₹100 Special Darshan)",
    description: "One of the most important Shiva temples in South India, known for its Vayu Linga (wind god). Located on the banks of the Swarnamukhi River at the foot of a sacred hill, it is one of the Pancha Bhuta Kshetras representing 'Vayu' (wind element). Famous for Rahu-Ketu pooja.",
    highlights: ["Pancha Bhuta Kshetra (Vayu)", "Famous Rahu-Ketu Sarpa Dosha pooja", "Ancient Chola-era architecture", "Swarnamukhi riverfront setting"],
    tips: "Famous for Rahu-Ketu pooja. Visit early morning to avoid crowds. Combine with Tirupati trip.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Srikalahasti_temple_Gopurams.jpg/1024px-Srikalahasti_temple_Gopurams.jpg",
    mapUrl: "https://maps.google.com/?q=13.7497,79.6978",
  },
  {
    id: 19,
    name: "Kanipakam Vinayaka Temple",
    nameTE: "కాణిపాకం వినాయక దేవాలయం",
    category: "temple",
    rating: 4.6,
    distance: "75 km from Tirupati",
    timing: "5:00 AM – 9:00 PM",
    entryFee: "Free",
    description: "A famous Ganesh temple where the self-manifested idol of Lord Vinayaka is believed to be growing in size. The swayambhu (self-manifested) idol was discovered in a water well. The temple is known as Varasiddhi Vinayaka and attracts devotees seeking fulfillment of wishes.",
    highlights: ["Self-growing Vinayaka idol", "Swayambhu (self-manifested) deity", "Sacred temple tank", "Grand Brahmotsavam celebrations"],
    tips: "Special pooja on Tuesdays and Chaturthi days. The growing idol phenomenon is fascinating.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Kanipakam_Temple_Tower.jpg/800px-Kanipakam_Temple_Tower.jpg",
    mapUrl: "https://maps.google.com/?q=13.3217,79.1267",
  },
  {
    id: 20,
    name: "TTD Gardens & Chakra Teertham",
    nameTE: "TTD ఉద్యానవనాలు & చక్ర తీర్థం",
    category: "nature",
    rating: 4.1,
    distance: "1.5 km from Tirumala Temple",
    timing: "7:00 AM – 7:00 PM",
    entryFee: "Free",
    description: "Beautifully maintained botanical gardens by TTD near the main temple complex featuring manicured lawns, flowering plants, medicinal herb gardens, and walking paths. Chakra Teertham is a sacred water body nearby, believed to be where Lord Vishnu's Sudarshana Chakra struck the earth.",
    highlights: ["Manicured TTD botanical gardens", "Sacred Chakra Teertham", "Medicinal herb garden", "Children's play area"],
    tips: "Pleasant morning or evening walk. Great to relax before or after darshan. Carry water.",
    photo: "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Tirumala_Hills_11.jpg/1280px-Tirumala_Hills_11.jpg",
    mapUrl: "https://maps.google.com/?q=13.684,79.347",
  },
];

export default function Explore() {
  const { t } = useLang();
  const [activeCategory, setActiveCategory] = useState("all");
  const [search, setSearch] = useState("");
  const [expandedId, setExpandedId] = useState(null);

  const filtered = useMemo(() => {
    return PLACES.filter((p) => {
      const matchesCat = activeCategory === "all" || p.category === activeCategory;
      const matchesSearch =
        !search ||
        p.name.toLowerCase().includes(search.toLowerCase()) ||
        p.description.toLowerCase().includes(search.toLowerCase()) ||
        (p.nameTE && p.nameTE.includes(search));
      return matchesCat && matchesSearch;
    });
  }, [activeCategory, search]);

  return (
    <div className="explore-page fade-in">
      {/* Header */}
      <div className="page-header">
        <div className="page-header-icon">🗺️</div>
        <h2>{t.exploreTitle || "Explore Tirumala & Tirupati"}</h2>
        <p className="page-subtitle">
          {t.exploreSub || "Discover sacred temples, stunning waterfalls, historic forts, and natural wonders around the seven hills"}
        </p>
      </div>

      {/* Search & Filter */}
      <div className="explore-controls card">
        <div className="explore-search">
          <MdSearch className="explore-search-icon" />
          <input
            type="text"
            placeholder={t.exploreSearch || "Search places..."}
            className="explore-search-input"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="explore-filters">
          <MdFilterList className="filter-icon-label" />
          {CATEGORIES.map((cat) => (
            <button
              key={cat.id}
              className={`explore-filter-btn ${activeCategory === cat.id ? "active" : ""}`}
              onClick={() => setActiveCategory(cat.id)}
            >
              <span>{cat.emoji}</span>
              <span>{cat.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Results count */}
      <div className="explore-count">
        Showing <strong>{filtered.length}</strong> of {PLACES.length} places
        {activeCategory !== "all" && (
          <span> in <em>{CATEGORIES.find((c) => c.id === activeCategory)?.label}</em></span>
        )}
      </div>

      {/* Places Grid */}
      <div className="explore-grid">
        {filtered.map((place) => {
          const isExpanded = expandedId === place.id;
          return (
            <article key={place.id} className={`explore-card ${isExpanded ? "expanded" : ""}`}>
              {/* Photo */}
              <div className="explore-card-img-wrap">
                <img
                  src={place.photo}
                  alt={place.name}
                  className="explore-card-img"
                  loading="lazy"
                  decoding="async"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 250'%3E%3Crect fill='%23f5f0e1' width='400' height='250'/%3E%3Ctext x='200' y='130' text-anchor='middle' fill='%238b1a1a' font-size='40'%3E🛕%3C/text%3E%3C/svg%3E";
                  }}
                />
                <span className="explore-card-category">
                  {CATEGORIES.find((c) => c.id === place.category)?.emoji}{" "}
                  {CATEGORIES.find((c) => c.id === place.category)?.label}
                </span>
                <span className="explore-card-rating">⭐ {place.rating}</span>
              </div>

              {/* Info */}
              <div className="explore-card-body">
                <h3 className="explore-card-title">{place.name}</h3>
                {place.nameTE && <div className="explore-card-te">{place.nameTE}</div>}

                <div className="explore-card-meta">
                  <span><MdPlace /> {place.distance}</span>
                  <span><MdAccessTime /> {place.timing}</span>
                  {place.entryFee && <span>🎟️ {place.entryFee}</span>}
                </div>

                <p className="explore-card-desc">
                  {isExpanded ? place.description : place.description.slice(0, 140) + "..."}
                </p>

                {isExpanded && (
                  <div className="explore-card-details fade-in">
                    {/* Highlights */}
                    <div className="explore-highlights">
                      <h4>✨ Highlights</h4>
                      <ul>
                        {place.highlights.map((h, i) => (
                          <li key={i}>{h}</li>
                        ))}
                      </ul>
                    </div>

                    {/* Tips */}
                    <div className="explore-tips">
                      <MdInfoOutline />
                      <span><strong>Tip:</strong> {place.tips}</span>
                    </div>

                    {/* Map link */}
                    <a
                      href={place.mapUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="explore-map-link"
                    >
                      <MdPlace /> Open in Google Maps
                    </a>
                  </div>
                )}

                <button
                  className="explore-card-toggle"
                  onClick={() => setExpandedId(isExpanded ? null : place.id)}
                >
                  {isExpanded ? "Show Less ▲" : "View Details ▼"}
                </button>
              </div>
            </article>
          );
        })}
      </div>

      {filtered.length === 0 && (
        <div className="explore-empty">
          <span style={{ fontSize: "2.5rem" }}>🔍</span>
          <p>No places found matching your search. Try a different keyword or category.</p>
        </div>
      )}
    </div>
  );
}
