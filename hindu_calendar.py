"""
Hindu Calendar & Festival Database for Tirumala Darshan Prediction.
Provides festival dates, lunar events, school holidays, and crowd-impact
annotations for calendar display and anomaly explanation.
"""

from datetime import date, timedelta

# ═══════════════════════════════════════════════════════════════
#  Telugu month names (overlapping with Gregorian months)
# ═══════════════════════════════════════════════════════════════
HINDU_MONTH_MAP = {
    1:  ("పుష్య – మాఘ",       "Pushya – Magha"),
    2:  ("మాఘ – ఫాల్గుణ",      "Magha – Phalguna"),
    3:  ("ఫాల్గుణ – చైత్ర",     "Phalguna – Chaitra"),
    4:  ("చైత్ర – వైశాఖ",       "Chaitra – Vaishakha"),
    5:  ("వైశాఖ – జ్యేష్ఠ",      "Vaishakha – Jyeshtha"),
    6:  ("జ్యేష్ఠ – ఆషాఢ",       "Jyeshtha – Ashadha"),
    7:  ("ఆషాఢ – శ్రావణ",       "Ashadha – Shravana"),
    8:  ("శ్రావణ – భాద్రపద",     "Shravana – Bhadrapada"),
    9:  ("భాద్రపద – ఆశ్వీజ",     "Bhadrapada – Ashvija"),
    10: ("ఆశ్వీజ – కార్తీక",      "Ashvija – Karthika"),
    11: ("కార్తీక – మార్గశిర",     "Karthika – Margashira"),
    12: ("మార్గశిర – పుష్య",      "Margashira – Pushya"),
}

# Impact levels → typical extra pilgrim multiplier
IMPACT = {
    "extreme": {"label": "🔴 Extreme",  "factor": 2.0},
    "very_high": {"label": "🟠 Very High", "factor": 1.7},
    "high":     {"label": "🟡 High",     "factor": 1.4},
    "moderate": {"label": "🟢 Moderate",  "factor": 1.2},
    "low":      {"label": "⚪ Low",      "factor": 1.05},
}

# ═══════════════════════════════════════════════════════════════
#  FESTIVALS & SPECIAL DAYS  (2025 – 2027)
#  Format: (month, day, name, name_te, type, impact)
# ═══════════════════════════════════════════════════════════════
FESTIVALS = {
    # ── 2025 ───────────────────────────────────────────────
    2025: [
        (1, 1,  "New Year's Day", "నూతన సంవత్సరం", "holiday", "very_high"),
        (1, 13, "Bhogi", "భోగి", "festival", "high"),
        (1, 14, "Sankranti / Pongal", "సంక్రాంతి", "festival", "extreme"),
        (1, 15, "Kanuma", "కనుమ", "festival", "very_high"),
        (1, 26, "Republic Day", "గణతంత్ర దినోత్సవం", "holiday", "high"),
        (2, 5,  "Rathasapthami", "రథసప్తమి", "festival", "extreme"),
        (2, 26, "Maha Shivaratri", "మహా శివరాత్రి", "festival", "very_high"),
        (3, 14, "Holi", "హోలీ", "festival", "high"),
        (3, 30, "Ugadi (Telugu New Year)", "ఉగాది", "festival", "extreme"),
        (4, 6,  "Sri Rama Navami", "శ్రీరామ నవమి", "festival", "very_high"),
        (4, 14, "Ambedkar Jayanti", "అంబేద్కర్ జయంతి", "holiday", "moderate"),
        (5, 1,  "May Day", "మే దినం", "holiday", "moderate"),
        (5, 12, "Vaishakha Purnima / Buddha Purnima", "వైశాఖ పూర్ణిమ", "festival", "high"),
        (6, 27, "Rath Yatra", "రథయాత్ర", "festival", "high"),
        (8, 15, "Independence Day", "స్వాతంత్ర్య దినోత్సవం", "holiday", "very_high"),
        (8, 16, "Krishna Janmashtami", "కృష్ణాష్టమి", "festival", "extreme"),
        (8, 27, "Vinayaka Chaturthi", "వినాయక చవితి", "festival", "very_high"),
        (9, 22, "Navaratri Begins", "నవరాత్రి ప్రారంభం", "festival", "very_high"),
        (10, 1, "Dussehra / Vijayadashami", "దసరా / విజయదశమి", "festival", "extreme"),
        (10, 2, "Gandhi Jayanti", "గాంధీ జయంతి", "holiday", "high"),
        (10, 3, "Annual Brahmotsavams Begin", "వార్షిక బ్రహ్మోత్సవాలు ప్రారంభం", "brahmotsavam", "extreme"),
        (10, 4, "Brahmotsavams Day 2", "బ్రహ్మోత్సవం 2వ రోజు", "brahmotsavam", "extreme"),
        (10, 5, "Brahmotsavams Day 3", "బ్రహ్మోత్సవం 3వ రోజు", "brahmotsavam", "extreme"),
        (10, 6, "Brahmotsavams Day 4", "బ్రహ్మోత్సవం 4వ రోజు", "brahmotsavam", "extreme"),
        (10, 7, "Brahmotsavams Day 5 – Garuda Seva", "గరుడ సేవ", "brahmotsavam", "extreme"),
        (10, 8, "Brahmotsavams Day 6", "బ్రహ్మోత్సవం 6వ రోజు", "brahmotsavam", "extreme"),
        (10, 9, "Brahmotsavams Day 7", "బ్రహ్మోత్సవం 7వ రోజు", "brahmotsavam", "extreme"),
        (10, 10,"Brahmotsavams Day 8", "బ్రహ్మోత్సవం 8వ రోజు", "brahmotsavam", "extreme"),
        (10, 11,"Brahmotsavams Day 9 – Chakra Snanam", "చక్ర స్నానం", "brahmotsavam", "extreme"),
        (10, 20,"Diwali / Deepavali", "దీపావళి", "festival", "extreme"),
        (11, 5, "Karthika Purnima / Laksha Deepotsavam", "కార్తీక పూర్ణిమ / లక్ష దీపోత్సవం", "festival", "extreme"),
        (12, 22,"Vaikuntha Ekadashi", "వైకుంఠ ఏకాదశి", "festival", "extreme"),
        (12, 25,"Christmas", "క్రిస్మస్", "holiday", "high"),
        (12, 31,"New Year's Eve", "సంవత్సరాంతం", "holiday", "very_high"),
    ],
    # ── 2026 ───────────────────────────────────────────────
    2026: [
        (1, 1,  "New Year's Day", "నూతన సంవత్సరం", "holiday", "very_high"),
        (1, 13, "Bhogi", "భోగి", "festival", "high"),
        (1, 14, "Sankranti / Pongal", "సంక్రాంతి", "festival", "extreme"),
        (1, 15, "Kanuma", "కనుమ", "festival", "very_high"),
        (1, 26, "Republic Day", "గణతంత్ర దినోత్సవం", "holiday", "high"),
        (1, 27, "Rathasapthami", "రథసప్తమి", "festival", "extreme"),
        (2, 15, "Maha Shivaratri", "మహా శివరాత్రి", "festival", "very_high"),
        (3, 3,  "Holi", "హోలీ", "festival", "high"),
        (3, 19, "Ugadi (Telugu New Year)", "ఉగాది", "festival", "extreme"),
        (3, 28, "Sri Rama Navami", "శ్రీరామ నవమి", "festival", "very_high"),
        (4, 14, "Ambedkar Jayanti", "అంబేద్కర్ జయంతి", "holiday", "moderate"),
        (5, 1,  "May Day", "మే దినం", "holiday", "moderate"),
        (5, 5,  "Akshaya Tritiya", "అక్షయ తృతీయ", "festival", "high"),
        (5, 31, "Vaishakha Purnima / Buddha Purnima", "వైశాఖ పూర్ణిమ", "festival", "high"),
        (7, 16, "Rath Yatra", "రథయాత్ర", "festival", "high"),
        (8, 15, "Independence Day / Janmashtami", "స్వాతంత్ర్య దినోత్సవం", "holiday", "extreme"),
        (9, 5,  "Vinayaka Chaturthi", "వినాయక చవితి", "festival", "very_high"),
        (9, 21, "Navaratri Begins", "నవరాత్రి ప్రారంభం", "festival", "very_high"),
        (9, 22, "Navaratri Day 2", "నవరాత్రి 2వ రోజు", "festival", "very_high"),
        (9, 23, "Navaratri Day 3", "నవరాత్రి 3వ రోజు", "festival", "very_high"),
        (9, 24, "Navaratri Day 4", "నవరాత్రి 4వ రోజు", "festival", "high"),
        (9, 25, "Navaratri Day 5", "నవరాత్రి 5వ రోజు", "festival", "high"),
        (9, 26, "Navaratri Day 6", "నవరాత్రి 6వ రోజు", "festival", "high"),
        (9, 27, "Navaratri Day 7", "నవరాత్రి 7వ రోజు", "festival", "very_high"),
        (9, 28, "Navaratri Day 8 – Durga Ashtami", "దుర్గాష్టమి", "festival", "very_high"),
        (9, 29, "Navaratri Day 9 – Mahanavami", "మహానవమి", "festival", "very_high"),
        (10, 1, "Dussehra / Vijayadashami", "దసరా / విజయదశమి", "festival", "extreme"),
        (10, 2, "Gandhi Jayanti", "గాంధీ జయంతి", "holiday", "high"),
        (10, 5, "Annual Brahmotsavams Begin", "వార్షిక బ్రహ్మోత్సవాలు ప్రారంభం", "brahmotsavam", "extreme"),
        (10, 6, "Brahmotsavams Day 2 – Pedda Sesha Vahanam", "పెద్ద శేష వాహనం", "brahmotsavam", "extreme"),
        (10, 7, "Brahmotsavams Day 3 – Simha Vahanam", "సింహ వాహనం", "brahmotsavam", "extreme"),
        (10, 8, "Brahmotsavams Day 4 – Mutyapu Pandiri", "ముత్యాల పందిరి", "brahmotsavam", "extreme"),
        (10, 9, "Brahmotsavams Day 5 – Garuda Seva ✦", "గరుడ సేవ ✦", "brahmotsavam", "extreme"),
        (10, 10,"Brahmotsavams Day 6 – Hanumantha Vahanam", "హనుమంత వాహనం", "brahmotsavam", "extreme"),
        (10, 11,"Brahmotsavams Day 7 – Gaja Vahanam", "గజ వాహనం", "brahmotsavam", "extreme"),
        (10, 12,"Brahmotsavams Day 8 – Surya Prabha", "సూర్యప్రభ వాహనం", "brahmotsavam", "extreme"),
        (10, 13,"Brahmotsavams Day 9 – Chakra Snanam", "చక్ర స్నానం", "brahmotsavam", "extreme"),
        (10, 20,"Diwali / Deepavali", "దీపావళి", "festival", "extreme"),
        (11, 2, "Tulasi Vivah", "తులసీ వివాహం", "festival", "moderate"),
        (11, 14,"Karthika Purnima / Laksha Deepotsavam", "కార్తీక పూర్ణిమ / లక్ష దీపోత్సవం", "festival", "extreme"),
        (12, 11,"Vaikuntha Ekadashi", "వైకుంఠ ఏకాదశి", "festival", "extreme"),
        (12, 25,"Christmas", "క్రిస్మస్", "holiday", "high"),
        (12, 31,"New Year's Eve", "సంవత్సరాంతం", "holiday", "very_high"),
    ],
    # ── 2027 ───────────────────────────────────────────────
    2027: [
        (1, 1,  "New Year's Day", "నూతన సంవత్సరం", "holiday", "very_high"),
        (1, 13, "Bhogi", "భోగి", "festival", "high"),
        (1, 14, "Sankranti / Pongal", "సంక్రాంతి", "festival", "extreme"),
        (1, 15, "Kanuma", "కనుమ", "festival", "very_high"),
        (1, 26, "Republic Day", "గణతంత్ర దినోత్సవం", "holiday", "high"),
        (2, 16, "Rathasapthami", "రథసప్తమి", "festival", "extreme"),
        (3, 7,  "Maha Shivaratri", "మహా శివరాత్రి", "festival", "very_high"),
        (3, 22, "Holi", "హోలీ", "festival", "high"),
        (4, 8,  "Ugadi (Telugu New Year)", "ఉగాది", "festival", "extreme"),
        (4, 16, "Sri Rama Navami", "శ్రీరామ నవమి", "festival", "very_high"),
        (4, 14, "Ambedkar Jayanti", "అంబేద్కర్ జయంతి", "holiday", "moderate"),
        (5, 1,  "May Day", "మే దినం", "holiday", "moderate"),
        (5, 24, "Akshaya Tritiya", "అక్షయ తృతీయ", "festival", "high"),
        (8, 15, "Independence Day", "స్వాతంత్ర్య దినోత్సవం", "holiday", "very_high"),
        (9, 4,  "Krishna Janmashtami", "కృష్ణాష్టమి", "festival", "extreme"),
        (9, 24, "Vinayaka Chaturthi", "వినాయక చవితి", "festival", "very_high"),
        (10, 2, "Gandhi Jayanti", "గాంధీ జయంతి", "holiday", "high"),
        (10, 11,"Navaratri Begins", "నవరాత్రి ప్రారంభం", "festival", "very_high"),
        (10, 20,"Dussehra / Vijayadashami", "దసరా / విజయదశమి", "festival", "extreme"),
        (11, 8, "Diwali / Deepavali", "దీపావళి", "festival", "extreme"),
        (11, 4, "Karthika Purnima / Laksha Deepotsavam", "కార్తీక పూర్ణిమ / లక్ష దీపోత్సవం", "festival", "extreme"),
        (12, 25,"Christmas", "క్రిస్మస్", "holiday", "high"),
        (12, 30,"Vaikuntha Ekadashi", "వైకుంఠ ఏకాదశి", "festival", "extreme"),
        (12, 31,"New Year's Eve", "సంవత్సరాంతం", "holiday", "very_high"),
    ],
}

# ═══════════════════════════════════════════════════════════════
#  LUNAR EVENTS (Purnima, Amavasya, Ekadashi) — 2025‑2027
# ═══════════════════════════════════════════════════════════════
PURNIMA = {
    2025: [(1,13),(2,12),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9),(9,7),(10,7),(11,5),(12,4)],
    2026: [(1,3),(2,1),(3,3),(4,2),(5,1),(5,31),(6,29),(7,29),(8,28),(9,26),(10,26),(11,24),(12,24)],
    2027: [(1,22),(2,20),(3,22),(4,20),(5,20),(6,18),(7,18),(8,17),(9,15),(10,15),(11,13),(12,13)],
}

AMAVASYA = {
    2025: [(1,29),(2,28),(3,29),(4,27),(5,27),(6,25),(7,24),(8,23),(9,21),(10,21),(11,20),(12,20)],
    2026: [(1,18),(2,17),(3,19),(4,17),(5,17),(6,15),(7,14),(8,13),(9,11),(10,11),(11,9),(12,9)],
    2027: [(1,7),(2,6),(3,8),(4,6),(5,6),(6,4),(7,4),(8,2),(9,1),(10,1),(10,31),(11,29),(12,29)],
}

# Shukla Ekadashi (11th day from new moon → ~new moon + 11 days)
# Krishna Ekadashi (11th day from full moon → ~full moon + 11 days)
EKADASHI = {
    2025: [
        (1,10),(1,25),(2,8),(2,23),(3,10),(3,24),(4,9),(4,22),
        (5,9),(5,22),(6,7),(6,21),(7,6),(7,20),(8,5),(8,19),
        (9,3),(9,17),(10,3),(10,17),(11,1),(11,16),(12,1),(12,16),
    ],
    2026: [
        (1,14),(1,29),(2,12),(2,28),(3,14),(3,30),(4,13),(4,28),
        (5,12),(5,28),(6,10),(6,26),(7,10),(7,25),(8,9),(8,24),
        (9,7),(9,22),(10,7),(10,22),(11,5),(11,20),(12,5),(12,20),
    ],
    2027: [
        (1,3),(1,18),(2,2),(2,17),(3,4),(3,19),(4,2),(4,17),
        (5,2),(5,17),(6,1),(6,15),(6,30),(7,15),(7,29),(8,13),
        (8,28),(9,12),(9,26),(10,12),(10,26),(11,10),(11,24),(12,10),(12,24),
    ],
}

# ═══════════════════════════════════════════════════════════════
#  SCHOOL / EXAM HOLIDAY SEASONS (approximate, recurring pattern)
# ═══════════════════════════════════════════════════════════════
SEASONAL_PERIODS = [
    # (start_month, start_day, end_month, end_day, name, name_te, impact)
    (4, 15, 6, 10, "Summer Holidays", "వేసవి సెలవులు", "high"),
    (10, 1, 10, 15, "Dasara Holidays", "దసరా సెలవులు", "very_high"),
    (12, 20, 1, 5, "Winter / Christmas Holidays", "శీతాకాల సెలవులు", "very_high"),
]


# ═══════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════

def get_hindu_month_info(gregorian_month: int) -> dict:
    """Return Telugu and English Hindu month name for a Gregorian month."""
    te, en = HINDU_MONTH_MAP.get(gregorian_month, ("", ""))
    return {"telugu": te, "english": en}


def _is_in_seasonal_period(year: int, month: int, day: int) -> list:
    """Check if a date falls in a school/seasonal holiday period."""
    events = []
    d = date(year, month, day)
    for sm, sd, em, ed, name, name_te, impact in SEASONAL_PERIODS:
        # Handle cross-year periods (Dec 20 → Jan 5)
        if sm > em:  # wraps around year boundary
            start1 = date(year, sm, sd)
            end1 = date(year, 12, 31)
            start2 = date(year, 1, 1)
            end2 = date(year, em, ed)
            if start1 <= d <= end1 or start2 <= d <= end2:
                events.append({
                    "name": name, "name_te": name_te,
                    "type": "school_holiday", "impact": impact,
                })
        else:
            start = date(year, sm, sd)
            end = date(year, em, ed)
            if start <= d <= end:
                events.append({
                    "name": name, "name_te": name_te,
                    "type": "school_holiday", "impact": impact,
                })
    return events


def get_events_for_date(year: int, month: int, day: int) -> list:
    """Return all events (festivals, lunar, seasonal) for a specific date."""
    events = []

    # 1. Festivals
    year_festivals = FESTIVALS.get(year, [])
    for fm, fd, name, name_te, ftype, impact in year_festivals:
        if fm == month and fd == day:
            events.append({
                "name": name, "name_te": name_te,
                "type": ftype, "impact": impact,
                "emoji": _type_emoji(ftype),
            })

    # 2. Purnima
    for pm, pd in PURNIMA.get(year, []):
        if pm == month and pd == day:
            events.append({
                "name": "Purnima (Full Moon)", "name_te": "పౌర్ణమి",
                "type": "lunar", "impact": "moderate",
                "emoji": "🌕",
            })

    # 3. Amavasya
    for am, ad in AMAVASYA.get(year, []):
        if am == month and ad == day:
            events.append({
                "name": "Amavasya (New Moon)", "name_te": "అమావాస్య",
                "type": "lunar", "impact": "moderate",
                "emoji": "🌑",
            })

    # 4. Ekadashi
    for em, ed in EKADASHI.get(year, []):
        if em == month and ed == day:
            events.append({
                "name": "Ekadashi", "name_te": "ఏకాదశి",
                "type": "lunar", "impact": "moderate",
                "emoji": "📿",
            })

    # 5. Seasonal / school holidays
    events.extend(_is_in_seasonal_period(year, month, day))

    return events


def get_max_impact(events: list) -> str:
    """Return the highest impact level from a list of events."""
    order = ["extreme", "very_high", "high", "moderate", "low"]
    for level in order:
        if any(e.get("impact") == level for e in events):
            return level
    return "low"


def get_impact_factor(year: int, month: int, day: int) -> float:
    """Return the crowd multiplier for a date based on festivals/events.
    1.0 = normal day, >1.0 = busier, <1.0 = quieter."""
    events = get_events_for_date(year, month, day)
    if not events:
        # Quieter weekdays (Tue/Wed/Thu with no events) get a slight dip
        d = date(year, month, day)
        if d.weekday() in (1, 2, 3):  # Tue, Wed, Thu
            return 0.90
        return 1.0
    max_impact = get_max_impact(events)
    return IMPACT[max_impact]["factor"]


def get_crowd_reason(events: list) -> str:
    """Build a human-readable explanation of why a day might be crowded."""
    if not events:
        return ""
    names = [e["name"] for e in events if e.get("type") != "school_holiday"]
    seasonal = [e["name"] for e in events if e.get("type") == "school_holiday"]
    parts = []
    if names:
        parts.append(", ".join(names))
    if seasonal:
        parts.append(f"({seasonal[0]})")
    return " + ".join(parts)


def _type_emoji(ftype: str) -> str:
    return {
        "festival": "🛕",
        "brahmotsavam": "🔱",
        "holiday": "🏛️",
        "school_holiday": "🏫",
        "lunar": "🌙",
    }.get(ftype, "📌")
