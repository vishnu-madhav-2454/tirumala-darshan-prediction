"""
Comprehensive TTD Website Scraper
Scrapes ALL pages from multiple official TTD websites and saves clean text
to ttd_scraped_data.txt.
Domains covered:
  - https://www.tirumala.org  (Main TTD website)
  - https://ttdevasthanams.ap.gov.in  (Official TTD Booking Portal)
  - http://www.svbcttd.com  (SVBC TV Channel)
  - https://srivariseva.tirumala.org  (Srivari Seva / Voluntary Services)
  - https://ebooks.tirumala.org  (e-Publications)
Uses BeautifulSoup + requests
"""

import requests
from bs4 import BeautifulSoup
import time
import re
import os

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttd_scraped_data.txt")

# Each entry is (title, full_url) — pages collected from multiple TTD domains.
PAGES = [
    # =========================================================================
    #  DOMAIN: https://www.tirumala.org  (Main TTD Website)
    # =========================================================================

    # --- TEMPLES ---
    ("Temple Legend (Lord Venkateswara)", "https://www.tirumala.org/TempleLegend.aspx"),
    ("Temple History", "https://www.tirumala.org/TTDTempleHistory.aspx"),
    ("Srinivasa Kalyanam (Sacred Wedding)", "https://www.tirumala.org/srk1.aspx"),
    ("Varaha Swamy Temple Legend", "https://www.tirumala.org/Varaha_TempleLegend.aspx"),
    ("Anjaneya Swamy Temple Legend", "https://www.tirumala.org/Anjaneya_TempleLegend.aspx"),
    ("Sri Padmavathi Temple Legend (Tiruchanoor)", "https://www.tirumala.org/PatAtThiruchanoorTempleLegend.aspx"),
    ("Sri Padmavathi Temple Sevas (Tiruchanoor)", "https://www.tirumala.org/PatAtThiruchanoorSevas.aspx"),
    ("Temples at Tiruchanoor", "https://www.tirumala.org/TemplesAtTiruchanoor.aspx"),
    ("Temples at Tirupathi", "https://www.tirumala.org/TemplesAtTirupathi.aspx"),
    ("More Temples", "https://www.tirumala.org/MoreTemples.aspx"),
    ("Places to Visit Around Tirumala", "https://www.tirumala.org/Places%20to%20see%20around%20Tirumala.aspx"),

    # --- DARSHAN ---
    ("Sarvadarshanam (Free General Darshan)", "https://www.tirumala.org/Sarvadarshanam.aspx"),
    ("Special Entry Darshan (Seeghra Darshan Rs.300)", "https://www.tirumala.org/SpecialEntryDarshan.aspx"),
    ("Divya Darshan (For Pedestrians Who Walk)", "https://www.tirumala.org/DivyaDarshan.aspx"),
    ("Special Darshan for Physically Disabled and Aged", "https://www.tirumala.org/SpecialDarshanForPhysicallyDisabledAndAged.aspx"),

    # --- SEVAS ---
    ("Arjitha Sevas (Paid Worship Services)", "https://www.tirumala.org/ArjithaSevas.aspx"),
    ("Daily Sevas", "https://www.tirumala.org/DailySevas.aspx"),
    ("Weekly Sevas", "https://www.tirumala.org/WeeklySevas.aspx"),
    ("Annual or Periodical Sevas", "https://www.tirumala.org/AnnualSevas.aspx"),

    # --- ACCOMMODATION ---
    ("Accommodation at Tirumala", "https://www.tirumala.org/AccommodationAtTirumala.aspx"),
    ("Rest Houses and Tariffs in Tirumala", "https://www.tirumala.org/RestHousesInTirumala.aspx"),
    ("Current Booking Information", "https://www.tirumala.org/Current_Booking.aspx"),
    ("Accommodation at Tirupati", "https://www.tirumala.org/AccommodationAtTirupati.aspx"),
    ("Advance Booking", "https://www.tirumala.org/Advancebooking.aspx"),

    # --- TRAVEL & TRANSPORT ---
    ("How to Reach Tirupati and Tirumala", "https://www.tirumala.org/Howtoreach_TirupatiandTirumala.aspx"),
    ("To Reach Tirupati", "https://www.tirumala.org/ToReachTirupati.aspx"),
    ("Travelling from Tirupati to Tirumala", "https://www.tirumala.org/TravellingfromTirupatitoTirumala.aspx"),
    ("Free Bus Service at Tirumala", "https://www.tirumala.org/FreeBusServiceAtTirumala.aspx"),
    ("Free Bus Service at Tirupathi", "https://www.tirumala.org/FreeBusServiceTirupathi.aspx"),
    ("Return Journey From Tirumala", "https://www.tirumala.org/ReturnJourneyFromTirumala.aspx"),
    ("Package Tours (Local and Surrounding Temples)", "https://www.tirumala.org/PackageTours.aspx"),
    ("Railway Booking Office at Tirumala", "https://www.tirumala.org/RailwayBookingOffice.aspx"),
    ("Automobile Clinic (Breakdown Help)", "https://www.tirumala.org/AutomobileClinic.aspx"),

    # --- PILGRIM SERVICES ---
    ("Sri Venkateswara Annaprasadam Trust (Free Food)", "https://www.tirumala.org/SRIVENKATESWARAANNAPRASADAMTRUST.aspx"),
    ("Annaprasadam Trust Details", "https://www.tirumala.org/AnnaprasadamTrust.aspx"),
    ("Kalyana Katta (Free Hair Tonsuring)", "https://www.tirumala.org/KalyanaKatta.aspx"),
    ("Medical Facilities", "https://www.tirumala.org/MedicalFacilities.aspx"),
    ("Fulfilling Vows (Mokku, Tulabharam, Hundi)", "https://www.tirumala.org/vows.aspx"),
    ("Laddu Prasadam", "https://www.tirumala.org/Laddu.aspx"),

    # --- FESTIVALS & UTSAVAMS ---
    ("Brahmotsavams and Utsavams", "https://www.tirumala.org/Utsavams.aspx"),
    ("Brahmotsavams Detailed", "https://www.tirumala.org/Brahmotsavams.aspx"),
    ("Vaikunta Ekadasi (Mukkoti Ekadasi)", "https://www.tirumala.org/VaikundaEkadasi.aspx"),
    ("Festivals", "https://www.tirumala.org/Festivals.aspx"),

    # --- DONATIONS & TRUSTS ---
    ("SRIVANI Trust (Temple Construction Donation)", "https://www.tirumala.org/SRIVANI.aspx"),
    ("Donor Privileges", "https://www.tirumala.org/Privileges.aspx"),
    ("Hundi Donation", "https://www.tirumala.org/HundiDonation.aspx"),
    ("More Links (Additional Trusts and Schemes)", "https://www.tirumala.org/MoreLinks.aspx"),

    # --- MUSEUMS ---
    ("SV Museum (Tirumala and Tirupati)", "https://www.tirumala.org/S.V.%20Museum.aspx"),

    # --- EDUCATION & SOCIAL ---
    ("Social Activities", "https://www.tirumala.org/SocialActivities.aspx"),
    ("Educational Trust and Institutions", "https://www.tirumala.org/EducationalTrust.aspx"),
    ("Religious Activities", "https://www.tirumala.org/ReligiousActivities.aspx"),
    ("Social Services Overview", "https://www.tirumala.org/SocialServices.aspx"),
    ("Hindu Dharma Prachara Parishad", "https://www.tirumala.org/hdpp.aspx"),

    # --- PUBLICATIONS ---
    ("TTD Publications", "https://www.tirumala.org/TTDs_Publications.aspx"),
    ("SA Publications", "https://www.tirumala.org/SAPublications.aspx"),
    ("Sri Annamacharya Recording Project", "https://www.tirumala.org/RecordingProject.aspx"),

    # --- GENERAL INFO ---
    ("Do's and Don'ts at Tirumala", "https://www.tirumala.org/Dos_Donot.aspx"),
    ("Frequently Asked Questions (FAQs)", "https://www.tirumala.org/FAQ.aspx"),
    ("Dress Code for Temple", "https://www.tirumala.org/DressCode.aspx"),
    ("Contact Us (Phone, Email, Address)", "https://www.tirumala.org/Contactus.aspx"),
    ("Panchagavya Products (Namami Govinda)", "https://www.tirumala.org/PanchagavyaProducts.aspx"),

    # --- ADMIN ---
    ("TTD Board Administration", "https://www.tirumala.org/TTDBoard.aspx"),
    ("TTD Trust Board Members", "https://www.tirumala.org/TTD%20Trust%20Board.aspx"),
    ("Srinivasa Kalyanam Photo Gallery", "https://www.tirumala.org/SrinivasaKalyanam.aspx"),
    ("Saranagathi Gadhyam", "https://www.tirumala.org/Saranagathi_Gadhyam.aspx"),

    # =========================================================================
    #  DOMAIN: https://ttdevasthanams.ap.gov.in  (Official TTD Booking Portal)
    # =========================================================================
    ("TTD Booking Portal - Dashboard", "https://ttdevasthanams.ap.gov.in/home/dashboard"),
    ("TTD Booking Portal - E-Services", "https://ttdevasthanams.ap.gov.in/home/eServices"),
    ("TTD Booking Portal - About TTD", "https://ttdevasthanams.ap.gov.in/home/aboutTTD"),

    # =========================================================================
    #  DOMAIN: http://www.svbcttd.com  (SVBC TV Channel)
    # =========================================================================
    ("SVBC TV Channel - Main Page", "http://www.svbcttd.com/"),
    ("SVBC TV Channel - About Us", "http://www.svbcttd.com/about-us"),
    ("SVBC TV Channel - Program Schedule", "http://www.svbcttd.com/programs"),

    # =========================================================================
    #  DOMAIN: https://srivariseva.tirumala.org  (Srivari Seva / Voluntary Services)
    # =========================================================================
    ("Srivari Seva - Main Page", "https://srivariseva.tirumala.org/"),
    ("Srivari Seva - About", "https://srivariseva.tirumala.org/about"),

    # =========================================================================
    #  DOMAIN: https://ebooks.tirumala.org  (e-Publications)
    # =========================================================================
    ("TTD e-Publications - Main Page", "https://ebooks.tirumala.org/"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Elements to remove from every page (navigation, headers, footers, scripts)
REMOVE_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "iframe", "svg"]
REMOVE_IDS = ["menu", "navbar", "footer", "sidebar", "header"]
REMOVE_CLASSES = [
    "menu", "navbar", "nav-", "footer", "sidebar", "header", "breadcrumb",
    "social", "cookie", "popup", "modal", "advertisement", "banner",
    "flash-news", "ticker", "scroll"
]


def clean_text(raw: str) -> str:
    """Clean scraped text: remove extra whitespace, blank lines, navigation junk."""
    # Remove common navigation/junk phrases
    junk_phrases = [
        r"Copyright.*?All Rights Reserved",
        r"Total Visitors\s*:\s*[\d,]+",
        r"Today's Visitors\s*:\s*[\d,]+",
        r"SCROLL TO TOP",
        r"Flash News.*?(?=\n\n|\Z)",
        r"prevnext\s*[\d\s]+",
        r"Home\s*>\s*",
        r"Skip to main content",
        r"Click here to.*",
    ]
    text = raw
    for pattern in junk_phrases:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Collapse multiple blank lines to max 2
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Strip each line
    lines = [line.strip() for line in text.split("\n")]
    # Remove very short junk lines (likely navigation remnants)
    lines = [l for l in lines if len(l) > 2 or l == ""]
    text = "\n".join(lines)
    return text.strip()


def extract_tables(soup) -> str:
    """Extract HTML tables as formatted text."""
    table_text = ""
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        table_text += "\n"
        for row in rows:
            cells = row.find_all(["td", "th"])
            cell_texts = [c.get_text(strip=True) for c in cells]
            # Skip empty rows
            if not any(cell_texts):
                continue
            table_text += " | ".join(cell_texts) + "\n"
        table_text += "\n"
    return table_text


def scrape_page(title: str, url: str) -> str:
    """Scrape a single page and return clean text content."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [SKIP] {url} -> {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove unwanted elements
    for tag_name in REMOVE_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()

    for el_id in REMOVE_IDS:
        for el in soup.find_all(id=re.compile(el_id, re.I)):
            el.decompose()

    for cls in REMOVE_CLASSES:
        for el in soup.find_all(class_=re.compile(cls, re.I)):
            el.decompose()

    # Extract tables separately (before get_text destroys structure)
    table_data = extract_tables(soup)

    # Find the main content area (try common content div patterns)
    content_area = None
    for selector in [
        "div.content-area", "div.main-content", "div#content",
        "div.ContentPlaceHolder", "div#ContentPlaceHolder1",
        "td.auto-style12", "td.auto-style7",
        "div.container", "div.row",
        "form#form1",
    ]:
        found = soup.select(selector)
        if found:
            # Use the one with the most text
            content_area = max(found, key=lambda x: len(x.get_text()))
            break

    if content_area is None:
        content_area = soup.body if soup.body else soup

    # Get text
    raw_text = content_area.get_text(separator="\n")

    # Combine with table data
    combined = raw_text
    if table_data.strip():
        combined += "\n\n--- TABLE DATA ---\n" + table_data

    cleaned = clean_text(combined)

    # Skip if too little content (likely empty/error page)
    if len(cleaned) < 50:
        print(f"  [SKIP] {title} - too little content ({len(cleaned)} chars)")
        return ""

    return cleaned


def scrape_all():
    """Scrape all TTD pages and write to output file."""
    print(f"Starting TTD website scraper...")
    print(f"Pages to scrape: {len(PAGES)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 60)

    all_data = []
    success = 0
    failed = 0

    for i, (title, url) in enumerate(PAGES, 1):
        print(f"[{i:02d}/{len(PAGES)}] Scraping: {title}...")
        content = scrape_page(title, url)

        if content:
            section = f"{'='*80}\n## {title}\n{'='*80}\n\n{content}"
            all_data.append(section)
            success += 1
            print(f"  OK ({len(content)} chars)")
        else:
            failed += 1

        # Polite delay to avoid overwhelming the server
        time.sleep(1.5)

    # Write all data to file
    print("\n" + "=" * 60)
    print(f"Writing to {OUTPUT_FILE}...")

    full_text = "\n\n\n".join(all_data)

    # Final header
    header = (
        "# TIRUMALA TIRUPATI DEVASTHANAMS (TTD) - COMPLETE INFORMATION\n"
        "# Sources:\n"
        "#   - https://www.tirumala.org  (Main TTD Website)\n"
        "#   - https://ttdevasthanams.ap.gov.in  (Official TTD Booking Portal)\n"
        "#   - http://www.svbcttd.com  (SVBC TV Channel)\n"
        "#   - https://srivariseva.tirumala.org  (Srivari Seva / Voluntary Services)\n"
        "#   - https://ebooks.tirumala.org  (e-Publications)\n"
        f"# Pages scraped: {success} / {len(PAGES)}\n"
        f"# Total characters: {len(full_text)}\n"
        "# This file contains comprehensive information about:\n"
        "#   - Temple Legend, History, Architecture\n"
        "#   - Darshan Types (Sarva, Special Entry, Divya, Disabled/Aged)\n"
        "#   - All Sevas (Daily, Weekly, Annual, Arjitha)\n"
        "#   - Accommodation (Tirumala & Tirupati with tariffs)\n"
        "#   - Transportation (Footpaths, Buses, Ghat Roads, Package Tours)\n"
        "#   - Annaprasadam (Free Food Service)\n"
        "#   - Kalyana Katta (Hair Tonsuring)\n"
        "#   - Medical Facilities\n"
        "#   - Festivals (Brahmotsavam, Vaikunta Ekadasi)\n"
        "#   - Donations and Trusts (SRIVANI, Privileges)\n"
        "#   - Museums, Education, Religious Activities\n"
        "#   - Publications, FAQs, Contact Info\n"
        "#   - Do's and Don'ts, Dress Code\n"
        "#   - Online Booking & E-Services (TTD Portal)\n"
        "#   - SVBC TV Channel & Program Schedule\n"
        "#   - Srivari Seva (Voluntary Services)\n"
        "#   - TTD e-Publications\n"
        "# ================================================================\n\n\n"
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(header + full_text)

    lines = full_text.count("\n") + 1
    print(f"\nDONE!")
    print(f"  Pages scraped successfully: {success}")
    print(f"  Pages failed/skipped: {failed}")
    print(f"  Total lines: {lines}")
    print(f"  Total characters: {len(full_text)}")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    scrape_all()
