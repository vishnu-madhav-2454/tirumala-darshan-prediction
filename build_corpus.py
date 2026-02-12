"""
Process the raw scraped TTD data into a clean, deduplicated corpus file.
Removes navigation boilerplate, excessive repetition, and formats nicely.
"""

import re
import os

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT  = os.path.join(BASE, "ttd_scraped_data.txt")
OUTPUT = os.path.join(BASE, "ttd_corpus.txt")

# Navigation / boilerplate phrases to strip out entirely
JUNK_PATTERNS = [
    r"Temples\s+Temple Legend\s+History.*?Saranagathi gadhyam",
    r"SEVAS\s+Arjitha Sevas.*?Annual / Periodical Sevas",
    r"DARSHAN\s+Sarvadarshan.*?Special Entry Darshan.*?\)",
    r"ACCOMMODATION\s+Accommodation at Tirumala.*?Rest Houses & Tariffs",
    r"TIRUMALA UPDATES.*?(?=\n[A-Z]{2,}|\n={3,}|\Z)",
    r"PANCHAGAVYA PRODUCTS\s+Dhoop.*?(?:more\.\.\.)",
    r"SCHEMES/TRUSTS\s+SRIVANI TRUST.*?(?:more\.\.\.)",
    r"SOCIAL SERVICE\s+.*?(?:more\.\.\.)",
    r"Day Schedules \d{2}-\d{2}-\d{4}.*?Ekanta Seva",
    r"Latest Updates.*?(?=\n={3,}|\n## |\Z)",
    r"Notifications\s+Tenders.*?(?=\n={3,}|\n## |\Z)",
    r"Other Links :.*?(?=\n)",
    r"Copyright.*?All Rights Reserved",
    r"Total Visitors\s*:\s*[\d,]+.*?(?=\n)",
    r"Today's Visitors\s*:\s*[\d,]+",
    r"SCROLL TO TOP",
    r"prevnext[\s\d]+",
    r"Annamaya Pataku Pattabhishekam.*?(?=\n\n)",
    r"Slotted Sarva Darshan.*?first-come-first-service basis",
    r"Flash News.*?(?=\n\n\n|\n[A-Z])",
    r"--- TABLE DATA ---",
    r"[|\-]{10,}",  # table border lines
    r"S\.No\s*\|\s*Content\s*\|.*",
    r"\d+\s*\|\s*\d+\s*\|.*?View/Download",  # tender rows
    r"(?:View|Download)\s*$",
    r"Skip to main content",
    r"Click here to.*",
    r"Please click on view.*",
    r"Advt No:.*",
]

# Known repeated large blocks (from sidebar/footer across all pages)
SIDEBAR_MARKERS = [
    "Akhanda Harinama Sankeerthana",
    "Sealed RFPs are invited",
    "EOI is invited for consultancy",
    "960 Net Immovable Properties",
    "TTD has introduced many Panchagavya",
    "TTD - SVIMS tie up",
    "Enriched bio-manure stocks",
    "Draft list of 1128 Immovable",
    "Yoga Vasishtam & Dhanvanthari",
    "Devotees / Organisers who intend",
    "Program List Day wise Troupes",
]


def remove_junk(text: str) -> str:
    """Remove navigation boilerplate and junk patterns."""
    for pat in JUNK_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def remove_sidebar_content(text: str) -> str:
    """Remove known sidebar repeated content blocks."""
    lines = text.split("\n")
    clean_lines = []
    skip_until_blank = False
    for line in lines:
        if skip_until_blank:
            if line.strip() == "":
                skip_until_blank = False
            continue
        if any(marker in line for marker in SIDEBAR_MARKERS):
            skip_until_blank = True
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)


def deduplicate_paragraphs(text: str) -> str:
    """Remove duplicate paragraphs (common from repeated page elements)."""
    paragraphs = text.split("\n\n")
    seen = set()
    unique = []
    for para in paragraphs:
        stripped = para.strip()
        if len(stripped) < 10:
            unique.append(para)
            continue
        # Use first 200 chars as fingerprint (handles near-duplicates)
        fingerprint = re.sub(r'\s+', ' ', stripped[:200]).lower()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(para)
    return "\n\n".join(unique)


def truncate_huge_sections(text: str, max_chars: int = 8000) -> str:
    """Truncate any single section that's unreasonably long (repeated content)."""
    sections = text.split("\n" + "=" * 80)
    truncated = []
    for sec in sections:
        if len(sec) > max_chars:
            # Keep only the meaningful first part
            sec = sec[:max_chars] + "\n\n[... Additional details available on tirumala.org ...]\n"
        truncated.append(sec)
    return ("\n" + "=" * 80).join(truncated)


def final_cleanup(text: str) -> str:
    """Final formatting cleanup."""
    # Remove excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Remove lines that are just whitespace
    lines = text.split("\n")
    lines = [l.rstrip() for l in lines]
    # Remove lines that are just single characters or pipe characters
    lines = [l for l in lines if not re.match(r'^[\s|_\-=]{1,3}$', l)]
    # Remove duplicate adjacent blank lines
    cleaned = []
    prev_blank = False
    for line in lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank
    return "\n".join(cleaned)


def build_corpus():
    """Read raw scraped data, clean it, write corpus."""
    print(f"Reading raw data from {INPUT}...")
    with open(INPUT, "r", encoding="utf-8") as f:
        raw = f.read()

    print(f"  Raw size: {len(raw):,} chars, {raw.count(chr(10)):,} lines")

    # Step 1: Remove junk patterns
    print("Step 1: Removing navigation boilerplate...")
    text = remove_junk(raw)
    print(f"  After junk removal: {len(text):,} chars")

    # Step 2: Remove sidebar content
    print("Step 2: Removing sidebar repeated content...")
    text = remove_sidebar_content(text)
    print(f"  After sidebar removal: {len(text):,} chars")

    # Step 3: Truncate huge sections
    print("Step 3: Truncating oversized sections...")
    text = truncate_huge_sections(text, max_chars=8000)
    print(f"  After truncation: {len(text):,} chars")

    # Step 4: Deduplicate paragraphs
    print("Step 4: Deduplicating paragraphs...")
    text = deduplicate_paragraphs(text)
    print(f"  After dedup: {len(text):,} chars")

    # Step 5: Final cleanup
    print("Step 5: Final cleanup...")
    text = final_cleanup(text)
    print(f"  After cleanup: {len(text):,} chars")

    lines = text.count("\n") + 1
    print(f"\nFinal corpus: {len(text):,} chars, {lines:,} lines")

    # Write output
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Written to: {OUTPUT}")
    return text


if __name__ == "__main__":
    build_corpus()
