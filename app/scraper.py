"""
Incremental scraper — fetches only NEW records from news.tirumala.org
and appends them to the existing CSV.

Usage:
    python -m app.scraper          # scrape new data
    python -m app.scraper --full   # re-scrape everything (113 pages)
"""
import re
import sys
import time
import argparse
import urllib3
from datetime import datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup

from app.config import DATA_CSV, SCRAPE_URL, SCRAPE_HEADERS, SCRAPE_DELAY

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ── Parsers (same logic as scrape_tirumala.py) ──────────────────────
def _parse_title(title: str):
    """Return (datetime, int|None) from a darshan title."""
    pat = r"(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4})\s*[:]\s*([\d,.\s]+)"
    m = re.search(pat, title)
    if not m:
        return None, None
    ds = m.group(1).replace("-", ".").replace("/", ".")
    try:
        dt = datetime.strptime(ds, "%d.%m.%Y")
    except ValueError:
        return None, None
    cs = m.group(2).strip().replace(",", "").replace(" ", "").replace(".", "")
    try:
        cnt = int(cs)
    except ValueError:
        return dt, None
    return dt, cnt


def _parse_body(text: str) -> dict:
    """Extract secondary fields from article body."""
    r = {"tonsures": None, "hundi_kanukalu_cr": None,
         "waiting_compartments": None, "approx_darshan_time_hours": None}
    if not text:
        return r
    text = text.replace("\n", " ").replace("\r", " ")

    m = re.search(r"[Tt]onsures?\s*[:]\s*([\d,.\s]+)", text)
    if m:
        v = m.group(1).strip().replace(",", "").replace(" ", "")
        try:
            r["tonsures"] = int(v)
        except ValueError:
            pass

    m = re.search(r"[Hh]undi\s*[Kk]anukalu\s*[:]\s*([\d,.]+)\s*CR", text, re.I)
    if m:
        try:
            r["hundi_kanukalu_cr"] = float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    m = re.search(
        r"[Ww]aiting\s+[Cc]ompartments?\s*[….\-:]+\s*(\d+|[Oo]ut\s*side\s+line[^.]*)",
        text,
    )
    if m:
        v = m.group(1).strip()
        r["waiting_compartments"] = v if not v.isdigit() else v

    m = re.search(
        r"[Aa]pprox[.\s]*[Dd]ars[ah]n\s+[Tt]ime.*?(\d+[\s\-–]*\d*)\s*H", text
    )
    if m:
        r["approx_darshan_time_hours"] = m.group(1).strip()

    return r


def scrape_page(page_num: int, cutoff: datetime | None = None):
    """Return list[dict] for one page. Stop if any date < cutoff."""
    url = SCRAPE_URL.format(page=page_num)
    records = []
    hit_cutoff = False
    try:
        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=30, verify=False)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERR] page {page_num}: {e}")
        return records, hit_cutoff

    soup = BeautifulSoup(resp.text, "lxml")
    for h2 in soup.find_all("h2"):
        link = h2.find("a")
        if not link:
            continue
        title = link.get_text(strip=True)
        if "darshan" not in title.lower():
            continue
        dt, cnt = _parse_title(title)
        if dt is None:
            continue
        if cutoff and dt < cutoff:
            hit_cutoff = True
            continue  # skip old, but keep scanning this page

        body = ""
        parent = h2.find_parent("article")
        if parent:
            body = parent.get_text(separator=" ", strip=True)
        bd = _parse_body(body)

        if cnt is None:
            _, cnt = _parse_title(body)

        records.append({
            "date": dt.strftime("%Y-%m-%d"),
            "total_pilgrims": cnt,
            **bd,
        })
    return records, hit_cutoff


# ── Public API ──────────────────────────────────────────────────────
def scrape_incremental(max_pages: int = 5) -> int:
    """Fetch only new records (pages 1..max_pages), append to CSV.

    Returns number of NEW records added.
    """
    # Load existing data
    try:
        existing = pd.read_csv(DATA_CSV, parse_dates=["date"])
        latest = existing["date"].max()
        print(f"  Existing data up to {latest.date()}")
    except FileNotFoundError:
        existing = pd.DataFrame()
        latest = datetime(2023, 1, 1)
        print("  No existing CSV — starting fresh")

    all_new = []
    for page in range(1, max_pages + 1):
        print(f"  Scraping page {page}/{max_pages} ...", end=" ")
        recs, hit_old = scrape_page(page, cutoff=latest - pd.Timedelta(days=1))
        print(f"{len(recs)} entries")
        all_new.extend(recs)
        if hit_old:
            print("  Reached already-seen dates — stopping.")
            break
        time.sleep(SCRAPE_DELAY)

    if not all_new:
        print("  No new records found.")
        return 0

    new_df = pd.DataFrame(all_new)
    new_df["date"] = pd.to_datetime(new_df["date"])
    new_df = new_df.drop_duplicates(subset=["date"], keep="first")

    # Merge with existing, keeping latest value for each date
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
    else:
        combined = new_df

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(DATA_CSV, index=False)
    added = len(combined) - len(existing)
    print(f"  ✅ {added} new records added (total: {len(combined)})")
    print(f"  Latest date: {combined['date'].max().date()}")
    return added


def scrape_full(end_page: int = 113):
    """Re-scrape all pages (like the original scraper)."""
    cutoff = datetime(2023, 1, 1)
    all_recs = []
    for page in range(1, end_page + 1):
        sys.stdout.write(f"\r  Scraping page {page}/{end_page} ...")
        sys.stdout.flush()
        recs, hit = scrape_page(page, cutoff=cutoff)
        all_recs.extend(recs)
        if hit:
            print(f" reached pre-2023 data at page {page}")
            break
        time.sleep(SCRAPE_DELAY)

    df = pd.DataFrame(all_recs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first")
    df = df[df["date"] >= "2023-01-01"].reset_index(drop=True)
    df.to_csv(DATA_CSV, index=False)
    print(f"\n  ✅ Full scrape: {len(df)} records saved.")


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tirumala scraper")
    parser.add_argument("--full", action="store_true", help="Full re-scrape (all pages)")
    parser.add_argument("--pages", type=int, default=5,
                        help="Max pages to check in incremental mode (default 5)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TIRUMALA INCREMENTAL SCRAPER")
    print("=" * 60)

    if args.full:
        scrape_full()
    else:
        scrape_incremental(max_pages=args.pages)
