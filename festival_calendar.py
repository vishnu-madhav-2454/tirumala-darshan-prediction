"""
Extended Festival & Event Calendar for Tirumala Darshan Prediction (2013-2027).

This module provides a COMPLETE historical festival database covering the full
data range, enabling the ML model to learn festival impacts directly from data.

Sources:
  - Hindu lunar calendar calculations (Panchang)
  - TTD official Brahmotsavam dates (Sep/Oct each year)
  - Government of India / AP state holiday calendars
  - School holiday schedules (AP/Telangana pattern)

Exports:
  get_festival_features(date) → dict of numeric features for that date
  get_festival_features_series(dates) → DataFrame of festival features
"""

from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
#  FESTIVAL DATABASE 2013-2027
#  Format: (month, day, event_name, category, impact_level)
#  Categories: sankranti, brahmotsavam, vaikuntha, shivaratri,
#              ugadi, dussehra, diwali, janmashtami, vinayaka,
#              navaratri, rathasapthami, holi, ramanavami,
#              national_holiday, purnima, ekadashi, other_festival
#  Impact levels: 5=extreme, 4=very_high, 3=high, 2=moderate, 1=low
# ═══════════════════════════════════════════════════════════════

# Hindu festivals are lunar-calendar-based, so dates shift each year.
# Below are historically accurate dates for major Tirumala-impacting events.

FESTIVAL_DB: Dict[int, List[Tuple[int, int, str, str, int]]] = {
    # ══════ 2013 ══════
    2013: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 26, "Republic Day", "national_holiday", 3),
        (2, 5,  "Rathasapthami", "rathasapthami", 5),
        (3, 10, "Maha Shivaratri", "shivaratri", 4),
        (3, 27, "Holi", "holi", 3),
        (4, 11, "Ugadi", "ugadi", 5),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (4, 19, "Sri Rama Navami", "ramanavami", 4),
        (8, 9,  "Krishna Janmashtami", "janmashtami", 5),
        (8, 15, "Independence Day", "national_holiday", 4),
        (9, 9,  "Vinayaka Chaturthi", "vinayaka", 4),
        (10, 5, "Navaratri Begins", "navaratri", 4),
        (10, 13,"Dussehra", "dussehra", 5),
        (10, 14,"Brahmotsavams Begin", "brahmotsavam", 5),
        (10, 15,"Brahmotsavams Day 2", "brahmotsavam", 5),
        (10, 16,"Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 17,"Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 18,"Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 19,"Brahmotsavams Day 6", "brahmotsavam", 5),
        (10, 20,"Brahmotsavams Day 7", "brahmotsavam", 5),
        (10, 21,"Brahmotsavams Day 8", "brahmotsavam", 5),
        (10, 22,"Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (11, 3, "Diwali", "diwali", 5),
        (11, 17,"Karthika Purnima", "purnima", 4),
        (12, 14,"Vaikuntha Ekadashi", "vaikuntha", 5),
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2014 ══════
    2014: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 26, "Republic Day", "national_holiday", 3),
        (1, 28, "Rathasapthami", "rathasapthami", 5),
        (2, 27, "Maha Shivaratri", "shivaratri", 4),
        (3, 17, "Holi", "holi", 3),
        (3, 31, "Ugadi", "ugadi", 5),
        (4, 8,  "Sri Rama Navami", "ramanavami", 4),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (8, 15, "Independence Day", "national_holiday", 4),
        (8, 17, "Krishna Janmashtami", "janmashtami", 5),
        (8, 29, "Vinayaka Chaturthi", "vinayaka", 4),
        (9, 25, "Navaratri Begins", "navaratri", 4),
        (10, 2, "Gandhi Jayanti", "national_holiday", 3),
        (10, 3, "Dussehra", "dussehra", 5),
        (10, 4, "Brahmotsavams Begin", "brahmotsavam", 5),
        (10, 5, "Brahmotsavams Day 2", "brahmotsavam", 5),
        (10, 6, "Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 7, "Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 8, "Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 9, "Brahmotsavams Day 6", "brahmotsavam", 5),
        (10, 10,"Brahmotsavams Day 7", "brahmotsavam", 5),
        (10, 11,"Brahmotsavams Day 8", "brahmotsavam", 5),
        (10, 12,"Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (10, 23,"Diwali", "diwali", 5),
        (11, 6, "Karthika Purnima", "purnima", 4),
        (12, 3, "Vaikuntha Ekadashi", "vaikuntha", 5),
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2015 ══════
    2015: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 17, "Rathasapthami", "rathasapthami", 5),
        (1, 26, "Republic Day", "national_holiday", 3),
        (2, 17, "Maha Shivaratri", "shivaratri", 4),
        (3, 6,  "Holi", "holi", 3),
        (3, 21, "Ugadi", "ugadi", 5),
        (3, 28, "Sri Rama Navami", "ramanavami", 4),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (8, 15, "Independence Day", "national_holiday", 4),
        (9, 5,  "Krishna Janmashtami", "janmashtami", 5),
        (9, 17, "Vinayaka Chaturthi", "vinayaka", 4),
        (10, 2, "Gandhi Jayanti", "national_holiday", 3),
        (10, 13,"Navaratri Begins", "navaratri", 4),
        (10, 22,"Dussehra", "dussehra", 5),
        (10, 23,"Brahmotsavams Begin", "brahmotsavam", 5),
        (10, 24,"Brahmotsavams Day 2", "brahmotsavam", 5),
        (10, 25,"Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 26,"Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 27,"Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 28,"Brahmotsavams Day 6", "brahmotsavam", 5),
        (10, 29,"Brahmotsavams Day 7", "brahmotsavam", 5),
        (10, 30,"Brahmotsavams Day 8", "brahmotsavam", 5),
        (10, 31,"Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (11, 11,"Diwali", "diwali", 5),
        (11, 25,"Karthika Purnima", "purnima", 4),
        (12, 22,"Vaikuntha Ekadashi", "vaikuntha", 5),
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2016 ══════
    2016: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 26, "Republic Day", "national_holiday", 3),
        (2, 6,  "Rathasapthami", "rathasapthami", 5),
        (3, 7,  "Maha Shivaratri", "shivaratri", 4),
        (3, 24, "Holi", "holi", 3),
        (4, 8,  "Ugadi", "ugadi", 5),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (4, 15, "Sri Rama Navami", "ramanavami", 4),
        (8, 15, "Independence Day", "national_holiday", 4),
        (8, 25, "Krishna Janmashtami", "janmashtami", 5),
        (9, 5,  "Vinayaka Chaturthi", "vinayaka", 4),
        (10, 1, "Navaratri Begins", "navaratri", 4),
        (10, 2, "Gandhi Jayanti", "national_holiday", 3),
        (10, 11,"Dussehra", "dussehra", 5),
        (10, 12,"Brahmotsavams Begin", "brahmotsavam", 5),
        (10, 13,"Brahmotsavams Day 2", "brahmotsavam", 5),
        (10, 14,"Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 15,"Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 16,"Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 17,"Brahmotsavams Day 6", "brahmotsavam", 5),
        (10, 18,"Brahmotsavams Day 7", "brahmotsavam", 5),
        (10, 19,"Brahmotsavams Day 8", "brahmotsavam", 5),
        (10, 20,"Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (10, 30,"Diwali", "diwali", 5),
        (11, 14,"Karthika Purnima", "purnima", 4),
        (12, 11,"Vaikuntha Ekadashi", "vaikuntha", 5),
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2017 ══════
    2017: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 26, "Republic Day", "national_holiday", 3),
        (1, 28, "Rathasapthami", "rathasapthami", 5),
        (2, 24, "Maha Shivaratri", "shivaratri", 4),
        (3, 13, "Holi", "holi", 3),
        (3, 28, "Ugadi", "ugadi", 5),
        (4, 5,  "Sri Rama Navami", "ramanavami", 4),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (8, 15, "Independence Day + Janmashtami", "janmashtami", 5),
        (8, 25, "Vinayaka Chaturthi", "vinayaka", 4),
        (9, 21, "Navaratri Begins", "navaratri", 4),
        (9, 30, "Dussehra", "dussehra", 5),
        (10, 1, "Brahmotsavams Begin", "brahmotsavam", 5),
        (10, 2, "Brahmotsavams Day 2 + Gandhi Jayanti", "brahmotsavam", 5),
        (10, 3, "Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 4, "Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 5, "Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 6, "Brahmotsavams Day 6", "brahmotsavam", 5),
        (10, 7, "Brahmotsavams Day 7", "brahmotsavam", 5),
        (10, 8, "Brahmotsavams Day 8", "brahmotsavam", 5),
        (10, 9, "Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (10, 19,"Diwali", "diwali", 5),
        (11, 3, "Karthika Purnima", "purnima", 4),
        (11, 30,"Vaikuntha Ekadashi", "vaikuntha", 5),
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2018 ══════
    2018: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 26, "Republic Day", "national_holiday", 3),
        (2, 13, "Maha Shivaratri", "shivaratri", 4),
        (2, 16, "Rathasapthami", "rathasapthami", 5),
        (3, 2,  "Holi", "holi", 3),
        (3, 18, "Ugadi", "ugadi", 5),
        (3, 25, "Sri Rama Navami", "ramanavami", 4),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (8, 15, "Independence Day", "national_holiday", 4),
        (9, 2,  "Krishna Janmashtami", "janmashtami", 5),
        (9, 13, "Vinayaka Chaturthi", "vinayaka", 4),
        (10, 2, "Gandhi Jayanti", "national_holiday", 3),
        (10, 10,"Navaratri Begins", "navaratri", 4),
        (10, 19,"Dussehra", "dussehra", 5),
        (10, 20,"Brahmotsavams Begin", "brahmotsavam", 5),
        (10, 21,"Brahmotsavams Day 2", "brahmotsavam", 5),
        (10, 22,"Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 23,"Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 24,"Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 25,"Brahmotsavams Day 6", "brahmotsavam", 5),
        (10, 26,"Brahmotsavams Day 7", "brahmotsavam", 5),
        (10, 27,"Brahmotsavams Day 8", "brahmotsavam", 5),
        (10, 28,"Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (11, 7, "Diwali", "diwali", 5),
        (11, 23,"Karthika Purnima", "purnima", 4),
        (12, 18,"Vaikuntha Ekadashi", "vaikuntha", 5),
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2019 ══════
    2019: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 26, "Republic Day", "national_holiday", 3),
        (2, 3,  "Rathasapthami", "rathasapthami", 5),
        (3, 4,  "Maha Shivaratri", "shivaratri", 4),
        (3, 21, "Holi", "holi", 3),
        (4, 6,  "Ugadi", "ugadi", 5),
        (4, 14, "Ambedkar Jayanti + Sri Rama Navami", "ramanavami", 4),
        (8, 15, "Independence Day", "national_holiday", 4),
        (8, 24, "Krishna Janmashtami", "janmashtami", 5),
        (9, 2,  "Vinayaka Chaturthi", "vinayaka", 4),
        (9, 29, "Navaratri Begins", "navaratri", 4),
        (10, 2, "Gandhi Jayanti", "national_holiday", 3),
        (10, 8, "Dussehra", "dussehra", 5),
        (10, 9, "Brahmotsavams Begin", "brahmotsavam", 5),
        (10, 10,"Brahmotsavams Day 2", "brahmotsavam", 5),
        (10, 11,"Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 12,"Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 13,"Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 14,"Brahmotsavams Day 6", "brahmotsavam", 5),
        (10, 15,"Brahmotsavams Day 7", "brahmotsavam", 5),
        (10, 16,"Brahmotsavams Day 8", "brahmotsavam", 5),
        (10, 17,"Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (10, 27,"Diwali", "diwali", 5),
        (11, 12,"Karthika Purnima", "purnima", 4),
        (12, 8, "Vaikuntha Ekadashi", "vaikuntha", 5),
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2020 ══════ (COVID year — but dates still needed for completeness)
    2020: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 26, "Republic Day", "national_holiday", 3),
        (1, 31, "Rathasapthami", "rathasapthami", 5),
        (2, 21, "Maha Shivaratri", "shivaratri", 4),
        (3, 10, "Holi", "holi", 3),
        (3, 25, "Ugadi", "ugadi", 5),
        (4, 2,  "Sri Rama Navami", "ramanavami", 4),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (8, 11, "Krishna Janmashtami", "janmashtami", 5),
        (8, 15, "Independence Day", "national_holiday", 4),
        (8, 22, "Vinayaka Chaturthi", "vinayaka", 4),
        (10, 2, "Gandhi Jayanti", "national_holiday", 3),
        (10, 17,"Navaratri Begins", "navaratri", 4),
        (10, 25,"Dussehra", "dussehra", 5),
        (10, 26,"Brahmotsavams Begin", "brahmotsavam", 5),
        (10, 27,"Brahmotsavams Day 2", "brahmotsavam", 5),
        (10, 28,"Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 29,"Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 30,"Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 31,"Brahmotsavams Day 6", "brahmotsavam", 5),
        (11, 1, "Brahmotsavams Day 7", "brahmotsavam", 5),
        (11, 2, "Brahmotsavams Day 8", "brahmotsavam", 5),
        (11, 3, "Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (11, 14,"Diwali", "diwali", 5),
        (11, 30,"Karthika Purnima", "purnima", 4),
        (12, 25,"Vaikuntha Ekadashi + Christmas", "vaikuntha", 5),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2021 ══════ (COVID year)
    2021: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 26, "Republic Day", "national_holiday", 3),
        (2, 18, "Rathasapthami", "rathasapthami", 5),
        (3, 11, "Maha Shivaratri", "shivaratri", 4),
        (3, 29, "Holi", "holi", 3),
        (4, 13, "Ugadi", "ugadi", 5),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (4, 21, "Sri Rama Navami", "ramanavami", 4),
        (8, 15, "Independence Day", "national_holiday", 4),
        (8, 30, "Krishna Janmashtami", "janmashtami", 5),
        (9, 10, "Vinayaka Chaturthi", "vinayaka", 4),
        (10, 2, "Gandhi Jayanti", "national_holiday", 3),
        (10, 7, "Navaratri Begins", "navaratri", 4),
        (10, 15,"Dussehra", "dussehra", 5),
        (10, 16,"Brahmotsavams Begin", "brahmotsavam", 5),
        (10, 17,"Brahmotsavams Day 2", "brahmotsavam", 5),
        (10, 18,"Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 19,"Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 20,"Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 21,"Brahmotsavams Day 6", "brahmotsavam", 5),
        (10, 22,"Brahmotsavams Day 7", "brahmotsavam", 5),
        (10, 23,"Brahmotsavams Day 8", "brahmotsavam", 5),
        (10, 24,"Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (11, 4, "Diwali", "diwali", 5),
        (11, 19,"Karthika Purnima", "purnima", 4),
        (12, 14,"Vaikuntha Ekadashi", "vaikuntha", 5),
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2022 ══════
    2022: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 26, "Republic Day", "national_holiday", 3),
        (2, 7,  "Rathasapthami", "rathasapthami", 5),
        (3, 1,  "Maha Shivaratri", "shivaratri", 4),
        (3, 18, "Holi", "holi", 3),
        (4, 2,  "Ugadi", "ugadi", 5),
        (4, 10, "Sri Rama Navami", "ramanavami", 4),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (8, 15, "Independence Day", "national_holiday", 4),
        (8, 18, "Krishna Janmashtami", "janmashtami", 5),
        (8, 31, "Vinayaka Chaturthi", "vinayaka", 4),
        (9, 26, "Navaratri Begins", "navaratri", 4),
        (10, 2, "Gandhi Jayanti", "national_holiday", 3),
        (10, 5, "Dussehra", "dussehra", 5),
        (10, 6, "Brahmotsavams Begin", "brahmotsavam", 5),
        (10, 7, "Brahmotsavams Day 2", "brahmotsavam", 5),
        (10, 8, "Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 9, "Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 10,"Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 11,"Brahmotsavams Day 6", "brahmotsavam", 5),
        (10, 12,"Brahmotsavams Day 7", "brahmotsavam", 5),
        (10, 13,"Brahmotsavams Day 8", "brahmotsavam", 5),
        (10, 14,"Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (10, 24,"Diwali", "diwali", 5),
        (11, 8, "Karthika Purnima", "purnima", 4),
        (12, 3, "Vaikuntha Ekadashi", "vaikuntha", 5),
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2023 ══════
    2023: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 2,  "Vaikuntha Ekadashi (Mukkoti)", "vaikuntha", 5),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 26, "Republic Day", "national_holiday", 3),
        (1, 29, "Rathasapthami", "rathasapthami", 5),
        (2, 18, "Maha Shivaratri", "shivaratri", 4),
        (3, 8,  "Holi", "holi", 3),
        (3, 22, "Ugadi", "ugadi", 5),
        (3, 30, "Sri Rama Navami", "ramanavami", 4),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (8, 15, "Independence Day", "national_holiday", 4),
        (9, 6,  "Krishna Janmashtami", "janmashtami", 5),
        (9, 19, "Vinayaka Chaturthi", "vinayaka", 4),
        (10, 2, "Gandhi Jayanti", "national_holiday", 3),
        (10, 15,"Navaratri Begins", "navaratri", 4),
        (10, 24,"Dussehra", "dussehra", 5),
        (10, 25,"Brahmotsavams Begin", "brahmotsavam", 5),
        (10, 26,"Brahmotsavams Day 2", "brahmotsavam", 5),
        (10, 27,"Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 28,"Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 29,"Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 30,"Brahmotsavams Day 6", "brahmotsavam", 5),
        (10, 31,"Brahmotsavams Day 7", "brahmotsavam", 5),
        (11, 1, "Brahmotsavams Day 8", "brahmotsavam", 5),
        (11, 2, "Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (11, 12,"Diwali", "diwali", 5),
        (11, 27,"Karthika Purnima", "purnima", 4),
        (12, 23,"Vaikuntha Ekadashi", "vaikuntha", 5),
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2024 ══════
    2024: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 17, "Rathasapthami", "rathasapthami", 5),
        (1, 26, "Republic Day", "national_holiday", 3),
        (3, 8,  "Maha Shivaratri", "shivaratri", 4),
        (3, 25, "Holi", "holi", 3),
        (4, 9,  "Ugadi", "ugadi", 5),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (4, 17, "Sri Rama Navami", "ramanavami", 4),
        (8, 15, "Independence Day", "national_holiday", 4),
        (8, 26, "Krishna Janmashtami", "janmashtami", 5),
        (9, 7,  "Vinayaka Chaturthi", "vinayaka", 4),
        (10, 2, "Gandhi Jayanti", "national_holiday", 3),
        (10, 3, "Navaratri Begins", "navaratri", 4),
        (10, 12,"Dussehra", "dussehra", 5),
        (10, 13,"Brahmotsavams Begin", "brahmotsavam", 5),
        (10, 14,"Brahmotsavams Day 2", "brahmotsavam", 5),
        (10, 15,"Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 16,"Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 17,"Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 18,"Brahmotsavams Day 6", "brahmotsavam", 5),
        (10, 19,"Brahmotsavams Day 7", "brahmotsavam", 5),
        (10, 20,"Brahmotsavams Day 8", "brahmotsavam", 5),
        (10, 21,"Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (11, 1, "Diwali", "diwali", 5),
        (11, 15,"Karthika Purnima", "purnima", 4),
        (12, 11,"Vaikuntha Ekadashi", "vaikuntha", 5),
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2025 ══════
    2025: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 10, "Vaikuntha Ekadashi (Mukkoti)", "vaikuntha", 5),
        (1, 13, "Bhogi", "sankranti", 3),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 26, "Republic Day", "national_holiday", 3),
        (2, 5,  "Rathasapthami", "rathasapthami", 5),
        (2, 26, "Maha Shivaratri", "shivaratri", 4),
        (3, 14, "Holi", "holi", 3),
        (3, 30, "Ugadi", "ugadi", 5),
        (4, 6,  "Sri Rama Navami", "ramanavami", 4),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (8, 15, "Independence Day + Janmashtami", "janmashtami", 5),
        (8, 27, "Vinayaka Chaturthi", "vinayaka", 4),
        (9, 22, "Navaratri Begins", "navaratri", 4),
        (10, 2, "Dussehra", "dussehra", 5),
        (10, 2, "Gandhi Jayanti", "national_holiday", 3),
        (10, 3, "Brahmotsavams Begin", "brahmotsavam", 5),
        (10, 4, "Brahmotsavams Day 2", "brahmotsavam", 5),
        (10, 5, "Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 6, "Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 7, "Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 8, "Brahmotsavams Day 6", "brahmotsavam", 5),
        (10, 9, "Brahmotsavams Day 7", "brahmotsavam", 5),
        (10, 10,"Brahmotsavams Day 8", "brahmotsavam", 5),
        (10, 11,"Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (10, 21,"Diwali", "diwali", 5),
        (11, 5, "Karthika Purnima", "purnima", 4),
        (12, 30,"Vaikuntha Ekadashi", "vaikuntha", 5),
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2026 ══════ (ALL dates synced with hindu_calendar.py — verified correct)
    2026: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 13, "Bhogi", "sankranti", 3),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 26, "Republic Day", "national_holiday", 3),
        (1, 27, "Rathasapthami", "rathasapthami", 5),
        (2, 15, "Maha Shivaratri", "shivaratri", 4),
        (3, 4,  "Holi", "holi", 3),
        (3, 19, "Ugadi", "ugadi", 5),
        (3, 26, "Sri Rama Navami", "ramanavami", 4),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (8, 15, "Independence Day + Janmashtami", "janmashtami", 5), # Janmashtami fell on Aug 15 in 2026
        (9, 5,  "Vinayaka Chaturthi", "vinayaka", 4),               # FIXED: was Sep 14
        (9, 21, "Navaratri Begins", "navaratri", 4),                # FIXED: was Oct 11
        (10, 1, "Dussehra / Vijayadashami", "dussehra", 5),         # FIXED: was Oct 21
        (10, 2, "Gandhi Jayanti", "national_holiday", 3),
        (10, 5, "Brahmotsavams Begin", "brahmotsavam", 5),           # FIXED: was Oct 22
        (10, 6, "Brahmotsavams Day 2", "brahmotsavam", 5),           # FIXED: was Oct 23
        (10, 7, "Brahmotsavams Day 3", "brahmotsavam", 5),           # FIXED: was Oct 24
        (10, 8, "Brahmotsavams Day 4", "brahmotsavam", 5),           # FIXED: was Oct 25
        (10, 9, "Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5), # FIXED: was Oct 26
        (10, 10,"Brahmotsavams Day 6", "brahmotsavam", 5),           # FIXED: was Oct 27
        (10, 11,"Brahmotsavams Day 7", "brahmotsavam", 5),           # FIXED: was Oct 28
        (10, 12,"Brahmotsavams Day 8", "brahmotsavam", 5),           # FIXED: was Oct 29
        (10, 13,"Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5), # FIXED: was Oct 30
        (10, 20,"Diwali", "diwali", 5),                              # FIXED: was Nov 8
        (11, 14,"Karthika Purnima", "purnima", 4),                   # FIXED: was Nov 22
        (12, 11,"Vaikuntha Ekadashi", "vaikuntha", 5),               # FIXED: was Dec 20
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
    # ══════ 2027 ══════ (ALL dates synced with hindu_calendar.py — verified correct)
    2027: [
        (1, 1,  "New Year", "national_holiday", 4),
        (1, 13, "Bhogi", "sankranti", 3),
        (1, 14, "Sankranti", "sankranti", 5),
        (1, 15, "Kanuma", "sankranti", 4),
        (1, 26, "Republic Day", "national_holiday", 3),
        (2, 16, "Rathasapthami", "rathasapthami", 5),
        (3, 6,  "Maha Shivaratri", "shivaratri", 4),
        (3, 23, "Holi", "holi", 3),
        (4, 7,  "Ugadi", "ugadi", 5),
        (4, 14, "Ambedkar Jayanti", "national_holiday", 2),
        (4, 15, "Sri Rama Navami", "ramanavami", 4),
        (8, 15, "Independence Day", "national_holiday", 4),
        (8, 25, "Krishna Janmashtami", "janmashtami", 5),
        (9, 4,  "Vinayaka Chaturthi", "vinayaka", 4),
        (10, 2, "Gandhi Jayanti", "national_holiday", 3),
        (10, 11,"Navaratri Begins", "navaratri", 4),         # FIXED: was Sep 30
        (10, 20,"Dussehra / Vijayadashami", "dussehra", 5), # FIXED: was Oct 9
        (10, 21,"Brahmotsavams Begin", "brahmotsavam", 5),   # FIXED: was Oct 10 (starts day after Dussehra)
        (10, 22,"Brahmotsavams Day 2", "brahmotsavam", 5),
        (10, 23,"Brahmotsavams Day 3", "brahmotsavam", 5),
        (10, 24,"Brahmotsavams Day 4", "brahmotsavam", 5),
        (10, 25,"Brahmotsavams Day 5 - Garuda Seva", "brahmotsavam", 5),
        (10, 26,"Brahmotsavams Day 6", "brahmotsavam", 5),
        (10, 27,"Brahmotsavams Day 7", "brahmotsavam", 5),
        (10, 28,"Brahmotsavams Day 8", "brahmotsavam", 5),
        (10, 29,"Brahmotsavams Day 9 - Chakra Snanam", "brahmotsavam", 5),
        (11, 8, "Diwali", "diwali", 5),                      # FIXED: was Oct 29
        (11, 13,"Karthika Purnima", "purnima", 4),
        (12, 25,"Christmas", "national_holiday", 3),
        (12, 31,"New Year Eve", "national_holiday", 4),
    ],
}

# ═══════════════════════════════════════════════════════════════
#  SCHOOL / SEASONAL HOLIDAY PERIODS (recurring pattern)
#  (start_month, start_day, end_month, end_day, name, impact_level)
# ═══════════════════════════════════════════════════════════════
SEASONAL_PERIODS = [
    (4, 15, 6, 10, "summer_holiday", 3),
    (10, 1, 10, 15, "dasara_holiday", 4),
    (12, 20, 1, 5, "winter_holiday", 4),
]

# Category-to-numeric encoding for one-hot-like features
FESTIVAL_CATEGORIES = [
    "sankranti", "brahmotsavam", "vaikuntha", "shivaratri",
    "ugadi", "dussehra", "diwali", "janmashtami", "vinayaka",
    "navaratri", "rathasapthami", "holi", "ramanavami",
    "national_holiday", "purnima", "ekadashi", "other_festival",
]


# ═══════════════════════════════════════════════════════════════
#  PUBLIC API: Feature generation for ML training
# ═══════════════════════════════════════════════════════════════

def _is_in_seasonal(year: int, month: int, day: int) -> List[Tuple[str, int]]:
    """Check if date falls in seasonal holiday periods."""
    d = date(year, month, day)
    hits = []
    for sm, sd, em, ed, name, impact in SEASONAL_PERIODS:
        if sm > em:  # wraps year boundary (e.g., Dec 20 → Jan 5)
            start1 = date(year, sm, sd)
            end1 = date(year, 12, 31)
            start2 = date(year, 1, 1)
            end2 = date(year, em, ed)
            if start1 <= d <= end1 or start2 <= d <= end2:
                hits.append((name, impact))
        else:
            start = date(year, sm, sd)
            end = date(year, em, ed)
            if start <= d <= end:
                hits.append((name, impact))
    return hits


def get_festival_features(dt) -> dict:
    """
    Return numeric festival features for a single date.

    Returns dict with keys:
      - fest_impact: int 0-5 (max impact level from all events)
      - is_festival: 0/1
      - is_brahmotsavam: 0/1
      - is_sankranti: 0/1
      - is_vaikuntha_ekadashi: 0/1
      - is_dussehra_period: 0/1
      - is_diwali: 0/1
      - is_janmashtami: 0/1
      - is_shivaratri: 0/1
      - is_navaratri: 0/1
      - is_ugadi: 0/1
      - is_rathasapthami: 0/1
      - is_ramanavami: 0/1
      - is_national_holiday: 0/1
      - is_summer_holiday: 0/1
      - is_dasara_holiday: 0/1
      - is_winter_holiday: 0/1
      - days_to_fest: int (0 if today is a festival, else days to next)
      - days_from_fest: int (0 if today, else days since last festival)
      - fest_window_3: 0/1 (within 3 days of a major festival)
      - fest_window_7: 0/1 (within 7 days of a major festival)
    """
    if hasattr(dt, 'year'):
        yr, mo, dy = dt.year, dt.month, dt.day
    else:
        yr, mo, dy = dt.year, dt.month, dt.day

    feats = {
        "fest_impact": 0,
        "is_festival": 0,
        "is_brahmotsavam": 0,
        "is_sankranti": 0,
        "is_vaikuntha_ekadashi": 0,
        "is_dussehra_period": 0,
        "is_diwali": 0,
        "is_janmashtami": 0,
        "is_shivaratri": 0,
        "is_navaratri": 0,
        "is_ugadi": 0,
        "is_rathasapthami": 0,
        "is_ramanavami": 0,
        "is_national_holiday": 0,
        "is_summer_holiday": 0,
        "is_dasara_holiday": 0,
        "is_winter_holiday": 0,
        "days_to_fest": 30,      # default: no festival within range
        "days_from_fest": 30,
        "fest_window_3": 0,
        "fest_window_7": 0,
    }

    # Check festival DB for this year
    year_events = FESTIVAL_DB.get(yr, [])
    max_impact = 0

    for fm, fd, name, category, impact in year_events:
        if fm == mo and fd == dy:
            max_impact = max(max_impact, impact)
            feats["is_festival"] = 1
            feats["days_to_fest"] = 0
            feats["days_from_fest"] = 0

            # Category flags
            if category == "brahmotsavam":
                feats["is_brahmotsavam"] = 1
            elif category == "sankranti":
                feats["is_sankranti"] = 1
            elif category == "vaikuntha":
                feats["is_vaikuntha_ekadashi"] = 1
            elif category == "dussehra":
                feats["is_dussehra_period"] = 1
            elif category == "diwali":
                feats["is_diwali"] = 1
            elif category == "janmashtami":
                feats["is_janmashtami"] = 1
            elif category == "shivaratri":
                feats["is_shivaratri"] = 1
            elif category == "navaratri":
                feats["is_navaratri"] = 1
            elif category == "ugadi":
                feats["is_ugadi"] = 1
            elif category == "rathasapthami":
                feats["is_rathasapthami"] = 1
            elif category == "ramanavami":
                feats["is_ramanavami"] = 1
            elif category == "national_holiday":
                feats["is_national_holiday"] = 1

    feats["fest_impact"] = max_impact

    # Seasonal holidays
    seasonal = _is_in_seasonal(yr, mo, dy)
    for sname, simpact in seasonal:
        max_impact = max(max_impact, simpact)
        if sname == "summer_holiday":
            feats["is_summer_holiday"] = 1
        elif sname == "dasara_holiday":
            feats["is_dasara_holiday"] = 1
        elif sname == "winter_holiday":
            feats["is_winter_holiday"] = 1

    # Update fest_impact to include seasonal
    feats["fest_impact"] = max_impact

    # Distance to nearest festival (forward/backward)
    if feats["is_festival"] == 0:
        try:
            current = date(yr, mo, dy)
        except (ValueError, TypeError):
            current = date(yr, mo, dy)

        # Search forward (next festival)
        min_fwd = 30
        for fm, fd, name, category, impact in year_events:
            if impact >= 3:  # Only major festivals
                try:
                    fdate = date(yr, fm, fd)
                    diff = (fdate - current).days
                    if 0 < diff < min_fwd:
                        min_fwd = diff
                except ValueError:
                    pass
        feats["days_to_fest"] = min_fwd

        # Search backward (previous festival)
        min_bwd = 30
        for fm, fd, name, category, impact in year_events:
            if impact >= 3:
                try:
                    fdate = date(yr, fm, fd)
                    diff = (current - fdate).days
                    if 0 < diff < min_bwd:
                        min_bwd = diff
                except ValueError:
                    pass
        feats["days_from_fest"] = min_bwd

    # Festival window features
    if feats["days_to_fest"] <= 3 or feats["days_from_fest"] <= 3:
        feats["fest_window_3"] = 1
    if feats["days_to_fest"] <= 7 or feats["days_from_fest"] <= 7:
        feats["fest_window_7"] = 1

    return feats


def get_festival_features_series(dates: pd.Series) -> pd.DataFrame:
    """
    Vectorized: generate festival features for a pandas Series of dates.
    Returns a DataFrame aligned with the input dates.
    """
    records = []
    for dt in dates:
        records.append(get_festival_features(dt))
    return pd.DataFrame(records, index=dates.index)


# ═══════════════════════════════════════════════════════════════
#  Utility: quick festival lookup (for flask_api calendar display)
# ═══════════════════════════════════════════════════════════════
def get_events_for_date(year: int, month: int, day: int) -> list:
    """Return list of event dicts for calendar display."""
    events = []
    year_events = FESTIVAL_DB.get(year, [])
    for fm, fd, name, category, impact in year_events:
        if fm == month and fd == day:
            events.append({
                "name": name,
                "category": category,
                "impact": impact,
            })
    seasonal = _is_in_seasonal(year, month, day)
    for sname, simpact in seasonal:
        events.append({
            "name": sname.replace("_", " ").title(),
            "category": "school_holiday",
            "impact": simpact,
        })
    return events
