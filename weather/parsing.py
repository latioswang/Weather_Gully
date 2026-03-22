"""Parsing helpers for weather event names and temperature buckets."""

import logging
import re
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Location aliases → canonical key
LOCATION_ALIASES = {
    # US cities
    "nyc": "NYC", "new york": "NYC", "laguardia": "NYC", "la guardia": "NYC",
    "chicago": "Chicago", "o'hare": "Chicago", "ohare": "Chicago",
    "seattle": "Seattle", "sea-tac": "Seattle", "seatac": "Seattle",
    "atlanta": "Atlanta", "hartsfield": "Atlanta",
    "dallas": "Dallas", "dfw": "Dallas",
    "miami": "Miami",
    # International cities
    "london": "London", "heathrow": "London", "lhr": "London",
    "paris": "Paris", "cdg": "Paris",
    "seoul": "Seoul", "incheon": "Seoul",
    "toronto": "Toronto", "pearson": "Toronto",
    "buenos aires": "BuenosAires", "buenosaires": "BuenosAires", "ezeiza": "BuenosAires",
    "sao paulo": "SaoPaulo", "saopaulo": "SaoPaulo", "são paulo": "SaoPaulo",
    "ankara": "Ankara", "esenboga": "Ankara",
    "wellington": "Wellington",
    # East Asia / Greater China
    "wuhan": "Wuhan", "tianhe": "Wuhan",
    "shanghai": "Shanghai", "pudong": "Shanghai",
    "chongqing": "Chongqing", "jiangbei": "Chongqing",
    "shenzhen": "Shenzhen", "bao'an": "Shenzhen", "baoan": "Shenzhen",
    "hong kong": "HongKong", "hongkong": "HongKong",
    "taipei": "Taipei", "taoyuan": "Taipei",
}

MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}


def parse_weather_event(event_name: str) -> dict | None:
    """Parse weather event name to extract location, date, metric.

    Returns ``{"location": str, "date": "YYYY-MM-DD", "metric": "high"|"low"}``
    or ``None`` if the event cannot be parsed.
    """
    if not event_name:
        return None

    event_lower = event_name.lower()

    # Determine metric
    if "lowest" in event_lower or "low temp" in event_lower:
        metric = "low"
    else:
        metric = "high"

    # Determine location
    location = None
    for alias, loc in LOCATION_ALIASES.items():
        if alias in event_lower:
            location = loc
            break

    if not location:
        return None

    # Extract date — expect "on <Month> <day>"
    month_day_match = re.search(r"on\s+([a-zA-Z]+)\s+(\d{1,2})", event_name, re.IGNORECASE)
    if not month_day_match:
        return None

    month_name = month_day_match.group(1).lower()
    day = int(month_day_match.group(2))

    month = MONTH_MAP.get(month_name)
    if not month:
        return None

    now = datetime.now(timezone.utc)
    year = now.year
    try:
        target_date = datetime(year, month, day, tzinfo=timezone.utc)
        # If the date is more than 30 days in the past, assume next year
        if target_date < now - timedelta(days=30):
            year += 1
            # Validate the date is valid in the new year (e.g. Feb 29 → non-leap year)
            datetime(year, month, day, tzinfo=timezone.utc)
        date_str = f"{year}-{month:02d}-{day:02d}"
    except ValueError:
        return None

    return {"location": location, "date": date_str, "metric": metric}


def parse_temperature_bucket(outcome_name: str) -> tuple[int, int] | None:
    """Parse temperature bucket from outcome name.

    Returns ``(low_bound, high_bound)`` or ``None``.
    Open-ended buckets use -999 / 999 as sentinel values.
    """
    if not outcome_name:
        return None

    # "X or below / or less"
    below_match = re.search(
        r"(-?\d+)\s*°?[fF]?\s*(?:or below|or less|and below|and under)",
        outcome_name,
        re.IGNORECASE,
    )
    if below_match:
        return (-999, int(below_match.group(1)))

    # "X or higher / or above / or more"
    above_match = re.search(
        r"(-?\d+)\s*°?[fF]?\s*(?:or higher|or above|or more|and above|and over|\+)",
        outcome_name,
        re.IGNORECASE,
    )
    if above_match:
        return (int(above_match.group(1)), 999)

    # "X - Y" or "X to Y"
    range_match = re.search(r"(-?\d+)\s*(?:[-\u2013]|to)\s*(-?\d+)", outcome_name)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        return (min(low, high), max(low, high))

    return None
