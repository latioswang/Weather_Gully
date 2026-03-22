"""Aviation Weather API client — METAR observations (real-time).

Fetches actual temperature observations from airport weather stations.
These are the same stations used by Polymarket for resolution,
making this the ground-truth data source.
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .config import LOCATIONS
from .http_client import fetch_json

logger = logging.getLogger(__name__)

AVIATION_API_BASE = "https://aviationweather.gov/api/data/metar"
_USER_AGENT = "WeatherGully/1.0"

# Map Polymarket location names to ICAO station identifiers
STATION_MAP: dict[str, str] = {
    # US cities
    "NYC": "KLGA",           # LaGuardia Airport
    "Chicago": "KORD",       # O'Hare International
    "Seattle": "KSEA",       # Seattle-Tacoma International
    "Atlanta": "KATL",       # Hartsfield-Jackson
    "Dallas": "KDAL",        # Love Field (Polymarket resolution station)
    "Miami": "KMIA",         # Miami International
    # International cities
    "London": "EGLC",        # London City Airport
    "Paris": "LFPG",         # Charles de Gaulle
    "Seoul": "RKSI",         # Incheon International
    "Toronto": "CYYZ",       # Toronto Pearson
    "BuenosAires": "SAEZ",   # Ezeiza International
    "SaoPaulo": "SBGR",      # Guarulhos International
    "Ankara": "LTAC",        # Esenboğa International
    "Wellington": "NZWN",    # Wellington International
    # East Asia / Greater China
    "Wuhan": "ZHHH",        # Tianhe International
    "Shanghai": "ZSPD",     # Pudong International
    "Chongqing": "ZUCK",    # Jiangbei International
    "Shenzhen": "ZGSZ",     # Bao'an International
    "HongKong": "VHHH",     # Hong Kong International
    "Taipei": "RCTP",       # Taoyuan International
}



def _celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return round(c * 9.0 / 5.0 + 32.0, 1)


async def get_metar_observations(
    locations: list[str],
    hours: int = 24,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[str, list[dict]]:
    """Fetch METAR observations for multiple stations in a single API call.

    Args:
        locations: List of canonical location names (e.g. ``["NYC", "Chicago"]``).
        hours: Number of hours of observations to fetch (default 24).
        max_retries: Number of retries on transient failures.
        base_delay: Base delay for exponential backoff.

    Returns:
        ``{location: [{"time": str, "temp_f": float, "raw": str}, ...]}``
        sorted by time ascending within each location.
    """
    # Map locations to station IDs
    stations: dict[str, str] = {}  # station_id → location_name
    for loc in locations:
        station = STATION_MAP.get(loc)
        if station:
            stations[station] = loc
        else:
            logger.warning("No METAR station mapped for location: %s", loc)

    if not stations:
        return {}

    ids_param = ",".join(stations.keys())
    url = f"{AVIATION_API_BASE}?ids={ids_param}&format=json&hours={hours}"

    data = await fetch_json(url, max_retries=max_retries, base_delay=base_delay)
    if data is None:
        logger.error("Aviation API returned no data")
        return {}

    # Group observations by location
    result: dict[str, list[dict]] = {loc: [] for loc in locations if loc in stations.values()}

    for obs in data:
        station_id = obs.get("icaoId", "")
        loc_name = stations.get(station_id)
        if not loc_name:
            continue

        temp_c = obs.get("temp")
        obs_time = obs.get("reportTime") or obs.get("obsTime")

        if temp_c is None or obs_time is None:
            continue

        try:
            temp_c = float(temp_c)
        except (ValueError, TypeError):
            continue

        result[loc_name].append({
            "time": str(obs_time),
            "temp_f": _celsius_to_fahrenheit(temp_c),
            "temp_c": temp_c,
            "station": station_id,
            "raw": obs.get("rawOb", ""),
        })

    # Sort by time ascending
    for loc in result:
        result[loc].sort(key=lambda x: x["time"])

    total_obs = sum(len(v) for v in result.values())
    logger.info("METAR: %d observations across %d stations", total_obs, len(stations))
    return result


def _utc_to_local_date(obs_time: str, tz_name: str) -> str:
    """Convert a UTC observation time string to a local date string.

    Args:
        obs_time: UTC time string like ``"2025-01-16T03:00:00Z"``.
        tz_name: IANA timezone name (e.g. ``"America/New_York"``).

    Returns:
        Local date string ``"YYYY-MM-DD"``.
    """
    utc_dt = datetime.fromisoformat(obs_time.replace("Z", "+00:00"))
    local_dt = utc_dt.astimezone(ZoneInfo(tz_name))
    return local_dt.strftime("%Y-%m-%d")


def compute_daily_extremes(
    observations: list[dict],
    target_date: str,
    tz_name: str = "",
) -> dict | None:
    """Compute daily high/low from METAR observations for a specific date.

    Args:
        observations: List of observation dicts (from ``get_metar_observations``).
        target_date: Date string ``"YYYY-MM-DD"`` to filter observations (local date).
        tz_name: IANA timezone name for converting UTC obs times to local dates.
            When empty, falls back to comparing the UTC date prefix (legacy behaviour).

    Returns:
        ``{"high": float, "low": float, "obs_count": int, "latest_obs_time": str}``
        or ``None`` if no observations match the target date.
    """
    day_obs = []
    for obs in observations:
        obs_time = obs.get("time", "")
        if tz_name:
            try:
                local_date = _utc_to_local_date(obs_time, tz_name)
            except (ValueError, KeyError):
                local_date = obs_time[:10]
        else:
            local_date = obs_time[:10]
        if local_date == target_date:
            day_obs.append(obs)

    if not day_obs:
        return None

    temps = [obs["temp_f"] for obs in day_obs]
    latest_time = max(obs["time"] for obs in day_obs)

    return {
        "high": max(temps),
        "low": min(temps),
        "obs_count": len(day_obs),
        "latest_obs_time": latest_time,
    }


async def get_aviation_daily_data(
    locations: list[str],
    hours: int = 24,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[str, dict[str, dict]]:
    """High-level: fetch METAR observations and compute daily extremes.

    Returns:
        ``{location: {date_str: {"obs_high": float, "obs_low": float,
        "obs_count": int, "latest_obs_time": str}}}``
    """
    all_obs = await get_metar_observations(
        locations, hours=hours,
        max_retries=max_retries, base_delay=base_delay,
    )

    result: dict[str, dict[str, dict]] = {}

    for loc, observations in all_obs.items():
        if not observations:
            continue

        # Resolve timezone for this location
        loc_info = LOCATIONS.get(loc, {})
        tz_name = loc_info.get("tz", "")

        # Collect all unique *local* dates from observations
        local_dates: set[str] = set()
        for obs in observations:
            obs_time = obs.get("time", "")
            if tz_name:
                try:
                    local_dates.add(_utc_to_local_date(obs_time, tz_name))
                except (ValueError, KeyError):
                    local_dates.add(obs_time[:10])
            else:
                local_dates.add(obs_time[:10])
        dates = sorted(local_dates)
        loc_data: dict[str, dict] = {}

        for date_str in dates:
            extremes = compute_daily_extremes(observations, date_str, tz_name=tz_name)
            if extremes:
                loc_data[date_str] = {
                    "obs_high": extremes["high"],
                    "obs_low": extremes["low"],
                    "obs_count": extremes["obs_count"],
                    "latest_obs_time": extremes["latest_obs_time"],
                }

        if loc_data:
            result[loc] = loc_data

    logger.info("Aviation daily data: %d locations with observations", len(result))
    return result
