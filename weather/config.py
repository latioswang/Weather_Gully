"""Configuration management — dataclass-based, CLOB direct."""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported locations (matching Polymarket resolution sources)
# US cities: resolve via NOAA/Weather Underground in °F
# International cities: resolve via Weather Underground in °C
LOCATIONS = {
    # --- US (°F, NOAA available) ---
    "NYC": {"lat": 40.7769, "lon": -73.8740, "name": "New York City (LaGuardia)", "tz": "America/New_York", "station": "KLGA", "unit": "F"},
    "Chicago": {"lat": 41.9742, "lon": -87.9073, "name": "Chicago (O'Hare)", "tz": "America/Chicago", "station": "KORD", "unit": "F"},
    "Seattle": {"lat": 47.4502, "lon": -122.3088, "name": "Seattle (Sea-Tac)", "tz": "America/Los_Angeles", "station": "KSEA", "unit": "F"},
    "Atlanta": {"lat": 33.6407, "lon": -84.4277, "name": "Atlanta (Hartsfield)", "tz": "America/New_York", "station": "KATL", "unit": "F"},
    "Dallas": {"lat": 32.8471, "lon": -96.8518, "name": "Dallas (Love Field)", "tz": "America/Chicago", "station": "KDAL", "unit": "F"},
    "Miami": {"lat": 25.7959, "lon": -80.2870, "name": "Miami (MIA)", "tz": "America/New_York", "station": "KMIA", "unit": "F"},
    # --- International (°C, no NOAA) ---
    # local_model: Open-Meteo regional NWP model (higher resolution than GFS/ECMWF)
    "London": {"lat": 51.5053, "lon": 0.0553, "name": "London (City Airport)", "tz": "Europe/London", "station": "EGLC", "unit": "C", "local_model": "icon_seamless"},
    "Paris": {"lat": 49.0097, "lon": 2.5479, "name": "Paris (CDG)", "tz": "Europe/Paris", "station": "LFPG", "unit": "C", "local_model": "meteofrance_seamless"},
    "Seoul": {"lat": 37.4602, "lon": 126.4407, "name": "Seoul (Incheon)", "tz": "Asia/Seoul", "station": "RKSI", "unit": "C", "local_model": "kma_seamless"},
    "Toronto": {"lat": 43.6777, "lon": -79.6248, "name": "Toronto (Pearson)", "tz": "America/Toronto", "station": "CYYZ", "unit": "C", "local_model": "gem_seamless"},
    "BuenosAires": {"lat": -34.8222, "lon": -58.5358, "name": "Buenos Aires (Ezeiza)", "tz": "America/Argentina/Buenos_Aires", "station": "SAEZ", "unit": "C"},
    "SaoPaulo": {"lat": -23.4356, "lon": -46.4731, "name": "São Paulo (Guarulhos)", "tz": "America/Sao_Paulo", "station": "SBGR", "unit": "C"},
    "Ankara": {"lat": 40.1281, "lon": 32.9951, "name": "Ankara (Esenboğa)", "tz": "Europe/Istanbul", "station": "LTAC", "unit": "C", "local_model": "icon_seamless"},
    "Wellington": {"lat": -41.3272, "lon": 174.8053, "name": "Wellington (Intl)", "tz": "Pacific/Auckland", "station": "NZWN", "unit": "C"},
    # --- East Asia / Greater China (°C) ---
    "Wuhan": {"lat": 30.7838, "lon": 114.2081, "name": "Wuhan (Tianhe)", "tz": "Asia/Shanghai", "station": "ZHHH", "unit": "C", "local_model": "jma_seamless"},
    "Shanghai": {"lat": 31.1434, "lon": 121.8052, "name": "Shanghai (Pudong)", "tz": "Asia/Shanghai", "station": "ZSPD", "unit": "C", "local_model": "jma_seamless"},
    "Chongqing": {"lat": 29.7192, "lon": 106.6413, "name": "Chongqing (Jiangbei)", "tz": "Asia/Shanghai", "station": "ZUCK", "unit": "C", "local_model": "jma_seamless"},
    "Shenzhen": {"lat": 22.6393, "lon": 113.8107, "name": "Shenzhen (Bao'an)", "tz": "Asia/Shanghai", "station": "ZGSZ", "unit": "C", "local_model": "jma_seamless"},
    "HongKong": {"lat": 22.3089, "lon": 113.9146, "name": "Hong Kong (VHHH)", "tz": "Asia/Hong_Kong", "station": "VHHH", "unit": "C", "local_model": "jma_seamless"},
    "Taipei": {"lat": 25.0777, "lon": 121.2325, "name": "Taipei (Taoyuan)", "tz": "Asia/Taipei", "station": "RCTP", "unit": "C", "local_model": "jma_seamless"},
}

# Polymarket constraints
MIN_SHARES_PER_ORDER = 2.0
MIN_TICK_SIZE = 0.01

# Environment variable mapping
_ENV_MAP = {
    "entry_threshold": "WEATHER_ENTRY_THRESHOLD",
    "exit_threshold": "WEATHER_EXIT_THRESHOLD",
    "max_position_usd": "WEATHER_MAX_POSITION",
    "sizing_pct": "WEATHER_SIZING_PCT",
    "max_trades_per_run": "WEATHER_MAX_TRADES",
    "locations": "WEATHER_LOCATIONS",
    "private_key": "POLY_PRIVATE_KEY",
}


@dataclass
class Config:
    # Auth (CLOB direct)
    private_key: str = ""
    creds_file: str = "creds.json"

    # Entry / exit
    entry_threshold: float = 0.15
    exit_threshold: float = 0.45
    max_position_usd: float = 5.00
    sizing_pct: float = 0.05

    # Kelly
    kelly_fraction: float = 0.25
    min_ev_threshold: float = 0.03
    min_probability: float = 0.25
    max_bucket_distance_sigma: float = 2.5

    # Rate limiting
    max_trades_per_run: int = 5

    # Retry
    max_retries: int = 3
    retry_base_delay: float = 1.0

    # Locations
    locations: str = "NYC"

    # Logging
    log_level: str = "INFO"

    # State
    state_file: str = "weather_state.json"

    # Strategy toggles
    max_days_ahead: int = 7
    seasonal_adjustments: bool = True
    adjacent_buckets: bool = True
    dynamic_exits: bool = True

    # Context safeguards
    slippage_max_pct: float = 0.25
    time_to_resolution_min_hours: int = 2

    # Multi-source forecasting (Open-Meteo)
    multi_source: bool = True

    # Forecast change detection
    forecast_change_threshold: float = 3.0  # °F change to trigger re-evaluation

    # Correlation guard (max 1 position per event)
    correlation_guard: bool = True

    # Stop-loss on forecast reversal
    stop_loss_reversal: bool = True
    stop_loss_reversal_threshold: float = 5.0  # °F shift away from our bucket

    # Max exposure for portfolio calculation
    max_exposure: float = 50.0

    # Aviation Weather observations (METAR)
    aviation_obs: bool = True
    aviation_obs_weight: float = 0.40
    aviation_hours: int = 24

    # Trading fees (Polymarket ~2% on gains)
    trading_fees: float = 0.02

    # Adaptive sigma (ensemble-based)
    adaptive_sigma: bool = True

    # Execution: fill verification
    fill_timeout_seconds: float = 30.0
    fill_poll_interval: float = 2.0

    # Circuit breaker (risk management)
    daily_loss_limit: float = 10.0           # Stop trading after $X daily loss
    max_positions_per_day: int = 20           # Max new positions per calendar day
    cooldown_hours_after_max_loss: float = 24.0  # Hours to wait after circuit break
    max_open_positions: int = 15              # Max simultaneous open positions

    # Inter-location correlation (sizing reduction)
    correlation_threshold: float = 0.3    # Ignore correlations below this
    correlation_discount: float = 0.5     # How much to reduce sizing (0=ignore, 1=full)

    # AR(1) autocorrelation correction in feedback bias
    ar_autocorrelation: bool = True

    # Kalman filter for dynamic sigma estimation
    kalman_sigma: bool = True

    # Mean-reversion timing (price Z-score sizing modifier)
    mean_reversion: bool = True

    # Adaptive execution (VWAP, depth sizing, edge-proportional slippage)
    slippage_edge_ratio: float = 0.8
    depth_fill_ratio: float = 0.5
    vwap_max_levels: int = 5

    # Low temperature markets
    trade_metrics: str = "high"

    # Portfolio: same-location sizing discount
    same_location_discount: float = 0.7
    same_location_horizon_window: int = 2

    # Model disagreement (NOAA vs Open-Meteo)
    model_disagreement_threshold: float = 3.0   # °F — boost sigma when models diverge more than this
    model_disagreement_multiplier: float = 1.5  # sigma multiplier when disagreement detected

    # Maker orders (hybrid taker/maker execution)
    maker_edge_threshold: float = 0.05     # Edge below this → use maker
    maker_spread_threshold: float = 0.10   # Spread above this → use maker
    maker_ttl_seconds: int = 900           # 15 min before cancel
    maker_tick_buffer: int = 1             # Ticks above best bid

    # Weather Underground
    wu_api_key: str = ""            # From creds.json or env var
    wu_cache_ttl: int = 3600        # 60 min cache

    # NOAA caching
    noaa_cache_ttl: int = 900       # 15 min cache

    # Dynamic model weighting
    dynamic_weights: bool = True    # Use performance-based weights
    wu_weight_bonus: float = 0.20   # +20% bonus for WU (resolution source)

    # International sigma boost (no NOAA → single source → higher uncertainty)
    international_sigma_boost: float = 1.3

    @property
    def active_locations(self) -> list[str]:
        """Return canonical location keys matching LOCATIONS dict keys.

        Input is case-insensitive: ``"nyc,chicago"`` → ``["NYC", "Chicago"]``.
        """
        canonical = {k.lower(): k for k in LOCATIONS}
        result = []
        for raw in self.locations.split(","):
            raw = raw.strip()
            if not raw:
                continue
            canon = canonical.get(raw.lower())
            if canon:
                result.append(canon)
            else:
                result.append(raw)  # Pass through unknown locations as-is
        return result

    @property
    def active_metrics(self) -> list[str]:
        """Return list of metrics to trade: ['high'], ['low'], or ['high', 'low']."""
        return [m.strip().lower() for m in self.trade_metrics.split(",") if m.strip()]

    @classmethod
    def load(cls, config_dir: str) -> "Config":
        """Load config with priority: config.json > env vars > defaults."""
        config_path = Path(config_dir) / "config.json"
        file_cfg: dict = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    file_cfg = json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Failed to load %s: %s", config_path, exc)

        kwargs: dict = {}
        field_types = {f.name: f.type for f in fields(cls)}

        for f in fields(cls):
            name = f.name
            # Priority 1: config.json
            if name in file_cfg:
                kwargs[name] = _coerce(file_cfg[name], field_types[name])
            # Priority 2: env vars
            elif name in _ENV_MAP:
                env_val = os.environ.get(_ENV_MAP[name])
                if env_val is not None:
                    kwargs[name] = _coerce(env_val, field_types[name])
            # else: default from dataclass

        return cls(**kwargs)

    def save(self, config_dir: str) -> None:
        """Persist current config to config.json (atomic write)."""
        config_path = Path(config_dir) / "config.json"
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data.pop("private_key", None)  # Never persist the private key
        fd, tmp = tempfile.mkstemp(dir=config_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, str(config_path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.info("Config saved to %s", config_path)

    def update(self, overrides: dict) -> None:
        """Apply key=value overrides (from --set CLI flag)."""
        field_types = {f.name: f.type for f in fields(self)}
        for key, value in overrides.items():
            if key not in field_types:
                logger.warning("Unknown config key: %s", key)
                continue
            setattr(self, key, _coerce(value, field_types[key]))

    def load_api_creds(self, config_dir: str) -> dict | None:
        """Load API creds from creds_file if it exists."""
        for path in [self.creds_file, str(Path(config_dir) / self.creds_file)]:
            if path and os.path.exists(path):
                try:
                    with open(path) as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError) as exc:
                    logger.warning("Failed to load API creds from %s: %s", path, exc)
        return None


def _coerce(value, type_hint: str):
    """Coerce a value to the declared field type."""
    if type_hint == "bool" or type_hint is bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes")
    if type_hint == "int" or type_hint is int:
        return int(value)
    if type_hint == "float" or type_hint is float:
        return float(value)
    return str(value)
