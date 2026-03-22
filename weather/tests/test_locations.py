"""Tests validating LOCATIONS config against Polymarket resolution sources.

Ensures METAR stations, coordinates, units, and Open-Meteo model assignments
are correct and consistent with Polymarket market resolution criteria.
"""

import unittest

from weather.config import LOCATIONS


# Ground truth: Polymarket resolution sources (Weather Underground stations)
# Verified from live Polymarket market pages on 2026-02-16
POLYMARKET_STATIONS = {
    # US cities (°F, NOAA available)
    "NYC":         {"station": "KLGA", "unit": "F", "name_contains": "LaGuardia"},
    "Chicago":     {"station": "KORD", "unit": "F", "name_contains": "O'Hare"},
    "Seattle":     {"station": "KSEA", "unit": "F", "name_contains": "Sea-Tac"},
    "Atlanta":     {"station": "KATL", "unit": "F", "name_contains": "Hartsfield"},
    "Dallas":      {"station": "KDAL", "unit": "F", "name_contains": "Love Field"},
    "Miami":       {"station": "KMIA", "unit": "F", "name_contains": "MIA"},
    # International cities (°C, no NOAA)
    "London":      {"station": "EGLC", "unit": "C", "name_contains": "City Airport"},
    "Paris":       {"station": "LFPG", "unit": "C", "name_contains": "CDG"},
    "Seoul":       {"station": "RKSI", "unit": "C", "name_contains": "Incheon"},
    "Toronto":     {"station": "CYYZ", "unit": "C", "name_contains": "Pearson"},
    "BuenosAires": {"station": "SAEZ", "unit": "C", "name_contains": "Ezeiza"},
    "SaoPaulo":    {"station": "SBGR", "unit": "C", "name_contains": "Guarulhos"},
    "Ankara":      {"station": "LTAC", "unit": "C", "name_contains": "Esenbo"},
    "Wellington":  {"station": "NZWN", "unit": "C", "name_contains": "Wellington"},
    # East Asia / Greater China (°C, Wunderground / NOAA resolution)
    "Wuhan":       {"station": "ZHHH", "unit": "C", "name_contains": "Tianhe"},
    "Shanghai":    {"station": "ZSPD", "unit": "C", "name_contains": "Pudong"},
    "Chongqing":   {"station": "ZUCK", "unit": "C", "name_contains": "Jiangbei"},
    "Shenzhen":    {"station": "ZGSZ", "unit": "C", "name_contains": "Bao'an"},
    "HongKong":    {"station": "VHHH", "unit": "C", "name_contains": "Hong Kong"},
    "Taipei":      {"station": "RCTP", "unit": "C", "name_contains": "Taoyuan"},
}

# Approximate airport coordinates (lat, lon) for sanity check (within 0.1°)
AIRPORT_COORDS = {
    "KLGA": (40.77, -73.87),
    "KORD": (41.97, -87.91),
    "KSEA": (47.45, -122.31),
    "KATL": (33.64, -84.43),
    "KDAL": (32.85, -96.85),
    "KMIA": (25.80, -80.29),
    "EGLC": (51.51, 0.05),
    "LFPG": (49.01, 2.55),
    "RKSI": (37.46, 126.44),
    "CYYZ": (43.68, -79.63),
    "SAEZ": (-34.82, -58.54),
    "SBGR": (-23.44, -46.47),
    "LTAC": (40.13, 32.99),
    "NZWN": (-41.33, 174.81),
    # East Asia / Greater China
    "ZHHH": (30.78, 114.21),
    "ZSPD": (31.14, 121.81),
    "ZUCK": (29.72, 106.64),
    "ZGSZ": (22.64, 113.81),
    "VHHH": (22.31, 113.91),
    "RCTP": (25.08, 121.23),
}


class TestLocationsComplete(unittest.TestCase):
    """Verify all Polymarket cities are configured."""

    def test_all_polymarket_cities_present(self):
        for city in POLYMARKET_STATIONS:
            self.assertIn(city, LOCATIONS, f"Missing Polymarket city: {city}")

    def test_no_extra_cities(self):
        for city in LOCATIONS:
            self.assertIn(city, POLYMARKET_STATIONS,
                          f"City {city} in config but not on Polymarket")


class TestStationAlignment(unittest.TestCase):
    """Verify METAR station codes match Polymarket resolution sources."""

    def test_all_stations_match_polymarket(self):
        for city, expected in POLYMARKET_STATIONS.items():
            loc = LOCATIONS[city]
            self.assertEqual(
                loc["station"], expected["station"],
                f"{city}: station mismatch — got {loc['station']}, "
                f"expected {expected['station']} (Polymarket resolution source)",
            )

    def test_dallas_is_love_field_not_dfw(self):
        """Regression: Polymarket Dallas resolves via KDAL (Love Field), not KDFW (DFW)."""
        self.assertEqual(LOCATIONS["Dallas"]["station"], "KDAL")
        self.assertNotEqual(LOCATIONS["Dallas"]["station"], "KDFW")


class TestUnitAlignment(unittest.TestCase):
    """Verify temperature unit matches Polymarket resolution criteria."""

    def test_us_cities_fahrenheit(self):
        us_cities = ["NYC", "Chicago", "Seattle", "Atlanta", "Dallas", "Miami"]
        for city in us_cities:
            self.assertEqual(LOCATIONS[city]["unit"], "F",
                             f"{city} should resolve in °F")

    def test_international_cities_celsius(self):
        intl_cities = ["London", "Paris", "Seoul", "Toronto",
                       "BuenosAires", "SaoPaulo", "Ankara", "Wellington",
                       "Wuhan", "Shanghai", "Chongqing", "Shenzhen",
                       "HongKong", "Taipei"]
        for city in intl_cities:
            self.assertEqual(LOCATIONS[city]["unit"], "C",
                             f"{city} should resolve in °C")


class TestCoordinateAlignment(unittest.TestCase):
    """Verify coordinates are near the resolution airport (within 0.15°)."""

    def test_coordinates_near_airport(self):
        for city, loc in LOCATIONS.items():
            station = loc["station"]
            expected_lat, expected_lon = AIRPORT_COORDS[station]
            self.assertAlmostEqual(
                loc["lat"], expected_lat, delta=0.15,
                msg=f"{city} lat mismatch: {loc['lat']} vs {expected_lat} ({station})",
            )
            self.assertAlmostEqual(
                loc["lon"], expected_lon, delta=0.15,
                msg=f"{city} lon mismatch: {loc['lon']} vs {expected_lon} ({station})",
            )


class TestRequiredFields(unittest.TestCase):
    """Verify every location has all required fields."""

    REQUIRED = {"lat", "lon", "name", "tz", "station", "unit"}

    def test_all_required_fields_present(self):
        for city, loc in LOCATIONS.items():
            for field in self.REQUIRED:
                self.assertIn(field, loc, f"{city} missing required field: {field}")

    def test_station_is_valid_icao(self):
        """ICAO codes are 4 uppercase letters."""
        for city, loc in LOCATIONS.items():
            station = loc["station"]
            self.assertEqual(len(station), 4,
                             f"{city} station {station} is not 4 chars")
            self.assertTrue(station.isalpha() and station.isupper(),
                            f"{city} station {station} is not uppercase alpha")

    def test_timezone_format(self):
        """Timezone should be IANA format (Area/City)."""
        for city, loc in LOCATIONS.items():
            tz = loc["tz"]
            self.assertIn("/", tz, f"{city} timezone {tz} is not IANA format")


class TestLocalModelAssignment(unittest.TestCase):
    """Verify regional NWP model assignments are reasonable."""

    EXPECTED_MODELS = {
        "London": "icon_seamless",
        "Paris": "meteofrance_seamless",
        "Seoul": "kma_seamless",
        "Toronto": "gem_seamless",
        "Ankara": "icon_seamless",
        "Wuhan": "jma_seamless",
        "Shanghai": "jma_seamless",
        "Chongqing": "jma_seamless",
        "Shenzhen": "jma_seamless",
        "HongKong": "jma_seamless",
        "Taipei": "jma_seamless",
    }
    # Cities without local models (no regional NWP available)
    NO_LOCAL_MODEL = ["BuenosAires", "SaoPaulo", "Wellington"]

    def test_cities_with_local_model(self):
        for city, expected_model in self.EXPECTED_MODELS.items():
            loc = LOCATIONS[city]
            self.assertEqual(loc.get("local_model"), expected_model,
                             f"{city} should use {expected_model}")

    def test_cities_without_local_model(self):
        for city in self.NO_LOCAL_MODEL:
            loc = LOCATIONS[city]
            self.assertNotIn("local_model", loc,
                             f"{city} should not have a local_model")

    def test_us_cities_no_local_model(self):
        """US cities use NOAA — they don't need a local Open-Meteo model."""
        us_cities = ["NYC", "Chicago", "Seattle", "Atlanta", "Dallas", "Miami"]
        for city in us_cities:
            self.assertNotIn("local_model", LOCATIONS[city],
                             f"US city {city} should not have local_model")


class TestPipelineSimulation(unittest.TestCase):
    """Simulate the full scoring pipeline for each city to verify integration."""

    def test_parse_bucket_fahrenheit_passthrough(self):
        """US city: bucket bounds pass through as-is."""
        from weather.strategy import _parse_bucket
        result = _parse_bucket("50-54", "NYC")
        self.assertIsNotNone(result)
        self.assertEqual(result, (50.0, 54.0))

    def test_parse_bucket_celsius_conversion(self):
        """International city: °C bucket bounds converted to °F."""
        from weather.strategy import _parse_bucket
        result = _parse_bucket("10-14", "London")
        self.assertIsNotNone(result)
        # 10°C = 50°F, 14°C = 57.2°F
        self.assertAlmostEqual(result[0], 50.0, places=1)
        self.assertAlmostEqual(result[1], 57.2, places=1)

    def test_probability_with_converted_bounds(self):
        """Full pipeline: probability estimate works with float bucket bounds."""
        from weather.probability import estimate_bucket_probability
        # Simulate London: forecast 12°C = 53.6°F, bucket 10-14°C = 50-57.2°F
        prob = estimate_bucket_probability(
            forecast_temp=53.6,
            bucket_low=50.0,
            bucket_high=57.2,
            forecast_date="2026-02-18",
        )
        self.assertGreater(prob, 0.1)  # Forecast is centered in bucket
        self.assertLess(prob, 1.0)

    def test_all_cities_have_valid_aviation_station(self):
        """Every configured station should produce a valid METAR API URL."""
        for city, loc in LOCATIONS.items():
            station = loc["station"]
            url = f"https://aviationweather.gov/api/data/metar?ids={station}&format=json&hours=24"
            self.assertIn(station, url)
            # Station must be valid ICAO
            self.assertTrue(len(station) == 4 and station.isalpha())

    def test_all_cities_open_meteo_coordinates(self):
        """All coordinates should be valid for Open-Meteo API."""
        for city, loc in LOCATIONS.items():
            lat, lon = loc["lat"], loc["lon"]
            self.assertGreaterEqual(lat, -90.0, f"{city} lat out of range")
            self.assertLessEqual(lat, 90.0, f"{city} lat out of range")
            self.assertGreaterEqual(lon, -180.0, f"{city} lon out of range")
            self.assertLessEqual(lon, 180.0, f"{city} lon out of range")

    def test_ensemble_with_local_model(self):
        """Local model weight is included in ensemble for international cities."""
        from weather.open_meteo import compute_ensemble_forecast
        om_data = {
            "ecmwf_high": 55.0, "ecmwf_low": 45.0,
            "gfs_high": 54.0, "gfs_low": 44.0,
            "local_high": 56.0, "local_low": 46.0,
        }
        high, spread = compute_ensemble_forecast(None, om_data, "high")
        low, _ = compute_ensemble_forecast(None, om_data, "low")
        # Local model (56) should pull high above ECMWF+GFS average (54.6)
        self.assertGreater(high, 54.5)
        self.assertIsNotNone(low)
        self.assertGreater(spread, 0.0)

    def test_ensemble_without_local_model(self):
        """US cities: no local model, ensemble uses ECMWF + GFS + NOAA."""
        from weather.open_meteo import compute_ensemble_forecast
        om_data = {
            "ecmwf_high": 55.0, "ecmwf_low": 45.0,
            "gfs_high": 54.0, "gfs_low": 44.0,
        }
        high, _ = compute_ensemble_forecast(53.0, om_data, "high")
        # With NOAA=53, GFS=54, ECMWF=55, result is weighted average
        self.assertGreater(high, 53.0)
        self.assertLess(high, 56.0)


if __name__ == "__main__":
    unittest.main()
