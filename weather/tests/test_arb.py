"""Tests for weather.arb — arbitrage scanner logic."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from weather.arb import (
    _best_price,
    _is_temperature_event,
    _parse_temp_event,
    scan_arbs,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ------------------------------------------------------------------
# Unit tests for helpers
# ------------------------------------------------------------------

class TestIsTemperatureEvent:
    def test_standard_highest_temp(self):
        assert _is_temperature_event("Highest temperature in London on March 23?")

    def test_case_insensitive(self):
        assert _is_temperature_event("highest temperature in NYC on March 25?")

    def test_non_temperature(self):
        assert not _is_temperature_event("Named storm forms before hurricane season?")
        assert not _is_temperature_event("How many 7.0 or above earthquakes in 2026?")
        assert not _is_temperature_event("5kt meteor strike in 2026?")

    def test_precipitation(self):
        assert not _is_temperature_event("Precipitation in Seattle in March?")

    def test_hottest_year(self):
        assert not _is_temperature_event("Where will 2026 rank among the hottest years on record?")


class TestParseTempEvent:
    def test_standard(self):
        result = _parse_temp_event("Highest temperature in London on March 23?")
        assert result == {"city": "London", "date_str": "March 23"}

    def test_multi_word_city(self):
        result = _parse_temp_event("Highest temperature in New York on March 25?")
        assert result == {"city": "New York", "date_str": "March 25"}

    def test_non_temp_returns_none(self):
        assert _parse_temp_event("5kt meteor strike?") is None


class TestBestPrice:
    def test_normal_asks(self):
        book = {"asks": [{"price": "0.42"}, {"price": "0.50"}], "bids": []}
        assert _best_price(book, "asks") == 0.42

    def test_normal_bids(self):
        book = {"asks": [], "bids": [{"price": "0.38"}, {"price": "0.30"}]}
        assert _best_price(book, "bids") == 0.38

    def test_empty_book(self):
        assert _best_price({"asks": [], "bids": []}, "asks") is None

    def test_none_book(self):
        assert _best_price(None, "asks") is None


# ------------------------------------------------------------------
# Fixtures for scan_arbs tests
# ------------------------------------------------------------------

def _make_event_raw(event_id, slug, title, markets_raw):
    """Build a raw Gamma event dict (as returned by /events endpoint)."""
    return {
        "id": event_id,
        "slug": slug,
        "title": title,
        "markets": markets_raw,
    }


def _make_raw_market(condition_id, group_item_title, uma_status=None):
    """Build a raw Gamma market dict (embedded in event).

    uma_status: None, "proposed", "resolved", etc.
    """
    raw = {
        "conditionId": condition_id,
        "groupItemTitle": group_item_title,
    }
    if uma_status is not None:
        raw["umaResolutionStatus"] = uma_status
    return raw


def _make_bridge_market(market_id, event_id, event_name, outcome_name,
                        token_yes="tok_yes", token_no="tok_no",
                        best_bid=0.0, best_ask=0.0):
    """Build a market dict as returned by bridge.fetch_weather_markets()."""
    return {
        "id": market_id,
        "event_id": event_id,
        "event_name": event_name,
        "outcome_name": outcome_name,
        "external_price_yes": 0.5,
        "token_id_yes": token_yes,
        "token_id_no": token_no,
        "end_date": "2026-03-25T00:00:00Z",
        "best_bid": best_bid,
        "best_ask": best_ask,
        "status": "active",
    }


def _make_orderbook(best_ask=0.0, best_bid=0.0):
    """Build a simple orderbook dict."""
    asks = [{"price": str(best_ask), "size": "100"}] if best_ask > 0 else []
    bids = [{"price": str(best_bid), "size": "100"}] if best_bid > 0 else []
    return {"asks": asks, "bids": bids}


def _build_clob_mock(orderbooks: dict[str, dict]):
    """Build a mock PublicClient that returns pre-defined orderbooks by token_id."""
    clob = AsyncMock()

    async def get_orderbook(token_id):
        return orderbooks.get(token_id, {"asks": [], "bids": []})

    clob.get_orderbook = get_orderbook
    return clob


# ------------------------------------------------------------------
# In-review filtering tests
# ------------------------------------------------------------------

class TestInReviewFiltering:
    """Markets with umaResolutionStatus='proposed' should be excluded."""

    def _setup_seattle_event(self, in_review_outcomes: list[str]):
        """Set up a Seattle temp event where specified outcomes are in review.

        Returns (bridge_mock, clob_mock, events_raw).
        """
        event_id = "285768"
        event_name = "Highest temperature in Seattle on March 23?"
        slug = "highest-temperature-in-seattle-on-march-23-2026"

        outcomes = [
            ("41°F or below", "cid_41", "tok_yes_41", "tok_no_41"),
            ("42-43°F", "cid_42", "tok_yes_42", "tok_no_42"),
            ("44-45°F", "cid_44", "tok_yes_44", "tok_no_44"),
            ("46-47°F", "cid_46", "tok_yes_46", "tok_no_46"),
            ("48-49°F", "cid_48", "tok_yes_48", "tok_no_48"),
        ]

        # Build raw Gamma event with umaResolutionStatus
        raw_markets = []
        for name, cid, _, _ in outcomes:
            uma = "proposed" if name in in_review_outcomes else None
            raw_markets.append(_make_raw_market(cid, name, uma_status=uma))

        events_raw = [_make_event_raw(event_id, slug, event_name, raw_markets)]

        # Build bridge markets
        bridge_markets = []
        for name, cid, tok_yes, tok_no in outcomes:
            bridge_markets.append(_make_bridge_market(
                cid, event_id, event_name, name,
                token_yes=tok_yes, token_no=tok_no,
            ))

        bridge = AsyncMock()
        bridge.fetch_weather_markets = AsyncMock(return_value=bridge_markets)

        # All orderbooks: YES ask=0.10, NO ask=0.90 (sum of NO = 4.5 for 5 buckets, < 4 threshold → arb)
        # But for in-review ones, we still set prices — scanner should skip them
        orderbooks = {}
        for name, cid, tok_yes, tok_no in outcomes:
            orderbooks[tok_yes] = _make_orderbook(best_ask=0.10, best_bid=0.08)
            orderbooks[tok_no] = _make_orderbook(best_ask=0.90, best_bid=0.88)

        clob = _build_clob_mock(orderbooks)
        return bridge, clob, events_raw

    def test_in_review_buckets_excluded_from_arb(self):
        """Buckets with umaResolutionStatus='proposed' should be skipped entirely."""
        bridge, clob, events_raw = self._setup_seattle_event(
            in_review_outcomes=["41°F or below"],
        )

        arbs, scanned = _run(scan_arbs(bridge, clob, events_raw))

        assert scanned == 1
        # With 5 buckets and NO ask=0.90 each, sum=4.50, threshold=4 → arb
        # With 1 excluded, 4 buckets, NO ask sum=3.60, threshold=3 → arb still
        # The key check: no bucket in the arb should be "41°F or below"
        for arb in arbs:
            if "buckets" in arb:
                outcomes_in_arb = [b["outcome"] for b in arb["buckets"]]
                assert "41°F or below" not in outcomes_in_arb, \
                    "In-review bucket '41°F or below' should be excluded"

    def test_in_review_adjusts_bucket_count(self):
        """N should be recalculated after excluding in-review buckets."""
        bridge, clob, events_raw = self._setup_seattle_event(
            in_review_outcomes=["41°F or below"],
        )

        arbs, scanned = _run(scan_arbs(bridge, clob, events_raw))

        for arb in arbs:
            if arb["type"] in ("all_yes", "all_no"):
                # Should be 4, not 5
                assert arb["n_buckets"] == 4, \
                    f"Expected 4 buckets (1 excluded), got {arb['n_buckets']}"

    def test_multiple_in_review_buckets(self):
        """Multiple in-review buckets should all be excluded."""
        bridge, clob, events_raw = self._setup_seattle_event(
            in_review_outcomes=["41°F or below", "42-43°F", "44-45°F"],
        )

        arbs, scanned = _run(scan_arbs(bridge, clob, events_raw))

        assert scanned == 1
        for arb in arbs:
            if "buckets" in arb:
                outcomes = [b["outcome"] for b in arb["buckets"]]
                assert "41°F or below" not in outcomes
                assert "42-43°F" not in outcomes
                assert "44-45°F" not in outcomes
                assert len(arb["buckets"]) == 2  # only 46-47°F and 48-49°F

    def test_all_in_review_skips_event(self):
        """If ALL buckets are in review, the event should be skipped (< 2 active)."""
        bridge, clob, events_raw = self._setup_seattle_event(
            in_review_outcomes=[
                "41°F or below", "42-43°F", "44-45°F", "46-47°F", "48-49°F",
            ],
        )

        arbs, scanned = _run(scan_arbs(bridge, clob, events_raw))

        # Event should be skipped entirely (0 active buckets < 2 minimum)
        assert scanned == 0 or len(arbs) == 0

    def test_no_in_review_all_buckets_included(self):
        """With no in-review markets, all buckets should be included."""
        bridge, clob, events_raw = self._setup_seattle_event(
            in_review_outcomes=[],
        )

        arbs, scanned = _run(scan_arbs(bridge, clob, events_raw))

        assert scanned == 1
        for arb in arbs:
            if "buckets" in arb:
                assert arb["n_buckets"] == 5


# ------------------------------------------------------------------
# Arb detection tests (existing logic, regression)
# ------------------------------------------------------------------

class TestArbDetection:
    """Test the core arb detection logic."""

    def _setup_event(self, n_buckets, yes_asks, no_asks, yes_bids=None, no_bids=None):
        """Create an event with specified prices."""
        event_id = "100"
        event_name = "Highest temperature in TestCity on March 25?"
        slug = "highest-temperature-in-testcity-on-march-25"

        if yes_bids is None:
            yes_bids = [0.0] * n_buckets
        if no_bids is None:
            no_bids = [0.0] * n_buckets

        raw_markets = []
        bridge_markets = []
        orderbooks = {}

        for i in range(n_buckets):
            cid = f"cid_{i}"
            tok_yes = f"tok_yes_{i}"
            tok_no = f"tok_no_{i}"
            outcome = f"Bucket {i}"

            raw_markets.append(_make_raw_market(cid, outcome))
            bridge_markets.append(_make_bridge_market(
                cid, event_id, event_name, outcome,
                token_yes=tok_yes, token_no=tok_no,
            ))
            orderbooks[tok_yes] = _make_orderbook(best_ask=yes_asks[i], best_bid=yes_bids[i])
            orderbooks[tok_no] = _make_orderbook(best_ask=no_asks[i], best_bid=no_bids[i])

        events_raw = [_make_event_raw(event_id, slug, event_name, raw_markets)]
        bridge = AsyncMock()
        bridge.fetch_weather_markets = AsyncMock(return_value=bridge_markets)
        clob = _build_clob_mock(orderbooks)

        return bridge, clob, events_raw

    def test_all_yes_arb_detected(self):
        """YES asks summing to less than 1.0 should trigger all_yes arb."""
        # 3 buckets, YES asks: 0.30 + 0.30 + 0.30 = 0.90 < 1.0
        bridge, clob, events_raw = self._setup_event(
            n_buckets=3,
            yes_asks=[0.30, 0.30, 0.30],
            no_asks=[0.70, 0.70, 0.70],
        )
        arbs, _ = _run(scan_arbs(bridge, clob, events_raw))
        all_yes = [a for a in arbs if a["type"] == "all_yes"]
        assert len(all_yes) == 1
        assert abs(all_yes[0]["profit"] - 0.10) < 0.001

    def test_all_no_arb_detected(self):
        """NO asks summing to less than N-1 should trigger all_no arb."""
        # 4 buckets, NO asks: 0.70 * 4 = 2.80 < 3 (N-1)
        bridge, clob, events_raw = self._setup_event(
            n_buckets=4,
            yes_asks=[0.30, 0.30, 0.30, 0.30],
            no_asks=[0.70, 0.70, 0.70, 0.70],
        )
        arbs, _ = _run(scan_arbs(bridge, clob, events_raw))
        all_no = [a for a in arbs if a["type"] == "all_no"]
        assert len(all_no) == 1
        assert abs(all_no[0]["profit"] - 0.20) < 0.001

    def test_yes_no_pair_arb(self):
        """YES ask + NO ask < 1.0 on one bucket should trigger yes_no_pair."""
        bridge, clob, events_raw = self._setup_event(
            n_buckets=3,
            yes_asks=[0.40, 0.30, 0.30],
            no_asks=[0.50, 0.70, 0.70],  # bucket 0: 0.40 + 0.50 = 0.90 < 1.0
        )
        arbs, _ = _run(scan_arbs(bridge, clob, events_raw))
        pairs = [a for a in arbs if a["type"] == "yes_no_pair"]
        assert len(pairs) >= 1
        assert any(abs(p["profit"] - 0.10) < 0.001 for p in pairs)

    def test_no_arb_when_prices_fair(self):
        """No arbs when prices sum correctly."""
        # 3 buckets, YES asks: 0.34 + 0.33 + 0.33 = 1.00, NO asks: 0.66*3 = 1.98 < 2 but close
        bridge, clob, events_raw = self._setup_event(
            n_buckets=3,
            yes_asks=[0.34, 0.33, 0.34],
            no_asks=[0.67, 0.67, 0.67],  # sum = 2.01 > 2 (N-1)
        )
        arbs, _ = _run(scan_arbs(bridge, clob, events_raw))
        # YES sum = 1.01 > 1.0, NO sum = 2.01 > 2 → no all_yes or all_no
        all_yes = [a for a in arbs if a["type"] == "all_yes"]
        all_no = [a for a in arbs if a["type"] == "all_no"]
        assert len(all_yes) == 0
        assert len(all_no) == 0

    def test_single_bucket_excluded(self):
        """Events with only 1 bucket should not produce arbs (need N≥2)."""
        bridge, clob, events_raw = self._setup_event(
            n_buckets=1,
            yes_asks=[0.50],
            no_asks=[0.40],
        )
        arbs, scanned = _run(scan_arbs(bridge, clob, events_raw))
        # 1-bucket event should be skipped
        assert len(arbs) == 0

    def test_crossed_book_detected(self):
        """YES bid > YES ask should trigger crossed_yes."""
        bridge, clob, events_raw = self._setup_event(
            n_buckets=3,
            yes_asks=[0.40, 0.30, 0.30],
            no_asks=[0.70, 0.70, 0.70],
            yes_bids=[0.50, 0.25, 0.25],  # bucket 0: bid 0.50 > ask 0.40
        )
        arbs, _ = _run(scan_arbs(bridge, clob, events_raw))
        crossed = [a for a in arbs if a["type"] == "crossed_yes"]
        assert len(crossed) >= 1
        assert crossed[0]["outcome"] == "Bucket 0"

    def test_sell_all_yes_detected(self):
        """YES bids summing to > 1.0 should trigger sell_all_yes."""
        bridge, clob, events_raw = self._setup_event(
            n_buckets=3,
            yes_asks=[0.40, 0.35, 0.35],
            no_asks=[0.70, 0.70, 0.70],
            yes_bids=[0.40, 0.35, 0.35],  # sum = 1.10 > 1.0
        )
        arbs, _ = _run(scan_arbs(bridge, clob, events_raw))
        sell_yes = [a for a in arbs if a["type"] == "sell_all_yes"]
        assert len(sell_yes) == 1
        assert not sell_yes[0]["executable"]  # requires inventory


# ------------------------------------------------------------------
# Depth analysis tests
# ------------------------------------------------------------------

from weather.arb import compute_arb_depth


class TestComputeArbDepth:
    """Test depth-aware arb sizing."""

    def test_uniform_depth(self):
        """3 buckets each with 100 shares at 0.30 → sum=0.90 < 1.0, all 100 shares valid."""
        books = [
            [(0.30, 100)],
            [(0.30, 100)],
            [(0.30, 100)],
        ]
        d = compute_arb_depth(books, threshold=1.0)
        assert d["max_shares"] == 100
        assert d["executable"]
        assert d["profit_per_share"] == pytest.approx(0.10, abs=0.001)
        assert d["total_profit"] == pytest.approx(10.0, abs=0.1)

    def test_price_increases_with_depth(self):
        """Prices increase at deeper levels, arb closes at some depth."""
        # 2 buckets: first 50 shares at 0.40, next 50 at 0.55
        books = [
            [(0.40, 50), (0.55, 50)],
            [(0.40, 50), (0.55, 50)],
        ]
        d = compute_arb_depth(books, threshold=1.0)
        # At depth 50: sum = 0.40 + 0.40 = 0.80 < 1.0 ✓
        # At depth 51: sum = 0.55 + 0.55 = 1.10 > 1.0 ✗
        assert d["max_shares"] == 50
        assert d["executable"]

    def test_empty_book_not_executable(self):
        """If any bucket has no asks, arb is not executable."""
        books = [
            [(0.30, 100)],
            [],  # no asks
            [(0.30, 100)],
        ]
        d = compute_arb_depth(books, threshold=1.0)
        assert d["max_shares"] == 0
        assert not d["executable"]
        assert "no asks" in d["limiting_factor"]

    def test_min_order_constraint(self):
        """Cheap bucket ($0.001) needs 1000+ shares for $1 min order."""
        books = [
            [(0.001, 500)],   # only 500 shares available
            [(0.90, 10000)],
        ]
        # sum = 0.001 + 0.90 = 0.901 < 1.0, arb exists
        d = compute_arb_depth(books, threshold=1.0)
        # But min order on cheap bucket = ceil(1/0.001) = 1001, only 500 available
        assert d["max_shares"] == 500
        assert d["min_shares_required"] == 1001
        assert not d["executable"]
        assert "need" in d["limiting_factor"]

    def test_all_no_threshold(self):
        """ALL_NO with 4 buckets: threshold = 3."""
        books = [
            [(0.70, 200)],
            [(0.70, 200)],
            [(0.70, 200)],
            [(0.70, 200)],
        ]
        d = compute_arb_depth(books, threshold=3.0)
        # sum = 2.80 < 3.0 → arb valid
        assert d["max_shares"] == 200
        assert d["executable"]
        assert d["profit_per_share"] == pytest.approx(0.20, abs=0.001)

    def test_asymmetric_depth(self):
        """Shallowest bucket limits overall depth."""
        books = [
            [(0.30, 10000)],
            [(0.30, 50)],      # shallow — only 50 shares
            [(0.30, 10000)],
        ]
        d = compute_arb_depth(books, threshold=1.0)
        assert d["max_shares"] == 50  # limited by bucket 1
        assert d["executable"]

    def test_no_buckets(self):
        d = compute_arb_depth([], threshold=1.0)
        assert d["max_shares"] == 0
        assert not d["executable"]
