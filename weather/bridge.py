"""CLOBWeatherBridge — CLOB + Gamma adapter for weather/strategy.py.

Provides the semantic interface that weather/strategy.py expects,
routing all operations through the Polymarket CLOB and Gamma APIs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .config import MIN_SHARES_PER_ORDER

if TYPE_CHECKING:
    from bot.gamma import GammaClient, GammaMarket
    from polymarket.client import PolymarketClient

logger = logging.getLogger(__name__)


def compute_available_depth(book_side: list[dict], max_levels: int = 5) -> float:
    """Compute total USD liquidity in the top N levels of an orderbook side."""
    total = 0.0
    for level in book_side[:max_levels]:
        try:
            price = float(level["price"])
            size = float(level["size"])
            total += size * price
        except (KeyError, ValueError, TypeError):
            continue
    return total


def compute_vwap(book_side: list[dict], target_usd: float) -> float:
    """Compute volume-weighted average price to fill target_usd.

    Walks the orderbook from best price, computing shares bought at each level.
    Returns total_cost / total_shares (true VWAP).
    """
    if not book_side:
        return 0.0

    total_shares = 0.0
    total_cost = 0.0
    remaining = target_usd
    for level in book_side:
        try:
            price = float(level["price"])
            size = float(level["size"])  # size is in shares
        except (KeyError, ValueError, TypeError):
            continue
        if price <= 0:
            continue
        available_usd = size * price
        take_usd = min(available_usd, remaining)
        take_shares = take_usd / price
        total_shares += take_shares
        total_cost += take_usd
        remaining -= take_usd
        if remaining <= 0:
            break

    if total_shares > 0:
        return total_cost / total_shares
    try:
        return float(book_side[0]["price"])
    except (KeyError, ValueError, TypeError, IndexError):
        return 0.0


class CLOBWeatherBridge:
    """Adapter: Polymarket CLOB + Gamma API for weather strategy.

    Args:
        clob_client: Authenticated (or public) PolymarketClient for order
            submission and orderbook queries.
        gamma_client: GammaClient for market discovery and metadata.
        max_exposure: Maximum total exposure in USD (used for portfolio).
    """

    def __init__(
        self,
        clob_client: PolymarketClient,
        gamma_client: GammaClient,
        max_exposure: float = 50.0,
    ):
        self.clob = clob_client
        self.gamma = gamma_client
        self.max_exposure = max_exposure
        # Cache: condition_id → GammaMarket (populated by fetch_weather_markets)
        self._market_cache: dict[str, GammaMarket] = {}
        # Exposure tracking (deducted from max_exposure for portfolio)
        self._total_exposure: float = 0.0
        self._position_count: int = 0
        self._known_positions: set[str] = set()  # market_ids with open positions
        self._neg_risk_cache: dict[str, bool] = {}  # token_id → neg_risk

    async def _get_neg_risk(self, token_id: str) -> bool:
        """Get neg_risk for a token, with caching to avoid repeated API calls."""
        if token_id not in self._neg_risk_cache:
            try:
                self._neg_risk_cache[token_id] = await self.clob.is_neg_risk(token_id)
            except Exception:
                self._neg_risk_cache[token_id] = True  # Safe default for weather markets
                logger.debug("neg_risk check failed for %s, defaulting to True", token_id[:16])
        return self._neg_risk_cache[token_id]

    def sync_exposure_from_state(self, trades: dict) -> None:
        """Initialize exposure tracking from persisted weather state trades.

        Args:
            trades: Dict of market_id → TradeRecord (from weather.state).
        """
        total = 0.0
        count = 0
        for trade in trades.values():
            cost_basis = getattr(trade, "cost_basis", 0.0)
            shares = getattr(trade, "shares", 0.0)
            total += cost_basis * shares
            count += 1
        self._total_exposure = total
        self._position_count = count
        self._known_positions = set()
        for trade in trades.values():
            market_id = getattr(trade, "market_id", "")
            if market_id:
                self._known_positions.add(market_id)
        if count > 0:
            logger.info("Bridge: synced exposure from state — $%.2f across %d positions",
                        total, count)

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    async def fetch_weather_markets(self) -> list[dict]:
        """Fetch active weather markets via Gamma API.

        Returns list of dicts compatible with the weather strategy:
            {id, event_id, event_name, outcome_name, external_price_yes,
             token_id_yes, token_id_no, end_date, best_bid, best_ask, status}
        """
        events, gamma_markets = await self.gamma.fetch_events_with_markets(
            tag_slug="weather", limit=100,
        )

        markets: list[dict] = []
        for gm in gamma_markets:
            if gm.closed or not gm.active:
                continue
            if not gm.clob_token_ids:
                continue

            # Use condition_id as market_id (unique per binary market)
            market_id = gm.condition_id

            # YES price is the first outcome price
            yes_price = gm.outcome_prices[0] if gm.outcome_prices else 0.0

            # Outcome name from group_item_title or question
            outcome_name = gm.group_item_title or gm.question

            token_id_yes = gm.clob_token_ids[0] if gm.clob_token_ids else ""
            token_id_no = gm.clob_token_ids[1] if len(gm.clob_token_ids) > 1 else ""

            # Cache for later lookups
            self._market_cache[market_id] = gm

            markets.append({
                "id": market_id,
                "event_id": gm.event_id,
                "event_name": gm.event_title,
                "outcome_name": outcome_name,
                "external_price_yes": yes_price,
                "token_id_yes": token_id_yes,
                "token_id_no": token_id_no,
                "end_date": gm.end_date,
                "best_bid": gm.best_bid,
                "best_ask": gm.best_ask,
                "status": "active",
            })

        logger.info("Bridge: fetched %d active weather markets", len(markets))
        self._last_events_raw = events
        return markets

    @property
    def last_events_raw(self) -> list[dict]:
        """Raw event dicts from the most recent fetch_weather_markets call."""
        return getattr(self, "_last_events_raw", [])

    # ------------------------------------------------------------------
    # Portfolio & positions
    # ------------------------------------------------------------------

    async def get_portfolio(self) -> dict:
        """Derive portfolio info from max_exposure minus tracked positions.

        Deducts exposure from open positions tracked via record_exposure().
        """
        return {
            "balance_usdc": max(0.0, self.max_exposure - self._total_exposure),
            "total_exposure": self._total_exposure,
            "positions_count": self._position_count,
        }

    def get_position(self, market_id: str) -> dict | None:
        """Return None — position tracking is done via state."""
        return None

    # ------------------------------------------------------------------
    # Market context
    # ------------------------------------------------------------------

    def get_market_context(self, market_id: str, **kwargs) -> dict | None:
        """Build market context from CLOB orderbook data.

        Returns a dict compatible with check_context_safeguards():
            {market: {time_to_resolution}, slippage: {estimates: [...]},
             edge: {}, warnings: [], discipline: {}}
        """
        gm = self._market_cache.get(market_id)
        if not gm:
            return None

        # Time to resolution
        time_str = ""
        if gm.end_date:
            try:
                end_dt = datetime.fromisoformat(gm.end_date.replace("Z", "+00:00"))
                delta = end_dt - datetime.now(timezone.utc)
                hours = max(0, delta.total_seconds() / 3600)
                days = int(hours // 24)
                remaining_hours = int(hours % 24)
                if days > 0:
                    time_str = f"{days}d {remaining_hours}h"
                else:
                    time_str = f"{remaining_hours}h"
            except (ValueError, TypeError):
                pass

        # Slippage estimate from spread
        spread = gm.spread
        bid_ask_sum = gm.best_bid + gm.best_ask
        mid = bid_ask_sum / 2 if bid_ask_sum > 0 else 0.0
        slippage_pct = spread / mid if mid > 0 else 0.0

        # Edge computation for adaptive slippage
        my_probability = kwargs.get("my_probability")
        edge_dict: dict = {}
        if my_probability is not None:
            market_price = gm.best_ask if gm.best_ask > 0 else (
                gm.outcome_prices[0] if gm.outcome_prices else 0.5
            )
            user_edge = my_probability - market_price
            edge_dict = {
                "user_edge": user_edge,
                "recommendation": "TRADE" if user_edge > 0.02 else ("HOLD" if user_edge > 0 else "SKIP"),
                "suggested_threshold": 0.02,
            }

        return {
            "market": {"time_to_resolution": time_str},
            "slippage": {"estimates": [{"slippage_pct": slippage_pct}]},
            "edge": edge_dict,
            "warnings": [],
            "discipline": {},
        }

    # ------------------------------------------------------------------
    # Fill verification
    # ------------------------------------------------------------------

    async def verify_fill(
        self,
        order_id: str,
        timeout_seconds: float = 30.0,
        poll_interval: float = 2.0,
    ) -> dict:
        """Poll order status until filled, partially filled, or timeout.

        Returns:
            {"filled": bool, "partial": bool, "size_matched": float,
             "original_size": float, "status": str}
        """
        deadline = time.monotonic() + timeout_seconds
        last_status = "UNKNOWN"
        size_matched = 0.0
        original_size = 0.0

        while time.monotonic() < deadline:
            try:
                order = await self.clob.get_order(order_id)
                if not order:
                    await asyncio.sleep(poll_interval)
                    continue

                last_status = order.get("status", "UNKNOWN")
                size_matched = float(order.get("size_matched", 0))
                original_size = float(order.get("original_size", order.get("size", 0)))

                if last_status == "MATCHED":
                    return {
                        "filled": True,
                        "partial": False,
                        "size_matched": size_matched,
                        "original_size": original_size,
                        "status": last_status,
                    }
                elif last_status == "CANCELLED":
                    return {
                        "filled": size_matched > 0,
                        "partial": 0 < size_matched < original_size,
                        "size_matched": size_matched,
                        "original_size": original_size,
                        "status": last_status,
                    }
            except Exception as exc:
                logger.debug("verify_fill poll error: %s", exc)

            await asyncio.sleep(poll_interval)

        # Timeout — check if partially filled
        return {
            "filled": size_matched > 0,
            "partial": 0 < size_matched < original_size,
            "size_matched": size_matched,
            "original_size": original_size,
            "status": f"TIMEOUT ({last_status})",
        }

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if cancellation succeeded."""
        try:
            await self.clob.cancel_order(order_id)
            logger.info("Cancelled order %s", order_id)
            return True
        except Exception as exc:
            logger.error("Failed to cancel order %s: %s", order_id, exc)
            return False

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    async def execute_trade(
        self,
        market_id: str,
        side: str,
        amount: float,
        fill_timeout: float = 30.0,
        fill_poll_interval: float = 2.0,
        depth_fill_ratio: float = 0.0,
        vwap_max_levels: int = 0,
        limit_price: float = 0.0,
    ) -> dict:
        """Execute a buy trade via CLOB limit order at fresh best ask.

        Re-fetches the orderbook before trading to avoid stale prices.
        When ``fill_timeout > 0``, verifies fill and cancels unfilled orders.

        Args:
            market_id: condition_id of the market.
            side: "yes" or "no".
            amount: USD amount to spend.
            fill_timeout: Seconds to wait for fill verification (default 30s).
            fill_poll_interval: Seconds between fill status polls.

        Returns:
            {"success": bool, "shares_bought": float, "trade_id": str}
        """
        gm = self._market_cache.get(market_id)
        if not gm:
            return {"error": f"Unknown market {market_id}", "success": False}

        # Determine token ID
        if side.lower() == "yes":
            if not gm.clob_token_ids:
                return {"error": "No YES token ID", "success": False}
            token_id = gm.clob_token_ids[0]
        else:
            if len(gm.clob_token_ids) < 2:
                return {"error": "No NO token ID", "success": False}
            token_id = gm.clob_token_ids[1]

        # Re-fetch fresh price from orderbook
        try:
            book = await self.clob.get_orderbook(token_id)
            asks = book.get("asks") or []
            if asks:
                # Depth-aware sizing: cap amount to available liquidity
                if depth_fill_ratio > 0:
                    depth = compute_available_depth(asks, max_levels=max(vwap_max_levels, 5))
                    max_from_depth = depth * depth_fill_ratio
                    if amount > max_from_depth > 0:
                        logger.info("Depth cap: $%.2f → $%.2f (%.0f%% of $%.2f depth)",
                                    amount, max_from_depth, depth_fill_ratio * 100, depth)
                        amount = max_from_depth
                    # Abort if depth-capped amount is too small for viable order
                    if amount < 0.05:
                        return {"error": "Depth too thin for minimum order", "success": False}

                # VWAP pricing across multiple levels
                if vwap_max_levels > 1:
                    price = compute_vwap(asks[:vwap_max_levels], amount)
                    if price <= 0:
                        price = float(asks[0]["price"])  # Fallback if VWAP failed
                else:
                    price = float(asks[0]["price"])
            else:
                # Fallback to cached price
                price = gm.best_ask if gm.best_ask > 0 else (
                    gm.outcome_prices[0] if gm.outcome_prices else 0.5
                )
        except Exception:
            price = gm.best_ask if gm.best_ask > 0 else (
                gm.outcome_prices[0] if gm.outcome_prices else 0.5
            )
            logger.warning("Could not refresh price for %s, using cached %.4f", token_id[:16], price)

        # Apply limit price cap (fair value from model)
        if limit_price > 0 and price > limit_price:
            logger.info("Limit cap: ask=$%.4f → limit=$%.4f (saving %.1f%%)",
                        price, limit_price, (price - limit_price) / price * 100)
            price = limit_price

        if price <= 0:
            return {"error": "Invalid price", "success": False}

        shares = amount / price

        try:
            neg_risk = await self._get_neg_risk(token_id)
            result = await self.clob.post_order(
                token_id=token_id,
                side="BUY",
                price=price,
                size=shares,
                neg_risk=neg_risk,
            )
            order_id = result.get("orderID", "")
            logger.info(
                "Bridge trade: BUY %.1f shares of %s @ %.4f (order %s)",
                shares, token_id[:16], price, order_id,
            )

            # Fill verification
            if fill_timeout > 0 and order_id:
                fill = await self.verify_fill(order_id, fill_timeout, fill_poll_interval)
                if fill["filled"] and not fill["partial"]:
                    actual_shares = fill["size_matched"]
                    logger.info("Order %s fully filled: %.1f shares", order_id, actual_shares)
                    shares = actual_shares
                elif fill["partial"]:
                    actual_shares = fill["size_matched"]
                    logger.warning("Order %s partially filled: %.1f/%.1f shares — cancelling remainder",
                                   order_id, actual_shares, fill["original_size"])
                    await self.cancel_order(order_id)
                    shares = actual_shares
                else:
                    logger.warning("Order %s not filled within %.1fs — cancelling", order_id, fill_timeout)
                    await self.cancel_order(order_id)
                    return {
                        "success": False,
                        "error": f"Order not filled within {fill_timeout}s",
                        "trade_id": order_id,
                    }
            elif fill_timeout == 0:
                logger.debug("No fill verification — exposure is estimated, not confirmed")

            self._total_exposure += price * shares
            if market_id not in self._known_positions:
                self._position_count += 1
                self._known_positions.add(market_id)
            return {
                "success": True,
                "shares_bought": shares,
                "trade_id": order_id,
            }
        except Exception as exc:
            logger.error("Bridge trade failed: %s", exc)
            return {"error": str(exc), "success": False}

    async def execute_sell(
        self,
        market_id: str,
        shares: float,
        side: str = "yes",
        fill_timeout: float = 30.0,
        fill_poll_interval: float = 2.0,
    ) -> dict:
        """Execute a sell trade via CLOB limit order at fresh best bid.

        Re-fetches the orderbook before selling to avoid stale prices.
        When ``fill_timeout > 0``, verifies fill and cancels unfilled orders.

        Args:
            market_id: condition_id of the market.
            shares: Number of shares to sell.
            side: "yes" or "no" — which token to sell (must match the buy side).
            fill_timeout: Seconds to wait for fill verification (default 30s).
            fill_poll_interval: Seconds between fill status polls.

        Returns:
            {"success": bool, "trade_id": str}
        """
        gm = self._market_cache.get(market_id)
        if not gm:
            return {"error": f"Unknown market {market_id}", "success": False}

        if not gm.clob_token_ids:
            return {"error": "No token ID", "success": False}

        # Sell the same token we bought
        if side.lower() == "no":
            if len(gm.clob_token_ids) < 2:
                return {"error": "No NO token ID", "success": False}
            token_id = gm.clob_token_ids[1]
        else:
            token_id = gm.clob_token_ids[0]

        # Re-fetch fresh price from orderbook
        try:
            book = await self.clob.get_orderbook(token_id)
            bids = book.get("bids") or []
            if bids:
                price = float(bids[0]["price"])
            else:
                price = gm.best_bid if gm.best_bid > 0 else (
                    gm.outcome_prices[0] if gm.outcome_prices else 0.5
                )
        except Exception:
            price = gm.best_bid if gm.best_bid > 0 else (
                gm.outcome_prices[0] if gm.outcome_prices else 0.5
            )
            logger.warning("Could not refresh bid for %s, using cached %.4f", token_id[:16], price)

        if price <= 0:
            return {"error": "Invalid bid price", "success": False}

        try:
            neg_risk = await self._get_neg_risk(token_id)
            result = await self.clob.post_order(
                token_id=token_id,
                side="SELL",
                price=price,
                size=shares,
                neg_risk=neg_risk,
            )
            order_id = result.get("orderID", "")
            logger.info(
                "Bridge sell: SELL %.1f shares of %s @ %.4f (order %s)",
                shares, token_id[:16], price, order_id,
            )

            # Fill verification
            actual_shares = shares
            if fill_timeout > 0 and order_id:
                fill = await self.verify_fill(order_id, fill_timeout, fill_poll_interval)
                if fill["filled"] and not fill["partial"]:
                    actual_shares = fill["size_matched"]
                    logger.info("Sell order %s fully filled: %.1f shares", order_id, actual_shares)
                elif fill["partial"]:
                    actual_shares = fill["size_matched"]
                    logger.warning("Sell order %s partially filled: %.1f/%.1f — cancelling remainder",
                                   order_id, actual_shares, fill["original_size"])
                    await self.cancel_order(order_id)
                else:
                    logger.warning("Sell order %s not filled within %.1fs — cancelling", order_id, fill_timeout)
                    await self.cancel_order(order_id)
                    return {
                        "success": False,
                        "error": f"Sell order not filled within {fill_timeout}s",
                        "trade_id": order_id,
                    }
            elif fill_timeout == 0:
                logger.debug("No fill verification — exposure is estimated, not confirmed")

            self._total_exposure = max(0.0, self._total_exposure - price * actual_shares)
            self._known_positions.discard(market_id)
            self._position_count = max(0, self._position_count - 1)
            return {
                "success": True,
                "trade_id": order_id,
                "fill_price": price,
                "shares_sold": actual_shares,
            }
        except Exception as exc:
            logger.error("Bridge sell failed: %s", exc)
            return {"error": str(exc), "success": False}

    async def execute_maker_order(
        self,
        market_id: str,
        side: str,
        amount: float,
        maker_price: float,
    ) -> dict:
        """Post a GTC postOnly bid. Returns immediately, no fill wait.

        Returns dict with keys: success, posted, order_id, price, size, token_id.
        If CLOB rejects postOnly (would cross spread), returns {"posted": False}.
        """
        gm = self._market_cache.get(market_id)
        if not gm or not gm.clob_token_ids:
            return {"success": False, "posted": False, "error": "no market data"}

        token_id = gm.clob_token_ids[0]
        price = round(maker_price, 2)
        size = round(amount / price, 1) if price > 0 else 0.0

        if size < MIN_SHARES_PER_ORDER:
            return {"success": False, "posted": False, "error": "size below minimum"}

        clob_side = "BUY" if side.lower() in ("yes", "buy") else "SELL"

        try:
            neg_risk = await self._get_neg_risk(token_id)
            result = await self.clob.post_order(
                token_id=token_id,
                side=clob_side,
                price=price,
                size=size,
                neg_risk=neg_risk,
                order_type="GTC",
                post_only=True,
            )
            order_id = result.get("orderID", "")

            if not order_id:
                logger.info("Maker order rejected (postOnly would cross): %s", result)
                return {"success": False, "posted": False, "error": "rejected"}

            logger.info(
                "Maker order posted: %s %.1f shares @ $%.2f (order %s)",
                clob_side, size, price, order_id,
            )
            return {
                "success": True,
                "posted": True,
                "order_id": order_id,
                "price": price,
                "size": size,
                "token_id": token_id,
            }

        except Exception as exc:
            logger.warning("Maker order failed: %s", exc)
            return {"success": False, "posted": False, "error": str(exc)}
