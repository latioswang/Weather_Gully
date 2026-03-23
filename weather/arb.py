"""Simple arbitrage scanner for Polymarket weather markets.

Scans multi-choice weather events for three arb conditions:
  1. Sum of YES ask prices < 1.0  →  buy all YES, one must win
  2. Sum of NO ask prices < (N-1) →  buy all NO, (N-1) must win
  3. Any single bucket YES ask + NO ask < 1.0  →  buy both sides

Verifies prices against live CLOB orderbook data.

Usage::

    python -m weather.arb
    python -m weather.arb --locations Wuhan,Shanghai
    python -m weather.arb --verbose
"""

import argparse
import asyncio
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from .config import LOCATIONS
from .parsing import parse_weather_event

logger = logging.getLogger(__name__)


async def _get_best_ask(clob, token_id: str) -> float | None:
    """Fetch best ask from live orderbook. Returns None if no asks."""
    try:
        book = await clob.get_orderbook(token_id)
        asks = book.get("asks", [])
        if asks:
            return float(asks[0]["price"])
    except Exception as e:
        logger.debug("Orderbook fetch failed for %s: %s", token_id, e)
    return None


async def scan_arbs(
    bridge,
    clob,
    locations: list[str],
    verbose: bool = False,
) -> list[dict]:
    """Scan weather markets for arbitrage opportunities.

    Returns list of arb dicts with details.
    """
    markets = await bridge.fetch_weather_markets()

    # Group by event_id
    events: dict[str, list[dict]] = defaultdict(list)
    for m in markets:
        events[m.get("event_id") or m.get("event_name", "unknown")].append(m)

    loc_set = {l.lower() for l in locations}
    arbs: list[dict] = []
    scanned = 0

    for event_id, event_markets in events.items():
        if not event_markets:
            continue

        # Parse event to get location / date / metric
        event_name = event_markets[0].get("event_name", "")
        info = parse_weather_event(event_name)
        if not info:
            continue

        location = info["location"]
        if location.lower() not in loc_set:
            continue

        scanned += 1
        date_str = info["date"]
        metric = info["metric"]
        n_buckets = len(event_markets)
        label = f"{location} {metric} on {date_str}"

        # Fetch live orderbook prices for each bucket
        buckets = []
        for m in event_markets:
            token_yes = m.get("token_id_yes", "")
            token_no = m.get("token_id_no", "")
            outcome = m.get("outcome_name", "?")

            # Fetch live asks (buy prices)
            yes_ask = await _get_best_ask(clob, token_yes) if token_yes else None
            no_ask = await _get_best_ask(clob, token_no) if token_no else None

            # Fallback to Gamma cached prices
            if yes_ask is None:
                yes_ask = m.get("best_ask") or m.get("external_price_yes") or 0.0
                logger.debug("Fallback YES price for %s: %.3f", outcome, yes_ask)
            if no_ask is None:
                # best_bid on YES side ≈ 1 - NO ask (approximate)
                bid = m.get("best_bid", 0.0)
                no_ask = (1.0 - bid) if bid else (1.0 - yes_ask)
                logger.debug("Fallback NO price for %s: %.3f", outcome, no_ask)

            buckets.append({
                "outcome": outcome,
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "market_id": m.get("id", ""),
            })

        # --- Check arb conditions ---

        yes_sum = sum(b["yes_ask"] for b in buckets)
        no_sum = sum(b["no_ask"] for b in buckets)
        found_arb = False

        # 1. All YES < 1.0
        if yes_sum < 1.0:
            profit = 1.0 - yes_sum
            arbs.append({
                "type": "all_yes",
                "label": label,
                "n_buckets": n_buckets,
                "yes_sum": yes_sum,
                "profit": profit,
                "buckets": buckets,
            })
            found_arb = True

        # 2. All NO < (N-1)
        if no_sum < n_buckets - 1:
            profit = (n_buckets - 1) - no_sum
            arbs.append({
                "type": "all_no",
                "label": label,
                "n_buckets": n_buckets,
                "no_sum": no_sum,
                "profit": profit,
                "buckets": buckets,
            })
            found_arb = True

        # 3. Individual YES + NO < 1.0
        for b in buckets:
            total = b["yes_ask"] + b["no_ask"]
            if total < 1.0:
                profit = 1.0 - total
                arbs.append({
                    "type": "yes_no_pair",
                    "label": label,
                    "outcome": b["outcome"],
                    "yes_ask": b["yes_ask"],
                    "no_ask": b["no_ask"],
                    "total": total,
                    "profit": profit,
                })
                found_arb = True

        if verbose and not found_arb:
            print(f"  OK: {label} ({n_buckets} buckets) — YES sum: {yes_sum:.3f}, NO sum: {no_sum:.3f}")

    return arbs, scanned


def _print_results(arbs: list[dict], scanned: int) -> None:
    """Pretty-print arb scan results."""
    if not arbs:
        print(f"\nNo arbitrage opportunities found across {scanned} events.")
        return

    print(f"\n{'=' * 60}")
    for arb in arbs:
        if arb["type"] == "all_yes":
            print(f"\n  ARB [ALL YES]: {arb['label']} ({arb['n_buckets']} buckets)")
            print(f"   YES sum: {arb['yes_sum']:.4f} < 1.0 -> profit: ${arb['profit']:.4f}/share")
            parts = [f"{b['outcome']} @{b['yes_ask']:.3f}" for b in arb["buckets"]]
            print(f"   Buckets: {' | '.join(parts)}")

        elif arb["type"] == "all_no":
            print(f"\n  ARB [ALL NO]: {arb['label']} ({arb['n_buckets']} buckets)")
            print(f"   NO sum: {arb['no_sum']:.4f} < {arb['n_buckets'] - 1} -> profit: ${arb['profit']:.4f}/share")
            parts = [f"{b['outcome']} @{b['no_ask']:.3f}" for b in arb["buckets"]]
            print(f"   Buckets: {' | '.join(parts)}")

        elif arb["type"] == "yes_no_pair":
            print(f"\n  ARB [YES+NO]: {arb['label']}")
            print(f"   Bucket {arb['outcome']}: YES @{arb['yes_ask']:.3f} + NO @{arb['no_ask']:.3f} = {arb['total']:.4f} -> profit: ${arb['profit']:.4f}/share")

    print(f"\n{'=' * 60}")
    print(f"Summary: {len(arbs)} arb opportunity(ies) found across {scanned} events")


async def _async_main(locations: list[str], verbose: bool) -> None:
    """Async entry point."""
    from bot.gamma import GammaClient
    from polymarket.public import PublicClient
    from .bridge import CLOBWeatherBridge

    gamma = GammaClient()
    clob = PublicClient()
    bridge = CLOBWeatherBridge(
        clob_client=clob,
        gamma_client=gamma,
        max_exposure=0,  # read-only, no trading
    )

    print(f"=== Weather Arbitrage Scanner ===")
    print(f"Scanning {len(locations)} locations...")

    try:
        arbs, scanned = await scan_arbs(bridge, clob, locations, verbose=verbose)
        _print_results(arbs, scanned)
    finally:
        from .http_client import close_session
        await close_session()
        await clob.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="weather.arb",
        description="Scan Polymarket weather markets for simple arbitrage opportunities",
    )
    parser.add_argument(
        "--locations", type=str, default=None,
        help="Comma-separated location keys (default: all cities)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show all events including non-arb",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    level = "DEBUG" if args.debug else "INFO"
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    if args.locations:
        loc_keys = [l.strip() for l in args.locations.split(",")]
        # Validate
        for l in loc_keys:
            if l not in LOCATIONS:
                print(f"Error: unknown location '{l}' (available: {', '.join(LOCATIONS)})")
                sys.exit(1)
    else:
        loc_keys = list(LOCATIONS.keys())

    asyncio.run(_async_main(loc_keys, verbose=args.verbose))


if __name__ == "__main__":
    main()
