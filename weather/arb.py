"""Arbitrage scanner + executor for Polymarket temperature markets.

Scans multi-choice "Highest temperature" events for guaranteed arb conditions:
  Buy-side (N≥2 buckets required):
    1. Sum of YES ask prices < 1.0  →  buy all YES, one must win
    2. Sum of NO ask prices < (N-1) →  buy all NO, (N-1) must win
    3. Any single bucket YES ask + NO ask < 1.0  →  buy both sides
  Sell-side (detection only — requires inventory):
    4. Sum of YES bid prices > 1.0  →  sell all YES, profit = sum - 1
    5. Sum of NO bid prices > (N-1) →  sell all NO, profit = sum - (N-1)
  Crossed book:
    6. Any bucket YES bid > YES ask  →  buy ask, sell bid
    7. Any bucket NO bid > NO ask    →  buy ask, sell bid

Verifies prices against live CLOB orderbook data.
Parallel orderbook fetches via asyncio.gather for speed.

Usage::

    python -m weather.arb                              # scan all temperature markets (dry-run)
    python -m weather.arb --locations Wuhan,Shanghai   # filter to specific locations
    python -m weather.arb --execute --max-buy 20       # actually place orders
    python -m weather.arb --verbose --debug            # full diagnostics
"""

import argparse
import asyncio
import logging
import re
import sys
from collections import defaultdict

logger = logging.getLogger(__name__)

# Max concurrent orderbook fetches to avoid rate-limiting
_ORDERBOOK_SEMAPHORE_LIMIT = 10
_FILL_TIMEOUT = 60.0
_FILL_POLL_INTERVAL = 2.0

# Pattern to identify temperature events
_TEMP_EVENT_RE = re.compile(
    r"highest temperature in .+ on ",
    re.IGNORECASE,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_temperature_event(event_name: str) -> bool:
    """Return True if this event is a 'Highest temperature in X on Y?' event."""
    return bool(_TEMP_EVENT_RE.search(event_name))


def _parse_temp_event(event_name: str) -> dict | None:
    """Parse 'Highest temperature in <city> on <Month> <day>?' into parts.

    Returns {"city": str, "date_str": str} or None.
    """
    m = re.match(
        r"Highest temperature in (.+?) on (.+?)(?:\?|$)",
        event_name,
        re.IGNORECASE,
    )
    if m:
        return {"city": m.group(1).strip(), "date_str": m.group(2).strip()}
    return None


def _polymarket_event_url(slug: str) -> str:
    """Build Polymarket event URL from slug."""
    if slug:
        return f"https://polymarket.com/event/{slug}"
    return "https://polymarket.com"


async def _get_orderbook(clob, token_id: str, sem: asyncio.Semaphore) -> dict | None:
    """Fetch full orderbook for a token with concurrency limiting."""
    if not token_id:
        return None
    async with sem:
        try:
            return await clob.get_orderbook(token_id)
        except Exception as e:
            logger.debug("Orderbook fetch failed for token %s: %s", token_id[:16], e)
            return None


def _best_price(book: dict | None, side: str) -> float | None:
    """Extract best bid or ask price from an orderbook dict.

    Returns None if no orders exist on that side — callers must treat None
    as "no data", never as zero.
    """
    if not book:
        return None
    levels = book.get(side, [])
    if levels:
        try:
            return float(levels[0]["price"])
        except (KeyError, ValueError, TypeError):
            pass
    return None


def _walk_book(levels: list[dict]) -> list[tuple[float, float]]:
    """Parse orderbook levels into [(price, size), ...]."""
    result = []
    for lvl in levels:
        try:
            result.append((float(lvl["price"]), float(lvl["size"])))
        except (KeyError, ValueError, TypeError):
            continue
    return result


def _bucket_asks(bucket: dict, side: str) -> list[tuple[float, float]]:
    """Get parsed ask levels for a bucket side ('yes' or 'no'). Lazy parse from raw book."""
    book = bucket.get(f"_{side}_book")
    if not book:
        return []
    return _walk_book(book.get("asks", []))


def _bucket_ask_depth(bucket: dict, side: str) -> int:
    """Total ask-side depth (shares) for a bucket."""
    return int(sum(s for _, s in _bucket_asks(bucket, side)))


def compute_arb_depth(
    bucket_books: list[list[tuple[float, float]]],
    threshold: float,
    min_order_usd: float = 1.0,
) -> dict:
    """Compute max shares where sum of marginal costs across buckets stays below threshold.

    For an all-YES arb with N buckets, threshold=1.0: we can keep buying 1 share
    of each bucket as long as the marginal cost of the Nth share across all buckets < 1.0.

    For all-NO arb with N buckets, threshold=N-1.

    We walk all books in lockstep: at each "depth step" we buy 1 share from each
    bucket at its current ask level, check if sum of prices < threshold.

    Args:
        bucket_books: List of ask-side books, one per bucket. Each is [(price, size), ...].
        threshold: Max allowed sum of prices across all buckets for one share of each.
        min_order_usd: Minimum order amount per bucket (Polymarket requires $1).

    Returns:
        {
            "max_shares": int — max shares you can buy of each bucket
            "total_cost": float — total cost across all buckets for max_shares
            "profit_per_share": float — threshold - sum_of_asks at best level
            "total_profit": float — profit_per_share * max_shares (approximate)
            "min_depth_bucket": str — which bucket runs out of depth first (index)
            "bucket_depths": list[int] — available shares per bucket at arb-valid prices
            "executable": bool — whether we can actually place orders (enough depth for min_order)
            "limiting_factor": str — what limits the arb size
        }
    """
    n_buckets = len(bucket_books)
    if n_buckets == 0:
        return {"max_shares": 0, "total_cost": 0, "profit_per_share": 0,
                "total_profit": 0, "executable": False, "limiting_factor": "no buckets",
                "bucket_depths": [], "min_depth_bucket": ""}

    # For each bucket, build a cumulative depth: [(cumulative_shares, marginal_price), ...]
    # This tells us: at depth D shares, the marginal price for the next share is P.
    bucket_cursors = []
    for book in bucket_books:
        # Expand levels into per-share marginal prices
        # e.g. [(0.99, 100), (0.995, 50)] → first 100 shares at 0.99, next 50 at 0.995
        cursor = {"levels": book, "level_idx": 0, "level_remaining": 0.0}
        if book:
            cursor["level_remaining"] = book[0][1]  # size of first level
        bucket_cursors.append(cursor)

    max_shares = 0
    total_cost = 0.0

    # Check if any bucket has zero depth
    empty_buckets = [i for i, b in enumerate(bucket_books) if not b]
    if empty_buckets:
        # Calculate best-level profit for reporting even if not executable
        best_prices = []
        for book in bucket_books:
            best_prices.append(book[0][0] if book else 0.0)
        return {
            "max_shares": 0,
            "total_cost": 0,
            "profit_per_share": threshold - sum(best_prices) if all(p > 0 for p in best_prices) else 0,
            "total_profit": 0,
            "executable": False,
            "limiting_factor": f"no asks on bucket(s) {empty_buckets}",
            "bucket_depths": [0 if not b else int(sum(s for _, s in b)) for b in bucket_books],
            "min_depth_bucket": str(empty_buckets[0]),
        }

    while True:
        # Get marginal price for each bucket at current depth
        marginal_prices = []
        exhausted = False
        for i, cur in enumerate(bucket_cursors):
            idx = cur["level_idx"]
            if idx >= len(cur["levels"]):
                exhausted = True
                break
            price = cur["levels"][idx][0]
            marginal_prices.append(price)

        if exhausted or len(marginal_prices) < n_buckets:
            break

        # Check if sum of marginal prices still beats the threshold
        price_sum = sum(marginal_prices)
        if price_sum >= threshold:
            break

        # Buy 1 share from each bucket
        # Actually, buy as many as the smallest remaining level allows
        min_remaining = min(cur["level_remaining"] for cur in bucket_cursors)
        if min_remaining <= 0:
            # Advance any exhausted levels
            for cur in bucket_cursors:
                if cur["level_remaining"] <= 0:
                    cur["level_idx"] += 1
                    idx = cur["level_idx"]
                    if idx < len(cur["levels"]):
                        cur["level_remaining"] = cur["levels"][idx][1]
            continue

        # Buy min_remaining shares at current marginal prices
        shares_to_buy = int(min_remaining)  # integer shares
        if shares_to_buy < 1:
            shares_to_buy = 1

        # Re-check with these exact prices
        step_cost = sum(marginal_prices) * shares_to_buy
        max_shares += shares_to_buy
        total_cost += step_cost

        # Deduct from all cursors
        for cur in bucket_cursors:
            cur["level_remaining"] -= shares_to_buy
            if cur["level_remaining"] <= 0:
                cur["level_idx"] += 1
                idx = cur["level_idx"]
                if idx < len(cur["levels"]):
                    cur["level_remaining"] = cur["levels"][idx][1]

    # Compute per-bucket depths (total shares available at any price)
    bucket_depths = [int(sum(s for _, s in b)) for b in bucket_books]
    min_depth_idx = min(range(n_buckets), key=lambda i: bucket_depths[i])

    # Check minimum order constraint: cheapest bucket must have cost >= min_order_usd
    # For each bucket, cost = price * shares. If price is $0.001 and min_order is $1,
    # need 1000 shares minimum.
    min_shares_per_bucket = []
    for book in bucket_books:
        if book:
            cheapest_price = book[0][0]
            if cheapest_price > 0:
                min_shares_per_bucket.append(int(min_order_usd / cheapest_price) + 1)
            else:
                min_shares_per_bucket.append(0)
        else:
            min_shares_per_bucket.append(0)

    min_required = max(min_shares_per_bucket) if min_shares_per_bucket else 0

    # Best-level profit
    best_prices = [book[0][0] if book else 0.0 for book in bucket_books]
    best_sum = sum(best_prices)
    profit_per_share = threshold - best_sum if best_sum < threshold else 0.0

    executable = max_shares >= min_required and max_shares > 0
    if max_shares == 0:
        limiting_factor = "no depth at arb-valid prices"
    elif max_shares < min_required:
        limiting_factor = f"need {min_required} shares for $1 min order but only {max_shares} at arb prices"
    else:
        limiting_factor = "none"

    return {
        "max_shares": max_shares,
        "total_cost": round(total_cost, 4),
        "profit_per_share": round(profit_per_share, 6),
        "total_profit": round(profit_per_share * max_shares, 4),
        "executable": executable,
        "limiting_factor": limiting_factor,
        "bucket_depths": bucket_depths,
        "min_depth_bucket": str(min_depth_idx),
        "min_shares_required": min_required,
    }


# ------------------------------------------------------------------
# Scanning
# ------------------------------------------------------------------

async def scan_arbs(
    bridge,
    clob,
    events_raw: list[dict] | None = None,
    locations: list[str] | None = None,
    verbose: bool = False,
) -> tuple[list[dict], int]:
    """Scan temperature markets for arbitrage opportunities.

    Args:
        bridge: CLOBWeatherBridge for market discovery.
        clob: PublicClient for orderbook reads.
        events_raw: Raw event dicts from Gamma API. If None, uses bridge.last_events_raw.
        locations: Optional location name filter. None = scan all.
        verbose: If True, print non-arb events too.

    Returns (arbs_list, events_scanned_count).
    """
    markets = await bridge.fetch_weather_markets()
    if events_raw is None:
        events_raw = bridge.last_events_raw
    logger.info("Fetched %d active weather markets from Gamma", len(markets))

    # Group by event_id
    events: dict[str, list[dict]] = defaultdict(list)
    for m in markets:
        events[m.get("event_id") or m.get("event_name", "unknown")].append(m)

    logger.info("Grouped into %d events total", len(events))

    # Build lookups from raw events:
    #   event_id → {slug, title}
    #   condition_id → umaResolutionStatus
    event_meta: dict[str, dict] = {}
    uma_status: dict[str, str] = {}  # condition_id → status
    for ev in events_raw:
        eid = str(ev.get("id", ""))
        event_meta[eid] = {
            "slug": ev.get("slug", ""),
            "title": ev.get("title", ""),
        }
        for rm in ev.get("markets") or []:
            cid = rm.get("conditionId", "")
            status = rm.get("umaResolutionStatus") or ""
            if cid and status:
                uma_status[cid] = status

    loc_set = {loc.lower() for loc in locations} if locations else None
    sem = asyncio.Semaphore(_ORDERBOOK_SEMAPHORE_LIMIT)
    arbs: list[dict] = []
    skipped_non_temp = 0

    # --- Phase 1: Filter events and collect orderbook fetch tasks ---
    scannable: list[dict] = []  # [{event_id, event_markets, label, slug, ...}]
    all_fetch_tasks = []
    task_index = []  # maps each task pair to its scannable index

    for event_id, event_markets in events.items():
        if not event_markets:
            continue

        event_name = event_markets[0].get("event_name", "")

        if not _is_temperature_event(event_name):
            skipped_non_temp += 1
            continue

        # Filter out in-review buckets
        active_markets = []
        in_review_count = 0
        for m in event_markets:
            cid = m.get("id", "")
            if uma_status.get(cid, "") == "proposed":
                in_review_count += 1
                logger.info("Skipping in-review bucket '%s' in '%s'",
                            m.get("outcome_name", "?"), event_name)
            else:
                active_markets.append(m)

        if len(active_markets) < 2:
            if in_review_count > 0:
                logger.info("Skipping '%s' — only %d active bucket(s) after excluding %d in-review",
                            event_name, len(active_markets), in_review_count)
            continue

        parsed = _parse_temp_event(event_name)
        city = parsed["city"] if parsed else event_name
        date_str = parsed["date_str"] if parsed else ""

        if loc_set and city.lower() not in loc_set:
            continue

        meta = event_meta.get(event_id, {})
        slug = meta.get("slug", "")
        label = f"{city} on {date_str}" if date_str else city

        token_pairs = [
            (m.get("token_id_yes", ""), m.get("token_id_no", ""),
             m.get("outcome_name", "?"), m.get("id", ""))
            for m in active_markets
        ]

        entry = {
            "event_id": event_id, "event_name": event_name,
            "label": label, "slug": slug,
            "event_url": _polymarket_event_url(slug),
            "event_markets": active_markets, "token_pairs": token_pairs,
            "n_buckets": len(active_markets),
            "fetch_start": len(all_fetch_tasks),
        }
        scannable.append(entry)

        for tok_yes, tok_no, _, _ in token_pairs:
            all_fetch_tasks.append(_get_orderbook(clob, tok_yes, sem))
            all_fetch_tasks.append(_get_orderbook(clob, tok_no, sem))

    logger.info("Fetching orderbooks for %d events (%d requests)...",
                len(scannable), len(all_fetch_tasks))

    # --- Phase 2: Single parallel fetch across ALL events ---
    all_books = await asyncio.gather(*all_fetch_tasks) if all_fetch_tasks else []

    # --- Phase 3: Process results and detect arbs ---
    scanned = len(scannable)
    for entry in scannable:
        event_id = entry["event_id"]
        event_name = entry["event_name"]
        label = entry["label"]
        slug = entry["slug"]
        event_url = entry["event_url"]
        n_buckets = entry["n_buckets"]
        token_pairs = entry["token_pairs"]
        fetch_start = entry["fetch_start"]

        logger.info("Scanning: %s (%d buckets) [event_id=%s, slug=%s]",
                     label, n_buckets, event_id, slug)

        buckets = []
        for i, (token_yes, token_no, outcome, market_id) in enumerate(token_pairs):
            idx = fetch_start + 2 * i
            yes_book = all_books[idx]
            no_book = all_books[idx + 1]

            bucket = {
                "outcome": outcome,
                "yes_ask": _best_price(yes_book, "asks"),
                "yes_bid": _best_price(yes_book, "bids"),
                "no_ask": _best_price(no_book, "asks"),
                "no_bid": _best_price(no_book, "bids"),
                "market_id": market_id,
                "token_id_yes": token_yes,
                "token_id_no": token_no,
                "event_id": event_id,
                "event_name": event_name,
                "event_slug": slug,
                "_yes_book": yes_book,
                "_no_book": no_book,
            }
            buckets.append(bucket)

            def _fmt(v): return f"{v:.3f}" if v is not None else "—"
            logger.debug(
                "  '%s' (market=%s): YES ask=%s bid=%s | NO ask=%s bid=%s",
                outcome, market_id[:16] if market_id else "?",
                _fmt(bucket["yes_ask"]), _fmt(bucket["yes_bid"]),
                _fmt(bucket["no_ask"]), _fmt(bucket["no_bid"]),
            )

        # --- Check arb conditions ---
        # Prices are None when no orders exist on that book side.
        # An arb is only valid when ALL required prices are real.

        all_yes_asks = [b["yes_ask"] for b in buckets if b["yes_ask"] is not None]
        all_no_asks = [b["no_ask"] for b in buckets if b["no_ask"] is not None]
        all_yes_bids = [b["yes_bid"] for b in buckets if b["yes_bid"] is not None]
        all_no_bids = [b["no_bid"] for b in buckets if b["no_bid"] is not None]

        has_all_yes_asks = len(all_yes_asks) == n_buckets
        has_all_no_asks = len(all_no_asks) == n_buckets
        has_all_yes_bids = len(all_yes_bids) == n_buckets
        has_all_no_bids = len(all_no_bids) == n_buckets

        yes_ask_sum = sum(all_yes_asks) if has_all_yes_asks else 0.0
        no_ask_sum = sum(all_no_asks) if has_all_no_asks else 0.0
        yes_bid_sum = sum(all_yes_bids) if has_all_yes_bids else 0.0
        no_bid_sum = sum(all_no_bids) if has_all_no_bids else 0.0
        found_arb = False

        base_arb = {
            "label": label,
            "event_id": event_id,
            "event_name": event_name,
            "event_slug": slug,
            "event_url": event_url,
            "n_buckets": n_buckets,
        }

        # 1. All YES asks < 1.0 — buy all YES (guaranteed: one must win)
        if has_all_yes_asks and yes_ask_sum < 1.0:
            profit = 1.0 - yes_ask_sum
            depth = compute_arb_depth(
                [_bucket_asks(b, "yes") for b in buckets], threshold=1.0,
            )
            arb = {
                **base_arb, "type": "all_yes",
                "sum": yes_ask_sum, "threshold": 1.0,
                "profit": profit, "buckets": buckets,
                "executable": depth["executable"],
                "depth": depth,
            }
            arbs.append(arb)
            found_arb = True
            logger.info("  ARB [ALL YES]: %s — sum=%.4f < 1.0, profit=$%.4f/share, "
                        "depth=%d shares ($%.2f total profit), %s  %s",
                        label, yes_ask_sum, profit,
                        depth["max_shares"], depth["total_profit"],
                        depth["limiting_factor"], event_url)

        # 2. All NO asks < (N-1) — buy all NO (guaranteed: N-1 must win)
        if has_all_no_asks and no_ask_sum < n_buckets - 1:
            profit = (n_buckets - 1) - no_ask_sum
            depth = compute_arb_depth(
                [_bucket_asks(b, "no") for b in buckets], threshold=n_buckets - 1,
            )
            arb = {
                **base_arb, "type": "all_no",
                "sum": no_ask_sum, "threshold": n_buckets - 1,
                "profit": profit, "buckets": buckets,
                "executable": depth["executable"],
                "depth": depth,
            }
            arbs.append(arb)
            found_arb = True
            logger.info("  ARB [ALL NO]: %s — sum=%.4f < %d, profit=$%.4f/share, "
                        "depth=%d shares ($%.2f total profit), %s  %s",
                        label, no_ask_sum, n_buckets - 1, profit,
                        depth["max_shares"], depth["total_profit"],
                        depth["limiting_factor"], event_url)

        # 3. Individual YES + NO asks < 1.0 — buy both sides
        for b in buckets:
            if b["yes_ask"] is None or b["no_ask"] is None:
                continue  # need real prices on both sides
            total = b["yes_ask"] + b["no_ask"]
            if total < 1.0:
                profit = 1.0 - total
                depth = compute_arb_depth(
                    [_bucket_asks(b, "yes"), _bucket_asks(b, "no")], threshold=1.0,
                )
                arbs.append({
                    **base_arb, "type": "yes_no_pair",
                    "outcome": b["outcome"],
                    "yes_ask": b["yes_ask"], "no_ask": b["no_ask"],
                    "total": total, "profit": profit,
                    "bucket": b, "executable": depth["executable"],
                    "depth": depth,
                })
                found_arb = True
                logger.info("  ARB [YES+NO]: %s '%s' — YES@%.3f + NO@%.3f = %.4f, profit=$%.4f, "
                            "depth=%d shares  %s",
                            label, b["outcome"], b["yes_ask"], b["no_ask"], total, profit,
                            depth["max_shares"], event_url)

        # 4. Sum of YES bids > 1.0 — sell all YES
        if has_all_yes_bids and yes_bid_sum > 1.0:
            profit = yes_bid_sum - 1.0
            arbs.append({
                **base_arb, "type": "sell_all_yes",
                "sum": yes_bid_sum, "threshold": 1.0,
                "profit": profit, "buckets": buckets, "executable": False,
            })
            found_arb = True
            logger.info("  ARB [SELL ALL YES]: %s — sum=%.4f > 1.0, profit=$%.4f (needs inventory)  %s",
                        label, yes_bid_sum, profit, event_url)

        # 5. Sum of NO bids > (N-1) — sell all NO
        if has_all_no_bids and no_bid_sum > n_buckets - 1:
            profit = no_bid_sum - (n_buckets - 1)
            arbs.append({
                **base_arb, "type": "sell_all_no",
                "sum": no_bid_sum, "threshold": n_buckets - 1,
                "profit": profit, "buckets": buckets, "executable": False,
            })
            found_arb = True
            logger.info("  ARB [SELL ALL NO]: %s — sum=%.4f > %d, profit=$%.4f (needs inventory)  %s",
                        label, no_bid_sum, n_buckets - 1, profit, event_url)

        # 6 & 7. Crossed books — bid > ask on same side
        for b in buckets:
            if b["yes_bid"] is not None and b["yes_ask"] is not None and b["yes_bid"] > b["yes_ask"]:
                profit = b["yes_bid"] - b["yes_ask"]
                arbs.append({
                    **base_arb, "type": "crossed_yes",
                    "outcome": b["outcome"],
                    "bid": b["yes_bid"], "ask": b["yes_ask"],
                    "profit": profit, "bucket": b, "executable": True,
                })
                found_arb = True
                logger.info("  ARB [CROSSED YES]: %s '%s' — bid=%.3f > ask=%.3f, profit=$%.4f",
                            label, b["outcome"], b["yes_bid"], b["yes_ask"], profit)

            if b["no_bid"] is not None and b["no_ask"] is not None and b["no_bid"] > b["no_ask"]:
                profit = b["no_bid"] - b["no_ask"]
                arbs.append({
                    **base_arb, "type": "crossed_no",
                    "outcome": b["outcome"],
                    "bid": b["no_bid"], "ask": b["no_ask"],
                    "profit": profit, "bucket": b, "executable": True,
                })
                found_arb = True
                logger.info("  ARB [CROSSED NO]: %s '%s' — bid=%.3f > ask=%.3f, profit=$%.4f",
                            label, b["outcome"], b["no_bid"], b["no_ask"], profit)

        if verbose and not found_arb:
            missing = []
            if not has_all_yes_asks: missing.append(f"{n_buckets - len(all_yes_asks)} YES asks missing")
            if not has_all_no_asks: missing.append(f"{n_buckets - len(all_no_asks)} NO asks missing")
            miss_str = f" ({', '.join(missing)})" if missing else ""
            print(f"  OK: {label} ({n_buckets} buckets){miss_str} — "
                  f"YES asks: {yes_ask_sum:.3f}/{n_buckets}, NO asks: {no_ask_sum:.3f}/{n_buckets}  "
                  f"{event_url}")

    logger.info("Scan complete: %d temperature events scanned, %d non-temperature skipped",
                scanned, skipped_non_temp)
    return arbs, scanned


# ------------------------------------------------------------------
# Pretty printing
# ------------------------------------------------------------------

def _print_depth(d: dict) -> None:
    """Print depth analysis details for an arb."""
    if not d:
        return
    print(f"   Depth: {d['max_shares']} shares @ arb-valid prices, "
          f"total profit: ${d['total_profit']:.4f}, total cost: ${d['total_cost']:.2f}")
    if d.get("min_shares_required", 0) > 0:
        print(f"   Min order: {d['min_shares_required']} shares "
              f"(Polymarket $1 minimum on cheapest bucket)")
    if d.get("limiting_factor", "none") != "none":
        print(f"   Limiting: {d['limiting_factor']}")


def _print_results(arbs: list[dict], scanned: int) -> None:
    """Pretty-print arb scan results with full context."""
    if not arbs:
        print(f"\nNo arbitrage opportunities found across {scanned} temperature events.")
        return

    print(f"\n{'=' * 70}")
    print(f"  ARBITRAGE OPPORTUNITIES ({len(arbs)} found across {scanned} temperature events)")
    print(f"{'=' * 70}")

    for arb in arbs:
        atype = arb["type"]
        url = arb.get("event_url", "")
        depth = arb.get("depth", {})
        exec_tag = ""
        if not arb.get("executable"):
            if depth and not depth.get("executable"):
                exec_tag = f" [NO DEPTH: {depth.get('limiting_factor', '?')}]"
            else:
                exec_tag = " [REQUIRES INVENTORY]"

        # Common header
        n_str = f" ({arb['n_buckets']} buckets)" if "n_buckets" in arb else ""
        print(f"\n  ARB [{atype.upper().replace('_', ' ')}]{exec_tag}: {arb['label']}{n_str}")
        print(f"   Event: {arb['event_name']}")
        print(f"   URL: {url}")

        # Type-specific details
        if atype in ("all_yes", "all_no"):
            side = "YES" if atype == "all_yes" else "NO"
            fmt = ".1f" if arb["threshold"] == 1.0 else ".0f"
            print(f"   {side} ask sum: {arb['sum']:.4f} < {arb['threshold']:{fmt}} -> profit: ${arb['profit']:.4f}/share")
            _print_depth(depth)
            price_key = "yes_ask" if atype == "all_yes" else "no_ask"
            tok_key = "token_id_yes" if atype == "all_yes" else "token_id_no"
            depth_side = "yes" if atype == "all_yes" else "no"
            for b in arb["buckets"]:
                print(f"     '{b['outcome']}' {side} ask @{b[price_key]:.3f}  "
                      f"(depth: {_bucket_ask_depth(b, depth_side)} shares, token: {b[tok_key][:20]}...)")

        elif atype == "yes_no_pair":
            b = arb.get("bucket", {})
            print(f"   Bucket '{arb['outcome']}': YES @{arb['yes_ask']:.3f} + NO @{arb['no_ask']:.3f} = {arb['total']:.4f} -> profit: ${arb['profit']:.4f}")
            _print_depth(depth)
            print(f"     YES token: {b.get('token_id_yes', '?')[:20]}...")
            print(f"     NO token:  {b.get('token_id_no', '?')[:20]}...")

        elif atype in ("sell_all_yes", "sell_all_no"):
            side = "YES" if atype == "sell_all_yes" else "NO"
            fmt = ".1f" if arb["threshold"] == 1.0 else ".0f"
            print(f"   {side} bid sum: {arb['sum']:.4f} > {arb['threshold']:{fmt}} -> profit: ${arb['profit']:.4f}/share")
            bid_key = "yes_bid" if atype == "sell_all_yes" else "no_bid"
            tok_key = "token_id_yes" if atype == "sell_all_yes" else "token_id_no"
            for b in arb["buckets"]:
                print(f"     '{b['outcome']}' {side} bid @{b[bid_key]:.3f}  (token: {b[tok_key][:20]}...)")

        elif atype in ("crossed_yes", "crossed_no"):
            side = "YES" if atype == "crossed_yes" else "NO"
            b = arb.get("bucket", {})
            tok_key = "token_id_yes" if atype == "crossed_yes" else "token_id_no"
            print(f"   Bucket '{arb['outcome']}': {side} bid @{arb['bid']:.3f} > ask @{arb['ask']:.3f} -> profit: ${arb['profit']:.4f}")
            print(f"     Token: {b.get(tok_key, '?')[:20]}...")

    print(f"\n{'=' * 70}")
    executable = sum(1 for a in arbs if a.get("executable"))
    detection_only = len(arbs) - executable
    print(f"Summary: {len(arbs)} arb(s) found across {scanned} temperature events "
          f"({executable} executable, {detection_only} detection-only)")


# ------------------------------------------------------------------
# Order execution & fill monitoring
# ------------------------------------------------------------------

async def _place_order(
    bridge,
    market_id: str,
    side: str,
    amount: float,
    outcome_name: str,
    event_name: str,
    token_id: str,
) -> dict:
    """Place a single order via bridge.execute_trade, with rich logging."""
    logger.info(
        "Placing order: BUY %s $%.2f on '%s' (market=%s, token=%s) — event: %s",
        side.upper(), amount, outcome_name, market_id[:16], token_id[:16], event_name,
    )
    result = await bridge.execute_trade(
        market_id=market_id,
        side=side,
        amount=amount,
        fill_timeout=0,  # We handle fill monitoring ourselves
    )
    order_id = result.get("trade_id", "")
    if result.get("success"):
        logger.info(
            "Order placed: %s on '%s' — order_id=%s, shares=%.1f (market=%s)",
            side.upper(), outcome_name, order_id, result.get("shares_bought", 0), market_id[:16],
        )
    else:
        logger.warning(
            "Order FAILED: %s on '%s' — %s (market=%s, token=%s)",
            side.upper(), outcome_name, result.get("error", "unknown"), market_id[:16], token_id[:16],
        )
    return {**result, "outcome_name": outcome_name, "market_id": market_id, "side": side, "token_id": token_id}


async def _monitor_and_unwind(
    orders: list[dict],
    bridge,
    timeout: float = _FILL_TIMEOUT,
) -> dict:
    """Monitor placed orders for fills. Cancel + unwind partial fills.

    Returns {"filled": [...], "unfilled": [...], "all_complete": bool, "unwind_results": [...]}
    """
    live_orders = [o for o in orders if o.get("success") and o.get("trade_id")]
    failed_orders = [o for o in orders if not o.get("success")]

    if not live_orders:
        logger.warning("No orders were successfully placed — nothing to monitor")
        return {"filled": [], "unfilled": failed_orders, "all_complete": False, "unwind_results": []}

    logger.info("Monitoring %d orders for fills (timeout=%ds)...", len(live_orders), int(timeout))

    fill_results = await asyncio.gather(*[
        bridge.verify_fill(o["trade_id"], timeout_seconds=timeout, poll_interval=_FILL_POLL_INTERVAL)
        for o in live_orders
    ])

    filled = []
    unfilled = []
    for order, fill in zip(live_orders, fill_results):
        if fill["filled"] and not fill["partial"]:
            filled.append({**order, "shares_filled": fill["size_matched"]})
            logger.info("FILLED: '%s' %s — %.1f shares (order=%s)",
                        order["outcome_name"], order["side"], fill["size_matched"], order["trade_id"])
        else:
            unfilled.append(order)
            if not fill["filled"]:
                logger.warning("NOT FILLED: '%s' %s — cancelling order %s",
                               order["outcome_name"], order["side"], order["trade_id"])
                await bridge.cancel_order(order["trade_id"])
            elif fill["partial"]:
                logger.warning("PARTIAL FILL: '%s' %s — %.1f/%.1f shares, cancelling (order=%s)",
                               order["outcome_name"], order["side"],
                               fill["size_matched"], fill["original_size"], order["trade_id"])
                await bridge.cancel_order(order["trade_id"])
                unfilled[-1]["shares_to_unwind"] = fill["size_matched"]

    unwind_results = []
    all_complete = len(unfilled) == 0 and len(failed_orders) == 0

    if filled and (unfilled or failed_orders):
        logger.warning("PARTIAL ARB: %d/%d legs filled — UNWINDING to avoid exposure",
                       len(filled), len(live_orders))
        for f in filled:
            shares = f.get("shares_filled", f.get("shares_bought", 0))
            if shares <= 0:
                continue
            logger.info("Unwind: SELL %s %.1f shares on '%s' (market=%s) at market price",
                        f["side"], shares, f["outcome_name"], f["market_id"][:16])
            sell_result = await bridge.execute_sell(
                market_id=f["market_id"], shares=shares, side=f["side"],
                fill_timeout=_FILL_TIMEOUT,
            )
            unwind_results.append({**sell_result, "outcome_name": f["outcome_name"]})
            if sell_result.get("success"):
                logger.info("Unwind SOLD: '%s' — %.1f shares @ $%.4f (order=%s)",
                            f["outcome_name"], sell_result.get("shares_sold", 0),
                            sell_result.get("fill_price", 0), sell_result.get("trade_id", ""))
            else:
                logger.error("Unwind FAILED: '%s' — %s",
                             f["outcome_name"], sell_result.get("error", "unknown"))

        for u in unfilled:
            shares = u.get("shares_to_unwind", 0)
            if shares <= 0:
                continue
            logger.info("Unwind partial: SELL %s %.1f shares on '%s' (market=%s)",
                        u["side"], shares, u["outcome_name"], u["market_id"][:16])
            sell_result = await bridge.execute_sell(
                market_id=u["market_id"], shares=shares, side=u["side"],
                fill_timeout=_FILL_TIMEOUT,
            )
            unwind_results.append({**sell_result, "outcome_name": u["outcome_name"]})

    return {"filled": filled, "unfilled": unfilled + failed_orders,
            "all_complete": all_complete, "unwind_results": unwind_results}


async def execute_arb(
    arb: dict,
    bridge,
    auth_clob,
    public_clob,
    dry_run: bool,
    session_spent: float,
    max_buy: float,
    current_open: int,
    max_open: int,
    amount_per_share: float = 1.0,
) -> dict:
    """Execute or dry-run log a detected arbitrage opportunity.

    Returns {"spent": float, "new_positions": int, "success": bool}
    """
    atype = arb["type"]
    label = arb["label"]
    event_name = arb.get("event_name", label)
    event_url = arb.get("event_url", "")

    # Sell-side arbs — detection only
    if not arb.get("executable", True):
        logger.info("[SKIP] %s on '%s' — requires inventory  %s", atype.upper(), label, event_url)
        if dry_run:
            print(f"\n  [DRY RUN][SKIP] {atype.upper()}: {label} — requires inventory")
            print(f"   {event_url}")
        return {"spent": 0, "new_positions": 0, "success": False}

    # Calculate cost
    if atype in ("all_yes", "all_no"):
        buckets = arb["buckets"]
        cost = sum(
            b["yes_ask"] if atype == "all_yes" else b["no_ask"]
            for b in buckets
        ) * amount_per_share
        new_positions = len(buckets)
    elif atype == "yes_no_pair":
        b = arb["bucket"]
        cost = (b["yes_ask"] + b["no_ask"]) * amount_per_share
        new_positions = 2
    elif atype in ("crossed_yes", "crossed_no"):
        cost = arb["ask"] * amount_per_share
        new_positions = 1
    else:
        cost = 0
        new_positions = 0

    # Check limits
    if session_spent + cost > max_buy:
        logger.warning("[LIMIT] Skipping %s '%s' — exceeds max-buy ($%.2f + $%.2f > $%.2f)",
                       atype, label, session_spent, cost, max_buy)
        if dry_run:
            print(f"\n  [DRY RUN][LIMIT] {atype.upper()}: {label} — exceeds --max-buy")
        return {"spent": 0, "new_positions": 0, "success": False}

    if current_open + new_positions > max_open:
        logger.warning("[LIMIT] Skipping %s '%s' — exceeds max-open (%d + %d > %d)",
                       atype, label, current_open, new_positions, max_open)
        if dry_run:
            print(f"\n  [DRY RUN][LIMIT] {atype.upper()}: {label} — exceeds --max-open")
        return {"spent": 0, "new_positions": 0, "success": False}

    # --- Dry run ---
    if dry_run:
        print(f"\n  [DRY RUN] {atype.upper()}: {label}")
        print(f"   Event: {event_name}")
        print(f"   URL: {event_url}")
        print(f"   Estimated cost: ${cost:.4f} | Profit: ${arb['profit']:.4f}/share")

        if atype == "all_yes":
            for b in arb["buckets"]:
                print(f"     Would BUY YES '{b['outcome']}' @ ${b['yes_ask']:.3f} "
                      f"(token: {b['token_id_yes'][:20]}..., market: {b['market_id'][:16]})")
        elif atype == "all_no":
            for b in arb["buckets"]:
                print(f"     Would BUY NO '{b['outcome']}' @ ${b['no_ask']:.3f} "
                      f"(token: {b['token_id_no'][:20]}..., market: {b['market_id'][:16]})")
        elif atype == "yes_no_pair":
            b = arb["bucket"]
            print(f"     Would BUY YES '{b['outcome']}' @ ${b['yes_ask']:.3f} "
                  f"(token: {b['token_id_yes'][:20]}...)")
            print(f"     Would BUY NO  '{b['outcome']}' @ ${b['no_ask']:.3f} "
                  f"(token: {b['token_id_no'][:20]}...)")
        elif atype in ("crossed_yes", "crossed_no"):
            b = arb["bucket"]
            side_label = "YES" if atype == "crossed_yes" else "NO"
            tok = b.get("token_id_yes" if "yes" in atype else "token_id_no", "?")
            print(f"     Would BUY {side_label} '{b['outcome']}' @ ${arb['ask']:.3f} "
                  f"then SELL @ ${arb['bid']:.3f} (token: {tok[:20]}...)")

        return {"spent": 0, "new_positions": 0, "success": True}

    # --- Live execution ---
    logger.info("=" * 60)
    logger.info("EXECUTING %s: %s (cost=$%.4f, profit=$%.4f/share)  %s",
                atype.upper(), label, cost, arb["profit"], event_url)

    placed_orders = []

    if atype == "all_yes":
        tasks = [
            _place_order(bridge, b["market_id"], "yes", b["yes_ask"] * amount_per_share,
                         b["outcome"], event_name, b["token_id_yes"])
            for b in arb["buckets"]
        ]
        placed_orders = list(await asyncio.gather(*tasks))

    elif atype == "all_no":
        tasks = [
            _place_order(bridge, b["market_id"], "no", b["no_ask"] * amount_per_share,
                         b["outcome"], event_name, b["token_id_no"])
            for b in arb["buckets"]
        ]
        placed_orders = list(await asyncio.gather(*tasks))

    elif atype == "yes_no_pair":
        b = arb["bucket"]
        tasks = [
            _place_order(bridge, b["market_id"], "yes", b["yes_ask"] * amount_per_share,
                         b["outcome"], event_name, b["token_id_yes"]),
            _place_order(bridge, b["market_id"], "no", b["no_ask"] * amount_per_share,
                         b["outcome"], event_name, b["token_id_no"]),
        ]
        placed_orders = list(await asyncio.gather(*tasks))

    elif atype in ("crossed_yes", "crossed_no"):
        b = arb["bucket"]
        side = "yes" if atype == "crossed_yes" else "no"
        tok_key = "token_id_yes" if atype == "crossed_yes" else "token_id_no"
        buy_result = await _place_order(
            bridge, b["market_id"], side, arb["ask"] * amount_per_share,
            b["outcome"], event_name, b[tok_key],
        )
        placed_orders = [buy_result]
        if buy_result.get("success"):
            shares = buy_result.get("shares_bought", 0)
            if shares > 0:
                logger.info("Crossed book: selling %s %.1f shares on '%s' at bid",
                            side, shares, b["outcome"])
                sell_result = await bridge.execute_sell(
                    market_id=b["market_id"], shares=shares, side=side,
                    fill_timeout=_FILL_TIMEOUT,
                )
                if sell_result.get("success"):
                    logger.info("Crossed book COMPLETE: sold %.1f shares @ $%.4f",
                                sell_result.get("shares_sold", 0), sell_result.get("fill_price", 0))
                else:
                    logger.error("Crossed book sell FAILED: %s", sell_result.get("error", "unknown"))
        return {"spent": cost if buy_result.get("success") else 0, "new_positions": 0,
                "success": buy_result.get("success", False)}

    # Monitor fills and unwind partials for multi-leg arbs
    monitor_result = await _monitor_and_unwind(placed_orders, bridge)

    if monitor_result["all_complete"]:
        logger.info("ARB COMPLETE: %s on %s — all %d legs filled!  %s",
                    atype.upper(), label, len(placed_orders), event_url)
        return {"spent": cost, "new_positions": new_positions, "success": True}
    else:
        logger.warning("ARB INCOMPLETE: %s on %s — %d/%d legs, unwound  %s",
                       atype.upper(), label, len(monitor_result["filled"]), len(placed_orders), event_url)
        return {"spent": 0, "new_positions": 0, "success": False}


# ------------------------------------------------------------------
# Entry points
# ------------------------------------------------------------------

async def _async_main(
    locations: list[str] | None,
    verbose: bool,
    dry_run: bool,
    max_buy: float,
    max_open: int,
) -> None:
    """Async entry point."""
    from bot.gamma import GammaClient
    from polymarket.public import PublicClient
    from .bridge import CLOBWeatherBridge

    gamma = GammaClient()
    public_clob = PublicClient()
    auth_clob = None
    current_open = 0

    if not dry_run:
        from polymarket.client import PolymarketClient
        from .config import Config

        cfg = Config.load(".")
        if not cfg.private_key:
            logger.error("No private key configured — set POLY_PRIVATE_KEY env var or add to config.json.")
            sys.exit(1)

        creds = cfg.load_api_creds(".")
        auth_clob = PolymarketClient(cfg.private_key, api_creds=creds)
        logger.info("Authenticated as %s", auth_clob.address)

        try:
            open_orders = await auth_clob.get_open_orders()
            current_open = len(open_orders)
            logger.info("Current open orders: %d (max allowed: %d)", current_open, max_open)
        except Exception as e:
            logger.warning("Could not fetch open orders: %s — assuming 0", e)

    bridge = CLOBWeatherBridge(
        clob_client=auth_clob or public_clob,
        gamma_client=gamma,
        max_exposure=max_buy,
    )

    mode_str = "DRY RUN" if dry_run else "LIVE EXECUTION"
    loc_str = ", ".join(locations) if locations else "ALL locations"
    print(f"=== Temperature Arbitrage Scanner [{mode_str}] ===")
    print(f"Scanning: {loc_str}")
    if not dry_run:
        print(f"Limits: max-buy=${max_buy:.2f}, max-open={max_open}")
        print(f"Open positions: {current_open}")
    print()

    try:
        arbs, scanned = await scan_arbs(bridge, public_clob, locations=locations, verbose=verbose)
        _print_results(arbs, scanned)

        if arbs:
            session_spent = 0.0
            executed = 0
            for arb in arbs:
                result = await execute_arb(
                    arb, bridge, auth_clob, public_clob,
                    dry_run, session_spent, max_buy,
                    current_open, max_open,
                )
                session_spent += result.get("spent", 0)
                current_open += result.get("new_positions", 0)
                if result.get("success"):
                    executed += 1

            print(f"\n{'=' * 70}")
            if dry_run:
                print(f"[DRY RUN] Would have executed {executed}/{len(arbs)} arbs")
            else:
                print(f"Executed {executed}/{len(arbs)} arbs, spent ${session_spent:.2f}")
    finally:
        from .http_client import close_session
        await close_session()
        await public_clob.close()
        if auth_clob:
            await auth_clob.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="weather.arb",
        description="Scan Polymarket temperature markets for arbitrage opportunities",
    )
    parser.add_argument(
        "--locations", type=str, default=None,
        help="Comma-separated city names to filter (default: scan all temperature events)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show all events including non-arb",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        dest="dry_run",
        help="Log arbs but don't trade (default)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually place orders (disables dry-run)",
    )
    parser.add_argument(
        "--max-buy", type=float, default=10.0,
        dest="max_buy",
        help="Max total USD to buy this session (default: 10)",
    )
    parser.add_argument(
        "--max-open", type=int, default=100,
        dest="max_open",
        help="Max total open positions allowed in wallet (default: 100)",
    )
    args = parser.parse_args()

    dry_run = not args.execute

    level = "DEBUG" if args.debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Location filter uses city names from event titles, not config keys
    loc_keys = None
    if args.locations:
        loc_keys = [loc.strip() for loc in args.locations.split(",")]

    asyncio.run(_async_main(loc_keys, verbose=args.verbose, dry_run=dry_run,
                            max_buy=args.max_buy, max_open=args.max_open))


if __name__ == "__main__":
    main()
