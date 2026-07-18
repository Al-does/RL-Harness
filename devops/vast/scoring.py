"""Offer gating + ranking (pure functions, no network — easy to unit test).

vast.ai offers only expose a coarse ``geolocation`` string ("California, US"),
so there is no lat/long and true geodistance is impossible. "Proximity" is
therefore an ordered region-preference list (``HOME_REGIONS``). By default that
list is only a near-price tiebreak; when the caller passes an explicit
``regions`` list with ``require_preferred_region=True`` (CLI ``--regions``),
offers outside those countries are dropped entirely.

After hard gates, candidate prices are restricted to the upper inner quartile
``[Q2, Q3]`` among distinct hosts (reliability over cheapest-host stinginess).
Small gated pools fall back to ``max(floor * mult, floor + pad)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import VastConfig


def build_query(
    cfg: VastConfig,
    disk: float,
    regions: Optional[list[str]] = None,
    max_price: Optional[float] = None,
    machine_id: Optional[int] = None,
) -> str:
    """Build a vast search query string (server-side pre-filter).

    The authoritative gating lives in :func:`rank_offers`; this just narrows the
    result set cheaply. We keep it permissive (no region filter here) because
    ``geolocation in [...]`` behaves inconsistently across the API and regions
    are only a soft tiebreak anyway.
    """
    parts = [
        f"gpu_name={cfg.GPU_NAME}",
        f"num_gpus={cfg.NUM_GPUS}",
        "verified=true",
        "rentable=true",
        "rented=false",
        "direct_port_count>=1",
        f"disk_space>={disk + cfg.DISK_HEADROOM_GB:g}",
        f"reliability>={cfg.MIN_RELIABILITY:g}",
    ]
    if max_price is not None:
        parts.append(f"dph_total<={max_price:g}")
    if machine_id is not None:
        parts.append(f"machine_id={int(machine_id)}")
    return " ".join(parts)


def country_code(geolocation: Optional[str]) -> Optional[str]:
    """Extract the trailing 2-letter country code from a geolocation string.

    "California, US" -> "US"; "The Netherlands, NL" -> "NL"; None -> None.
    """
    if not geolocation:
        return None
    tail = geolocation.replace(",", " ").split()[-1].strip().upper()
    return tail if len(tail) == 2 and tail.isalpha() else None


def region_rank(cc: Optional[str], regions: list[str]) -> int:
    """Index of the country in the preference list; len(regions) if unlisted."""
    upper = [r.upper() for r in regions]
    if cc in upper:
        return upper.index(cc)
    return len(upper)


def effective_price(
    offer: dict,
    offer_type: str,
    bid: Optional[float],
    cfg: VastConfig,
) -> float:
    """Price used for ranking: on-demand list price, or the (auto) bid."""
    if offer_type == "interruptible":
        if bid is not None:
            return float(bid)
        return round(float(offer.get("min_bid") or 0.0) * cfg.BID_MARGIN, 4)
    return float(offer.get("dph_total") or 0.0)


def percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile; ``sorted_vals`` must be non-empty and sorted."""
    if not sorted_vals:
        raise ValueError("percentile on empty sequence")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    p = min(1.0, max(0.0, p))
    order = (len(sorted_vals) - 1) * p
    lo = int(order)
    hi = min(lo + 1, len(sorted_vals) - 1)
    weight = order - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * weight)


def price_band_bounds(
    host_prices: list[float], cfg: VastConfig
) -> tuple[float, float, str]:
    """Return ``(lo, hi, mode)`` for the reliability-oriented price window.

    * ``upper_inner_quartile`` — ``[Q2, Q3]`` when enough distinct hosts exist
    * ``floor_fallback`` — ``[floor, max(floor*mult, floor+pad)]`` otherwise
    """
    prices = sorted(float(p) for p in host_prices)
    if not prices:
        return 0.0, 0.0, "empty"
    floor = prices[0]
    if len(prices) < cfg.PRICE_BAND_MIN_HOSTS:
        hi = max(floor * cfg.PRICE_BAND_FLOOR_MULT, floor + cfg.PRICE_BAND_FLOOR_PAD)
        return floor, hi, "floor_fallback"
    q2 = percentile(prices, 0.50)
    q3 = percentile(prices, 0.75)
    return q2, q3, "upper_inner_quartile"


@dataclass
class RankedOffer:
    offer: dict
    price: float          # effective price used for ranking ($/hr)
    region: Optional[str]  # 2-letter country code
    region_rank: int
    sort_key: tuple

    @property
    def id(self) -> int:
        return int(self.offer["id"])

    @property
    def machine_id(self):
        return self.offer.get("machine_id")


def _passes_gates(
    offer: dict,
    cfg: VastConfig,
    disk: float,
    price: float,
    max_price: Optional[float],
) -> bool:
    if float(offer.get("reliability2") or 0.0) < cfg.MIN_RELIABILITY:
        return False
    # The `verified` field comes back null even for verified hosts; the real
    # signal is the `verification` string == "verified".
    if str(offer.get("verification") or "").lower() != "verified":
        return False
    if float(offer.get("duration") or 0.0) / 86400.0 < cfg.MIN_DAYS:
        return False
    if float(offer.get("disk_space") or 0.0) < disk + cfg.DISK_HEADROOM_GB:
        return False
    if int(offer.get("direct_port_count") or 0) < 1:
        return False
    if float(offer.get("cuda_max_good") or 0.0) < cfg.MIN_CUDA:
        return False
    if float(offer.get("cpu_cores_effective") or 0.0) < cfg.MIN_CPU_CORES:
        return False
    if not bool(offer.get("rentable")):
        return False
    if max_price is not None and price > max_price:
        return False
    return True


def _offer_public_ip(offer: dict) -> Optional[str]:
    ip = offer.get("public_ipaddr") or offer.get("public_ip")
    if not ip:
        return None
    text = str(ip).strip()
    return text or None


def rank_offers(
    offers: list[dict],
    cfg: VastConfig,
    disk: float,
    count: int,
    regions: Optional[list[str]] = None,
    offer_type: str = "ondemand",
    bid: Optional[float] = None,
    max_price: Optional[float] = None,
    excluded_machine_ids: Optional[set[int]] = None,
    excluded_public_ips: Optional[set[str]] = None,
    require_preferred_region: bool = False,
    apply_price_band: bool = True,
    log=None,
) -> list[RankedOffer]:
    """Gate then rank offers, returning the best ``count`` across distinct hosts.

    After hard gates (and optional region / exclusion filters), prices are
    restricted to the upper inner quartile among distinct hosts when the pool
    is large enough; otherwise a modest floor-relative cap is used. Inside the
    band, rank by reliability, CPU, mild download signal, region preference,
    then price.
    """
    regions = list(regions) if regions is not None else list(cfg.HOME_REGIONS)
    preferred = {r.upper() for r in regions}
    excluded = {str(machine_id) for machine_id in excluded_machine_ids or ()}
    excluded_ips = {str(ip).strip() for ip in excluded_public_ips or () if str(ip).strip()}

    gated: list[RankedOffer] = []
    for o in offers:
        if str(o.get("machine_id")) in excluded:
            continue
        public_ip = _offer_public_ip(o)
        if public_ip and public_ip in excluded_ips:
            continue
        price = effective_price(o, offer_type, bid, cfg)
        if not _passes_gates(o, cfg, disk, price, max_price):
            continue
        cc = country_code(o.get("geolocation"))
        if require_preferred_region and (cc is None or cc not in preferred):
            continue
        rr = region_rank(cc, regions)
        gated.append(
            RankedOffer(
                offer=o,
                price=price,
                region=cc,
                region_rank=rr,
                sort_key=(),  # filled after banding
            )
        )

    if not gated:
        return []

    # One price per host for quartile math (cheapest listing on that machine).
    host_floor: dict[str, float] = {}
    for ranked in gated:
        key = str(ranked.machine_id)
        prev = host_floor.get(key)
        if prev is None or ranked.price < prev:
            host_floor[key] = ranked.price

    if apply_price_band:
        lo, hi, mode = price_band_bounds(list(host_floor.values()), cfg)
        if log is not None:
            log(
                f"  price band [${lo:.3f}, ${hi:.3f}]/hr via {mode} "
                f"({len(host_floor)} gated host(s))"
            )
        gated = [r for r in gated if lo <= r.price <= hi]
        if not gated:
            return []

    for ranked in gated:
        offer = ranked.offer
        ranked.sort_key = (
            -float(offer.get("reliability2") or 0.0),
            -float(offer.get("cpu_cores_effective") or 0.0),
            -float(offer.get("inet_down") or 0.0),
            ranked.region_rank,
            ranked.price,
        )

    gated.sort(key=lambda r: r.sort_key)

    picked: list[RankedOffer] = []
    seen_machines: set = set()
    for r in gated:
        mid = r.machine_id
        if mid in seen_machines:
            continue
        seen_machines.add(mid)
        picked.append(r)
        if len(picked) >= count:
            break
    return picked
