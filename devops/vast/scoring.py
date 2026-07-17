"""Offer gating + ranking (pure functions, no network — easy to unit test).

vast.ai offers only expose a coarse ``geolocation`` string ("California, US"),
so there is no lat/long and true geodistance is impossible. "Proximity" is
therefore an ordered region-preference list (``HOME_REGIONS``); it only acts as
a tiebreak between near-equal prices.
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
    offer_id: Optional[int] = None,
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
    if offer_id is not None:
        parts.append(f"id={int(offer_id)}")
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
) -> list[RankedOffer]:
    """Gate then rank offers, returning the best ``count`` across distinct hosts.

    Sort key = ``(round(price / PRICE_TOLERANCE), region_rank, price)`` so that
    prices within one tolerance band are considered equal and proximity (region
    preference) breaks the tie; exact price is the final tiebreak.
    """
    regions = list(regions) if regions is not None else list(cfg.HOME_REGIONS)
    excluded = {str(machine_id) for machine_id in excluded_machine_ids or ()}
    ranked: list[RankedOffer] = []
    for o in offers:
        if str(o.get("machine_id")) in excluded:
            continue
        price = effective_price(o, offer_type, bid, cfg)
        if not _passes_gates(o, cfg, disk, price, max_price):
            continue
        cc = country_code(o.get("geolocation"))
        rr = region_rank(cc, regions)
        band = round(price / cfg.PRICE_TOLERANCE) if cfg.PRICE_TOLERANCE > 0 else price
        ranked.append(
            RankedOffer(
                offer=o,
                price=price,
                region=cc,
                region_rank=rr,
                sort_key=(band, rr, price),
            )
        )

    ranked.sort(key=lambda r: r.sort_key)

    # Pick the top N across distinct hosts (avoid two offers on one machine).
    picked: list[RankedOffer] = []
    seen_machines: set = set()
    for r in ranked:
        mid = r.machine_id
        if mid in seen_machines:
            continue
        seen_machines.add(mid)
        picked.append(r)
        if len(picked) >= count:
            break
    return picked
