"""vast.ai provisioning CLI — find, rank, rent, bootstrap, and connect to boxes.

    uv run --group devops python -m devops.vast.provision up -n 2 --dry-run
    uv run --group devops python -m devops.vast.provision up -n 1 \
        --run "rl-harness experiments.mess3_belief_geometry_2026_07.reward_only.experiment --seed 0 --smoke" --yes
    uv run --group devops python -m devops.vast.provision status
    uv run --group devops python -m devops.vast.provision destroy --all

See devops/vast/README.md for the full flag reference and cost/teardown notes.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .config import CONFIG, VastConfig
from .quarantine import active_exclusions, record_failure
from .redaction import redact_sensitive
from .scoring import RankedOffer, build_query, rank_offers

_HERE = Path(__file__).resolve().parent
BOOTSTRAP_PATH = _HERE / "bootstrap.sh"


# ---------------------------------------------------------------------------
# state.json (gitignored record of rented boxes)
# ---------------------------------------------------------------------------

def load_state(cfg: VastConfig) -> dict:
    if cfg.STATE_PATH.exists():
        try:
            return json.loads(cfg.STATE_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {"instances": []}


def save_state(cfg: VastConfig, state: dict) -> None:
    cfg.STATE_PATH.write_text(json.dumps(state, indent=2))


def _record(state: dict, entry: dict) -> None:
    state["instances"] = [i for i in state.get("instances", []) if i.get("id") != entry["id"]]
    state["instances"].append(entry)


def _unrecord(state: dict, instance_id: int) -> None:
    """Drop an instance id from local state (after a failed-readiness destroy)."""
    state["instances"] = [
        entry for entry in state.get("instances", [])
        if entry.get("id") != instance_id
    ]


# ---------------------------------------------------------------------------
# ref + token + onstart helpers
# ---------------------------------------------------------------------------

def resolve_ref(args, cfg: VastConfig, log=print) -> str:
    """Git ref to clone on the box: --commit > --branch > current local HEAD sha."""
    if args.commit:
        return args.commit
    if args.branch:
        return args.branch
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    on_remote = subprocess.run(
        ["git", "branch", "-r", "--contains", sha], capture_output=True, text=True
    ).stdout.strip()
    if not on_remote:
        log(f"WARNING: HEAD {sha[:10]} is not on any remote branch. The box clones "
            f"from {cfg.REPO_URL}, so it will fail to check this ref out. Push it "
            "first, or pass --branch main.")
    return sha


def resolve_github_token(args) -> Optional[str]:
    """Token resolution: --github-token > GITHUB_TOKEN env > `gh auth token`."""
    if getattr(args, "github_token", None):
        return args.github_token
    import os

    if os.environ.get("GITHUB_TOKEN"):
        return os.environ["GITHUB_TOKEN"]
    try:
        tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        if tok.returncode == 0 and tok.stdout.strip():
            return tok.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def build_onstart(cfg: VastConfig) -> str:
    """Inline bootstrap.sh (base64) into a compact onstart command.

    Inlining (vs. curling from the repo) means the box does not need the ref to
    be fetchable before it has cloned — the setup script travels with the create
    call, fully unattended.
    """
    raw = BOOTSTRAP_PATH.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    # tee to container stdout as well as the file so `vastai logs <id>` shows
    # bootstrap progress even when SSH is not (yet) reachable.
    return (
        f"echo {b64} | base64 -d > /root/bootstrap.sh && "
        "bash /root/bootstrap.sh 2>&1 | tee /root/bootstrap.log"
    )


def build_env(
    cfg: VastConfig,
    ref: str,
    run_cmd: Optional[str],
    self_destruct: bool,
    instance_label: str,
    run_name: str,
    results_branch: str,
    github_token: Optional[str],
    api_key: Optional[str],
    teardown_on_error: bool = False,
    max_age_s: float = 0.0,
) -> dict:
    env = {
        "VAST_REPO_URL": cfg.REPO_URL,
        "VAST_REPO_SLUG": cfg.REPO_SLUG,
        "VAST_GIT_REF": ref,
        "VAST_INSTANCE_LABEL": instance_label,
        "VAST_UV_SYNC_TIMEOUT_S": str(int(cfg.UV_SYNC_TIMEOUT_S)),
        "VAST_UV_SYNC_STALL_S": str(int(cfg.UV_SYNC_STALL_S)),
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
    }
    if run_cmd:
        env["VAST_RUN_CMD"] = run_cmd
    # Max-age watchdog: needs the API key on the box to REST-destroy itself. This
    # is independent of self-destruct (a box with no --run and no --self-destruct
    # still gets a hard lifetime cap). The key is visible to the host — same
    # tradeoff already accepted for self-destruct boxes.
    if max_age_s and max_age_s > 0:
        env["VAST_MAX_AGE_S"] = str(int(max_age_s))
        if api_key:
            env["VAST_API_KEY"] = api_key
    if self_destruct:
        env.update({
            "VAST_SELF_DESTRUCT": "1",
            "VAST_RUN_NAME": run_name,
            "VAST_RESULTS_BRANCH": results_branch,
            "GIT_USER_NAME": cfg.GIT_USER_NAME,
            "GIT_USER_EMAIL": cfg.GIT_USER_EMAIL,
        })
        if teardown_on_error:
            env["VAST_TEARDOWN_ON_ERROR"] = "1"
        if api_key:
            env["VAST_API_KEY"] = api_key
        if github_token:
            env["GITHUB_TOKEN"] = github_token
    return env


# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------

def print_offer_table(picked: list[RankedOffer], offer_type: str, log=print) -> None:
    if not picked:
        log("No offers passed the gates. Relax --max-price / regions or retry later.")
        return
    log("")
    log(f"  {'#':<3}{'offer_id':<11}{'$/hr':<8}{'region':<8}{'reliab':<8}"
        f"{'cpu':<7}{'cuda':<7}{'inet↓':<9}{'machine':<10}")
    log("  " + "-" * 79)
    for i, r in enumerate(picked, 1):
        o = r.offer
        log(f"  {i:<3}{r.id:<11}{r.price:<8.3f}{(r.region or '?'):<8}"
            f"{float(o.get('reliability2') or 0):<8.3f}"
            f"{float(o.get('cpu_cores_effective') or 0):<7.1f}"
            f"{float(o.get('cuda_max_good') or 0):<7.1f}"
            f"{float(o.get('inet_down') or 0):<9.1f}"
            f"{str(o.get('machine_id')):<10}")
    total = sum(r.price for r in picked)
    log("")
    log(f"  {len(picked)} box(es), ~${total:.3f}/hr total ({offer_type}).")


# ---------------------------------------------------------------------------
# readiness (SSH sentinel poll)
# ---------------------------------------------------------------------------

def wait_for_ready_ssh(
    host: str, port: int, identity: Path, cfg: VastConfig, log=print
) -> bool:
    """Poll over SSH until /root/.vast_ready exists (env fully installed)."""
    base = [
        "ssh", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10",
        "-o", "LogLevel=ERROR", "-i", str(identity), "-p", str(port), f"root@{host}",
    ]
    deadline = time.time() + cfg.READY_TIMEOUT_S
    announced = False
    while time.time() < deadline:
        failed = subprocess.run(base + ["test -f /root/.vast_bootstrap_failed"],
                                capture_output=True, text=True)
        if failed.returncode == 0:
            reason = subprocess.run(base + ["cat /root/.vast_bootstrap_failed"],
                                    capture_output=True, text=True).stdout.strip()
            log(f"  bootstrap FAILED on {host}: {reason}")
            return False
        ready = subprocess.run(base + ["test -f /root/.vast_ready"],
                               capture_output=True, text=True)
        if ready.returncode == 0:
            log(f"  {host}:{port} env ready")
            return True
        if not announced:
            log(f"  {host}:{port} reachable; waiting for uv sync to finish...")
            announced = True
        time.sleep(cfg.POLL_INTERVAL_S)
    log(f"  {host}:{port} not ready after {cfg.READY_TIMEOUT_S:.0f}s")
    return False


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_up(args, cfg: VastConfig) -> int:
    from .terminals import BoxConn, open_terminals, write_ssh_config
    from .vast_client import VastClient, VastClientError, resolve_api_key

    log = print
    offer_type = "interruptible" if args.mode == "interruptible" else "ondemand"
    disk = float(args.disk or cfg.DISK_GB)
    image = args.image or cfg.IMAGE
    explicit_regions = bool(args.regions)
    regions = [r.strip() for r in args.regions.split(",")] if args.regions else list(cfg.HOME_REGIONS)
    ref = resolve_ref(args, cfg, log)
    if args.offer_id is not None and args.count != 1:
        log("--offer-id selects one offer and requires --count 1.")
        return 2
    if args.machine_id is not None and args.count != 1:
        log("--machine-id selects one host and requires --count 1.")
        return 2
    if args.offer_id is not None and args.machine_id is not None:
        log("Use only one of --offer-id or --machine-id.")
        return 2

    client = VastClient(cfg)
    api_key = client.api_key or resolve_api_key(cfg)

    excluded_machines = set(args.exclude_machine or ())
    quarantined_machines, quarantined_ips = active_exclusions(cfg)
    if quarantined_machines:
        excluded_machines |= quarantined_machines
        shown = ", ".join(str(mid) for mid in sorted(quarantined_machines))
        log(f"  quarantine: excluding machine(s) {shown}")
    if quarantined_ips:
        shown = ", ".join(sorted(quarantined_ips))
        log(f"  quarantine: excluding public IP(s) {shown}")
    query = build_query(
        cfg, disk, regions, args.max_price, machine_id=args.machine_id
    )
    log(f"searching {offer_type} offers: {query}")
    offers = client.search_offers(query, offer_type=offer_type)
    log(f"  {len(offers)} raw offer(s) returned")
    pinned = args.offer_id is not None or args.machine_id is not None
    if args.offer_id is not None:
        offers = [
            offer for offer in offers
            if int(offer.get("id") or -1) == args.offer_id
        ]
        if not offers:
            log(f"Exact offer {args.offer_id} was not returned; it may no longer be rentable.")
    if args.machine_id is not None:
        offers = [
            offer for offer in offers
            if int(offer.get("machine_id") or -1) == args.machine_id
        ]
        if not offers:
            log(f"Machine {args.machine_id} has no rentable offers right now.")
    if excluded_machines:
        excluded = ", ".join(str(machine_id) for machine_id in sorted(excluded_machines))
        log(f"  excluding machine(s): {excluded}")
    if explicit_regions:
        log(f"  requiring region(s): {', '.join(regions)}")
    # Rank a candidate *pool* larger than the request so we can fall through to
    # the next-best offer when a top pick turns out to be unavailable on-demand.
    pool_size = (
        1 if pinned else max(args.count * 6, args.count + 6)
    )
    pool = rank_offers(
        offers, cfg, disk=disk, count=pool_size,
        regions=regions, offer_type=offer_type, bid=args.bid, max_price=args.max_price,
        excluded_machine_ids=excluded_machines,
        excluded_public_ips=quarantined_ips,
        require_preferred_region=explicit_regions,
        apply_price_band=not pinned,
        log=log,
    )
    picked = pool[:args.count]
    print_offer_table(picked, offer_type, log)

    if args.dry_run:
        log("\n--dry-run: no instances rented.")
        return 0
    if not pool:
        return 1

    if not args.yes:
        resp = input(f"\nRent {len(picked)} box(es)? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            log("aborted.")
            return 1

    # self-destruct prerequisites
    github_token = None
    if args.self_destruct:
        github_token = resolve_github_token(args)
        if not github_token:
            log("WARNING: --self-destruct set but no GitHub token found "
                "(--github-token / GITHUB_TOKEN / `gh auth token`); results push will be skipped.")
    results_branch = args.results_branch or cfg.DEFAULT_RESULTS_BRANCH
    run_name = args.run_name or f"run-{time.strftime('%Y%m%d-%H%M%S')}"

    max_age_hours = cfg.MAX_AGE_HOURS if args.max_age is None else args.max_age
    max_age_s = float(max_age_hours) * 3600.0 if max_age_hours and max_age_hours > 0 else 0.0
    if max_age_s > 0:
        log(f"max-age cap: boxes self-destruct after {max_age_hours:g}h "
            "(on-box watchdog; `reap` is the local backstop)")
    else:
        log("WARNING: max-age cap disabled; boxes have no automatic lifetime limit")

    pubkey = client.ensure_ssh_key(cfg.SSH_KEY_PATH)
    onstart = build_onstart(cfg)
    state = load_state(cfg)

    identity = Path(cfg.SSH_KEY_PATH).expanduser()
    identity = identity.with_suffix("") if identity.suffix == ".pub" else identity
    offers_left = list(pool)
    failed_machines = {str(mid) for mid in excluded_machines}
    failed_public_ips: set[str] = set()
    boxes: list[BoxConn] = []
    shot = 0

    def offer_public_ip(ranked: RankedOffer) -> Optional[str]:
        ip = ranked.offer.get("public_ipaddr") or ranked.offer.get("public_ip")
        return str(ip).strip() if ip else None

    def create_one(ranked: RankedOffer) -> Optional[dict]:
        nonlocal shot
        shot += 1
        instance_label = f"rllib-{run_name}-{shot}-{uuid.uuid4().hex[:6]}"
        bid = None
        if offer_type == "interruptible":
            bid = (
                args.bid if args.bid is not None
                else round(float(ranked.offer.get("min_bid") or 0) * cfg.BID_MARGIN, 4)
            )
        env = build_env(
            cfg, ref, args.run, args.self_destruct, instance_label,
            run_name, results_branch, github_token, api_key,
            teardown_on_error=args.teardown_on_error, max_age_s=max_age_s,
        )
        log(f"renting offer {ranked.id} (${ranked.price:.3f}/hr, {ranked.region}) "
            f"-> label {instance_label}")
        try:
            iid = client.create_instance(
                ranked.id, image=image, disk=disk, env=env, label=instance_label,
                onstart_cmd=onstart, bid=bid,
            )
        except VastClientError as e:
            log(f"  offer {ranked.id} skipped: {e}")
            return None
        entry = {
            "id": iid, "label": instance_label, "offer_id": ranked.id,
            "machine_id": ranked.machine_id, "price": ranked.price, "mode": offer_type,
            "region": ranked.region, "run_name": run_name, "ref": ref,
            "public_ip": offer_public_ip(ranked),
            "self_destruct": bool(args.self_destruct),
            "created_at": time.time(),
            "max_age_s": max_age_s,
        }
        client.attach_ssh_key(iid, pubkey)
        _record(state, entry)
        save_state(cfg, state)
        log(f"  created instance {iid}")
        return entry

    def connect_box(entry: dict):
        iid = entry["id"]
        log(f"waiting for instance {iid} to run...")
        readiness_client = VastClient(cfg, api_key=api_key)
        try:
            inst = readiness_client.wait_until_running(iid, log=log)
        except VastClientError as e:
            log(f"  {e}")
            return entry, False
        host, port = readiness_client.connection_info(inst, probe=True)
        entry["host"], entry["port"] = host, port
        if inst.get("public_ipaddr"):
            entry["public_ip"] = str(inst.get("public_ipaddr")).strip()
        ready = wait_for_ready_ssh(host, port, identity, cfg, log)
        entry["ready"] = ready
        return entry, ready

    def destroy_unready(entry: dict, reason: str) -> None:
        iid = entry["id"]
        machine = entry.get("machine_id")
        public_ip = entry.get("public_ip") or entry.get("host")
        log(f"  destroying unready instance {iid} ({reason}); "
            f"excluding machine {machine}"
            + (f" / ip {public_ip}" if public_ip else ""))
        try:
            client.destroy_instance(iid)
        except Exception as error:  # noqa: BLE001 — best-effort cleanup
            detail = redact_sensitive(error, secrets=(api_key,))
            log(f"  warning: destroy {iid} failed: {detail}")
        if machine is not None:
            failed_machines.add(str(machine))
        host = entry.get("host")
        quarantine_ip = entry.get("public_ip")
        if not quarantine_ip and host and str(host).replace(".", "").isdigit():
            quarantine_ip = str(host)
        if quarantine_ip:
            failed_public_ips.add(str(quarantine_ip))
        record_failure(
            cfg,
            machine_id=machine,
            public_ip=quarantine_ip,
            reason=reason,
        )
        _unrecord(state, iid)
        save_state(cfg, state)

    # Create a concurrent batch, wait for readiness, then replace any failed
    # hosts from the remaining ranked pool so one bad machine does not consume
    # the whole request (and keep billing after a readiness timeout).
    while len(boxes) < args.count and offers_left:
        need = args.count - len(boxes)
        batch: list[dict] = []
        while offers_left and len(batch) < need:
            ranked = offers_left.pop(0)
            if str(ranked.machine_id) in failed_machines:
                continue
            public_ip = offer_public_ip(ranked)
            if public_ip and public_ip in failed_public_ips:
                continue
            entry = create_one(ranked)
            if entry is not None:
                batch.append(entry)
        if not batch:
            break

        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(connect_box, entry): entry
                for entry in batch
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    entry, ready = future.result()
                except Exception as error:  # noqa: BLE001 — isolate one host failure
                    ready = False
                    detail = redact_sensitive(error, secrets=(api_key,))
                    log(f"  readiness failed for instance {entry['id']}: {detail}")
                if ready:
                    alias = f"vast-{len(boxes) + 1}"
                    entry["alias"] = alias
                    boxes.append(BoxConn(
                        alias=alias,
                        host=entry["host"],
                        port=int(entry["port"]),
                        instance_id=entry["id"],
                    ))
                    _record(state, entry)
                    save_state(cfg, state)
                else:
                    destroy_unready(entry, "readiness timeout or bootstrap failure")

    if not boxes:
        log("No ready instances (candidate offers unavailable or failed readiness).")
        return 1
    if len(boxes) < args.count:
        log(f"note: only {len(boxes)}/{args.count} boxes became ready; "
            f"remaining offers were unavailable or failed readiness.")

    boxes.sort(key=lambda box: box.alias)
    write_ssh_config(boxes, cfg, log)
    if not args.no_open:
        open_terminals(boxes, log)

    log("")
    log("=" * 70)
    log("Boxes are LIVE and BILLING. Tear them down when done:")
    log("  uv run --group devops python -m devops.vast.provision destroy --all")
    if offer_type == "interruptible":
        log("Interruptible: if outbid, boxes go to 'stopped' (disk still billed); "
            "destroy cleans them up.")
    for b in boxes:
        log(f"  {b.alias}: ssh root@{b.host} -p {b.port}   (or: ssh {b.alias})")
    log("=" * 70)
    return 0 if len(boxes) == args.count else 1


def cmd_destroy(args, cfg: VastConfig) -> int:
    from .vast_client import VastClient

    log = print
    state = load_state(cfg)
    tracked = state.get("instances", [])
    if args.id:
        targets = [i for i in tracked if str(i["id"]) in {str(x) for x in args.id}]
        # allow destroying an untracked id passed explicitly
        known = {str(i["id"]) for i in tracked}
        for x in args.id:
            if str(x) not in known:
                targets.append({"id": int(x), "label": "(untracked)"})
    elif args.all:
        targets = list(tracked)
    else:
        log("Specify --all or --id <id> [<id> ...]")
        return 1

    if not targets:
        log("Nothing to destroy.")
        return 0

    log("Will destroy:")
    for t in targets:
        log(f"  instance {t['id']} ({t.get('label', '?')})")
    if not args.yes:
        resp = input(f"Destroy {len(targets)} instance(s)? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            log("aborted.")
            return 1

    client = VastClient(cfg)
    remaining = list(tracked)
    for t in targets:
        try:
            client.destroy_instance(t["id"])
            log(f"destroyed {t['id']}")
            remaining = [i for i in remaining if i["id"] != t["id"]]
        except Exception as e:  # noqa: BLE001
            log(f"failed to destroy {t['id']}: {e}")
    state["instances"] = remaining
    save_state(cfg, state)
    return 0


def cmd_reap(args, cfg: VastConfig) -> int:
    """Local backstop for the on-box watchdog: destroy tracked boxes older than
    the max-age cap. Cron/loop this so a box whose on-box timer never fired (e.g.
    a stopped interruptible box, or a crashed bootstrap) still gets freed.
    """
    from .vast_client import VastClient

    log = print
    default_hours = cfg.MAX_AGE_HOURS if args.max_age is None else args.max_age
    state = load_state(cfg)
    tracked = state.get("instances", [])
    now = time.time()

    stale: list[dict] = []
    for t in tracked:
        created = t.get("created_at")
        if not created:
            continue  # can't age a box we didn't timestamp; leave it be
        # Prefer the per-box cap recorded at creation; fall back to the CLI/config.
        cap_s = t.get("max_age_s") or (float(default_hours) * 3600.0 if default_hours else 0.0)
        if cap_s and cap_s > 0 and (now - created) > cap_s:
            t["_age_h"] = (now - created) / 3600.0
            stale.append(t)

    if not stale:
        log(f"No tracked boxes exceed the max-age cap ({default_hours:g}h).")
        return 0

    log("Will reap (older than cap):")
    for t in stale:
        log(f"  instance {t['id']} ({t.get('label', '?')}) — age {t['_age_h']:.1f}h")
    if not args.yes:
        resp = input(f"Destroy {len(stale)} stale box(es)? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            log("aborted.")
            return 1

    client = VastClient(cfg)
    remaining = list(tracked)
    for t in stale:
        try:
            client.destroy_instance(t["id"])
            log(f"reaped {t['id']} (age {t['_age_h']:.1f}h)")
            remaining = [i for i in remaining if i["id"] != t["id"]]
        except Exception as e:  # noqa: BLE001
            log(f"failed to reap {t['id']}: {e}")
    state["instances"] = remaining
    save_state(cfg, state)
    return 0


def cmd_status(args, cfg: VastConfig) -> int:
    from .vast_client import VastClient

    log = print
    state = load_state(cfg)
    tracked = state.get("instances", [])
    if not tracked:
        log("No tracked instances (state.json is empty).")
        return 0
    client = VastClient(cfg)
    log(f"  {'id':<10}{'label':<32}{'status':<12}{'$/hr':<8}{'ssh'}")
    log("  " + "-" * 78)
    for t in tracked:
        inst = None
        try:
            inst = client.show_instance(t["id"])
        except Exception:  # noqa: BLE001
            pass
        status = (inst or {}).get("actual_status", "gone")
        host, port = VastClient.connection_info(inst) if inst else (t.get("host"), t.get("port"))
        ssh = f"ssh root@{host} -p {port}" if host else "-"
        log(f"  {str(t['id']):<10}{t.get('label', '?')[:31]:<32}{str(status):<12}"
            f"{float(t.get('price') or 0):<8.3f}{ssh}")
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m devops.vast.provision",
        description="Find, rank, rent, bootstrap, and connect to vast.ai RTX 4090 boxes.",
    )
    sub = p.add_subparsers(dest="command")

    up = sub.add_parser("up", help="search, rank, and rent boxes (default command)")
    up.add_argument("-n", "--count", type=int, default=1, help="number of boxes")
    up.add_argument("--mode", choices=["ondemand", "interruptible"], default="ondemand")
    up.add_argument("--bid", type=float, default=None,
                    help="interruptible bid $/hr (default: auto = min_bid * margin)")
    up.add_argument("--disk", type=float, default=None, help="disk GB (default: config)")
    up.add_argument("--image", default=None, help="docker image (default: config)")
    up.add_argument("--branch", default=None, help="git branch to clone on the box")
    up.add_argument("--commit", default=None, help="git commit sha to clone on the box")
    up.add_argument("--run", default=None, metavar="CMD",
                    help="command to run in tmux inside the activated, pre-synced environment")
    up.add_argument("--max-price", type=float, default=None, help="hard cap on $/hr")
    up.add_argument("--regions", default=None,
                    help="comma-separated country codes to require, e.g. US,CA "
                         "(hard filter when set; default HOME_REGIONS is tiebreak-only)")
    up.add_argument("--offer-id", type=int, default=None,
                    help="rent one exact offer ID from the search results")
    up.add_argument("--machine-id", type=int, default=None,
                    help="rent one exact provider machine ID from the search results")
    up.add_argument("--exclude-machine", type=int, nargs="+", action="extend", default=[],
                    metavar="ID", help="exclude one or more known-bad machine IDs")
    up.add_argument("--dry-run", action="store_true", help="print ranked candidates, rent nothing")
    up.add_argument("--yes", action="store_true", help="skip the rent confirmation prompt")
    up.add_argument("--no-open", action="store_true", help="do not auto-open terminal tabs")
    # self-destruct
    up.add_argument("--self-destruct", action="store_true",
                    help="inject teardown env + enable the training push+destroy hook")
    up.add_argument("--run-name", default=None, help="per-shot results subdir + commit label")
    up.add_argument("--results-branch", default=None,
                    help="branch the box pushes results to (default: 'results')")
    up.add_argument("--github-token", default=None, help="write token (else GITHUB_TOKEN / gh auth token)")
    up.add_argument("--teardown-on-error", action="store_true",
                    help="also push+destroy if the run raises (off by default)")
    up.add_argument("--max-age", type=float, default=None, metavar="HOURS",
                    help="wall-clock lifetime cap; box self-destructs after this "
                         "many hours (default: config MAX_AGE_HOURS; 0 disables)")

    d = sub.add_parser("destroy", help="destroy tracked (or specified) instances")
    d.add_argument("--all", action="store_true", help="destroy all tracked instances")
    d.add_argument("--id", nargs="+", type=int, help="specific instance id(s)")
    d.add_argument("--yes", action="store_true", help="skip confirmation")

    r = sub.add_parser("reap", help="destroy tracked boxes older than the max-age cap")
    r.add_argument("--max-age", type=float, default=None, metavar="HOURS",
                   help="override the age cap in hours (default: config MAX_AGE_HOURS)")
    r.add_argument("--yes", action="store_true", help="skip confirmation")

    sub.add_parser("status", help="show status of tracked instances")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    # Line-buffer so progress is visible in real time even when piped/redirected
    # (provisioning has long waits — a silent block-buffered pipe looks hung).
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    # `up` is the default command
    if not argv or argv[0] not in {"up", "destroy", "reap", "status", "-h", "--help"}:
        argv = ["up"] + argv
    args = build_parser().parse_args(argv)
    cfg = CONFIG
    if args.command == "destroy":
        return cmd_destroy(args, cfg)
    if args.command == "reap":
        return cmd_reap(args, cfg)
    if args.command == "status":
        return cmd_status(args, cfg)
    return cmd_up(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
