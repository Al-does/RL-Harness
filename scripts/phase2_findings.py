"""Assemble the Phase-2 report ladder from completed runs.

    uv run python scripts/phase2_findings.py

Reads results/phase2/<arm>/seed*/probe_curve.json + progress.jsonl and the
Phase-1 analytic table, prints the ladder (random 1/3 -> memoryless -> B-M1
-> B-R1 -> B-SL -> filter ceiling), the gap decomposition (architecture tax,
RL tax), and probe metrics per arm.  Writes results/phase2/ladder.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
P2 = REPO / "results" / "phase2"


def load_arm(arm: str) -> list[dict]:
    out = []
    for seed_dir in sorted(P2.glob(f"{arm}/seed*")):
        pc = seed_dir / "probe_curve.json"
        if pc.exists():
            curve = json.loads(pc.read_text())
            out.append({"seed": seed_dir.name, "final": curve[-1], "curve": curve})
    return out


def agg(runs: list[dict], key: str):
    vals = [r["final"].get(key) for r in runs if r["final"].get(key) is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)


def fmt(m, s):
    if m is None:
        return "—"
    return f"{m:.4f}" + (f" ± {s:.4f}" if s else "")


def main():
    # Phase-1 analytic anchors (Environment B).
    table = {}
    with open(REPO / "results" / "phase1" / "stateguess_table.csv") as f:
        for row in csv.DictReader(f):
            table[int(row["delay"])] = {
                "memoryless": float(row["memoryless"]),
                "filter": float(row["filter_ceiling"]),
            }

    ladder = {"random": 1 / 3, "memoryless_d1": table[1]["memoryless"],
              "filter_d1": table[1]["filter"],
              "memoryless_d0": table[0]["memoryless"],
              "filter_d0": table[0]["filter"]}

    print("=== Environment B ladder (greedy accuracy; mean ± sd over seeds) ===")
    rows = {}
    for arm in ["b_m1", "b_r1", "b_r1_g0", "b_r0", "b_sl"]:
        runs = load_arm(arm)
        if not runs:
            print(f"{arm:10s}  (no completed runs yet)")
            continue
        # Ladder number: greedy reward for RL arms; aux-head state accuracy
        # for the supervised twin.
        key = "aux_acc_state" if arm == "b_sl" else "reward_greedy"
        m, s = agg(runs, key)
        g, gs = agg(runs, "r2_global")
        fine, fs = agg(runs, "r2_fine")
        km, _ = agg(runs, "decoded_kmeans3_explained")
        rows[arm] = {"n_seeds": len(runs), "ladder_value": m, "ladder_sd": s,
                     "r2_global": g, "r2_fine": fine, "decoded_kmeans3": km}
        print(f"{arm:10s}  n={len(runs)}  value={fmt(m, s)}  "
              f"globalR2={fmt(g, gs)}  fineR2={fmt(fine, fs)}  kmeans3={km}")

    print("\n=== analytic anchors ===")
    for k, v in ladder.items():
        print(f"{k:15s} {v:.4f}")

    out = {"anchors": ladder, "arms": rows}
    if "b_sl" in rows and "b_r1_g0" in rows:
        sl = rows["b_sl"]["ladder_value"]
        r1 = rows["b_r1_g0"]["ladder_value"]
        out["architecture_tax"] = ladder["filter_d1"] - sl
        out["rl_tax"] = sl - r1
        print(f"\narchitecture tax (filter - SL): {out['architecture_tax']:+.4f}")
        print(f"RL tax (SL - R1@gamma0):        {out['rl_tax']:+.4f}")

    with open(P2 / "ladder.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {P2}/ladder.json")


if __name__ == "__main__":
    main()
