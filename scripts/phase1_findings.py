"""Phase-1 gate verification: check the sweep artifacts against the
known-good values from prior rounds and, if everything reproduces, write
results/phase1/GATE_PASSED (which scripts/train.py requires before any
phase >= 2 launch; the separate REVIEW_APPROVED marker is written by the PI
after the review stop).

    uv run python scripts/phase1_findings.py [--dir results/phase1]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

KNOWN_GOOD = {  # at (beta=4, w_max2=5, delay=1)
    "reactive": (0.192, 2e-3),
    "belief_ceiling": (0.381, 1.5e-3),
    "oracle": (0.4625, 5e-4),
}
STACK2_PRIOR_FLOOR = 0.295  # this round's optimizer found 0.3055 (better table)


def fail(msg):
    print(f"GATE FAIL: {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phase1")
    args = ap.parse_args()
    d = Path(args.dir)

    rows = json.loads((d / "gate_sweep.json").read_text())
    by_key = {(r["beta"], r["w_max2"], r["delay"]): r for r in rows}
    op = by_key.get((4.0, 5.0, 1)) or fail("operating point (4, 5, 1) missing from sweep")

    checks = []

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")

    print("known-good regression at (beta=4, w_max2=5, delay=1):")
    for key, (target, tol) in KNOWN_GOOD.items():
        v = op[key]
        check(key, abs(v - target) <= tol, f"{v:.4f} vs prior {target} (tol {tol})")
    check("stack2 >= prior", op["stack2"] >= STACK2_PRIOR_FLOOR - 1e-3,
          f"{op['stack2']:.4f} vs prior 0.295 (better table found; see FINDINGS)")

    print("gate thresholds at the operating point:")
    check("premium_reactive >= 0.15", op["premium_reactive"] >= 0.15,
          f"{op['premium_reactive']:.3f}")
    check("premium_stack2 >= 0.08", op["premium_stack2"] >= 0.08,
          f"{op['premium_stack2']:.3f}")

    print("known sweep behavior:")
    d1 = {(r["beta"], r["w_max2"]): r for r in rows if r["delay"] == 1}
    d0 = [r for r in rows if r["delay"] == 0]
    check("all delay=0 fail the gate", not any(r["gate_pass"] for r in d0),
          f"{sum(r['gate_pass'] for r in d0)}/{len(d0)} pass")
    beta2 = [r for (b, w), r in d1.items() if b == 2.0]
    check("beta=2 collapses (no premium)", not any(r["gate_pass"] for r in beta2),
          "premiums " + ", ".join(f"{r['premium_reactive']:.3f}" for r in beta2))
    beta48 = [r for (b, w), r in d1.items() if b in (4.0, 8.0)]
    check("beta in {4,8} at delay=1 pass", all(r["gate_pass"] for r in beta48),
          f"{sum(r['gate_pass'] for r in beta48)}/{len(beta48)} pass")
    check("~96% boundary saturation at w_max2=5 (prior obs)",
          abs((1 - d1[(4.0, 5.0)]["interior_share_w0"]) - 0.96) < 0.02,
          f"boundary share {1 - d1[(4.0, 5.0)]['interior_share_w0']:.3f}")
    check("smaller box raises interior share (hypothesis)",
          d1[(4.0, 3.0)]["interior_share_any"] > d1[(4.0, 5.0)]["interior_share_any"],
          f"w_max2=3: {d1[(4.0, 3.0)]['interior_share_any']:.3f} vs "
          f"w_max2=5: {d1[(4.0, 5.0)]['interior_share_any']:.3f}")

    sg = (d / "stateguess_table.csv").read_text().strip().splitlines()[1:]
    mem0 = float(sg[0].split(",")[2])
    check("StateGuess memoryless(delay=0) == 0.85 exactly", abs(mem0 - 0.85) < 1e-9,
          f"{mem0}")

    if all(ok for _, ok, _ in checks):
        stamp = datetime.now(timezone.utc).isoformat()
        (d / "GATE_PASSED").write_text(
            f"phase 1 gate passed at {stamp}\n"
            + "".join(f"{n}: {det}\n" for n, _, det in checks)
        )
        print(f"\nALL CHECKS PASSED -> wrote {d / 'GATE_PASSED'}")
        print("Phase 1 now STOPS for review; create REVIEW_APPROVED to unlock training.")
    else:
        fail("one or more checks failed; GATE_PASSED not written")


if __name__ == "__main__":
    main()
