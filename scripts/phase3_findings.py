"""Assemble the Phase-3 report from completed Environment-A runs.

    uv run python scripts/phase3_findings.py

Reads results/phase3/<arm>/seed*/probe_curve.json plus the Phase-1 analytic
anchors and writes results/phase3/arms_table.json + FINDINGS_phase3.md:
reward ladder against analytic anchors, probe geometry per arm (global /
fine R^2, within-branch action variance), and the aux-lambda dose response.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
P3 = REPO / "results" / "phase3"

ARMS = ["a_main", "a_nodelay", "a_aux_0p1", "a_aux_0p5", "a_pred",
        "a_oracle", "a_beliefobs", "a_stack2", "a_stack4", "a_stack8",
        "a_stack16", "a_lstm"]

METRICS = ["reward_mean", "reward_greedy", "r2_global", "r2_fine",
           "r2_fine_depth1", "within_branch_action_var_frac", "aux_acc_state"]


def anchors() -> dict:
    for r in json.loads((REPO / "results/phase1/gate_sweep.json").read_text()):
        if r["beta"] == 4.0 and r["w_max2"] == 5.0 and r["delay"] == 1:
            a = {k: r[k] for k in ["constant", "reactive", "stack2",
                                   "belief_ceiling", "oracle"]}
    for r in json.loads((REPO / "results/phase1/gate_sweep.json").read_text()):
        if r["beta"] == 4.0 and r["w_max2"] == 5.0 and r["delay"] == 0:
            a["belief_ceiling_delay0"] = r["belief_ceiling"]
    # Best periodic open-loop value (clock but no observations); computed in
    # Phase 4 -- exceeds the constant no-info optimum because scrambled agents
    # retain a clock through their own action history.
    a["openloop_period4"] = 0.1966
    return a


def load_arm(arm: str) -> list[dict]:
    out = []
    for seed_dir in sorted(P3.glob(f"{arm}/seed*")):
        pc = seed_dir / "probe_curve.json"
        if pc.exists():
            curve = json.loads(pc.read_text())
            out.append({"seed": seed_dir.name, "final": curve[-1]})
    return out


def agg(runs, key):
    vals = [r["final"].get(key) for r in runs if r["final"].get(key) is not None]
    if not vals:
        return None
    return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)


def fmt(ms, digits=3):
    if ms is None:
        return "—"
    m, s = ms
    return f"{m:.{digits}f}" + (f" ± {s:.{digits}f}" if s else "")


def main():
    a = anchors()
    table = {}
    lines = [
        "# Phase 3 findings: Environment A main arms",
        "",
        "Operating point beta=4, w_max2=5, delay=1 (alpha=0.85, episodes of 1024).",
        "",
        "## Analytic anchors (Phase 1)",
        "",
    ] + [f"- {k}: {v:.4f}" for k, v in a.items()] + [
        "",
        "## Arm table (mean ± sd over seeds; rewards are greedy rollouts)",
        "",
        "| arm | seeds | reward (greedy) | global R^2 | fine R^2 | within-branch act var |",
        "|-----|------:|----------------:|-----------:|---------:|----------------------:|",
    ]
    for arm in ARMS:
        runs = load_arm(arm)
        if not runs:
            lines.append(f"| {arm} | 0 | — | — | — | — |")
            continue
        row = {m: agg(runs, m) for m in METRICS}
        row["n_seeds"] = len(runs)
        table[arm] = row
        lines.append(
            f"| {arm} | {len(runs)} | {fmt(row['reward_greedy'])} "
            f"| {fmt(row['r2_global'])} | {fmt(row['r2_fine'])} "
            f"| {fmt(row['within_branch_action_var_frac'])} |"
        )

    lines += ["", "## Readings", ""]

    def m(arm, key):
        return table[arm][key][0] if arm in table and table[arm][key] else None

    if "a_main" in table:
        r = m("a_main", "reward_greedy")
        lines.append(
            f"- **Headline (a_main)**: greedy reward {r:.3f} = "
            f"{100*(r - a['reactive'])/(a['belief_ceiling'] - a['reactive']):.0f}% of the "
            f"reactive-to-belief-ceiling gap; fine R^2 {m('a_main','r2_fine'):.3f} with "
            f"{100*m('a_main','within_branch_action_var_frac'):.0f}% within-branch action "
            "variance (policy hedges on decision-relevant belief coordinates)."
        )
    if "a_nodelay" in table:
        lines.append(
            f"- **Premium removal (a_nodelay)**: greedy reward "
            f"{m('a_nodelay','reward_greedy'):.3f} vs its own ceiling "
            f"{a['belief_ceiling_delay0']:.3f}, but fine R^2 collapses to "
            f"{m('a_nodelay','r2_fine'):.3f} — reward alone does not buy fine belief "
            "geometry once the memory premium is gone."
        )
    doses = [(0.0, "a_main"), (0.1, "a_aux_0p1"), (0.5, "a_aux_0p5"), (1.0, "a_pred")]
    have = [(lam, arm) for lam, arm in doses if arm in table]
    if len(have) >= 3:
        lines.append(
            "- **Aux dose response (lambda -> fine R^2)**: "
            + ", ".join(f"{lam:g} -> {m(arm, 'r2_fine'):.3f}" for lam, arm in have)
            + " (a_pred is prediction-only, random actions)."
        )
    stacks = [(k, f"a_stack{k}") for k in (2, 4, 8, 16) if f"a_stack{k}" in table]
    if stacks:
        lines.append(
            "- **Stack ladder (k -> greedy reward | fine R^2)**: "
            + ", ".join(f"{k} -> {m(arm, 'reward_greedy'):.3f} | "
                        f"{m(arm, 'r2_fine'):.3f}" for k, arm in stacks)
        )
    if "a_oracle" in table:
        lines.append(
            f"- **Oracle sanity (a_oracle)**: {m('a_oracle','reward_greedy'):.3f} vs "
            f"analytic {a['oracle']:.4f}; belief-obs MLP "
            f"{fmt(table.get('a_beliefobs', {}).get('reward_greedy')) if 'a_beliefobs' in table else '—'} vs "
            f"ceiling {a['belief_ceiling']:.4f}."
        )

    payload = {"anchors": a, "arms": {k: {mk: v for mk, v in row.items()}
                                      for k, row in table.items()}}
    (P3 / "arms_table.json").write_text(json.dumps(payload, indent=2))
    (P3 / "FINDINGS_phase3.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {P3}/arms_table.json, FINDINGS_phase3.md")


if __name__ == "__main__":
    main()
