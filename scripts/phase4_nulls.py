"""Phase-4 null suite: N-init re-analysis + the null-bracket table.

    uv run python scripts/phase4_nulls.py --probe   # fill missing probe curves
    uv run python scripts/phase4_nulls.py           # assemble tables + figures

- N-init: every arm's probe metrics at EVERY saved checkpoint including step 0
  (probe_arm.py already handles one run; --probe drives it across all runs).
- Null bracket: [initialization floor, trained-on-noise floor] per metric per
  architecture; initialization floor = step-0 checkpoints across arms of that
  architecture; trained-on-noise floor = n_scramble final checkpoints.
- Deliverable: results/phase4/null_bracket.json + null_table.md restating
  every headline R^2 against the bracket, and per-arm probe-curve figures
  (written by probe_arm.py into each run directory).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]


def all_runs(finished_only: bool = False):
    """finished_only: require module_state_final.pt (skips in-progress runs;
    used by --probe). Assembly instead takes any run with a probe curve, since
    some early runs were STOP-ed before writing a final checkpoint."""
    for phase in ["phase2", "phase3", "phase4"]:
        for run in sorted((REPO / "results" / phase).glob("*/seed*")):
            if not (run / "blueprint.json").exists():
                continue
            if finished_only:
                if (run / "module_state_final.pt").exists():
                    yield run
            elif (run / "probe_curve.json").exists():
                yield run


def ensure_probed(run: Path, min_points: int) -> bool:
    pc = run / "probe_curve.json"
    if pc.exists() and len(json.loads(pc.read_text())) >= min_points:
        return True
    n_ckpts = len(list(run.glob("module_state_*.pt")))
    print(f"probing {run} ({n_ckpts} ckpts)...", flush=True)
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "probe_arm.py"), "--run", str(run)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr[-500:]}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    outdir = REPO / "results" / "phase4"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.probe:
        for run in all_runs(finished_only=True):
            n_ckpts = len(list(run.glob("module_state_*.pt")))
            ensure_probed(run, min_points=max(2, n_ckpts - 2))
        return

    # ---- assemble ---------------------------------------------------------
    per_arm: dict[str, list] = {}
    for run in all_runs():
        pc = run / "probe_curve.json"
        if not pc.exists():
            continue
        bp = json.loads((run / "blueprint.json").read_text())
        curve = json.loads(pc.read_text())
        architecture = bp["model"]["class"].rsplit(":", maxsplit=1)[-1]
        per_arm.setdefault(bp["name"], []).append(
            {"seed": run.name, "arch": architecture, "curve": curve}
        )

    # Initialization floors per architecture (step-0 entries).
    floors: dict[str, dict[str, list]] = {}
    for arm, entries in per_arm.items():
        for e in entries:
            first = e["curve"][0]
            if first["env_steps"] == 0:
                f = floors.setdefault(e["arch"], {"r2_global": [], "r2_fine": []})
                f["r2_global"].append(first["r2_global"])
                f["r2_fine"].append(first["r2_fine"])
    init_floor = {
        arch: {k: [float(np.mean(v)), float(np.min(v)), float(np.max(v))]
               for k, v in d.items()}
        for arch, d in floors.items()
    }

    # Trained-on-noise floor (n_scramble finals).
    noise_floor = None
    if "n_scramble" in per_arm:
        finals = [e["curve"][-1] for e in per_arm["n_scramble"]]
        noise_floor = {
            "r2_global": float(np.mean([f["r2_global"] for f in finals])),
            "r2_fine": float(np.mean([f["r2_fine"] for f in finals])),
            "reward_mean": float(np.mean([f["reward_mean"] for f in finals])),
        }

    # Headline table.
    lines = [
        "# Null-bracket table (Phase 4)",
        "",
        "Fine R^2 = branch depth 2 (last two visible tokens) unless noted.",
        "",
        "| arm | seeds | global R^2 (final) | fine R^2 (final) | fine R^2 (init) |",
        "|-----|------:|-------------------:|-----------------:|----------------:|",
    ]
    table = {}
    for arm in sorted(per_arm):
        entries = per_arm[arm]
        fin_g = [e["curve"][-1]["r2_global"] for e in entries]
        fin_f = [e["curve"][-1]["r2_fine"] for e in entries]
        ini_f = [e["curve"][0]["r2_fine"] for e in entries if e["curve"][0]["env_steps"] == 0]
        table[arm] = {
            "n_seeds": len(entries),
            "r2_global_final": [float(np.mean(fin_g)), float(np.std(fin_g))],
            "r2_fine_final": [float(np.mean(fin_f)), float(np.std(fin_f))],
            "r2_fine_init": [float(np.mean(ini_f)), float(np.std(ini_f))] if ini_f else None,
        }
        lines.append(
            f"| {arm} | {len(entries)} | {np.mean(fin_g):.3f} ± {np.std(fin_g):.3f} "
            f"| {np.mean(fin_f):.3f} ± {np.std(fin_f):.3f} "
            f"| {(f'{np.mean(ini_f):.3f}' if ini_f else '—')} |"
        )
    lines += ["", "## Null bracket", ""]
    for arch, d in init_floor.items():
        lines.append(f"- initialization floor ({arch}): global "
                     f"{d['r2_global'][0]:.3f}, fine {d['r2_fine'][0]:.3f} "
                     f"(range {d['r2_fine'][1]:.3f}..{d['r2_fine'][2]:.3f})")
    if noise_floor:
        lines.append(f"- trained-on-noise floor (n_scramble): global "
                     f"{noise_floor['r2_global']:.3f}, fine {noise_floor['r2_fine']:.3f}, "
                     f"reward {noise_floor['reward_mean']:.4f} "
                     "(no-info optima: constant 0.1415, best periodic open-loop "
                     "0.1966 at period 4 -- scramble agents retain a clock via "
                     "action history, so the open-loop value is the right ceiling)")

    payload = {"per_arm": table, "init_floor": init_floor, "noise_floor": noise_floor}
    (outdir / "null_bracket.json").write_text(json.dumps(payload, indent=2))
    (outdir / "null_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {outdir}/null_bracket.json, null_table.md")


if __name__ == "__main__":
    main()
