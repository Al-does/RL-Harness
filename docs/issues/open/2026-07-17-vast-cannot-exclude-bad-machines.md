---
status: open
severity: high
area: "devops/vast/{provision.py,scoring.py,vast_client.py,config.py}"
discovered: 2026-07-17
reproduction: partial
---

# Vast retries cannot exclude known-bad machines

## Context

- **Git revision / worktree:** `d91bdffc7aa6839a8bd45161e2b03920c270a5f5`
  on `MESS3-supervised`; dirty only in unrelated pre-existing paths
  (`AGENTS.md`, `.cursor/Dockerfile`, `.cursor/environment.json`)
- **Command:** `uv run --group devops python -m devops.vast.provision up -n 1 --regions US,CA --dry-run`,
  followed by the same command with `--yes --no-open --branch MESS3-supervised`
  instead of `--dry-run`
- **Environment:** local macOS client provisioning Vast RTX 4090 Ubuntu/CUDA
  hosts; provider and Python/Ray versions were not relevant because training
  never started
- **Training context:** experiment `experiments.MESS3_supervised.experiment`;
  full run; seed not reached; one RTX 4090 per candidate
- **Related records:** `docs/issues/open/2026-07-17-vast-http-errors-leak-api-key.md`;
  no durable run manifest (all failures occurred during provisioning)

## Expected behavior

After a host is observed to be unusable, a retry should be able to exclude its
machine ID or select a specific offer from the displayed ranking. Provisioning
multiple candidates should also observe their readiness concurrently, so one
stalled candidate does not serialize progress on the others.

## Observed behavior

The 2026-07-17 supervised MESS3 provisioning attempts repeatedly selected or
re-ranked hosts that had already failed:

- Machine `140297`, offer `40912244`, instance `45181205` (US, `$0.259/hr`,
  reliability `0.995`, advertised `inet_down=438.7 Mbps`) reached running and
  SSH. Clone took about 6m45s (`16:34:11`-`16:40:56`), then dependency setup
  exhausted the hard 1200-second limit:

  ```text
  uv sync timed out after 1200s (host network too slow)
  ```

- Machine `9020`, offer `31475021`, instance `45182832` (CA, `$0.280/hr`,
  reliability `0.983`, advertised `inet_down=856.2 Mbps`) kept
  `actual_status=None` and did not reach running within
  `RUNNING_TIMEOUT_S=900`.
- Machine `140968`, offer `41890592`, instance `45183642` (US, `$0.259/hr`,
  reliability `0.988`, advertised `inet_down=876.3 Mbps`) reached running and
  SSH, but `git clone`/`git index-pack` remained active for more than 15
  minutes. It was manually destroyed before the readiness timeout.
- Machines `140607`, `9020`, and `140968` appeared again in later rankings
  after destruction.

Every failed instance was destroyed, and `status` was confirmed empty before
the next retry. The host failures are provider-dependent and are not claimed
to occur deterministically on every rental. The repository defect is that
`up` exposes neither `--offer-id` nor `--exclude-machine` (nor a minimum-price
filter), so the operator cannot act on known-bad host evidence without changing
code. `--regions` only changes a near-price tie-break. For `-n N`, instances
are created first but running/readiness waits are then processed serially.

## Minimal reproduction

1. Display the current ranking:

   ```bash
   uv run --group devops python -m devops.vast.provision up -n 1 --regions US,CA --dry-run
   ```

2. Observe that the CLI has no supported way to select a displayed offer ID or
   exclude a displayed machine ID:

   ```bash
   uv run --group devops python -m devops.vast.provision up --help
   ```

3. Rent normally:

   ```bash
   uv run --group devops python -m devops.vast.provision up -n 1 --regions US,CA --yes --no-open --branch MESS3-supervised
   ```

4. If the selected provider host stalls, destroy it and repeat. The particular
   host failure is nondeterministic, but the inability to exclude that machine
   or select another ranked offer is deterministic.

## Suspected cause and scope

`build_parser()` and `cmd_up()` support only automatic ranking plus a
maximum-price cap. `rank_offers()` deduplicates hosts within one search result
but accepts no caller-supplied machine exclusion set, and there is no persistent
failure quarantine. Region preference is deliberately only a tie-break in the
sort key. Finally, `cmd_up()` loops over created entries and calls
`wait_until_running()` and `wait_for_ready_ssh()` synchronously for each.

This affects any experiment provisioned through `devops.vast`, not MESS3
specifically. The provider should remain responsible for bad hosts, but the
toolkit should support exact offer selection and repeatable machine exclusions
(and ideally concurrent candidate readiness) so retries do not rent known-bad
machines or wait serially.

## Resolution history

- 2026-07-17 — Recorded from repeated supervised MESS3 provisioning attempts;
  all failed instances were destroyed before retrying.
