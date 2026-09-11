"""RunPod Pod bootstrap: clone exact refs, then run their durable lifecycle.

This baked file intentionally depends only on Python's standard library. The
actual batch lifecycle is imported from the checked-out harness commit so
runner fixes no longer require rebuilding the environment image.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

WORK_DIR = Path("/workspace")
LIBRARY_DIR = WORK_DIR / "rl-harness"
VENV_DIR = Path(os.environ.get("RUNPOD_VENV_DIR", str(WORK_DIR / ".venv")))
MLFLOW_DIR = WORK_DIR / "mlruns"


def log(message: str) -> None:
    print(
        f"[runpod {time.strftime('%H:%M:%S', time.gmtime())}] {message}",
        flush=True,
    )


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def terminate_self(reason: str) -> bool:
    """Terminate this Pod with its RunPod-provided Pod-scoped API key."""
    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not pod_id or not api_key:
        log(
            "ERROR: cannot self-terminate because RUNPOD_POD_ID or the "
            "Pod-scoped RUNPOD_API_KEY is missing"
        )
        return False
    query = (
        "mutation { podTerminate(input: { podId: "
        f"{json.dumps(pod_id)}"
        " }) }"
    )
    request = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": query}).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "rl-harness-runpod/1.0 (Pod self-terminator)",
        },
    )
    log(f"terminating Pod {pod_id} ({reason})")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
        if payload.get("errors"):
            log("ERROR: self-termination GraphQL mutation returned errors")
            return False
        return True
    except urllib.error.HTTPError as error:
        log(f"ERROR: self-termination returned HTTP {error.code}")
    except Exception as error:  # noqa: BLE001
        # Do not print exception URLs: they can include authentication details.
        log(f"ERROR: self-termination failed ({type(error).__name__})")
    return False


def start_watchdog(max_age_s: int) -> threading.Event:
    stop = threading.Event()

    def watch() -> None:
        if not stop.wait(max_age_s):
            terminate_self(f"hard max-age {max_age_s}s reached")

    thread = threading.Thread(
        target=watch,
        name="runpod-max-age-watchdog",
        daemon=True,
    )
    thread.start()
    return stop


def start_ssh_server() -> None:
    """Start key-only SSH for an explicitly interactive Pod."""
    public_key = os.environ.get("SSH_PUBLIC_KEY", "").strip()
    if not public_key or "\n" in public_key:
        raise RuntimeError("interactive Pod requires one SSH_PUBLIC_KEY")
    if public_key.partition(" ")[0] not in {
        "ssh-ed25519",
        "ssh-rsa",
        "ecdsa-sha2-nistp256",
        "ecdsa-sha2-nistp384",
        "ecdsa-sha2-nistp521",
    }:
        raise RuntimeError("interactive Pod received unsupported SSH key type")
    ssh_dir = Path("/root/.ssh")
    ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    authorized_keys = ssh_dir / "authorized_keys"
    authorized_keys.write_text(public_key + "\n")
    authorized_keys.chmod(0o600)
    Path("/run/sshd").mkdir(parents=True, exist_ok=True)
    Path("/etc/ssh/sshd_config.d/99-runpod-interactive.conf").write_text(
        "PermitRootLogin prohibit-password\n"
        "PasswordAuthentication no\n"
        "KbdInteractiveAuthentication no\n"
        "PubkeyAuthentication yes\n"
    )
    run(["ssh-keygen", "-A"])
    run(["/usr/sbin/sshd", "-t"])
    run(["/usr/sbin/sshd"])
    log("interactive SSH server started with key-only authentication")


def git_auth_env() -> dict[str, str]:
    """Pass GitHub auth in process env without writing it into .git/config."""
    token = required("GH_TOKEN")
    encoded = base64.b64encode(
        f"x-access-token:{token}".encode("utf-8")
    ).decode("ascii")
    env = dict(os.environ)
    env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {encoded}",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return env


def checkout(
    *,
    url: str,
    ref: str,
    target: Path,
    sparse_experiment: bool = False,
) -> str:
    env = git_auth_env()
    if target.exists():
        raise RuntimeError(f"checkout target already exists: {target}")
    clone = ["git", "clone", "--depth", "1"]
    if sparse_experiment:
        clone.extend(["--filter=blob:none", "--sparse", "--no-checkout"])
    clone.extend([url, str(target)])
    run(clone, env=env)
    if sparse_experiment:
        run(
            [
                "git",
                "sparse-checkout",
                "set",
                "--cone",
                "experiments",
                "scripts",
                "tests",
                "pyproject.toml",
                "README.md",
                ".gitignore",
                "uv.lock",
            ],
            cwd=target,
            env=env,
        )
    run(["git", "fetch", "--depth", "1", "origin", ref], cwd=target, env=env)
    run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=target)
    completed = run(
        ["git", "rev-parse", "HEAD"],
        cwd=target,
        capture_output=True,
    )
    sha = completed.stdout.strip()
    log(f"checked out {url.rsplit('/', 1)[-1]} at {sha}")
    return sha


def ensure_system_tools() -> None:
    missing = [
        command
        for command in ("git", "curl")
        if subprocess.run(
            ["bash", "-lc", f"command -v {shlex.quote(command)}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        != 0
    ]
    if not missing:
        return
    log(f"installing missing system tools: {', '.join(missing)}")
    run(["apt-get", "update", "-y"])
    run(
        [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "git",
            "curl",
            "ca-certificates",
        ]
    )


def install_environment(library_dir: Path, experiment_dir: Path) -> Path:
    ensure_system_tools()
    uv = Path("/root/.local/bin/uv")
    if not uv.exists():
        log("installing uv")
        run(
            [
                "bash",
                "-lc",
                "curl -LsSf https://astral.sh/uv/install.sh | sh",
            ]
        )
    if not uv.exists():
        raise RuntimeError("uv installation did not produce /root/.local/bin/uv")

    python = VENV_DIR / "bin" / "python"
    if not python.exists():
        log("installing Python 3.13 and the pinned training environment")
        run([str(uv), "python", "install", "3.13"])
        run([str(uv), "venv", "--python", "3.13", str(VENV_DIR)])
        pins = [
            f"ray[rllib]=={required('RUNPOD_RAY_VERSION')}",
            f"torch=={required('RUNPOD_TORCH_VERSION')}",
            f"gymnasium=={required('RUNPOD_GYMNASIUM_VERSION')}",
            "matplotlib==3.11.1",
            "scipy==1.18.0",
            f"boto3=={required('RUNPOD_BOTO3_VERSION')}",
            f"mlflow-skinny=={required('RUNPOD_MLFLOW_VERSION')}",
        ]
        run([str(uv), "pip", "install", "--python", str(python), *pins])
    else:
        log(f"using baked training environment at {VENV_DIR}")
    run(
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "-e",
            str(library_dir),
            "-e",
            str(experiment_dir),
        ]
    )
    run(
        [
            str(python),
            "-c",
            (
                "import gymnasium, ray, torch; "
                "print('frameworks', 'ray='+ray.__version__, "
                "'torch='+torch.__version__, "
                "'gymnasium='+gymnasium.__version__, flush=True); "
                "print('cuda', torch.version.cuda, "
                "torch.cuda.is_available(), "
                "torch.cuda.get_device_name(0) if torch.cuda.is_available() "
                "else None, flush=True); "
                "assert ray.__version__ == "
                f"{required('RUNPOD_RAY_VERSION')!r}; "
                "assert torch.cuda.is_available()"
            ),
        ]
    )
    return python


def main() -> int:
    started = time.monotonic()
    max_age_s = int(required("RUNPOD_MAX_AGE_S"))
    if max_age_s <= 0:
        raise RuntimeError("RUNPOD_MAX_AGE_S must be positive")
    watchdog = start_watchdog(max_age_s)
    experiment_dir: Path | None = None
    run_name = required("RUNPOD_RUN_NAME")
    interactive = os.environ.get("RUNPOD_INTERACTIVE") == "1"
    exit_code = 1
    cleanup_allowed = False
    cleanup_reason = "bootstrap failed"
    stage = "checkout library"
    try:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        if interactive:
            stage = "start SSH"
            start_ssh_server()
            stage = "checkout library"
        library_sha = checkout(
            url=required("RUNPOD_LIBRARY_REPO_URL"),
            ref=required("RUNPOD_LIBRARY_GIT_REF"),
            target=LIBRARY_DIR,
        )
        stage = "checkout experiment"
        experiment_url = required("RUNPOD_EXPERIMENT_REPO_URL")
        experiment_name = experiment_url.rstrip("/").rsplit("/", 1)[-1]
        experiment_name = experiment_name.removesuffix(".git")
        experiment_dir = WORK_DIR / experiment_name
        experiment_sha = checkout(
            url=experiment_url,
            ref=required("RUNPOD_EXPERIMENT_GIT_REF"),
            target=experiment_dir,
            sparse_experiment=True,
        )
        stage = "install environment"
        python = install_environment(LIBRARY_DIR, experiment_dir)
        if interactive:
            log(
                "interactive CUDA workspace ready; "
                f"experiment={experiment_sha}, library={library_sha}, "
                f"workspace={experiment_dir}"
            )
            while True:
                time.sleep(3600)
        stage = "durable batch lifecycle"
        from devops.runpod.pods.remote_runner import run_batch_job

        outcome = run_batch_job(
            experiment_repo=experiment_dir,
            python=python,
            mlflow_dir=MLFLOW_DIR,
            experiment_sha=experiment_sha,
            library_sha=library_sha,
            log=log,
        )
        exit_code = outcome.exit_code
        cleanup_allowed = outcome.cleanup_allowed
        cleanup_reason = outcome.reason
        log(
            "batch lifecycle finished: "
            f"workload_exit={exit_code}, "
            f"publication={outcome.publication_status}, "
            f"canonical_manifest={outcome.canonical_manifest_key or 'missing'}"
        )
    except Exception as error:  # noqa: BLE001
        log(f"ERROR: runner failed during {stage} ({type(error).__name__})")
        exit_code = 1
    finally:
        elapsed_h = (time.monotonic() - started) / 3600.0
        hourly = float(os.environ.get("RUNPOD_ESTIMATED_PRICE_PER_HOUR", "0"))
        log(
            f"estimated compute cost=${elapsed_h * hourly:.4f} "
            f"({elapsed_h:.3f}h at ${hourly:.3f}/h); authoritative billing "
            "is collected by local status/reap"
        )
        if cleanup_allowed and os.environ.get("RUNPOD_AUTO_TERMINATE") == "1":
            log(
                f"job lifecycle finished ({cleanup_reason}); "
                "flushing logs before teardown"
            )
            time.sleep(5)
            if terminate_self(cleanup_reason):
                watchdog.set()
        elif os.environ.get("RUNPOD_AUTO_TERMINATE") == "1":
            log(
                "automatic teardown withheld because required persistence "
                "did not complete; hard max-age remains active"
            )
            while True:
                time.sleep(3600)
        else:
            watchdog.set()
            log("automatic teardown disabled; provider max-age remains active")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
