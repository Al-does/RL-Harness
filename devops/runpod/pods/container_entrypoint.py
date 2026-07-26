"""RunPod Pod entrypoint: clone, train, persist, and always terminate.

This file is sent to the Pod by the provisioning API (and copied into the
optional custom image). It intentionally depends only on Python's standard
library until the pinned training environment has been installed.
"""

from __future__ import annotations

import base64
import json
import os
import random
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


def start_mlflow_run(
    python: Path,
    tags: dict[str, str],
) -> str:
    env = dict(os.environ)
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    env["RUNPOD_MLFLOW_TAGS"] = json.dumps(tags, sort_keys=True)
    script = (
        "import json, os, mlflow; "
        f"mlflow.set_tracking_uri({('file:' + str(MLFLOW_DIR))!r}); "
        "mlflow.set_experiment('runpod-pods'); "
        "run=mlflow.start_run(run_name=os.environ['RUNPOD_RUN_NAME']); "
        "mlflow.set_tags(json.loads(os.environ['RUNPOD_MLFLOW_TAGS'])); "
        "print(run.info.run_id)"
    )
    result = run(
        [str(python), "-c", script],
        env=env,
        capture_output=True,
    )
    run_id = result.stdout.strip().splitlines()[-1]
    log(f"MLflow run started: {run_id}")
    return run_id


def finish_mlflow_run(
    python: Path | None,
    run_id: str | None,
    status: str,
) -> None:
    if not python or not run_id:
        return
    env = dict(os.environ)
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    env["MLFLOW_RUN_ID"] = run_id
    script = (
        "import os, mlflow; "
        f"mlflow.set_tracking_uri({('file:' + str(MLFLOW_DIR))!r}); "
        "mlflow.tracking.MlflowClient().set_terminated("
        "os.environ['MLFLOW_RUN_ID'], "
        f"status={status!r})"
    )
    try:
        run([str(python), "-c", script], env=env)
    except Exception as error:  # noqa: BLE001
        log(f"WARNING: could not finish MLflow run ({type(error).__name__})")


def upload_mlflow(python: Path | None, run_name: str) -> None:
    if not python or not MLFLOW_DIR.exists():
        return
    required_keys = (
        "B2_BUCKET",
        "B2_ENDPOINT",
        "B2_APPLICATION_KEY_ID",
        "B2_APPLICATION_KEY",
    )
    if not all(os.environ.get(key) for key in required_keys):
        log("MLflow metadata not uploaded: B2 is not configured")
        return
    prefix_root = os.environ.get("B2_PREFIX", "").strip("/")
    prefix = "/".join(
        part for part in (prefix_root, "runpod", "mlflow", run_name) if part
    )
    script = """
import os
from pathlib import Path
import boto3
from botocore.config import Config

root = Path(os.environ["RUNPOD_MLFLOW_DIR"])
endpoint = os.environ["B2_ENDPOINT"]
if not endpoint.startswith(("http://", "https://")):
    endpoint = "https://" + endpoint
client = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ["B2_APPLICATION_KEY_ID"],
    aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
    config=Config(signature_version="s3v4"),
)
for path in root.rglob("*"):
    if path.is_file():
        relative = path.relative_to(root).as_posix()
        client.upload_file(
            str(path),
            os.environ["B2_BUCKET"],
            os.environ["RUNPOD_MLFLOW_PREFIX"] + "/" + relative,
        )
"""
    env = dict(os.environ)
    env["RUNPOD_MLFLOW_DIR"] = str(MLFLOW_DIR)
    env["RUNPOD_MLFLOW_PREFIX"] = prefix
    try:
        run([str(python), "-c", script], env=env)
        log(f"MLflow metadata uploaded to s3://{env['B2_BUCKET']}/{prefix}/")
    except Exception as error:  # noqa: BLE001
        log(f"WARNING: MLflow upload failed ({type(error).__name__})")


def push_results(experiment_dir: Path, run_name: str) -> None:
    if os.environ.get("RUNPOD_PUSH_RESULTS") != "1":
        return
    branch = os.environ.get("RUNPOD_RESULTS_BRANCH", "results")
    env = git_auth_env()
    run(["git", "config", "user.name", "runpod-bot"], cwd=experiment_dir)
    run(
        ["git", "config", "user.email", "runpod-bot@users.noreply.github.com"],
        cwd=experiment_dir,
    )
    run(["git", "add", "-A", "--", "experiments/"], cwd=experiment_dir)
    if (
        run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=experiment_dir,
            check=False,
        ).returncode
        == 0
    ):
        log("no compact experiment results to push")
        return
    run(
        [
            "git",
            "commit",
            "-m",
            f"results: {run_name} (RunPod {os.environ.get('RUNPOD_POD_ID', '?')})",
        ],
        cwd=experiment_dir,
    )
    delay = 1.0
    for attempt in range(1, 7):
        fetched = run(
            ["git", "fetch", "origin", branch],
            cwd=experiment_dir,
            env=env,
            check=False,
        )
        if fetched.returncode == 0:
            rebased = run(
                ["git", "rebase", "--autostash", "FETCH_HEAD"],
                cwd=experiment_dir,
                check=False,
            )
            if rebased.returncode != 0:
                run(
                    ["git", "rebase", "--abort"],
                    cwd=experiment_dir,
                    check=False,
                )
        pushed = run(
            ["git", "push", "origin", f"HEAD:refs/heads/{branch}"],
            cwd=experiment_dir,
            env=env,
            check=False,
        )
        if pushed.returncode == 0:
            log(f"pushed compact results to {branch}")
            return
        log(f"results push rejected (attempt {attempt}/6)")
        time.sleep(delay + random.uniform(0, delay))
        delay = min(delay * 2, 30.0)
    log("WARNING: compact results push failed after 6 attempts")


def main() -> int:
    started = time.monotonic()
    max_age_s = int(required("RUNPOD_MAX_AGE_S"))
    if max_age_s <= 0:
        raise RuntimeError("RUNPOD_MAX_AGE_S must be positive")
    watchdog = start_watchdog(max_age_s)
    python: Path | None = None
    mlflow_run_id: str | None = None
    experiment_dir: Path | None = None
    run_name = required("RUNPOD_RUN_NAME")
    status = "FAILED"
    exit_code = 1
    stage = "checkout library"
    try:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
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
        stage = "start MLflow run"
        tags = {
            "git.commit": experiment_sha,
            "git.experiment_commit": experiment_sha,
            "git.library_commit": library_sha,
            "container.image.digest": required("RUNPOD_IMAGE_DIGEST"),
            "runpod.pod_id": os.environ.get("RUNPOD_POD_ID", ""),
            "runpod.cloud": "COMMUNITY",
            "runpod.interruptible": "false",
            "runpod.gpu.requested": required("RUNPOD_GPU_TYPE_IDS"),
        }
        mlflow_run_id = start_mlflow_run(python, tags)
        stage = "run experiment"
        command = required("RUNPOD_RUN_CMD")
        log(f"starting experiment command: {command}")
        env = dict(os.environ)
        env["PATH"] = f"{VENV_DIR / 'bin'}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(VENV_DIR)
        env["MLFLOW_ALLOW_FILE_STORE"] = "true"
        env["MLFLOW_TRACKING_URI"] = f"file:{MLFLOW_DIR}"
        env["MLFLOW_RUN_ID"] = mlflow_run_id
        completed = run(
            ["bash", "-lc", command],
            cwd=experiment_dir,
            env=env,
            check=False,
        )
        exit_code = completed.returncode
        status = "FINISHED" if exit_code == 0 else "FAILED"
        log(f"experiment command exited with status {exit_code}")
    except Exception as error:  # noqa: BLE001
        # Never interpolate exception details here: subprocess/HTTP errors can
        # carry secret-bearing environment or headers.
        log(f"ERROR: runner failed during {stage} ({type(error).__name__})")
        exit_code = 1
        status = "FAILED"
    finally:
        finish_mlflow_run(python, mlflow_run_id, status)
        upload_mlflow(python, run_name)
        if experiment_dir is not None:
            try:
                push_results(experiment_dir, run_name)
            except Exception as error:  # noqa: BLE001
                log(f"WARNING: results push failed ({type(error).__name__})")
        elapsed_h = (time.monotonic() - started) / 3600.0
        hourly = float(os.environ.get("RUNPOD_ESTIMATED_PRICE_PER_HOUR", "0"))
        log(
            f"estimated compute cost=${elapsed_h * hourly:.4f} "
            f"({elapsed_h:.3f}h at ${hourly:.3f}/h); authoritative billing "
            "is collected by local status/reap"
        )
        watchdog.set()
        reason = "job completed" if exit_code == 0 else "job failed"
        log(f"job lifecycle finished ({reason}); flushing logs before teardown")
        time.sleep(5)
        terminate_self(reason)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
