"""Local connect UX: write SSH aliases and open one terminal tab per box.

Writes ``~/.ssh/config.d/vast.conf`` with ``vast-1..N`` Host aliases (and makes
sure ``~/.ssh/config`` Includes it), then opens a terminal tab per box already
SSH'd in — iTerm2 if installed, else Terminal.app. ``--no-open`` skips the tabs.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import CONFIG, VastConfig

ITERM_APP = Path("/Applications/iTerm.app")
_INCLUDE_LINE = "Include config.d/*.conf"


@dataclass
class BoxConn:
    alias: str        # e.g. "vast-1"
    host: str
    port: int
    instance_id: int
    user: str = "root"


def _identity_file(cfg: VastConfig) -> Path:
    """Private key path from the configured public key (strip the .pub)."""
    pub = Path(cfg.SSH_KEY_PATH).expanduser()
    return pub.with_suffix("") if pub.suffix == ".pub" else pub


def write_ssh_config(boxes: list[BoxConn], cfg: VastConfig = CONFIG, log=print) -> Path:
    """Write vast.conf Host aliases and ensure ~/.ssh/config Includes it."""
    conf_path = Path(cfg.SSH_CONFIG_PATH).expanduser()
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    identity = _identity_file(cfg)

    blocks = [
        "# Managed by devops/vast — regenerated on each `provision up`. Do not edit.",
        "",
    ]
    for b in boxes:
        blocks += [
            f"Host {b.alias}",
            f"    HostName {b.host}",
            f"    Port {b.port}",
            f"    User {b.user}",
            f"    IdentityFile {identity}",
            "    StrictHostKeyChecking accept-new",
            "    UserKnownHostsFile ~/.ssh/known_hosts",
            "",
        ]
    conf_path.write_text("\n".join(blocks))
    log(f"wrote {len(boxes)} ssh alias(es) -> {conf_path}")

    _ensure_include(conf_path.parent.parent / "config", log)
    return conf_path


def _ensure_include(ssh_config: Path, log=print) -> None:
    """Prepend an Include for config.d/*.conf to ~/.ssh/config if absent."""
    ssh_config = Path(ssh_config).expanduser()
    existing = ssh_config.read_text() if ssh_config.exists() else ""
    if "config.d/" in existing:
        return
    ssh_config.parent.mkdir(parents=True, exist_ok=True)
    ssh_config.write_text(f"{_INCLUDE_LINE}\n\n{existing}")
    try:
        ssh_config.chmod(0o600)
    except OSError:
        pass
    log(f"added '{_INCLUDE_LINE}' to {ssh_config}")


def open_terminals(boxes: list[BoxConn], log=print) -> bool:
    """Open one terminal tab per box, each SSH'd into its alias."""
    if not boxes:
        return False
    aliases = [b.alias for b in boxes]
    if ITERM_APP.exists():
        script = _iterm_script(aliases)
        app = "iTerm2"
    else:
        script = _terminal_script(aliases)
        app = "Terminal.app"
    try:
        subprocess.run(["osascript", "-"], input=script, text=True, check=True,
                       capture_output=True)
        log(f"opened {len(aliases)} {app} tab(s): {', '.join(aliases)}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        detail = getattr(e, "stderr", "") or str(e)
        log(f"could not auto-open {app} ({detail.strip()}); connect manually: "
            + "; ".join(f"ssh {a}" for a in aliases))
        return False


def _iterm_script(aliases: list[str]) -> str:
    lines = [
        'tell application "iTerm"',
        "    activate",
        "    set newWindow to (create window with default profile)",
        f'    tell current session of newWindow to write text "ssh {aliases[0]}"',
    ]
    for alias in aliases[1:]:
        lines += [
            "    tell newWindow",
            "        create tab with default profile",
            f'        tell current session to write text "ssh {alias}"',
            "    end tell",
        ]
    lines.append("end tell")
    return "\n".join(lines)


def _terminal_script(aliases: list[str]) -> str:
    lines = ['tell application "Terminal"', "    activate"]
    for alias in aliases:
        lines.append(f'    do script "ssh {alias}"')
    lines.append("end tell")
    return "\n".join(lines)
