"""Local connect UX: write SSH aliases and open one terminal tab per box.

Writes ``~/.ssh/config.d/vast.conf`` with per-instance ``vast-<id>`` Host
aliases (and makes sure ``~/.ssh/config`` Includes it), then opens a terminal
tab per box already SSH'd in — iTerm2 if installed, else Terminal.app.
``--no-open`` skips the tabs.

Aliases are merged into the shared file so concurrent agents do not overwrite
each other's connection targets. Destroy/reap prune only the aliases they
remove.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .config import CONFIG, VastConfig

ITERM_APP = Path("/Applications/iTerm.app")
_INCLUDE_LINE = "Include config.d/*.conf"
_HEADER = (
    "# Managed by devops/vast — merge/prune on provision up/destroy. "
    "Do not edit Host blocks."
)
_HOST_START = re.compile(r"^Host\s+(\S+)\s*$")


@dataclass
class BoxConn:
    alias: str        # e.g. "vast-45325012"
    host: str
    port: int
    instance_id: int
    user: str = "root"


def ssh_alias_for_instance(instance_id: int) -> str:
    """Stable SSH Host alias for one vast instance (unique across agents)."""
    return f"vast-{int(instance_id)}"


def _identity_file(cfg: VastConfig) -> Path:
    """Private key path from the configured public key (strip the .pub)."""
    pub = Path(cfg.SSH_KEY_PATH).expanduser()
    return pub.with_suffix("") if pub.suffix == ".pub" else pub


def _format_host_block(box: BoxConn, identity: Path) -> str:
    return "\n".join(
        [
            f"Host {box.alias}",
            f"    HostName {box.host}",
            f"    Port {box.port}",
            f"    User {box.user}",
            f"    IdentityFile {identity}",
            "    StrictHostKeyChecking accept-new",
            "    UserKnownHostsFile ~/.ssh/known_hosts",
            "",
        ]
    )


def _parse_host_blocks(text: str) -> dict[str, str]:
    """Parse Host blocks from an ssh config fragment into ``{alias: block}``."""
    blocks: dict[str, str] = {}
    current_alias: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        match = _HOST_START.match(line)
        if match:
            if current_alias is not None:
                blocks[current_alias] = "\n".join(current_lines).rstrip() + "\n"
            current_alias = match.group(1)
            current_lines = [line]
            continue
        if current_alias is not None:
            current_lines.append(line)
    if current_alias is not None:
        blocks[current_alias] = "\n".join(current_lines).rstrip() + "\n"
    return blocks


def write_ssh_config(
    boxes: list[BoxConn],
    cfg: VastConfig = CONFIG,
    log=print,
    *,
    prune_aliases: Optional[Iterable[str]] = None,
) -> Path:
    """Merge Host aliases into vast.conf; optionally prune destroyed aliases."""
    conf_path = Path(cfg.SSH_CONFIG_PATH).expanduser()
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    identity = _identity_file(cfg)

    existing = conf_path.read_text() if conf_path.exists() else ""
    by_alias = _parse_host_blocks(existing)
    for alias in prune_aliases or ():
        by_alias.pop(alias, None)
    for box in boxes:
        by_alias[box.alias] = _format_host_block(box, identity)

    pruned = [alias for alias in (prune_aliases or ()) if alias]
    ordered = sorted(by_alias.items(), key=lambda item: item[0])
    body = "\n".join(block for _, block in ordered)
    conf_path.write_text(f"{_HEADER}\n\n{body}".rstrip() + ("\n" if body else "\n"))
    if boxes or pruned:
        log(
            f"ssh config: {len(boxes)} upserted, {len(pruned)} pruned -> "
            f"{conf_path} ({len(by_alias)} total alias(es))"
        )

    _ensure_include(conf_path.parent.parent / "config", log)
    return conf_path


def prune_ssh_aliases(
    aliases: Iterable[str],
    cfg: VastConfig = CONFIG,
    log=print,
) -> Path:
    """Remove specific Host aliases from the shared vast.conf."""
    return write_ssh_config([], cfg, log, prune_aliases=aliases)


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
