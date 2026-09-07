"""
Secrets come from a .env file at the project root.

The alternative is asking whoever starts aiRon to export a key first, which
is a step to forget on a robot that should just come up when it is powered
on. A file next to the code survives reboots and lets a systemd unit or a
double-clicked launcher work the same way a shell does.

Deliberately dependency-free: this runs on a Jetson that is not always
online, and pulling in a package to split a string on "=" is not worth an
install step. The format is the usual one - KEY=value per line, # comments,
blank lines ignored, optional surrounding quotes, optional "export" prefix.

Anything already in the environment wins. An exported key is a deliberate
act - overriding it from a file would make aiRon ignore what the person in
front of the terminal just told it.
"""

from __future__ import annotations

import os
from pathlib import Path

# airon/core/env.py -> airon/core -> airon -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ENV = PROJECT_ROOT / ".env"


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_env(path: Path | str | None = None) -> list[str]:
    """Read a .env into os.environ. Returns the names it set - never the
    values, because this gets printed at startup and a key that is echoed
    to a terminal is a key in somebody's scrollback."""
    path = Path(path) if path is not None else DEFAULT_ENV
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []

    loaded = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            continue
        if key in os.environ:
            continue
        os.environ[key] = _unquote(value)
        loaded.append(key)
    return loaded
