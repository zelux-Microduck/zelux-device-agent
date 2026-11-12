#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
version = (ROOT / "VERSION").read_text("utf-8").strip()
init = (ROOT / "software/zeluxd/__init__.py").read_text("utf-8")
pyproject = (ROOT / "pyproject.toml").read_text("utf-8")
agent = re.search(r'__version__\s*=\s*"([^"]+)"', init).group(1)
project = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE).group(1)
if len({version, agent, project}) != 1:
    raise SystemExit(f"版本不一致：VERSION={version}, agent={agent}, project={project}")
print(f"version {version} OK")
