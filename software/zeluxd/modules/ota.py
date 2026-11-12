"""升级（OTA）：版本目录 + 软链 + 回滚。

这套做法抄的是参考产品验证过的模型，但把"校验"写得更好：
* `/opt/zelux/releases/<版本>/` 每个版本一个完整目录，`/opt/zelux/current` 是指向它的软链；
* 升级 = 下载 → 校验 sha256 → 解到新目录 → 原子切软链 → 重启服务；
* 任何一步失败都保持旧版本不动（切换是 `os.replace` 一个软链，原子）；
* **前置条件必须校验**：鸭子得站好、策略得关掉，否则舵机在升级重启时会突然泄力摔下来。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from .. import __version__
from ..errors import ZeluxError, denied
from .base import BaseModule

ROOT = Path(os.environ.get("ZELUX_INSTALL_ROOT", "/opt/zelux"))
RELEASES = ROOT / "releases"
CURRENT = ROOT / "current"


class Module(BaseModule):
    name = "ota"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.job: dict | None = None
        self._store = self.load_store()

    # ---------------- 状态 ----------------
    def _installed(self) -> str:
        try:
            if CURRENT.exists():
                target = CURRENT.resolve().name
                if target:
                    return target
        except Exception:
            pass
        return __version__

    def _available_releases(self) -> list[str]:
        if not RELEASES.exists():
            return []
        return sorted((p.name for p in RELEASES.iterdir() if p.is_dir()), reverse=True)

    def state(self) -> dict:
        return {"ota": {
            "installed": self._installed(),
            "source": self._store.get("source"),
            "releases": self._available_releases()[:8],
            "staged": self._store.get("staged"),
            "candidate": self._store.get("candidate"),
            "busy": bool(self.job and self.job.get("phase") not in (None, "done", "error")),
            "job": self.job,
            "can_rollback": len(self._available_releases()) > 1,
        }}

    # ---------------- 命令 ----------------
    async def cmd(self, op: str, args: dict) -> dict:
        if op == "check":
            return await self._check(args.get("url"))
        if op == "stage":
            return await self._stage(args)
        if op == "apply":
            return await self._apply(args)
        if op == "rollback":
            return await self._rollback()
        if op == "clean":
            return await self._clean()
        raise ZeluxError(f"ota 不支持 {op}", "unsupported")

    async def _check(self, url: str | None) -> dict:
        url = url or self._store.get("source")
        if not url:
            raise denied("还没配升级源。这一版的镜像用 `sudo zelux-update check` 走命令行，或在设置里填一个 manifest.json 地址")
        try:
            raw = await asyncio.wait_for(self._get(url), timeout=6)
            manifest = json.loads(raw)
        except asyncio.TimeoutError:
            raise ZeluxError("升级源没响应（6 秒超时）。鸭子可能没连外网", "hardware")
        except Exception as exc:
            raise ZeluxError(f"读升级源失败：{exc}", "hardware")
        current = self._installed()
        latest = manifest.get("version")
        self._store.update({"source": url, "candidate": manifest if latest and latest != current else None})
        self.save_store(self._store)
        return {"installed": current, "latest": latest, "update": bool(latest and latest != current),
                "changelog": manifest.get("changelog"), "size": manifest.get("size")}

    async def _stage(self, args: dict) -> dict:
        manifest = self._store.get("candidate") or args.get("manifest")
        if not manifest:
            raise denied("没有可安装的版本，先检查更新")
        if self.job and self.job.get("phase") == "downloading":
            raise ZeluxError("已经在下载了", "busy")
        self.job = {"phase": "downloading", "started_at": time.time(), "version": manifest.get("version")}
        try:
            data = await asyncio.wait_for(self._get(manifest["url"]), timeout=180)
        except Exception as exc:
            self.job = {"phase": "error", "error": f"下载失败：{exc}"}
            raise ZeluxError(f"下载失败：{exc}", "hardware")
        want = (manifest.get("sha256") or "").lower()
        got = hashlib.sha256(data).hexdigest()
        if want and got != want:
            self.job = {"phase": "error", "error": "校验对不上，包可能下坏了"}
            raise ZeluxError("校验对不上，包可能下坏了（也可能被人改过），已经丢弃", "denied")
        staged = self.config.state_dir / "staged"
        staged.mkdir(parents=True, exist_ok=True)
        path = staged / f"zelux-{manifest['version']}.tar.gz"
        path.write_bytes(data)
        self._store["staged"] = {"version": manifest["version"], "path": str(path), "sha256": got,
                                 "changelog": manifest.get("changelog"), "at": time.time()}
        self.save_store(self._store)
        self.job = {"phase": "done", "version": manifest["version"]}
        self.log_event(f"新版本 {manifest['version']} 已下载并校验通过", cat="ota")
        return {"ok": True, "version": manifest["version"]}

    async def _apply(self, args: dict) -> dict:
        staged = self._store.get("staged")
        if not staged:
            raise denied("还没有下载好的版本")
        # 前置条件：鸭子必须站好、没有在走、没有策略在跑
        hub = self.ctx.hub
        if hub.posture not in ("holding", "sitting"):
            raise denied(f"现在鸭子是「{hub.posture}」，升级会重启服务让舵机泄力。先让它站好或坐下再来")
        if hub.estop.get("state") == "pending":
            raise ZeluxError("正在卸力，等它完成", "busy")
        helper = shutil.which("zelux-update")
        if not helper:
            raise ZeluxError("这台机器上没有 zelux-update 助手（不是镜像装的系统？），请用 install.sh 重装", "unsupported")
        self.job = {"phase": "applying", "version": staged["version"]}
        try:
            proc = await asyncio.create_subprocess_exec("sudo", "-n", helper, "apply", staged["path"],
                                                        stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.STDOUT)
            out = (await asyncio.wait_for(proc.communicate(), timeout=120))[0].decode(errors="ignore")
        except asyncio.TimeoutError:
            self.job = {"phase": "error", "error": "切换版本超时"}
            raise ZeluxError("切换版本超时，旧版本还在跑", "timeout")
        if proc.returncode != 0:
            self.job = {"phase": "error", "error": out[-400:]}
            raise ZeluxError(f"升级失败，旧版本没动：{out.strip()[-200:]}", "failed")
        self.job = {"phase": "done", "version": staged["version"], "output": out[-300:]}
        self._store["staged"] = None
        self.save_store(self._store)
        return {"ok": True, "note": "服务会在 3 秒内重启，页面会自动重连"}

    async def _rollback(self) -> dict:
        releases = self._available_releases()
        current = self._installed()
        others = [r for r in releases if r != current]
        if not others:
            raise denied("只有一个版本，没有可回滚的目标")
        helper = shutil.which("zelux-update")
        if not helper:
            raise ZeluxError("没有 zelux-update 助手，回滚要手动改 /opt/zelux/current 软链", "unsupported")
        proc = await asyncio.create_subprocess_exec("sudo", "-n", helper, "switch", others[0],
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out = (await asyncio.wait_for(proc.communicate(), timeout=60))[0].decode(errors="ignore")
        if proc.returncode != 0:
            raise ZeluxError(f"回滚失败：{out[-200:]}", "failed")
        return {"ok": True, "to": others[0]}

    async def _clean(self) -> dict:
        releases = self._available_releases()
        current = self._installed()
        removed = []
        for old in releases[2:]:
            if old == current:
                continue
            try:
                shutil.rmtree(RELEASES / old)
                removed.append(old)
            except Exception:
                pass
        return {"ok": True, "removed": removed}

    # ---------------- 工具 ----------------
    async def _get(self, url: str) -> bytes:
        if url.startswith("file://"):
            return Path(url[7:]).read_bytes()
        if "://" not in url:
            return Path(url).read_bytes()

        def fetch() -> bytes:
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": f"zeluxd/{__version__}"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()

        return await asyncio.to_thread(fetch)
