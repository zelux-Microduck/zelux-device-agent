"""系统模块：不涉及硬件的那些（时间、WiFi 扫描、重启、版本、日志）。

放这里的原因：这些能力每个镜像都要有，但它们跟"鸭子"没关系，单独一页"系统"。
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import time

from .. import __version__
from ..errors import ZeluxError
from .base import BaseModule


class Module(BaseModule):
    name = "sys"

    def __init__(self, ctx):
        super().__init__(ctx)
        self._cache: dict = {}

    def state(self) -> dict:
        return {"sys": {"version": __version__, "uptime_s": round(time.time() - self.ctx.hub.started_at),
                        "backend": self.config.mode, "hostname": self._hostname()}}

    def _hostname(self) -> str:
        try:
            import socket

            return socket.gethostname()
        except Exception:
            return "zelux-duck"

    async def cmd(self, op: str, args: dict) -> dict:
        if op == "info":
            return {"version": __version__, "backend": self.config.mode, "state_dir": str(self.config.state_dir)}
        if op == "logs":
            items = list(self.ctx.hub.log_bus.items)[-int(args.get("n", 200)):]
            return {"items": items}
        if op == "time":
            return {"now": time.time(), "text": time.strftime("%Y-%m-%d %H:%M:%S")}
        if op == "wifi_scan":
            return {"networks": await self._wifi_scan()}
        if op == "reboot":
            if not args.get("confirm"):
                raise ZeluxError("重启前请确认：舵机会保持当前姿势（不断电）", "denied")
            self.log_event("正在重启主控…", cat="sys")
            asyncio.create_task(self._later("reboot"))
            return {"ok": True, "note": "约 40 秒后回来"}
        if op == "poweroff":
            if not args.get("confirm"):
                raise ZeluxError("关机前请先让鸭子坐下或扶着它", "denied")
            asyncio.create_task(self._later("poweroff"))
            return {"ok": True}
        if op == "sim_hook":
            hook = getattr(self.hal, "sim_hook", None)
            if not hook:
                raise ZeluxError("只有模拟模式才有仿真钩子", "unsupported")
            return await hook(str(args.get("name")), args)
        raise ZeluxError(f"sys 不支持 {op}", "unsupported")

    async def _later(self, what: str) -> None:
        await asyncio.sleep(1.0)
        try:
            subprocess.Popen(["systemctl", what], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            self.log_event(f"{what} 失败：{exc}", level="error", cat="sys")

    async def _wifi_scan(self) -> list:
        if not shutil.which("nmcli"):
            return []
        try:
            out = await asyncio.wait_for(
                asyncio.create_subprocess_exec("nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi",
                                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL), timeout=8)
            raw = (await out.communicate())[0].decode(errors="ignore")
        except Exception:
            return []
        nets = []
        for line in raw.splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[0].strip():
                nets.append({"ssid": parts[0], "signal": int(parts[1] or 0), "security": parts[2] or "open"})
        return sorted(nets, key=lambda n: -n["signal"])[:20]
