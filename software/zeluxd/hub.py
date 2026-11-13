"""状态总线：把所有模块的缓存拼成一份快照，按客户端可见性变频推送；同时做命令分发。

为什么要有这一层：
* 一条 WebSocket 同时承载"推"和"控"，手机端省电、省连接；
* 页面不可见时降到 1 Hz（这个细节参考镜像做得很好，我们照做）；
* 命令按模块加锁，长任务（拍照、训练导入）不会把 10 Hz 状态流堵死。
"""
from __future__ import annotations

import asyncio
import collections
import importlib
import logging
import os
import pkgutil
import platform
import socket
import time
import types
from typing import Any

from .errors import ZeluxError
from .hal import create_hal, load_driver_selection
from .drivers import DriverRegistry
from .firmware import FirmwareManager

log = logging.getLogger("zeluxd.hub")


# ---------------------------------------------------------------- 日志总线
class LogBus(logging.Handler):
    """把 stdlib logging 收进环形缓冲，页面日志页直接读，不用去翻 journald。"""

    CATS = {"zeluxd": "系统", "zeluxd.servos": "总线", "zeluxd.imu": "IMU",
            "zeluxd.pad": "手柄", "zeluxd.tof": "ToF", "zeluxd.fan": "风扇",
            "zeluxd.vision": "视觉", "zeluxd.ai": "AI", "zeluxd.train": "训练",
            "zeluxd.ota": "更新"}

    def __init__(self, size: int = 600) -> None:
        super().__init__()
        self.items: collections.deque = collections.deque(maxlen=size)
        self.n = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.n += 1
            self.items.append({
                "n": self.n,
                "ts": record.created,
                "t": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "cat": self.CATS.get(record.name, "系统"),
                "level": record.levelname.lower(),
                "msg": record.getMessage(),
                "raw": None,
                "hint": getattr(record, "hint", None),
            })
        except Exception:  # 日志不许反噬
            pass


# ---------------------------------------------------------------- 客户端
class Client:
    def __init__(self, ws, peer: str) -> None:
        self.ws = ws
        self.peer = peer
        self.visible = True
        self.route = "/"
        self.focus: str | None = None
        self.last_hb = time.monotonic()
        self.last_sent = 0.0

    def interval(self) -> float:
        return 0.1 if self.visible else 1.0

    def wants(self, module: str) -> bool:
        """某些模块（如 ToF 大帧）只在对应页面可见时才推全量。"""
        return self.visible and (self.focus is None or self.focus == module)


# ---------------------------------------------------------------- Hub
class Hub:
    def __init__(self, config) -> None:
        self.config = config
        self.log_bus = LogBus()
        self.clients: set[Client] = set()
        self.modules: dict[str, Any] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.mode = "commissioning"
        self.posture = "absent"
        self.estop = {"state": "ready", "why": ""}
        self.started_at = time.time()
        self.duck = {"label": "zelux-duck", "color": "#00D8A0"}
        self._stop = asyncio.Event()
        self.hal = None
        self.driver_registry = None
        self.firmware_manager = None
        self.ctx = None
        self.imu_publisher = None

    # ---------------- 组装 ----------------
    async def boot(self) -> None:
        cfg = self.config
        cfg.ensure_dirs()
        self.ctx = types.SimpleNamespace(config=cfg, log=logging.getLogger("zeluxd"), hub=self,
                                         state={}, hal=None, version=__import__("zeluxd").__version__)
        selection = load_driver_selection(cfg.drivers_dir / "selection.json")
        self.hal = create_hal(self.ctx, selection=selection)
        self.ctx.hal = self.hal
        self.driver_registry = DriverRegistry(cfg.mode, self.hal, state_dir=cfg.drivers_dir,
                                              agent_version=self.ctx.version,
                                              allow_untrusted=cfg.dev)
        self.firmware_manager = FirmwareManager(cfg.firmware_dir, self.driver_registry,
                                                allow_untrusted=cfg.dev)
        self._load_modules()
        await self.hal.start()
        if cfg.robotd_imu_publish:
            from .imu_publish import ImuPublisher

            self.imu_publisher = ImuPublisher(self.hal, cfg.robotd_imu_socket)
            await self.imu_publisher.start()
        for m in self.modules.values():
            try:
                await m.start()
            except Exception as exc:
                log.error("模块 %s 起不来：%s", m.name, exc)
        asyncio.create_task(self._broadcast_loop())
        log.info("ZELUX Duck Studio %s 启动 · %s · %s", self.ctx.version, cfg.mode, socket.gethostname())

    def _load_modules(self) -> None:
        from .modules.base import BaseModule

        pkg = importlib.import_module("zeluxd.modules")
        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name in ("base", "__init__"):
                continue
            try:
                mod = importlib.import_module(f"zeluxd.modules.{info.name}")
            except Exception as exc:
                log.warning("模块 %s 导入失败：%s", info.name, exc)
                continue
            cls = getattr(mod, "Module", None)
            if not cls or not (isinstance(cls, type) and issubclass(cls, BaseModule)):
                continue
            inst = cls(self.ctx)
            self.modules[inst.name] = inst
            self.locks[inst.name] = asyncio.Lock()
            # 单复数别名：协议里写的是 servo.op，模块名叫 servos。两种都认，别让前端为难。
            # 只给写明的这几个开别名，"sys" 不能被削成 "sy"。
            alias = {"servos": "servo", "servo": "servos"}.get(inst.name)
            if alias:
                self.modules.setdefault(alias, inst)
        log.info("已加载模块：%s", "、".join(sorted(self.modules)) or "（无）")

    async def shutdown(self) -> None:
        self._stop.set()
        for m in self.modules.values():
            try:
                await m.stop()
            except Exception:
                pass
        if self.hal:
            if self.imu_publisher:
                await self.imu_publisher.stop()
            await self.hal.stop()

    # ---------------- 状态 ----------------
    def snapshot(self) -> dict:
        root: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "mode": self.mode,
            "posture": self.posture,
            "estop": dict(self.estop),
            "battery": self.hal.snapshot_power() if self.hal else {},
            "cpu": {"temp": self.hal.snapshot_temp() if self.hal else None,
                    "load": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None},
            "net": {"ip": _primary_ip(), "host": socket.gethostname()},
            "servo_session": None,
        }
        for name, m in self.modules.items():
            try:
                part = m.state()
            except Exception as exc:
                part = {name: {"error": str(exc)}}
            if isinstance(part, dict):
                root.update(part)
        return root

    async def _broadcast_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.clients:
                    snap = self.snapshot()
                    now = time.monotonic()
                    for c in list(self.clients):
                        if now - c.last_sent >= c.interval():
                            payload = self._for_client(snap, c)
                            c.last_sent = now
                            try:
                                await c.ws.send_json({"t": "state", "data": payload})
                            except Exception:
                                self.clients.discard(c)
            except Exception:
                log.exception("广播循环出错")
            await asyncio.sleep(0.1)

    def _for_client(self, snap: dict, c: Client) -> dict:
        """按页面裁数据：ToF 64 格、舵机 15 行只在需要的页面全量推，省 3G/弱 WiFi 的流量。"""
        out = dict(snap)
        tof = snap.get("tof") or {}
        if len(tof.get("distance_mm") or []) == 64 and not c.wants("tof"):
            out["tof"] = {k: v for k, v in tof.items() if k not in ("distance_mm", "status")}
            out["tof"]["zones_omitted"] = True
        servos = snap.get("servos") or {}
        if len(servos.get("items") or []) and not c.wants("servos"):
            out["servos"] = {"online": servos.get("online"), "summary": servos.get("summary"),
                             "items_omitted": True}
        return out

    # ---------------- 命令 ----------------
    async def dispatch(self, name: str, args: dict) -> dict:
        args = args or {}
        if name == "relax":
            return await self.relax()
        if "." in name:
            mod_name, op = name.split(".", 1)
        else:
            mod_name, op = name, ""
        mod = self.modules.get(mod_name)
        if not mod:
            raise ZeluxError(f"没有 {mod_name} 这个模块", "unsupported")
        lock = self.locks.setdefault(mod_name, asyncio.Lock())
        if lock.locked():
            raise ZeluxError(f"{mod_name} 正在处理上一个请求", "busy")
        async with lock:
            result = await mod.cmd(op, args)
        # 急停之后要能重新握持：用户明确让舵机使能/摆姿势，就等于重新接管了控制权。
        # 不在这里复位的话，一次急停会让避障和语音动作永久失效（只能重启）。
        if self.estop["state"] == "confirmed" and mod.name in ("servos", "servo"):
            if op in ("pose", "torque") and args.get("on", True) is not False:
                self.estop = {"state": "ready", "why": ""}
                log.info("急停已复位：舵机重新使能")
        return result

    async def relax(self) -> dict:
        self.estop = {"state": "pending", "why": "正在卸力"}
        try:
            res = await asyncio.wait_for(self.hal.relax(), timeout=3.0)
        except Exception as exc:
            self.estop = {"state": "unavailable", "why": f"总线没响应：{exc}"}
            return {"ok": False, "error": self.estop["why"], "state": "unavailable"}
        self.estop = {"state": "confirmed", "why": "已全部卸力"}
        log.info("急停：全部舵机已卸力  %s", res or "")
        return {"ok": True, "state": "confirmed"}

    # ---------------- 推送给页面 ----------------
    def emit_event(self, name: str, data: dict) -> None:
        for c in list(self.clients):
            asyncio.create_task(_safe_send(c, {"t": "event", "name": name, "data": data}))

    def log_line(self, msg: str, level: str = "info", cat: str = "zeluxd", **extra) -> None:
        rec = logging.LogRecord("zeluxd." + cat if cat != "zeluxd" else "zeluxd",
                                logging.INFO, __file__, 0, msg, None, None)
        rec.levelname = level.upper()
        rec.hint = extra.get("hint")
        self.log_bus.emit(rec)
        for c in list(self.clients):
            asyncio.create_task(_safe_send(c, {"t": "event", "name": "log",
                                               "data": {"msg": msg, "level": level, "cat": cat}}))


async def _safe_send(c: Client, payload: dict) -> None:
    try:
        await c.ws.send_json(payload)
    except Exception:
        pass


def _primary_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("223.5.5.5", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def system_info() -> dict:
    return {"python": platform.python_version(), "machine": platform.machine(),
            "system": platform.system(), "host": socket.gethostname()}
