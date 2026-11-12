"""模块基类。

一个模块 = 一个页面需要的后端能力。约定：

* `state()` 必须是**纯同步、微秒级**的，只读缓存；10 Hz 状态总线每 tick 都会调它。
* `cmd(op, args)` 是协程，可以慢、可以抛 `ZeluxError`。
* 需要后台轮询的自己 `asyncio.create_task`，在 `stop()` 里收。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..errors import ZeluxError


class BaseModule:
    name = "base"

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.hal = ctx.hal
        self.config = ctx.config
        self.log = logging.getLogger(f"zeluxd.{self.name}")
        self._tasks: list = []

    # ---------- 生命周期 ----------
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()

    def spawn(self, coro) -> None:
        import asyncio

        self._tasks.append(asyncio.create_task(coro))

    # ---------- 协议 ----------
    def state(self) -> dict:
        return {}

    async def cmd(self, op: str, args: dict) -> dict:
        raise ZeluxError(f"{self.name} 不支持 {op}", "unsupported")

    # ---------- 小工具 ----------
    @property
    def store_path(self) -> Path:
        d = self.config.state_dir
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.name}.json"

    def load_store(self) -> dict:
        try:
            return json.loads(self.store_path.read_text("utf-8"))
        except Exception:
            return {}

    def save_store(self, data: dict) -> None:
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(self.store_path)

    def emit(self, name: str, data: dict) -> None:
        """给前端推一条事件（日志/播报/进度）。"""
        self.ctx.hub.emit_event(name, data)

    def log_event(self, msg: str, level: str = "info", cat: str | None = None, **extra) -> None:
        self.ctx.hub.log_line(msg, level=level, cat=cat or self.name, **extra)

    @staticmethod
    def now() -> float:
        return time.time()
