"""风扇页（`name = "fan"`）：两路 PWM 调速 + 主板那颗的转速表 + 温控曲线。

为什么页面上要同时有"自动曲线"和"手动百分比"：
调机时人手要知道"现在到底转多少"，而跑起来以后没人想看第二眼。所以默认 auto
（按 `[[58,70],[64,85],[70,100]]` 阶梯走，58℃ 以下 60% 兜底），
手动一旦点了就锁在手动档，重启回到 auto——一台会自己改回自动的风扇才不会烧主板。

没有风扇的机器（模拟模式、或者镜像没起 zelux-fand）不许让页面崩：
一律返回 `available:false` + 一句能行动的原因；这种情况下手动调速只写一个"演示值"
并在 state 里标 `virtual:true`，让人一眼看出这不是真的在转。
"""
from __future__ import annotations

import asyncio

from ..errors import ZeluxError
from ..hal.radxa_fan import (DEFAULT_PCT, FanUnavailable, clamp_pct, curve_pct,
                             normalize_curve)
from .base import BaseModule

AUTO_HZ = 0.5            # 曲线 2 秒判一次就够：温度惯性比这慢得多


class Module(BaseModule):
    name = "fan"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        stored = self.load_store() or {}
        self.curve = normalize_curve(stored.get("curve") if isinstance(stored, dict) else None)
        self.mode = "auto" if str(stored.get("mode", "auto")) != "manual" else "manual"
        self._manual: dict[int, int] = {}

        # 两条调用路径都认：契约里的 hal.fan_set/fan_read，或者 RadxaFan 驱动对象
        self._driver = getattr(self.hal, "fan", None)
        self._set = getattr(self.hal, "fan_set", None)
        self._read = getattr(self.hal, "fan_read", None)
        self._control = "hal" if (self._set or self._driver) else "none"
        self._last_pct: int | None = None
        self._task: asyncio.Task | None = None

    # ---------------- 状态 ----------------
    def state(self) -> dict:
        snap = dict(self.hal.snapshot_fan() or {})
        items = [dict(i) for i in (snap.get("items") or [])]
        temp = self.hal.snapshot_temp()

        # 没有真实控制器时把手动值标成"演示"，别让人以为真改了转速
        virtual = self._control == "none" and self.mode == "manual"
        if virtual:
            for i, item in enumerate(items):
                item["pct"] = self._manual.get(i, item.get("pct", DEFAULT_PCT))
                item["virtual"] = True
                item["rpm"] = None

        available, reason, hint = self._availability()
        snap.update({
            "mode": self.mode,
            "curve": [list(r) for r in self.curve],
            "temp": temp if temp is not None else snap.get("temp"),
            "items": items,
            "available": available,
            "reason": "" if available else reason,
            "hint": "" if available else hint,
            "virtual": virtual,
            "source": getattr(self._driver, "source", None) or ("hal" if self._control == "hal" else "none"),
            "pct": self._last_pct if self._last_pct is not None else snap.get("pct", DEFAULT_PCT),
            "pin_config": {"channels": [dict(item) for item in getattr(self.config, "fan_channels", [])],
                           "restart_required": False},
        })
        return {"fan": snap}

    def _availability(self) -> tuple[bool, str, str]:
        if self._driver is not None and hasattr(self._driver, "available"):
            return (bool(getattr(self._driver, "available")),
                    str(getattr(self._driver, "reason", "风扇不可用")),
                    str(getattr(self._driver, "hint", "")))
        if self._control == "hal":
            return True, "", ""
        if self.config.mode == "sim":
            return False, "模拟模式没有真风扇：转速只是演示值", "接真机后才会真的转"
        return (False,
                "HAL 没提供风扇接口（缺少 fan_set/fan_read）",
                "确认 hal/radxa.py 里挂了 RadxaFan；或检查 zelux-fand 是否在跑")

    # ---------------- 生命周期 ----------------
    async def start(self) -> None:
        if self.mode == "manual":
            return
        self._ensure_loop()

    async def stop(self) -> None:
        await super().stop()
        self._task = None

    def _ensure_loop(self) -> None:
        if self._control != "hal":
            return                    # 没有真实控制器就别空转，省电省日志
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._auto_loop(), name="fan-auto")

    async def _auto_loop(self) -> None:
        while self.mode == "auto":
            try:
                temp = self.hal.snapshot_temp()
                pct = curve_pct(temp, self.curve)
                if pct != self._last_pct or self._last_pct is None:
                    # 只在档位变化时写 PWM：sysfs 每次写都要走一遍内核，没必要 2 秒敲一次
                    await self._apply_all(pct)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.log.warning("温控循环出错：%s", exc)
            await asyncio.sleep(1.0 / AUTO_HZ)

    # ---------------- 命令 ----------------
    async def cmd(self, op: str, args: dict) -> dict:
        args = args or {}
        if op == "op":
            # 契约：`fan.op` + `{op:"manual", index:0, pct:80}`
            op = str(args.get("op", "") or args.get("mode", "") or "status")
        op = op.strip().lower()

        if op == "auto":
            return await self._set_mode("auto", args)
        if op == "manual":
            return await self._set_mode("manual", args)
        if op == "curve":
            return self._set_curve(args)
        if op in ("pin_config", "pins"):
            return self._set_pin_config(args)
        if op in ("self_test", "selftest", "test"):
            return await self._self_test(args)
        if op == "read":
            return await self._read_one(args)
        if op in ("status", ""):
            return {"ok": True, "fan": self.state()["fan"]}
        raise ZeluxError(f"fan 不支持 {op}（可用：auto/manual/curve/read/self_test/status）",
                         "unsupported")

    def _set_pin_config(self, args: dict) -> dict:
        channels = args.get("channels")
        if not isinstance(channels, list) or len(channels) != 2:
            raise ZeluxError("风扇引脚必须包含两路配置", "denied")
        out = []
        for index, item in enumerate(channels):
            if not isinstance(item, dict):
                raise ZeluxError(f"第 {index} 路配置格式错误", "denied")
            try:
                pwm, gpio = int(item["pwm"]), int(item["gpio"])
                tach = None if item.get("tach") in (None, "") else int(item["tach"])
            except (KeyError, TypeError, ValueError):
                raise ZeluxError(f"第 {index} 路 PWM/GPIO/Tach 必须是整数", "denied") from None
            if not (0 <= pwm <= 255 and 0 <= gpio <= 1023 and (tach is None or 0 <= tach <= 1023)):
                raise ZeluxError(f"第 {index} 路引脚超出允许范围", "denied")
            out.append({"name": str(item.get("name") or f"风扇 {index + 1}")[:32], "pwm": pwm,
                        "gpio": gpio, "pin": str(item.get("pin") or "")[:64], "tach": tach})
        self.config.save_hardware("fan", {"channels": out})
        return {"ok": True, "channels": out, "restart_required": True,
                "note": "配置已保存，重启后重新初始化 PWM 和 Tach"}

    async def _set_mode(self, mode: str, args: dict) -> dict:
        if mode == "auto":
            self.mode = "auto"
            self._save()
            self._ensure_loop()
            temp = self.hal.snapshot_temp()
            pct = curve_pct(temp, self.curve)
            applied = await self._apply_all(pct)
            return {"ok": True, "mode": "auto", "pct": pct, "temp": temp,
                    "applied": applied, **self._avail_note()}

        index = int(args.get("index", 0) or 0)
        if args.get("pct") is None:
            raise ZeluxError("手动档要给 pct（0–100）", "denied")
        pct = clamp_pct(args.get("pct"))
        self.mode = "manual"
        self._task = None             # 曲线循环见到 mode 变了会自己退出
        self._save()
        result = await self._apply(index, pct)
        if result.get("ok"):
            self.log_event(f"风扇 {index} 手动 {pct}%（{result.get('source', '?')}）")
        return {"ok": bool(result.get("ok")), "mode": "manual", "index": index, "pct": pct,
                "detail": result, **self._avail_note()}

    def _set_curve(self, args: dict) -> dict:
        self.curve = normalize_curve(args.get("curve"))
        self._save()
        self.log_event("温控曲线已更新：" + "、".join(f"{t}℃→{p}%" for t, p in self.curve))
        return {"ok": True, "curve": [list(r) for r in self.curve], "mode": self.mode,
                "note": "阶梯判据是 temp ≥ 台阶温度；都不满足时走 60%"}

    async def _read_one(self, args: dict) -> dict:
        index = int(args.get("index", 1) or 0)
        if self._driver is not None and hasattr(self._driver, "read"):
            return {"ok": True, "item": await self._driver.read(index)}
        if self._read:
            return {"ok": True, "item": await self._read(index)}
        items = self.state()["fan"]["items"]
        if index < 0 or index >= len(items):
            raise ZeluxError(f"没有第 {index} 号风扇", "denied")
        return {"ok": True, "item": items[index], "virtual": self._control == "none"}

    async def _self_test(self, args: dict) -> dict:
        """全速 1 秒看 rpm 是不是 > 300——这是"风扇到底转没转"唯一靠谱的判据。"""
        if self._driver is not None and hasattr(self._driver, "self_test"):
            try:
                res = await self._driver.self_test(args.get("index"))
            except FanUnavailable as exc:
                raise ZeluxError(exc.message, "denied") from None
            self.log_event("风扇自检：" + ("通过" if res.get("ok") else "有问题，看详情"))
            return res
        return {"ok": None, "available": False,
                "reason": "这台机器没法自检（没有 RadxaFan 驱动）",
                "hint": "真机上启用 zelux-fand 后，自检会全速 1 秒读 tach 判断是否真的在转"}

    # ---------------- 下发 ----------------
    async def _apply(self, index: int, pct: int) -> dict:
        if self._control == "none":
            self._manual[index] = clamp_pct(pct)
            self._last_pct = self._manual[index]
            return {"ok": True, "virtual": True, "index": index, "pct": self._manual[index],
                    "reason": "没有真实控制器，只记了一个演示值"}
        try:
            if self._set:
                res = await self._set(index, clamp_pct(pct))
            else:
                res = await self._driver.set_pct(index, clamp_pct(pct))
        except FanUnavailable as exc:
            self.log.warning("风扇调速失败：%s", exc.message)
            return {"ok": False, "index": index, "pct": clamp_pct(pct),
                    "reason": exc.message, "hint": exc.hint}
        except Exception as exc:
            return {"ok": False, "index": index, "pct": clamp_pct(pct),
                    "reason": f"风扇调速出错：{exc}"}
        if res.get("ok"):
            self._last_pct = clamp_pct(pct)
        return res if isinstance(res, dict) else {"ok": bool(res), "pct": clamp_pct(pct)}

    async def _apply_all(self, pct: int) -> list:
        count = len(self.state()["fan"]["items"]) or 1
        return [await self._apply(i, pct) for i in range(count)]

    def _avail_note(self) -> dict:
        available, reason, hint = self._availability()
        return {} if available else {"available": False, "reason": reason, "hint": hint}

    def _save(self) -> None:
        try:
            self.save_store({"curve": [list(r) for r in self.curve], "mode": self.mode})
        except OSError as exc:
            self.log.warning("风扇配置没存下来：%s", exc)
