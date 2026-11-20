"""风扇 Studio API：实例清单与风扇控制都由 root fanctl 服务提供。

为什么页面上要同时有"自动曲线"和"手动百分比"：
调机时人手要知道"现在到底转多少"，而跑起来以后没人想看第二眼。所以默认 auto
（按 `[[58,70],[64,85],[70,100]]` 阶梯走，58℃ 以下 60% 兜底），
手动一旦点了就锁在手动档，重启回到 auto——一台会自己改回自动的风扇才不会烧主板。

没有风扇的机器（模拟模式、或者镜像没起 zelux-fand）不许让页面崩：
一律返回 `available:false` + 一句能行动的原因；这种情况下手动调速只写一个"演示值"
并在 state 里标 `virtual:true`，让人一眼看出这不是真的在转。
"""
from __future__ import annotations

from ..errors import ZeluxError
from ..hal.radxa_fan import DEFAULT_PCT, FanUnavailable, clamp_pct, normalize_curve
from .base import BaseModule

try:
    from zelux_drivers.radxa.fan import FanUnavailable as PackageFanUnavailable
except ImportError:
    PackageFanUnavailable = FanUnavailable
FAN_ERRORS = (FanUnavailable, PackageFanUnavailable)

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
        self._control = "service" if self._driver else ("hal" if self._set else "none")
        self._last_pct: int | None = None

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
            "mode": snap.get("mode", self.mode) if self._driver else self.mode,
            "curve": snap.get("curve", self.curve) if self._driver else [list(r) for r in self.curve],
            "temp": temp if temp is not None else snap.get("temp"),
            "items": items,
            "available": available,
            "reason": "" if available else reason,
            "hint": "" if available else hint,
            "virtual": virtual,
            "source": getattr(self._driver, "source", None) or ("hal" if self._control == "hal" else "none"),
            "pct": self._last_pct if self._last_pct is not None else snap.get("pct", DEFAULT_PCT),
            "pin_config": {"instances": [dict(item) for item in
                                           ((snap.get("inventory") or {}).get("instances") or [])],
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
        if self._driver is not None and hasattr(self._driver, "start"):
            await self._driver.start()
            snap = self.state()["fan"]
            self.mode = snap.get("mode", self.mode)
            self.curve = normalize_curve(snap.get("curve"))

    async def stop(self) -> None:
        await super().stop()
        if self._driver is not None and hasattr(self._driver, "stop"):
            await self._driver.stop()
        # Fan Agent has no PWM or temperature loop; zelux-fan owns both.

    # ---------------- 命令 ----------------
    async def cmd(self, op: str, args: dict) -> dict:
        try:
            return await self._cmd(op, args)
        except FAN_ERRORS as exc:
            raise ZeluxError(exc.message, getattr(exc, "code", "hardware")) from None

    async def _cmd(self, op: str, args: dict) -> dict:
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
            return await self._set_curve(args)
        if op in ("pin_config", "pins"):
            return await self._set_pin_config(args)
        if op in ("pwm_readiness", "hardware_readiness"):
            return await self._pwm_readiness()
        if op in ("pwm_setup_status", "pwm_setup_preview", "pwm_setup_apply", "pwm_setup_rollback"):
            return await self._pwm_setup(op, args)
        if op == "instance_create":
            return await self._instance_create(args)
        if op == "instance_update":
            return await self._instance_update(args)
        if op == "instance_delete":
            return await self._instance_delete(args)
        if op in ("instance_enable", "instance_disable"):
            driver = self._require_instance_driver("set_enabled")
            identifier = args.get("instance_id")
            if not isinstance(identifier, str) or not identifier:
                raise ZeluxError("instance_id 参数不能为空", "denied")
            if op == "instance_enable":
                try:
                    snapshot = await driver.refresh()
                except FAN_ERRORS as exc:
                    raise ZeluxError(exc.message, getattr(exc, "code", "hardware")) from None
                inventory = snapshot.get("inventory", {}) if isinstance(snapshot, dict) else {}
                existing = next((item for item in inventory.get("instances", [])
                                 if item.get("id") == identifier), None)
                if not existing:
                    raise ZeluxError("找不到待启用的风扇实例，请刷新状态", "not_found")
                await self._assert_pwm_mapping_ready(existing)
            try:
                result = await driver.set_enabled(identifier, op == "instance_enable")
            except FAN_ERRORS as exc:
                raise ZeluxError(exc.message, getattr(exc, "code", "hardware")) from None
            self._raise_on_driver_failure(result, "修改风扇实例状态失败")
            return {"ok": True, "result": result, "fan": self.state()["fan"]}
        if op in ("self_test", "selftest", "test"):
            return await self._self_test(args)
        if op == "read":
            return await self._read_one(args)
        if op in ("status", ""):
            return {"ok": True, "fan": self.state()["fan"]}
        raise ZeluxError(f"fan 不支持 {op}（可用：auto/manual/curve/read/self_test/status）",
                         "unsupported")

    async def _set_pin_config(self, args: dict) -> dict:
        """Deprecated: use instance_create/update/delete instead."""
        raise ZeluxError(
            "pin_config 已废弃，请使用 instance_create/instance_update/instance_delete",
            "unsupported"
        )

    def _require_instance_driver(self, method: str | None = None):
        """Return driver or raise ZeluxError if unavailable or method not callable.

        Args:
            method: Optional method name to check for callability (e.g., 'create_instance')

        Returns:
            The driver object if available and method is callable (if specified)

        Raises:
            ZeluxError: If driver is None or method is not callable
        """
        if self._driver is None:
            raise ZeluxError("当前没有可管理风扇实例的 fanctl 服务", "unavailable")
        if method is not None:
            if not hasattr(self._driver, method):
                raise ZeluxError(
                    f"当前 fanctl 驱动不支持 {method} 操作",
                    "unsupported"
                )
            if not callable(getattr(self._driver, method)):
                raise ZeluxError(
                    f"当前 fanctl 驱动的 {method} 不可调用",
                    "unsupported"
                )
        return self._driver

    async def _pwm_readiness(self) -> dict:
        driver = self._require_instance_driver("pwm_readiness")
        try:
            result = await driver.pwm_readiness()
        except FAN_ERRORS as exc:
            raise ZeluxError(exc.message, getattr(exc, "code", "hardware")) from None
        return {"ok": True, **result}

    async def _pwm_setup(self, op: str, args: dict) -> dict:
        driver = self._require_instance_driver("pwm_setup")
        setup_op = op.removeprefix("pwm_setup_")
        pin_id = args.get("pin_id", "")
        token = args.get("token", "")
        if not isinstance(pin_id, str) or len(pin_id) > 64 or not isinstance(token, str) or len(token) > 128:
            raise ZeluxError("PWM 配置请求字段无效", "denied")
        if setup_op in ("preview", "apply", "rollback") and not pin_id:
            raise ZeluxError("必须指定物理 PWM 引脚档案", "denied")
        if setup_op in ("apply", "rollback") and args.get("confirm") is not True:
            raise ZeluxError("该操作必须由用户明确确认", "denied")
        try:
            return await driver.pwm_setup(setup_op, pin_id=pin_id, token=token,
                                          confirm=args.get("confirm") is True)
        except FAN_ERRORS as exc:
            raise ZeluxError(exc.message, getattr(exc, "code", "hardware")) from None

    async def _instance_create(self, args: dict) -> dict:
        driver = self._require_instance_driver("create_instance")
        self._require_instance_driver("pwm_setup")
        instance = args.get("instance")
        if not isinstance(instance, dict):
            raise ZeluxError("instance 参数必须是字典", "denied")
        if not instance:
            raise ZeluxError("instance 参数不能为空", "denied")
        instance_id = instance.get("id")
        if not instance_id:
            raise ZeluxError("instance 必须包含有效的 id", "denied")
        pin_id = instance.get("pwm_pin")
        mapping = instance.get("pwm")
        if not isinstance(pin_id, str) or not pin_id or not isinstance(mapping, dict):
            raise ZeluxError("必须选择已验证就绪的物理 PWM 引脚", "denied")
        await self._assert_pwm_mapping_ready(instance)
        try:
            result = await driver.create_instance(instance)
        except FAN_ERRORS as exc:
            raise ZeluxError(exc.message, getattr(exc, "code", "hardware")) from None
        self._raise_on_driver_failure(result, "创建风扇实例失败")
        return {"ok": True, "result": result, "fan": self.state()["fan"]}

    async def _assert_pwm_mapping_ready(self, instance: dict) -> None:
        driver = self._require_instance_driver("pwm_setup")
        pin_id, mapping = instance.get("pwm_pin"), instance.get("pwm")
        if not isinstance(pin_id, str) or not pin_id or not isinstance(mapping, dict):
            raise ZeluxError("必须选择已验证就绪的物理 PWM 引脚", "denied")
        try:
            readiness = await driver.pwm_setup("status")
        except FAN_ERRORS as exc:
            raise ZeluxError(exc.message, getattr(exc, "code", "hardware")) from None
        profile = readiness.get("pwm_setup", {})
        pin = next((item for item in profile.get("mapped_pins", [])
                    if item.get("id") == pin_id), None)
        if not pin or pin.get("ready") is not True:
            raise ZeluxError("所选 PWM 物理引脚尚未通过设备端运行状态验证；请先配置 overlay、重启并刷新状态", "not_ready")
        if mapping.get("chip") != pin.get("chip") or mapping.get("channel") != pin.get("channel"):
            raise ZeluxError("PWM 引脚映射与设备端硬件档案不一致，请刷新引脚状态后重新选择", "mapping_mismatch")

    async def _instance_update(self, args: dict) -> dict:
        driver = self._require_instance_driver("update_instance")
        instance_id = args.get("instance_id")
        if not instance_id:
            raise ZeluxError("instance_id 参数不能为空", "denied")
        instance = args.get("instance")
        if not isinstance(instance, dict):
            raise ZeluxError("instance 参数必须是字典", "denied")
        if not instance:
            raise ZeluxError("instance 参数不能为空", "denied")
        instance_body_id = instance.get("id")
        if not instance_body_id:
            raise ZeluxError("instance 必须包含 id 字段", "denied")
        if str(instance_body_id) != str(instance_id):
            raise ZeluxError(
                f"instance_id({instance_id}) 与 instance.id({instance_body_id}) 不匹配",
                "denied"
            )
        if instance.get("enabled") is not False:
            await self._assert_pwm_mapping_ready(instance)
        try:
            result = await driver.update_instance(str(instance_id), instance)
        except FAN_ERRORS as exc:
            raise ZeluxError(exc.message, getattr(exc, "code", "hardware")) from None
        self._raise_on_driver_failure(result, "更新风扇实例失败")
        return {"ok": True, "result": result, "fan": self.state()["fan"]}

    async def _instance_delete(self, args: dict) -> dict:
        driver = self._require_instance_driver("delete_instance")
        instance_id = args.get("instance_id")
        if not instance_id:
            raise ZeluxError("instance_id 参数不能为空", "denied")
        if args.get("confirm") is not True:
            raise ZeluxError("删除风扇会停止并释放该实例，请先确认", "denied")
        try:
            result = await driver.delete_instance(str(instance_id))
        except FAN_ERRORS as exc:
            raise ZeluxError(exc.message, getattr(exc, "code", "hardware")) from None
        self._raise_on_driver_failure(result, "删除风扇实例失败")
        return {"ok": True, "result": result, "fan": self.state()["fan"]}

    @staticmethod
    def _raise_on_driver_failure(result: dict, fallback: str) -> None:
        if isinstance(result, dict) and result.get("ok") is False:
            raise ZeluxError(str(result.get("error") or result.get("reason") or fallback),
                             str(result.get("code") or "hardware"))

    async def _set_mode(self, mode: str, args: dict) -> dict:
        if mode == "auto":
            if self._driver is not None and hasattr(self._driver, "set_auto"):
                try:
                    result = await self._driver.set_auto()
                except FAN_ERRORS as exc:
                    raise ZeluxError(exc.message, getattr(exc, "code", "hardware")) from None
                self.mode = "auto"
                return {"ok": True, **result, "fan": self.state()["fan"]}
            self.mode = "auto"
            return {"ok": True, "mode": "auto", "virtual": self._control == "none"}

        target = args.get("instance_id", args.get("index", 0))
        if args.get("pct") is None:
            raise ZeluxError("手动档要给 pct（0–100）", "denied")
        pct = clamp_pct(args.get("pct"))
        self.mode = "auto" if self._driver else "manual"
        result = await self._apply(target, pct)
        if result.get("ok"):
            self.log_event(f"风扇 {target} 手动 {pct}%（{result.get('source', '?')}）")
        if not result.get("ok"):
            raise ZeluxError(result.get("reason") or "风扇调速失败", result.get("code", "hardware"))
        return {"ok": True, "mode": self.mode, "instance_mode": "manual", "instance_id": target, "pct": pct,
                "detail": result, **self._avail_note()}

    async def _set_curve(self, args: dict) -> dict:
        self.curve = normalize_curve(args.get("curve"))
        if self._driver is not None and hasattr(self._driver, "set_curve"):
            try:
                result = await self._driver.set_curve(self.curve)
            except FAN_ERRORS as exc:
                raise ZeluxError(exc.message, getattr(exc, "code", "hardware")) from None
        else:
            result = {"ok": self._control == "none", "virtual": True}
        self.log_event("温控曲线已更新：" + "、".join(f"{t}℃→{p}%" for t, p in self.curve))
        return {**result, "curve": [list(r) for r in self.curve], "mode": self.mode}

    async def _read_one(self, args: dict) -> dict:
        index = args.get("instance_id", args.get("index", 0))
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
                res = await self._driver.self_test(args.get("instance_id", args.get("index")))
            except FAN_ERRORS as exc:
                raise ZeluxError(exc.message, "denied") from None
            self.log_event("风扇自检：" + ("通过" if res.get("ok") else "有问题，看详情"))
            return res
        return {"ok": None, "available": False,
                "reason": "这台机器没法自检（没有 RadxaFan 驱动）",
                "hint": "真机上启用 zelux-fand 后，自检会全速 1 秒读 tach 判断是否真的在转"}

    # ---------------- 下发 ----------------
    async def _apply(self, index: Any, pct: int) -> dict:
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
        except FAN_ERRORS as exc:
            self.log.warning("风扇调速失败：%s", exc.message)
            return {"ok": False, "index": index, "pct": clamp_pct(pct),
                    "reason": exc.message, "hint": exc.hint, "code": getattr(exc, "code", "hardware")}
        except Exception as exc:
            return {"ok": False, "index": index, "pct": clamp_pct(pct),
                    "reason": f"风扇调速出错：{exc}"}
        if res.get("ok"):
            self._last_pct = clamp_pct(pct)
        return res if isinstance(res, dict) else {"ok": bool(res), "pct": clamp_pct(pct)}

    async def _apply_all(self, pct: int) -> list:
        items = self.state()["fan"]["items"]
        return [await self._apply(item.get("id", i), pct) for i, item in enumerate(items)]

    def _avail_note(self) -> dict:
        available, reason, hint = self._availability()
        return {} if available else {"available": False, "reason": reason, "hint": hint}
