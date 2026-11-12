"""外设固件包管理；当前只验证、暂存和执行安全前置检查。"""
from __future__ import annotations

from ..errors import ZeluxError
from ..firmware import FirmwareError
from .base import BaseModule


class Module(BaseModule):
    name = "firmware"

    def state(self) -> dict:
        manager = getattr(self.ctx.hub, "firmware_manager", None)
        return {"firmware": manager.state() if manager else {"write_enabled": False}}

    async def cmd(self, op: str, args: dict) -> dict:
        manager = getattr(self.ctx.hub, "firmware_manager", None)
        if not manager:
            raise ZeluxError("固件管理器还没启动", "unavailable")
        try:
            if op in ("list", "status"):
                return manager.state()
            if op == "check_manifest":
                return {"ok": True, "manifest": manager.check_manifest(args.get("manifest"))}
            if op == "install":
                return manager.install_package(str(args.get("filename", "")))
            if op == "precheck":
                result = manager.precheck(str(args.get("firmware_id", "")),
                                          str(args.get("version", "")))
                if result["device_class"] == "servo":
                    snapshot = self.hal.snapshot_servos()
                    torque_off = not any(item.get("torque") for item in snapshot.get("items", []))
                    safe_posture = self.ctx.hub.posture in ("holding", "sitting")
                    result["checks"].update({"servo_torque_off": torque_off,
                                             "safe_posture": safe_posture,
                                             "estop_not_pending": self.ctx.hub.estop.get("state") != "pending"})
                    result["ready"] = all(result["checks"].values())
                return result
            if op in ("apply", "rollback"):
                raise ZeluxError("真实固件刷写尚未启用", "unsupported")
        except FirmwareError as exc:
            raise ZeluxError(str(exc), "denied") from exc
        raise ZeluxError(f"firmware 不支持 {op}", "unsupported")

