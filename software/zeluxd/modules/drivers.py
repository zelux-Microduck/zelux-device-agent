"""驱动清单模块：第一阶段提供只读查询和探测。"""
from __future__ import annotations

from ..drivers import ManifestError
from ..errors import ZeluxError
from .base import BaseModule


class Module(BaseModule):
    name = "drivers"

    def state(self) -> dict:
        registry = getattr(self.ctx.hub, "driver_registry", None)
        return {"drivers": registry.state() if registry else {"items": []}}

    async def cmd(self, op: str, args: dict) -> dict:
        registry = getattr(self.ctx.hub, "driver_registry", None)
        if not registry:
            raise ZeluxError("驱动注册表还没启动", "unavailable")
        if op in ("list", "status"):
            return registry.state()
        if op == "get":
            item = registry.get(str(args.get("device_class", "")))
            if not item:
                raise ZeluxError("没有这个设备类型", "unsupported")
            return item
        if op == "probe":
            return {"items": registry.probe(args.get("device_class"))}
        if op == "self_test":
            try:
                return registry.self_test(str(args.get("device_class", "")))
            except ManifestError as exc:
                raise ZeluxError(str(exc), "denied") from exc
        if op == "check_manifest":
            try:
                return {"ok": True, "manifest": registry.check_manifest(args.get("manifest"))}
            except ManifestError as exc:
                raise ZeluxError(str(exc), "denied") from exc
        if op == "select":
            try:
                return registry.select(str(args.get("device_class", "")),
                                       str(args.get("driver_id", "")))
            except ManifestError as exc:
                raise ZeluxError(str(exc), "denied") from exc
        if op == "install":
            try:
                return registry.install_package(str(args.get("filename", "")))
            except ManifestError as exc:
                raise ZeluxError(str(exc), "denied") from exc
        if op in ("rollback", "firmware"):
            raise ZeluxError("驱动回滚和固件更新尚未启用", "unsupported")
        raise ZeluxError(f"drivers 不支持 {op}", "unsupported")
