"""HAL 工厂：同一份代码在板子上（radxa）和本机（sim）跑不同后端。"""
from __future__ import annotations

import json
from pathlib import Path

from .base import JOINT_IDS, JOINT_NAMES, Hal

_DRIVER_ID_ALIASES = {
    "zelux.servo.feetech-sts3215": "zelux.servo.feetech-sts-scs",
    "zelux.tof.vl53l5cx": "zelux.tof.st-vl53l5cx",
    "zelux.camera.v4l2": "zelux.camera.v4l2-libcamera",
    "zelux.fan.radxa-pwm": "zelux.fan.radxa-zero3w-dual-pwm",
    "zelux.imu.qmi8658-compatible": "zelux.imu.multi-i2c",
}


def load_driver_selection(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(Path(path).read_text("utf-8"))
        if isinstance(raw, dict):
            return {str(key): _DRIVER_ID_ALIASES.get(str(value), str(value)) for key, value in raw.items()
                    if key in {"servo", "tof", "camera", "fan", "imu"} and value}
    except Exception:
        pass
    return {}


def create_hal(ctx, selection: dict[str, str] | None = None) -> Hal:
    if ctx.config.mode == "radxa":
        try:
            from .radxa import RadxaHal  # 真机实现（延迟导入，缺依赖时降级）

            return RadxaHal(ctx, selection=selection)
        except Exception as exc:  # 板子上缺依赖不至于把网页也弄挂
            ctx.log.error("真机 HAL 起不来（%s），降级为模拟", exc)
    from .sim import SimHal

    return SimHal(ctx)


__all__ = ["Hal", "JOINT_IDS", "JOINT_NAMES", "create_hal", "load_driver_selection"]
