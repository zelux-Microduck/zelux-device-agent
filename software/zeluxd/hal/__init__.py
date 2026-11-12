"""HAL 工厂：同一份代码在板子上（radxa）和本机（sim）跑不同后端。"""
from __future__ import annotations

from .base import JOINT_IDS, JOINT_NAMES, Hal


def create_hal(ctx) -> Hal:
    if ctx.config.mode == "radxa":
        try:
            from .radxa import RadxaHal  # 真机实现（延迟导入，缺依赖时降级）

            return RadxaHal(ctx)
        except Exception as exc:  # 板子上缺依赖不至于把网页也弄挂
            ctx.log.error("真机 HAL 起不来（%s），降级为模拟", exc)
    from .sim import SimHal

    return SimHal(ctx)


__all__ = ["Hal", "JOINT_IDS", "JOINT_NAMES", "create_hal"]
