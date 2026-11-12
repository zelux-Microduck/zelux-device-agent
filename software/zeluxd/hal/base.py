"""硬件抽象层。

原则：
1. 所有模块只能通过这里访问硬件，绝不自己开串口 / 开 i2c。
2. **读**走缓存（`snapshot_*`，永远不阻塞，10 Hz 状态总线在用）；
   **写**是协程（`servo_write` 等，可以等，可以失败）。
3. 缓存由 HAL 自己的后台任务刷新；模块不要在里面做 I/O。
"""
from __future__ import annotations

import abc
from typing import Any

from ..errors import hardware

# 15 颗舵机的总线 ID（与官方 microduck 一致：右腿 10–14、左腿 20–24、头颈 30–34）
JOINT_IDS = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34]
JOINT_NAMES = [
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ankle",
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ankle",
    "neck_pan", "neck_tilt", "head_pan", "head_tilt", "beak",
]


class Hal(abc.ABC):
    """所有硬件的唯一入口。子类必须实现抽象方法。"""

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.log = ctx.log

    # ---------- 生命周期 ----------
    async def start(self) -> None:  # noqa: D401
        """起后台采集任务。失败不抛，降级并写日志。"""

    async def stop(self) -> None:
        """停采集任务、关总线、卸力。"""

    # ---------- 舵机 ----------
    async def servo_take(self) -> None:
        raise hardware("这台机器没有舵机总线")

    async def servo_release(self) -> None:
        raise hardware("这台机器没有舵机总线")

    async def servo_read(self, sid: int) -> dict:
        raise hardware("这台机器没有舵机总线")

    async def servo_write(self, sid: int, reg: int, value: int, eeprom: bool = False) -> None:
        raise hardware("这台机器没有舵机总线")

    async def servo_torque(self, on: bool, ids: list[int] | None = None) -> None:
        raise hardware("这台机器没有舵机总线")

    async def servo_sync_goal(self, goals: dict[int, int]) -> None:
        raise hardware("这台机器没有舵机总线")

    # ---------- 传感器（读缓存） ----------
    def snapshot_servos(self) -> dict:
        return {"online": 0, "items": [], "bus": "none"}

    def snapshot_imu(self) -> dict:
        return {"ok": False, "reason": "没有 IMU"}

    def snapshot_tof(self) -> dict:
        return {"state": "unavailable", "reason": "没有 ToF", "zones": []}

    def snapshot_fan(self) -> dict:
        return {"mode": "auto", "pct": 0, "items": []}

    def snapshot_power(self) -> dict:
        return {"pct": None, "v": None}

    def snapshot_temp(self) -> float | None:
        return None

    # ---------- 视觉 / 音频 ----------
    async def camera_snapshot(self, path: str) -> bool:
        return False

    def camera_stream_url(self) -> str | None:
        return None

    async def speak(self, text: str) -> bool:
        return False

    # ---------- 运动意图（实时层接口） ----------
    async def move(self, vx: float, vy: float, wz: float) -> None:
        """速度意图，m/s 与 rad/s。没有实时层时直接返回。"""

    async def pose(self, name: str) -> None:
        """姿势意图：zero/stand/sit/fold。"""

    async def relax(self) -> dict:
        """急停：全部卸力。必须尽量幂等、尽量快。"""
        await self.servo_torque(False)
        return {"ok": True}

    # ---------- 工具 ----------
    @staticmethod
    def clamp(v: float, lo: float, hi: float) -> float:
        return lo if v < lo else hi if v > hi else v

    def info(self) -> dict[str, Any]:
        return {"backend": type(self).__name__}
