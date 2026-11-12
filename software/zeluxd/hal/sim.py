"""模拟 HAL：没有硬件也能把整套页面跑起来（开发、演示、评审看效果）。

它模拟的是"行为"而不是"数据"：舵机会按速度限幅往目标走、风扇按温度阶梯调速、
ToF 会有一个障碍物周期性靠近、IMU 有噪声和温漂。所以前端的动画、告警、避障逻辑
在模拟器里就能验证，不用等板子。
"""
from __future__ import annotations

import asyncio
import math
import random
import struct
import time
import zlib
from pathlib import Path

from .base import JOINT_IDS, Hal

CENTER = 2048


class SimHal(Hal):
    backend = "sim"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        rnd = random.Random(7)
        self.pos = {sid: float(CENTER) for sid in JOINT_IDS}
        self.goal = {sid: float(CENTER) for sid in JOINT_IDS}
        self.torque = {sid: False for sid in JOINT_IDS}
        self.temp = {sid: 32.0 + rnd.random() * 4 for sid in JOINT_IDS}
        self.volt = {sid: 7.9 for sid in JOINT_IDS}
        self.bus_taken = False
        self._cache_servos = {"online": 0, "items": [], "bus": "sim"}
        self._cache_imu = {"ok": True, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                           "chip": "sim-qmi8658", "calibrated": False}
        self._cache_tof = {"state": "ready", "sensor": "sim-vl53l5cx", "rows": 8, "cols": 8,
                           "hz": 15, "distance_mm": [], "status": [], "nearest_mm": None,
                           "valid_count": 0, "seq": 0, "age_ms": 0}
        self._cache_fan = {"mode": "auto", "pct": 60, "temp": 48.0, "curve": [[58, 70], [64, 85], [70, 100]],
                           "items": [{"name": "顶部风扇", "pwm": 14, "gpio": 116, "pct": 60, "rpm": None},
                                     {"name": "主板散热", "pwm": 12, "gpio": 149, "pct": 60, "rpm": 2100,
                                      "tach_hz": 70.0}]}
        self._cache_power = {"pct": 78.0, "v": 7.85}
        self._t0 = time.time()
        self._tasks: list[asyncio.Task] = []
        self._tilt = {"roll": 0.0, "pitch": 0.0}
        self._fan_manual: dict[int, int | None] = {0: None, 1: None}
        self._temp = 47.0
        self._seq = 0
        self._moved = False

    # ---------------- 生命周期 ----------------
    async def start(self) -> None:
        self._tasks = [asyncio.create_task(self._loop(0.02, self._step_servos)),
                       asyncio.create_task(self._loop(0.05, self._step_sensors)),
                       asyncio.create_task(self._loop(0.066, self._step_tof)),
                       asyncio.create_task(self._loop(1.0, self._step_slow))]
        self.log.info("模拟硬件已启动（%d 颗舵机、8×8 ToF、双风扇）", len(JOINT_IDS))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()

    async def _loop(self, period: float, fn) -> None:
        while True:
            try:
                await fn()
            except asyncio.CancelledError:
                raise
            except Exception:  # 模拟器不许把守护进程带崩
                self.log.exception("sim 循环出错")
            await asyncio.sleep(period)

    # ---------------- 舵机 ----------------
    async def servo_take(self) -> None:
        self.bus_taken = True
        self.log.info("模拟：已接管舵机总线")

    async def servo_release(self) -> None:
        self.bus_taken = False

    async def servo_read(self, sid: int) -> dict:
        return {"id": sid, "pos": int(self.pos[sid]), "goal": int(self.goal[sid]),
                "load": round(abs(self.goal[sid] - self.pos[sid]) * 0.4, 1),
                "volt": round(self.volt[sid], 2), "temp": round(self.temp[sid], 1),
                "torque": self.torque[sid], "offset": 0, "speed": 0}

    async def servo_write(self, sid: int, reg: int, value: int, eeprom: bool = False) -> None:
        if reg == 41:          # 目标位置
            self.goal[sid] = float(self.clamp(value, 0, 4095))
        elif reg == 40:        # 扭矩开关
            self.torque[sid] = bool(value)
        await asyncio.sleep(0.004)

    async def servo_torque(self, on: bool, ids: list[int] | None = None) -> None:
        for sid in (ids or JOINT_IDS):
            if sid in self.torque:
                self.torque[sid] = on
        if not on:
            for sid in self.pos:
                self.goal[sid] = self.pos[sid]
        await asyncio.sleep(0.05)

    async def servo_sync_goal(self, goals: dict[int, int]) -> None:
        for sid, val in goals.items():
            if sid in self.goal:
                self.goal[sid] = float(self.clamp(val, 0, 4095))

    async def pose(self, name: str) -> None:
        poses = {
            "zero": [0] * 15,
            "stand": [0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491, 0, 0, 0, 0, 0.0873, 0.4579, 0.0049, -0.453],
            "sit": [0, 0, -0.5236, 1.0472, 0, 0.5, 1.6, 0, 0, 0, 0, 0, 0.5236, -1.0472, 0],
            "fold": [0, 0, 1.57, 1.57, 0, 1, 1, 0, 0, 0, 0, 0, -1.57, -1.57, 0],
        }
        rad = poses.get(name)
        if not rad:
            return
        for sid, r in zip(JOINT_IDS, rad):
            self.goal[sid] = CENTER + r * 651.9   # 2π ↔ 4096
            self.torque[sid] = True

    async def relax(self) -> dict:
        await self.servo_torque(False)
        return {"ok": True, "released": len(JOINT_IDS)}

    async def move(self, vx: float, vy: float, wz: float) -> None:
        self._moved = abs(vx) + abs(vy) + abs(wz) > 1e-4

    # ---------------- 传感器 ----------------
    def snapshot_servos(self) -> dict:
        return self._cache_servos

    def snapshot_imu(self) -> dict:
        return self._cache_imu

    def snapshot_tof(self) -> dict:
        return self._cache_tof

    def snapshot_fan(self) -> dict:
        return self._cache_fan

    def snapshot_power(self) -> dict:
        return self._cache_power

    def snapshot_temp(self) -> float:
        return round(self._temp, 1)

    def info(self) -> dict:
        return {"backend": "sim", "note": "模拟硬件：没有真机也能看全部页面"}

    async def _step_servos(self) -> None:
        for sid, cur in self.pos.items():
            tgt = self.goal[sid]
            if self.torque[sid]:
                step = 120.0
                delta = tgt - cur
                cur += max(-step, min(step, delta))
            self.pos[sid] = cur
            moving = abs(tgt - cur) > 1
            self.temp[sid] += (0.6 if moving else -0.15) + random.uniform(-0.03, 0.03)
            self.temp[sid] = self.clamp(self.temp[sid], 28.0, 78.0)
        items = [{"id": sid, "pos": int(self.pos[sid]), "goal": int(self.goal[sid]),
                  "load": round(abs(self.goal[sid] - self.pos[sid]) * 0.35, 1),
                  "volt": round(self.volt[sid], 2), "temp": round(self.temp[sid], 1),
                  "torque": self.torque[sid]} for sid in JOINT_IDS]
        self._cache_servos = {"online": len(JOINT_IDS), "items": items,
                              "bus": "taken" if self.bus_taken else "shared",
                              "holding": sum(1 for v in self.torque.values() if v)}

    async def _step_sensors(self) -> None:
        t = time.time() - self._t0
        noise = lambda a: random.uniform(-a, a)
        roll = self._tilt["roll"] + math.sin(t * 1.7) * 0.6 + noise(0.12)
        pitch = self._tilt["pitch"] + math.cos(t * 1.3) * 0.8 + noise(0.12)
        yaw = (t * 1.2) % 360 - 180
        self._cache_imu = {"ok": True, "roll": round(roll, 2), "pitch": round(pitch, 2),
                           "yaw": round(yaw, 1), "chip": "sim-qmi8658",
                           "calibrated": self.ctx.state.get("imu_calibrated", False)}
        self._temp += (0.35 if (self._moved or self._fan_manual[0]) else -0.08) + random.uniform(-0.03, 0.03)
        self._temp = self.clamp(self._temp, 38.0, 82.0)
        pct, mode = self._fan_pct()
        self._cache_fan = {"mode": mode, "pct": pct, "temp": round(self._temp, 1),
                           "curve": [[58, 70], [64, 85], [70, 100]],
                           "items": [
                               {"name": "顶部风扇", "pwm": 14, "gpio": 116, "pct": pct, "rpm": None},
                               {"name": "主板散热", "pwm": 12, "gpio": 149, "pct": pct,
                                "rpm": int(pct * 34 + random.uniform(-30, 30)) if pct else 0,
                                "tach_hz": round(pct * 1.13, 1)}]}
        drain = 0.0009 if self._moved else 0.0004
        v = self._cache_power["v"] - drain
        if v < 6.4:
            v = 8.1
        self._cache_power = {"pct": round(max(0.0, (v - 6.4) / (8.1 - 6.4) * 100), 0), "v": round(v, 2)}

    def _fan_pct(self) -> tuple[int, str]:
        manual = self._fan_manual[0]
        if manual is not None:
            return manual, "manual"
        t = self._temp
        if t < 58:
            return 60, "auto"
        if t < 64:
            return 70, "auto"
        if t < 70:
            return 85, "auto"
        return 100, "auto"

    async def _step_tof(self) -> None:
        t = time.time() - self._t0
        # 障碍物：每 12 秒靠近一次，最近 180 mm
        phase = (t % 12) / 12
        near = 180 + 1400 * (1 - min(1.0, abs(phase - 0.15) * 3.2))
        dist, status = [], []
        self._seq += 1
        for r in range(8):
            for c in range(8):
                base = 1500 + 500 * math.sin((t * 0.3) + c * 0.4)
                d = near if (2 <= c <= 5 and 1 <= r <= 6) else base + random.uniform(-60, 60)
                if d > 3800:
                    dist.append(0)
                    status.append(255)
                else:
                    dist.append(int(d))
                    status.append(5)
        valid = [d for d, s in zip(dist, status) if s == 5]
        self._cache_tof = {"state": "ready", "sensor": "sim-vl53l5cx", "rows": 8, "cols": 8, "hz": 15,
                           "distance_mm": dist, "status": status, "seq": self._seq, "age_ms": 12,
                           "nearest_mm": min(valid) if valid else None,
                           "valid_count": len(valid),
                           "simulated": True}

    async def _step_slow(self) -> None:
        # 让"电快没电了"这种极端状态偶尔能被看到
        if self._cache_power["v"] and self._cache_power["v"] < 6.55:
            self.log.warning("模拟：电量低 %.2f V", self._cache_power["v"])

    # ---------------- 视觉 ----------------
    async def camera_snapshot(self, path: str) -> bool:
        """画一张带网格的 PNG，用来验证"拍照—存档—回放"整条链路。"""
        w, h = 640, 480
        px = bytearray()
        for y in range(h):
            px.append(0)  # 每行滤波器类型 0
            for x in range(w):
                g = int(18 + 26 * (x / w) + 18 * (y / h))
                if x % 80 == 0 or y % 80 == 0:
                    px += bytes((0, 216, 160))
                elif abs(x - w // 2) < 3 or abs(y - h // 2) < 3:
                    px += bytes((61, 123, 255))
                else:
                    px += bytes((g, g + 6, g + 10))

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (struct.pack(">I", len(data)) + tag + data
                    + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

        raw = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        raw += chunk(b"IDAT", zlib.compress(bytes(px), 6)) + chunk(b"IEND", b"")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(raw)
        return True

    def camera_stream_url(self) -> str | None:
        return None

    async def speak(self, text: str) -> bool:
        self.log.info("模拟 TTS：%s", text)
        await asyncio.sleep(0.2)
        return True

    # ---------------- 给标定向导用的仿真钩子 ----------------
    async def sim_hook(self, name: str, args: dict) -> dict:
        if name == "tilt":
            self._tilt["roll"] = float(args.get("roll", 0.0))
            self._tilt["pitch"] = float(args.get("pitch", 0.0))
            return {"ok": True, "tilt": dict(self._tilt)}
        if name == "flat":
            self._tilt = {"roll": 0.0, "pitch": 0.0}
            return {"ok": True}
        return {"ok": False, "error": "未知的仿真钩子"}
