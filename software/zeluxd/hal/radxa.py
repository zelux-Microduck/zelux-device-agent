"""真机 HAL：把四块硬件拼起来，并对外保持与 `sim.SimHal` 完全一致的接口。

分工：
* `radxa_servo.RadxaServo` —— 15 颗总线舵机（/dev/ttyS2 @ 1 Mbps）
* `radxa_imu.RadxaIMU`     —— I²C 外部 IMU（QMI8658 / MPU6050 / ICM42688 自动探测）
* `radxa_tof.RadxaTof`     —— VL53L5CX 8×8（I²C-3 / 0x52）
* `radxa_fan.RadxaFan`     —— 双路 PWM + tach（经 root 助手 /run/zelux/fan.sock）

"拼"这件事本身有风险，所以：
1. 任何一块起不来都只降级那一块，不许拉着整机崩；
2. 电池没有专门的检测芯片时，用舵机回报的母线电压估算（会标注 `source:"servo_volt"` 和误差）；
3. 摄像头优先 ffmpeg(v4l2)，其次 libcamera-still；都没有就如实返回 False。
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from .base import JOINT_IDS, Hal
try:
    # 正式路径：驱动由独立仓库/包发布，Agent 不随驱动版本一起升级。
    from zelux_drivers import create as create_driver, descriptor as driver_descriptor
    from zelux_drivers.radxa import RadxaCamera, RadxaFan, RadxaIMU, RadxaServo, RadxaTof

    DRIVER_SOURCE = "zelux-device-drivers"
except ImportError:
    # 迁移期兼容：旧镜像尚未安装独立驱动包时继续可启动。
    from .radxa_fan import RadxaFan
    from .radxa_imu import RadxaIMU
    from .radxa_servo import RadxaServo
    from .radxa_tof import RadxaTof

    RadxaCamera = None
    create_driver = None
    driver_descriptor = None

    DRIVER_SOURCE = "builtin-fallback"

SERVO_MIN_V = 6.4     # 2S 锂电保护点
SERVO_MAX_V = 8.1     # 满电
DEFAULT_DRIVER_IDS = {
    "servo": "zelux.servo.feetech-sts-scs",
    "tof": "zelux.tof.st-vl53l5cx",
    "camera": "zelux.camera.v4l2-libcamera",
    "fan": "zelux.fan.radxa-zero3w-dual-pwm",
    "imu": "zelux.imu.multi-i2c",
}


class RadxaHal(Hal):
    backend = "radxa"

    def __init__(self, ctx, selection: dict[str, str] | None = None, factory=None) -> None:
        super().__init__(ctx)
        self._selection = dict(selection or {})
        self._factory = factory if factory is not None else create_driver
        self.driver_report: dict[str, dict] = {}
        self.servo = self._construct("servo")
        self.imu = self._construct("imu")
        self.tof = self._construct("tof")
        self.fan = self._construct("fan")
        self.camera = self._construct("camera")
        self._power = {"pct": None, "v": None, "source": "none"}
        self._temp: float | None = None
        self._motion_sock = Path("/run/zelux/motiond.sock")
        self._tasks: list[asyncio.Task] = []
        self._intent = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "ts": 0.0}

    def _args(self, kind: str) -> tuple[tuple, dict]:
        if kind == "tof":
            return (self.ctx,), {"bus": self.ctx.config.i2c_bus, "addr": self.ctx.config.tof_addr}
        if kind == "fan":
            channels = getattr(self.ctx.config, "fan_channels", None)
            return ((self.ctx,), {"fans": tuple(dict(item) for item in channels)}) if channels else ((self.ctx,), {})
        return (self.ctx,), {}

    def _builtin(self, kind: str):
        classes = {"servo": RadxaServo, "imu": RadxaIMU, "tof": RadxaTof,
                   "fan": RadxaFan, "camera": RadxaCamera}
        cls = classes[kind]
        if cls is None:
            return None
        args, kwargs = self._args(kind)
        return cls(*args, **kwargs)

    def _construct(self, kind: str, *, force_default: bool = False):
        wanted = DEFAULT_DRIVER_IDS[kind] if force_default else self._selection.get(kind, DEFAULT_DRIVER_IDS[kind])
        args, kwargs = self._args(kind)
        try:
            if self._factory is None:
                raise ImportError("独立驱动包未安装")
            part = self._factory(wanted, *args, **kwargs)
            version = driver_descriptor(wanted).version if driver_descriptor else None
            self.driver_report[kind] = {"requested": self._selection.get(kind, wanted), "loaded": wanted,
                                        "version": version,
                                        "fallback": force_default, "error": None}
            return part
        except Exception as exc:
            if wanted != DEFAULT_DRIVER_IDS[kind]:
                self.log.error("%s 驱动 %s 构造失败（%s），回退默认驱动", kind, wanted, exc)
                part = self._construct(kind, force_default=True)
                self.driver_report[kind].update({"requested": wanted, "fallback": True,
                                                 "error": f"构造失败：{exc}"})
                return part
            try:
                part = self._builtin(kind)
                self.driver_report[kind] = {"requested": self._selection.get(kind, wanted),
                                            "loaded": wanted if part is not None else None,
                                            "fallback": True, "error": f"独立包加载失败：{exc}"}
                return part
            except Exception as fallback_exc:
                self.driver_report[kind] = {"requested": self._selection.get(kind, wanted), "loaded": None,
                                            "fallback": True, "error": f"默认驱动构造失败：{fallback_exc}"}
                return None

    # ---------------- 生命周期 ----------------
    async def start(self) -> None:
        parts = [("servo", "舵机"), ("imu", "IMU"), ("tof", "ToF"),
                 ("fan", "风扇"), ("camera", "摄像头")]
        for kind, name in parts:
            part = getattr(self, kind)
            if part is None:
                continue
            try:
                await part.start()
                self.log.info("%s 已启动", name)
            except Exception as exc:
                report = self.driver_report[kind]
                if report.get("loaded") != DEFAULT_DRIVER_IDS[kind]:
                    self.log.error("%s 的所选驱动起不来（%s），尝试默认驱动", name, exc)
                    try:
                        await part.stop()
                    except Exception:
                        pass
                    fallback = self._construct(kind, force_default=True)
                    setattr(self, kind, fallback)
                    try:
                        await fallback.start()
                        self.driver_report[kind].update({"requested": report.get("requested"), "fallback": True,
                                                         "error": f"启动失败：{exc}"})
                        continue
                    except Exception as fallback_exc:
                        self.driver_report[kind]["error"] = f"所选驱动：{exc}；默认驱动：{fallback_exc}"
                if not self.driver_report[kind].get("error"):
                    self.driver_report[kind]["error"] = f"启动失败：{exc}"
                self.log.error("%s 起不来（%s），这一块会显示不可用", name, exc)
        self._tasks = [asyncio.create_task(self._poll_power()), asyncio.create_task(self._poll_temp())]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for part in (self.servo, self.imu, self.tof, self.fan, self.camera):
            if part is None:
                continue
            try:
                await part.stop()
            except Exception:
                pass

    def info(self) -> dict:
        return {"backend": "radxa", "driver_source": DRIVER_SOURCE, "drivers": self.driver_report,
                "servo": self.servo.info() if self.servo else {"available": False},
                "imu": self.imu.info() if self.imu else {"available": False},
                "tof": self.tof.info() if self.tof else {"available": False},
                "fan": self.fan.info() if self.fan else {"available": False},
                "camera": self.camera.info() if self.camera else {"available": False, "driver": "builtin"},
                "motiond": self._motion_sock.exists()}

    # ---------------- 舵机 ----------------
    async def servo_take(self) -> dict:
        return await self.servo.take()

    async def servo_release(self) -> dict:
        return await self.servo.release()

    async def servo_read(self, sid: int) -> dict:
        return await self.servo.servo_read(sid)

    async def servo_write(self, sid: int, reg: int, value: int, eeprom: bool = False) -> None:
        return await self.servo.servo_write(sid, reg, value, eeprom)

    async def servo_torque(self, on: bool, ids: list[int] | None = None) -> None:
        return await self.servo.servo_torque(on, ids)

    async def servo_sync_goal(self, goals: dict[int, int]) -> None:
        return await self.servo.servo_sync_goal(goals)

    def snapshot_servos(self) -> dict:
        return self.servo.snapshot_servos()

    # ---------------- 传感器 ----------------
    def snapshot_imu(self) -> dict:
        return self.imu.snapshot_imu()

    def snapshot_tof(self) -> dict:
        return self.tof.snapshot_tof()

    def snapshot_fan(self) -> dict:
        return self.fan.snapshot_fan()

    def snapshot_temp(self) -> float | None:
        return self._temp

    def snapshot_power(self) -> dict:
        return self._power

    async def _poll_power(self) -> None:
        """优先读电池管理芯片；没有就用 15 颗舵机回报的母线电压平均（这是最接近真实的近似）。"""
        while True:
            try:
                cap = self._battery_capacity()
                if cap is not None:
                    self._power = {"pct": cap, "v": None, "source": "fuel_gauge"}
                else:
                    volts = [it.get("volt") for it in (self.snapshot_servos().get("items") or [])
                             if it.get("volt")]
                    if volts:
                        v = sum(volts) / len(volts)
                        pct = max(0.0, min(100.0, (v - SERVO_MIN_V) / (SERVO_MAX_V - SERVO_MIN_V) * 100))
                        self._power = {"pct": round(pct, 0), "v": round(v, 2),
                                       "source": "servo_volt", "note": "查表估算，误差约 ±15%"}
            except Exception:
                pass
            await asyncio.sleep(3)

    @staticmethod
    def _battery_capacity() -> float | None:
        for p in Path("/sys/class/power_supply").glob("*"):
            cap = p / "capacity"
            if cap.exists():
                try:
                    return float(cap.read_text().strip())
                except Exception:
                    continue
        return None

    async def _poll_temp(self) -> None:
        while True:
            best = None
            for z in Path("/sys/class/thermal").glob("thermal_zone*"):
                try:
                    val = int((z / "temp").read_text().strip()) / 1000.0
                except Exception:
                    continue
                if 0 < val < 150 and (best is None or val > best):
                    best = val
            self._temp = round(best, 1) if best else None
            await asyncio.sleep(2)

    # ---------------- 运动意图 ----------------
    async def move(self, vx: float, vy: float, wz: float) -> None:
        """把速度意图交给实时层。没有 motiond 时只记录，不假装动了。"""
        self._intent = {"vx": vx, "vy": vy, "wz": wz, "ts": time.time()}
        await self._to_motiond({"t": "move", "vx": vx, "vy": vy, "wz": wz})

    async def pose(self, name: str) -> None:
        if await self._to_motiond({"t": "pose", "name": name}):
            return
        # 没有实时层：直接写舵机（仅调试用，没有平衡控制）
        rad = {
            "zero": [0] * 15,
            "stand": [0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491, 0, 0, 0, 0, 0.0873, 0.4579, 0.0049, -0.453],
            "sit": [0, 0, -0.5236, 1.0472, 0, 0.5, 1.6, 0, 0, 0, 0, 0, 0.5236, -1.0472, 0],
            "fold": [0, 0, 1.57, 1.57, 0, 1, 1, 0, 0, 0, 0, 0, -1.57, -1.57, 0],
        }.get(name)
        if not rad:
            return
        goals = {sid: int(2048 + r * 4096 / 6.283185) for sid, r in zip(JOINT_IDS, rad)}
        await self.servo.servo_torque(True)
        await self.servo.servo_sync_goal(goals)
        self.log.warning("没有 motiond：姿势 %s 是直接写舵机摆出来的，没有平衡控制", name)

    async def relax(self) -> dict:
        await self._to_motiond({"t": "relax"})
        await self.servo.servo_torque(False)
        return {"ok": True, "released": len(JOINT_IDS)}

    async def _to_motiond(self, payload: dict) -> bool:
        if not self._motion_sock.exists():
            return False
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self._motion_sock)), timeout=0.3)
            writer.write((json.dumps(payload) + "\n").encode())
            await writer.drain()
            writer.close()
            return True
        except Exception:
            return False

    # ---------------- 视觉 ----------------
    async def camera_snapshot(self, path: str) -> bool:
        if self.camera is not None:
            return await self.camera.snapshot(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dev = self._video_device()
        cmds = []
        if dev and shutil.which("ffmpeg"):
            cmds.append(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2",
                         "-i", dev, "-frames:v", "1", "-y", path])
        if shutil.which("libcamera-still"):
            cmds.append(["libcamera-still", "-n", "-t", "1", "--width", "1280", "--height", "720",
                         "-o", path.replace(".png", ".jpg")])
        for cmd in cmds:
            try:
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.DEVNULL,
                                                            stderr=subprocess.PIPE)
                await asyncio.wait_for(proc.wait(), timeout=8)
                target = Path(cmd[-2] if cmd[-2] != "-y" else cmd[-1])
                if proc.returncode == 0 and (target.exists() or Path(path).exists()):
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _video_device() -> str | None:
        for p in sorted(Path("/dev").glob("video*")):
            return str(p)
        return None

    def camera_stream_url(self) -> str | None:
        """摄像头实时流由 media 侧（WebRTC/go2rtc）提供；没起就返回 None，前端显示占位。"""
        if self.camera is not None:
            return self.camera.stream_url()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.15)
        try:
            if sock.connect_ex(("127.0.0.1", 8443)) == 0:
                return f"/camera/stream"
        except Exception:
            pass
        finally:
            sock.close()
        return None

    async def speak(self, text: str) -> bool:
        """最小可用 TTS：有 piper/espeak 就本地合成。语音主链路在 modules/ai.py，这里只兜底。"""
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            exe = shutil.which("espeak-ng") or shutil.which("espeak")
            try:
                proc = await asyncio.create_subprocess_exec(exe, "-v", "zh", text,
                                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                await asyncio.wait_for(proc.wait(), timeout=10)
                return proc.returncode == 0
            except Exception:
                return False
        return False
