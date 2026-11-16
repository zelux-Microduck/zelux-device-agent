"""全局配置。环境变量优先，便于同一个代码在板子和本机跑不同后端。"""
from __future__ import annotations

import os
import json
from pathlib import Path

VERSION = "1.0.0"
PROTOCOL = 1


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Config:
    def __init__(self) -> None:
        # sim = 本机模拟（无硬件）；radxa = 真机
        self.mode = os.environ.get("ZELUX_MODE", "sim").strip().lower()
        self.host = os.environ.get("ZELUX_HOST", "0.0.0.0")
        self.port = int(os.environ.get("ZELUX_PORT", "8000"))
        # 状态目录：放口令哈希、标定草稿、技能、模型、照片
        self.state_dir = Path(os.environ.get("ZELUX_STATE", "var")).resolve()
        # 前端目录：默认取仓库里的 software/web
        here = Path(__file__).resolve().parent
        default_web = here.parent / "web"
        self.web_dir = Path(os.environ.get("ZELUX_WEB", str(default_web))).resolve()
        # 技能库目录
        self.skills_dir = Path(os.environ.get("ZELUX_SKILLS", str(here.parent / "skills"))).resolve()
        self.models_dir = self.state_dir / "models"
        self.shots_dir = self.state_dir / "shots"
        self.drivers_dir = self.state_dir / "drivers"
        self.firmware_dir = self.state_dir / "firmware"
        self.hardware_path = self.state_dir / "hardware.json"
        # 初始口令（首次启动写入哈希；之后改口令走页面）
        self.initial_password = os.environ.get("ZELUX_PASSWORD", "zelux")
        self.allow_no_password = _flag("ZELUX_NO_PASSWORD", False)
        # 调试：跳过 Origin 校验（仅本机开发）
        self.dev = _flag("ZELUX_DEV", False)
        # 硬件参数
        self.servo_port = os.environ.get("ZELUX_SERVO_PORT", "/dev/ttyS2")
        self.servo_baud = int(os.environ.get("ZELUX_SERVO_BAUD", "1000000"))
        self.i2c_bus = int(os.environ.get("ZELUX_I2C_BUS", "3"))
        self.tof_addr = int(os.environ.get("ZELUX_TOF_ADDR", "0x52"), 16)
        self.fan_socket = os.environ.get("ZELUX_FAN_SOCK", "/run/zelux/fan.sock")
        self.robotctl = os.environ.get("ZELUX_ROBOTCTL", "/opt/robot/daemon/current/bin/robotctl")
        self.robotd_config = Path(os.environ.get("ZELUX_ROBOTD_CONFIG", "/etc/robot/robotd.toml"))
        self.robotd_socket = Path(os.environ.get("ZELUX_ROBOTD_SOCKET", "/run/robotd.sock"))
        self.robotd_imu_socket = os.environ.get(
            "ZELUX_ROBOTD_IMU_SOCK", "/run/zelux/robotd-imu.sock")
        self.robotd_imu_publish = _flag("ZELUX_ROBOTD_IMU_PUBLISH", self.mode == "radxa")
        self.fan_channels = [
            {"name": "顶部风扇", "pwm": 14, "gpio": 116, "pin": "GPIO3_C4", "tach": None},
            {"name": "主板散热", "pwm": 12, "gpio": 149, "pin": "GPIO4_C6/PIN_24", "tach": 150},
        ]
        self._load_hardware()

    def _load_hardware(self) -> None:
        try:
            raw = json.loads(self.hardware_path.read_text("utf-8"))
        except Exception:
            return
        tof = raw.get("tof") if isinstance(raw, dict) else None
        if isinstance(tof, dict):
            try:
                bus, address = int(tof.get("i2c_bus")), int(tof.get("address"))
                if 0 <= bus <= 32 and 0x08 <= address <= 0xFE:
                    self.i2c_bus, self.tof_addr = bus, address
            except (TypeError, ValueError):
                pass
        fan = raw.get("fan") if isinstance(raw, dict) else None
        if isinstance(fan, dict) and isinstance(fan.get("channels"), list) and len(fan["channels"]) == 2:
            try:
                channels = []
                for item in fan["channels"]:
                    pwm, gpio = int(item["pwm"]), int(item["gpio"])
                    tach = None if item.get("tach") in (None, "") else int(item["tach"])
                    if not (0 <= pwm <= 255 and 0 <= gpio <= 1023 and (tach is None or 0 <= tach <= 1023)):
                        raise ValueError
                    channels.append({"name": str(item.get("name") or "风扇")[:32], "pwm": pwm, "gpio": gpio,
                                     "pin": str(item.get("pin") or "")[:64], "tach": tach})
                self.fan_channels = channels
            except (KeyError, TypeError, ValueError):
                pass

    def save_hardware(self, section: str, value: dict) -> None:
        try:
            raw = json.loads(self.hardware_path.read_text("utf-8"))
        except Exception:
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        raw[section] = value
        self.hardware_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.hardware_path.with_suffix(".tmp")
        temp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), "utf-8")
        temp.replace(self.hardware_path)

    def ensure_dirs(self) -> None:
        for d in (self.state_dir, self.models_dir, self.shots_dir, self.drivers_dir,
                  self.firmware_dir):
            d.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
