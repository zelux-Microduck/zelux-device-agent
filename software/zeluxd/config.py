"""全局配置。环境变量优先，便于同一个代码在板子和本机跑不同后端。"""
from __future__ import annotations

import os
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

    def ensure_dirs(self) -> None:
        for d in (self.state_dir, self.models_dir, self.shots_dir, self.drivers_dir,
                  self.firmware_dir):
            d.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
