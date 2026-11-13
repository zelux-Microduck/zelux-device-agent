"""Publish the selected HAL IMU to robotd without sharing the hardware handle."""
from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from pathlib import Path

log = logging.getLogger("zeluxd.imu_publish")


def packet(snapshot: dict, sequence: int, driver: dict | None = None) -> dict | None:
    """Build the versioned control packet; incomplete UI-only snapshots are not measurements."""
    required = ("gyro_rad_s", "gravity", "quaternion_wxyz")
    if not snapshot.get("ok") or any(key not in snapshot for key in required):
        return None
    identity = dict(driver or {})
    age_ms = snapshot.get("age_ms")
    return {
        "schema": "zelux.robotd-imu/v1",
        "sequence": sequence,
        "monotonic_ns": time.monotonic_ns(),
        "gyro_rad_s": list(snapshot["gyro_rad_s"]),
        "gravity": list(snapshot["gravity"]),
        "quaternion_wxyz": list(snapshot["quaternion_wxyz"]),
        "ready": (bool(snapshot.get("calibrated"))
                  and isinstance(age_ms, (int, float)) and 0 <= age_ms <= 100),
        "stale": {"total": 0, "consecutive": 0},
        "driver": {
            "id": identity.get("loaded") or "zelux.imu.unknown",
            "version": identity.get("version") or "unknown",
            "model": snapshot.get("chip") or "unknown",
            "transport": "i2c",
            "api": "zelux.imu-source/v1",
        },
        "error": None,
    }


class ImuPublisher:
    def __init__(self, hal, path: str | Path, hz: int = 50) -> None:
        self.hal = hal
        self.path = str(path)
        self.period = 1.0 / max(1, hz)
        self._sequence = 0
        self._task: asyncio.Task | None = None
        self._socket: socket.socket | None = None

    async def start(self) -> None:
        if not hasattr(socket, "AF_UNIX"):
            log.warning("当前系统不支持 Unix socket，IMU 不会发布给 robotd")
            return
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._task = asyncio.create_task(self._loop(), name="robotd-imu-publish")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._socket:
            self._socket.close()
            self._socket = None

    async def _loop(self) -> None:
        warned = False
        while True:
            started = time.monotonic()
            report = getattr(self.hal, "driver_report", {}).get("imu", {})
            message = packet(self.hal.snapshot_imu() or {}, self._sequence, report)
            if message is not None and self._socket is not None:
                try:
                    self._socket.sendto(
                        json.dumps(message, separators=(",", ":")).encode(), self.path)
                    self._sequence += 1
                    warned = False
                except (FileNotFoundError, ConnectionRefusedError):
                    # robotd has not selected the IPC source yet; this is a normal idle state.
                    pass
                except OSError as exc:
                    if not warned:
                        log.warning("向 robotd 发布 IMU 失败：%s", exc)
                        warned = True
            await asyncio.sleep(max(0.0, self.period - (time.monotonic() - started)))
