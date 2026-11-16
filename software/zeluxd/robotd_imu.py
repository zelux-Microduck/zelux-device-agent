"""IMU management bridge. Only robotctl/IPC; never opens the shared serial port."""
from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

from .errors import ZeluxError


class RobotdImu:
    def __init__(self, config, runner=None):
        self.config = config
        self._runner = runner or self._process
        self._catalog = None
        self._status = None
        self._updated = None
        self._error = "尚未读取 robotd"
        self._lock = asyncio.Lock()
        self._last_source = None

    async def _process(self, *args):
        try:
            process = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        except OSError as exc:
            raise ZeluxError(f"无法执行 robotd 管理工具，请检查安装与权限：{exc}", "unavailable") from exc
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 20 if args[0] == "systemctl" else 4)
        except BaseException as exc:
            if process.returncode is None:
                process.kill()
            await process.communicate()
            if isinstance(exc, asyncio.TimeoutError):
                raise ZeluxError("robotd 管理操作超时；请刷新确认状态，不要重复重启", "unavailable") from exc
            raise
        if process.returncode:
            raise ZeluxError(stderr.decode(errors="replace")[:1000] or "robotd 管理操作失败", "unavailable")
        if args[0] == "systemctl":
            return {}
        try:
            return json.loads(stdout)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ZeluxError("robotctl 返回了无效状态", "unavailable") from exc

    async def _read(self):
        try:
            await self._fetch()
        except BaseException as exc:
            self._status, self._updated = None, None
            self._error = str(exc)
            raise

    async def _fetch(self):
        catalog = await self._runner(self.config.robotctl, "imu-config", "--file", str(self.config.robotd_config))
        if not isinstance(catalog, dict) or catalog.get("schema") != "zelux.robotd-imu-management/v1":
            raise ZeluxError("robotctl 尚不支持 IMU 管理契约", "unsupported")
        self._catalog = catalog
        status = await self._runner(self.config.robotctl, "--robot-socket", str(self.config.robotd_socket), "imu-status")
        if not isinstance(status, dict) or status.get("schema") != "zelux.robotd-imu-status/v1":
            raise ZeluxError("robotd 尚不支持 IMU 运行状态", "unsupported")
        health = status.get("health")
        if not isinstance(health, dict) or not isinstance(health.get("imu_runtime"), dict):
            raise ZeluxError("robotd 未报告实际加载的 IMU；需要更新 robotd", "unsupported")
        runtime = health["imu_runtime"]
        if (not isinstance(runtime.get("instance_id"), str) or not runtime["instance_id"]
                or not isinstance(runtime.get("revision"), str)
                or runtime.get("source") not in ("ipc", "embedded", "imu-dxl", "unavailable")
                or type(runtime.get("simulated")) is not bool
                or not isinstance(runtime.get("config_path"), str)):
            raise ZeluxError("robotd 运行身份不完整，不能确认实际生效状态", "unavailable")
        if Path(runtime.get("config_path", "")).resolve() != Path(self.config.robotd_config).resolve():
            raise ZeluxError("Agent 配置路径与 robotd 实际读取路径不一致", "denied")
        self._last_source = runtime.get("source")
        self._status = status
        self._updated = time.monotonic()
        self._error = ""

    async def refresh(self):
        async with self._lock:
            try:
                await self._read()
            except Exception as exc:
                self._status = None  # Never leave a cached ready=true after a failed read.
                self._updated = None
                self._error = str(exc)
            return self.view()

    def view(self):
        age = None if self._updated is None else max(0.0, time.monotonic() - self._updated)
        fresh = age is not None and age <= 10 and self._status is not None
        status = self._status if fresh else {}
        health = status.get("health", {})
        runtime = health.get("imu_runtime")
        catalog = self._catalog or {}
        simulated = bool(runtime and runtime.get("simulated"))
        pending = (catalog.get("revision") != runtime.get("revision")) if runtime and not simulated else None
        sensor = health.get("imu") or {}
        ticks = (health.get("control_loop") or {}).get("ticks", 0)
        ticking = isinstance(ticks, int) and not isinstance(ticks, bool) and ticks > 0
        configured_source = catalog.get("config", {}).get("imu", {}).get("source")
        agent_calibration = (self.config.mode != "radxa" or
                             (fresh and configured_source == "ipc" and self._last_source == "ipc"))
        imu_config = catalog.get("config", {}).get("imu", {})
        correction = imu_config.get("mount_correction_wxyz")
        model = next((item for item in catalog.get("models", [])
                      if item.get("id") == imu_config.get("model")), {})
        calibration_supported = (configured_source == "imu-dxl"
                                 and model.get("calibration_supported") is True)
        if not calibration_supported or not fresh:
            calibration_status = "unknown"
        elif correction is None:
            calibration_status = "not_calibrated"
        elif pending is True:
            calibration_status = "pending_restart"
        elif pending is False:
            calibration_status = "applied"
        else:
            calibration_status = "unknown"
        return {"schema": "zelux.studio-imu-management/v1", "available": fresh,
                "catalog": catalog, "runtime": runtime, "pending_restart": pending,
                "sensor": sensor, "simulated": simulated, "age_s": age,
                "ready": fresh and not simulated and ticking and sensor.get("ready") is True,
                "restart_safe": fresh and not simulated and status.get("restart", {}).get("safe") is True,
                "error": self._error if fresh or self._error else "robotd 状态已过期，请刷新",
                "calibration": "agent" if agent_calibration else "robotd-managed",
                "mount_calibration": {
                    "status": calibration_status,
                    "correction_wxyz": correction,
                    "method": "guided-gravity-axes-v1" if correction is not None else None,
                }}

    async def capture_gravity(self, count=12):
        """Read a stable gravity window from robotd's published state, never the serial bus."""
        await self.refresh()
        view = self.view()
        config = view.get("catalog", {}).get("config", {}).get("imu", {})
        runtime = view.get("runtime") or {}
        model = next((item for item in view.get("catalog", {}).get("models", [])
                      if item.get("id") == config.get("model")), {})
        if (not view["available"] or view["simulated"] or not view["ready"]
                or view.get("pending_restart") is not False
                or config.get("source") != "imu-dxl" or model.get("calibration_supported") is not True
                or runtime.get("source") != "imu-dxl" or runtime.get("model") != config.get("model")):
            raise ZeluxError("robotd IMU 未就绪，不能采集机体对齐姿态", "hardware")
        try:
            process = await asyncio.create_subprocess_exec(
                self.config.robotctl, "--robot-socket", str(self.config.robotd_socket),
                "monitor", "--json", "--hz", "10",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        except OSError as exc:
            raise ZeluxError(f"无法订阅 robotd 状态：{exc}", "unavailable") from exc
        rows = []
        try:
            while len(rows) < count:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=8)
                if not line:
                    detail = (await process.stderr.read()).decode(errors="replace")[:500]
                    raise ZeluxError(detail or "robotd 状态流已结束", "unavailable")
                try:
                    packet = json.loads(line)
                    state = packet.get("params", {})
                    gravity = state.get("safety", {}).get("gravity")
                    imu = state.get("imu")
                    gyro = imu.get("gyro") if isinstance(imu, dict) else None
                    if (isinstance(gravity, list) and len(gravity) == 3
                            and isinstance(gyro, list) and len(gyro) == 3
                            and all(isinstance(v, (int, float)) and math.isfinite(v)
                                    for v in gravity + gyro)):
                        rows.append((list(map(float, gravity)), list(map(float, gyro))))
                except (ValueError, TypeError, AttributeError):
                    continue
        except asyncio.TimeoutError as exc:
            raise ZeluxError("等待 robotd 姿态样本超时", "unavailable") from exc
        finally:
            if process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        gravity = [row[0] for row in rows[-count:]]
        gyro = [row[1] for row in rows[-count:]]
        mean = [sum(v[i] for v in gravity) / count for i in range(3)]
        norm = math.sqrt(sum(v * v for v in mean))
        if not math.isfinite(norm) or norm < 0.8:
            raise ZeluxError("重力方向无效，请等待 IMU 就绪后重试", "hardware")
        mean = [v / norm for v in mean]
        spread = max(math.degrees(math.acos(max(-1.0, min(1.0,
            sum(g[i] * mean[i] for i in range(3)) /
            max(1e-9, math.sqrt(sum(v * v for v in g))))))) for g in gravity)
        motion = max(math.sqrt(sum(v * v for v in row)) for row in gyro)
        if spread > 2.5 or motion > 0.15:
            raise ZeluxError(
                f"姿态不够稳定（重力波动 {spread:.1f}°，角速度 {motion:.2f} rad/s），保持静止后重试",
                "busy")
        return {"gravity": mean, "spread_deg": round(spread, 2),
                "gyro_max_rad_s": round(motion, 4), "samples": count}

    async def save_mount_correction(self, correction, expected):
        if self.config.mode != "radxa":
            raise ZeluxError("模拟模式不能修改真实 robotd 配置", "denied")
        async with self._lock:
            await self._read()
            view = self.view()
            config = self._catalog.get("config", {}).get("imu", {})
            model = next((item for item in self._catalog.get("models", [])
                          if item.get("id") == config.get("model")), {})
            if (not view["available"] or view["simulated"]
                    or model.get("calibration_supported") is not True
                    or config.get("source") != "imu-dxl"):
                raise ZeluxError("当前 robotd 型号目录未开放此 IMU 的机体对齐标定", "denied")
            if view.get("pending_restart") is not False or not view.get("ready"):
                raise ZeluxError("请先应用当前型号配置，并确认 IMU 状态就绪", "busy")
            if expected != self._catalog.get("revision"):
                raise ZeluxError("配置已改变，请刷新后重新保存标定", "conflict")
            raw = json.dumps(correction, separators=(",", ":"))
            saved = await self._runner(
                self.config.robotctl, "imu-config", "--file", str(self.config.robotd_config),
                "--mount-correction-wxyz", raw, "--expected", expected)
            if not isinstance(saved, dict) or saved.get("schema") != "zelux.robotd-imu-management/v1":
                raise ZeluxError("robotctl 未确认标定配置保存结果，请刷新核对", "unavailable")
            self._catalog = saved
            return {"ok": True, "saved": True, "management": self.view()}

    async def select(self, model, expected):
        if self.config.mode != "radxa":
            raise ZeluxError("模拟模式不能修改真实 robotd 配置", "denied")
        async with self._lock:
            try:
                await self._read()
                view = self.view()
                if view["simulated"]:
                    raise ZeluxError("robotd 运行在模拟模式，拒绝修改真实配置", "denied")
                if not isinstance(expected, str) or expected != self._catalog.get("revision"):
                    raise ZeluxError("配置已改变，请刷新后重新选择", "conflict")
                choices = self._catalog.get("models", [])
                if not any(m.get("id") == model and m.get("supported") is True and m.get("compatible") is True for m in choices):
                    raise ZeluxError("该型号未验证或与总线协议不兼容", "denied")
                saved = await self._runner(self.config.robotctl, "imu-config", "--file", str(self.config.robotd_config),
                                           "--model", model, "--expected", expected)
                if not isinstance(saved, dict) or saved.get("schema") != "zelux.robotd-imu-management/v1":
                    raise ZeluxError("保存结果无效，配置可能已写入，请刷新核对", "unavailable")
                self._catalog = saved
                return {"ok": True, "saved": True, "management": self.view()}
            except Exception:
                self._status = None
                self._updated = None
                raise

    async def restart(self, expected, confirm, supported):
        if self.config.mode != "radxa" or confirm is not True or supported is not True:
            raise ZeluxError("请先确认机械支撑、停止运动和现场断电看护，再重启 robotd", "denied")
        async with self._lock:
            await self._read()
            view = self.view()
            if expected != self._catalog.get("revision"):
                raise ZeluxError("配置已改变，请刷新后重新确认", "conflict")
            if not view["restart_safe"]:
                raise ZeluxError("robotd 当前不允许安全重启或正在模拟运行", "busy")
            config = self._catalog.get("config", {}).get("imu", {})
            choices = self._catalog.get("models", [])
            if config.get("source") != "imu-dxl" or not any(
                    item.get("id") == config.get("model")
                    and item.get("supported") is True and item.get("compatible") is True
                    for item in choices):
                raise ZeluxError("此入口仅应用 robotd 型号目录中已验证且兼容的总线 IMU", "denied")
            instance = view["runtime"].get("instance_id")
            self._status, self._updated = None, None
            self._error = "正在重启；尚未验证新进程已加载配置"
            await self._runner("systemctl", "--no-ask-password", "restart", "robotd.service")
            # A successful service command is not proof of a healthy sensor.
            try:
                await self._read()
            except Exception as exc:
                self._status, self._updated = None, None
                self._error = str(exc)
            current = self.view()
            applied = bool(current["runtime"] and current["runtime"].get("instance_id") != instance
                           and current["pending_restart"] is False and not current["simulated"])
            return {"ok": True, "restart_requested": True, "applied": applied, "management": current}
