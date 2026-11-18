"""内置驱动注册表。

第一阶段把现有单体 HAL 描述成可查询的驱动清单。后续各设备驱动迁入独立
进程时，业务模块和 Studio 继续消费相同的 registry/state 结构。
"""
from __future__ import annotations

import json
import base64
import hashlib
import os
import re
import stat
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .protocol import (DRIVER_API_SUPPORTED, DRIVER_API_VERSION, satisfies,
                       version_tuple)

DEVICE_CLASSES = {"servo", "tof", "camera", "fan", "imu"}
_DRIVER_ID = re.compile(r"^zelux\.[a-z0-9_-]+\.[a-z0-9_.-]+$")
_KEY_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_BYTES = 48 * 1024 * 1024
MAX_MANIFEST_BYTES = 128 * 1024
MAX_ZIP_ENTRIES = 32


@dataclass(frozen=True)
class DriverDescriptor:
    id: str
    name: str
    version: str
    device_class: str
    provider: str = "builtin"
    driver_api: str = DRIVER_API_SUPPORTED
    firmware_version: str | None = None
    firmware_upgrade_supported: bool = False
    models: tuple[str, ...] = ()
    capabilities: dict[str, Any] = field(default_factory=dict)

    def public(self, *, active: bool = True, health: str = "unknown") -> dict[str, Any]:
        data = asdict(self)
        data.update({"active": active, "health": health})
        return data


class ManifestError(ValueError):
    pass


def validate_manifest(raw: Any, *, agent_version: str) -> dict[str, Any]:
    """校验驱动 manifest；不执行 entrypoint，也不信任包内路径。"""
    if not isinstance(raw, dict):
        raise ManifestError("驱动清单必须是 JSON 对象")
    required = ("schema", "id", "name", "version", "device_class", "driver_api",
                "agent", "entrypoint", "artifact", "signature")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ManifestError(f"驱动清单缺少：{', '.join(missing)}")
    if raw["schema"] != "zelux.driver-manifest/v1":
        raise ManifestError("不支持的驱动清单版本")
    if not _DRIVER_ID.match(str(raw["id"])):
        raise ManifestError("驱动 id 格式不合法")
    try:
        version_tuple(str(raw["version"]))
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc
    if raw["device_class"] not in DEVICE_CLASSES:
        raise ManifestError("device_class 必须是 servo/tof/camera/fan/imu")
    try:
        driver_api_ok = satisfies(DRIVER_API_VERSION, str(raw["driver_api"]))
        agent_ok = satisfies(agent_version, str(raw["agent"]))
    except ValueError as exc:
        raise ManifestError(f"版本范围不合法：{exc}") from exc
    if not driver_api_ok:
        raise ManifestError(f"驱动要求 Driver API {raw['driver_api']}，设备是 {DRIVER_API_VERSION}")
    if not agent_ok:
        raise ManifestError(f"驱动要求 Agent {raw['agent']}，设备是 {agent_version}")
    artifact = raw.get("artifact")
    if not isinstance(artifact, dict) or not artifact.get("file") or not artifact.get("sha256"):
        raise ManifestError("artifact 必须包含 file 和 sha256")
    filename = str(artifact["file"])
    if Path(filename).name != filename or filename in (".", ".."):
        raise ManifestError("artifact.file 只能是包内文件名")
    digest = str(artifact["sha256"])
    if not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
        raise ManifestError("artifact.sha256 必须是 64 位十六进制")
    result = dict(raw)
    result["version"] = str(raw["version"])
    return result


def _canonical_manifest(manifest: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in manifest.items() if key != "signature"}
    return json.dumps(unsigned, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def verify_ed25519(manifest: dict[str, Any], trusted_keys_dir: Path) -> tuple[bool, str]:
    signature = manifest.get("signature")
    if not isinstance(signature, dict):
        return False, "签名不是结构化 Ed25519 签名"
    if signature.get("algorithm") != "ed25519":
        return False, "只支持 ed25519 签名"
    key_id = str(signature.get("key_id", ""))
    if not _KEY_ID.fullmatch(key_id):
        return False, "签名 key_id 不合法"
    key_path = trusted_keys_dir / f"{key_id}.pem"
    if not key_path.is_file():
        return False, f"设备不信任签名密钥 {key_id}"
    try:
        value = base64.b64decode(str(signature.get("value", "")), validate=True)
        from cryptography.hazmat.primitives import serialization

        public_key = serialization.load_pem_public_key(key_path.read_bytes())
        public_key.verify(value, _canonical_manifest(manifest))
    except ImportError:
        return False, "设备缺少 cryptography，无法验证 Ed25519 签名"
    except Exception:
        return False, "驱动包签名验证失败"
    return True, key_id


def zip_entry_is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)


SIM_DRIVERS = (
    DriverDescriptor("zelux.sim.servo", "模拟舵机", "1.0.0", "servo",
                     models=("SIM-SERVO",), capabilities={"count": 15, "bus": "sim", "firmware": False}),
    DriverDescriptor("zelux.sim.tof", "模拟 8×8 ToF", "1.0.0", "tof",
                     models=("SIM-TOF",), capabilities={"resolutions": [[8, 8]], "max_hz": 15, "firmware": False}),
    DriverDescriptor("zelux.sim.camera", "模拟摄像头", "1.0.0", "camera",
                     models=("SIM-CAMERA",), capabilities={"snapshot": True, "stream": False, "firmware": False}),
    DriverDescriptor("zelux.sim.fan", "模拟双风扇", "1.0.0", "fan",
                     models=("SIM-FAN",), capabilities={"channels": 2, "pwm": True, "tach": [False, True]}),
    DriverDescriptor("zelux.sim.imu", "模拟 IMU", "1.0.0", "imu",
                     models=("SIM-IMU",), capabilities={"calibration": True, "firmware": False}),
)

RADXA_DRIVERS = (
    DriverDescriptor("zelux.servo.feetech-sts-scs", "Feetech STS/SCS", "1.1.0", "servo",
                     models=("STS3215", "STS", "SCS"),
                     capabilities={"count": 15, "bus": "serial", "baud": 1_000_000,
                                   "firmware": False}),
    DriverDescriptor("zelux.tof.st-vl53l5cx", "VL53L5CX", "1.1.0", "tof",
                     models=("VL53L5CX",),
                     capabilities={"resolutions": [[8, 8]], "max_hz": 15, "firmware": False}),
    DriverDescriptor("zelux.camera.v4l2-libcamera", "V4L2 / libcamera", "1.1.0", "camera",
                     models=("UVC", "V4L2", "libcamera"),
                     capabilities={"snapshot": True, "stream": True, "firmware": False}),
    DriverDescriptor("zelux.fan.radxa-pwm", "Radxa PWM 风扇", "1.1.0", "fan",
                     models=("Zero 3W dual PWM fan",),
                     capabilities={"channels": 2, "pwm": True, "tach": [False, True]}),
    DriverDescriptor("zelux.imu.multi-i2c", "QMI8658/MPU6050/ICM42688", "1.1.0", "imu",
                     models=("QMI8658", "MPU6050", "ICM42688"),
                     capabilities={"calibration": True, "firmware": False}),
)


class DriverRegistry:
    def __init__(self, mode: str, hal, *, state_dir: Path | None = None,
                 agent_version: str = "0.0.0", allow_untrusted: bool = False) -> None:
        self.mode = mode
        self.hal = hal
        self.agent_version = agent_version
        self.allow_untrusted = allow_untrusted
        self.state_dir = Path(state_dir) if state_dir else None
        self.selection_path = self.state_dir / "selection.json" if self.state_dir else None
        self.inbox_dir = self.state_dir / "inbox" if self.state_dir else None
        self.releases_dir = self.state_dir / "releases" if self.state_dir else None
        self.trusted_keys_dir = self.state_dir / "trusted_keys" if self.state_dir else None
        descriptors = SIM_DRIVERS if mode == "sim" else RADXA_DRIVERS
        self._available = {item.id: item for item in descriptors}
        if mode != "sim":
            try:
                from zelux_drivers import descriptors as package_descriptors

                for item in package_descriptors():
                    caps = {name: True for name in item.capabilities}
                    candidate = DriverDescriptor(item.id, item.id, item.version, item.kind.value,
                                                 provider=item.vendor, driver_api=item.protocol,
                                                 models=item.models, capabilities=caps,
                                                 firmware_upgrade_supported="firmware-upgrade" in item.capabilities)
                    self._available.setdefault(candidate.id, candidate)
            except ImportError:
                pass
        self._active = {item.device_class: item for item in descriptors}
        self._pending: dict[str, DriverDescriptor] = {}
        self._load_selection()
        self._sync_hal_report()

    def _sync_hal_report(self) -> None:
        report = getattr(self.hal, "driver_report", None)
        if not isinstance(report, dict):
            return
        for device_class, detail in report.items():
            loaded = self._available.get(str(detail.get("loaded") or ""))
            requested = self._available.get(str(detail.get("requested") or ""))
            if loaded and loaded.device_class == device_class:
                self._active[device_class] = loaded
            if requested and loaded is not requested:
                self._pending[device_class] = requested
            elif requested is loaded:
                self._pending.pop(device_class, None)

    def _load_selection(self) -> None:
        if not self.selection_path or not self.selection_path.exists():
            return
        try:
            raw = json.loads(self.selection_path.read_text("utf-8"))
            for device_class, driver_id in raw.items():
                candidate = self._available.get(driver_id)
                if candidate and candidate.device_class == device_class:
                    self._active[device_class] = candidate
        except Exception:
            # 损坏的选择文件不能阻止设备启动；保留内置默认驱动。
            return

    def _save_selection(self, selected: dict[str, DriverDescriptor] | None = None) -> None:
        if not self.selection_path:
            return
        self.selection_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.selection_path.with_suffix(".tmp")
        desired = selected or self._active
        tmp.write_text(json.dumps({key: value.id for key, value in desired.items()},
                                  ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(self.selection_path)

    def _health(self, device_class: str) -> str:
        try:
            if device_class == "servo":
                snap = self.hal.snapshot_servos()
                return "ready" if snap.get("online", 0) > 0 or self.mode == "sim" else "unavailable"
            if device_class == "tof":
                return "ready" if self.hal.snapshot_tof().get("state") == "ready" else "unavailable"
            if device_class == "fan":
                return "ready" if self.hal.snapshot_fan().get("items") else "unavailable"
            if device_class == "imu":
                return "ready" if self.hal.snapshot_imu().get("ok") else "unavailable"
            if device_class == "camera":
                return "ready"  # 摄像头无阻塞式快照，具体可用性由 vision 模块报告
        except Exception:
            return "error"
        return "unknown"

    def capabilities(self) -> dict[str, dict[str, Any]]:
        return {
            name: {**driver.capabilities, "driver": driver.id,
                   "driver_version": driver.version, "api": DRIVER_API_VERSION,
                   "health": self._health(name)}
            for name, driver in self._active.items()
        }

    def _detected_models(self, device_class: str, driver: DriverDescriptor) -> list[str]:
        if self.mode == "sim":
            return list(driver.models[:1])
        try:
            if device_class == "imu":
                chip = self.hal.snapshot_imu().get("chip")
                return [str(chip).upper()] if chip and chip != "?" else []
            if device_class == "tof":
                sensor = self.hal.snapshot_tof().get("sensor")
                return [str(sensor).upper()] if sensor else []
            if device_class in ("servo", "fan"):
                return list(driver.models[:1])
            if device_class == "camera":
                info = self.hal.info().get("camera", {})
                return [str(info.get("driver", "V4L2"))] if info.get("available") else []
        except Exception:
            return []
        return []

    def _public(self, device_class: str, driver: DriverDescriptor, *, active: bool = True) -> dict[str, Any]:
        out = driver.public(active=active, health=self._health(device_class))
        detected = self._detected_models(device_class, driver) if active else []
        out.update({"detected_models": detected, "current_model": detected[0] if detected else None,
                    "compatible": True, "compatibility_reason": "Driver API compatible"})
        report = getattr(self.hal, "driver_report", {}).get(device_class, {}) if active else {}
        out.update({"load_fallback": bool(report.get("fallback")),
                    "load_error": report.get("error")})
        return out

    def state(self) -> dict[str, Any]:
        self._sync_hal_report()
        pending = {name: item.id for name, item in self._pending.items()
                   if self._active.get(name) is not item}
        return {
            "api_version": DRIVER_API_VERSION,
            "api_supported": DRIVER_API_SUPPORTED,
            "items": [self._public(name, driver)
                      for name, driver in sorted(self._active.items())],
            "available": [self._public(driver.device_class, driver,
                                        active=driver is self._active.get(driver.device_class))
                          for driver in sorted(self._available.values(), key=lambda item: item.id)],
            "packages": self.installed_packages(),
            "inbox": self.inbox(),
            "pending": pending,
            "restart_required": bool(pending),
        }

    def get(self, device_class: str) -> dict[str, Any] | None:
        self._sync_hal_report()
        driver = self._active.get(device_class)
        return self._public(device_class, driver) if driver else None

    def probe(self, device_class: str | None = None) -> list[dict[str, Any]]:
        classes = [device_class] if device_class else sorted(self._active)
        result = []
        for name in classes:
            driver = self._active.get(name)
            if not driver:
                continue
            result.append({"device_class": name, "driver": driver.id,
                           "confidence": 1.0 if self.mode == "sim" else 0.8,
                           "health": self._health(name),
                           "evidence": ["configured_profile", f"backend={self.mode}"]})
        return result

    def self_test(self, device_class: str) -> dict[str, Any]:
        driver = self._active.get(device_class)
        if not driver:
            raise ManifestError("没有这个设备类型")
        health = self._health(device_class)
        return {"ok": health == "ready", "device_class": device_class,
                "driver": driver.id, "health": health, "destructive": False}

    def select(self, device_class: str, driver_id: str) -> dict[str, Any]:
        candidate = self._available.get(driver_id)
        if not candidate:
            raise ManifestError("驱动尚未安装或没有可执行实现")
        if candidate.device_class != device_class:
            raise ManifestError("驱动类型与设备类型不一致")
        old = self._active.get(device_class)
        check = self.self_test(device_class) if old and old.id == candidate.id else None
        desired = dict(self._active)
        desired.update(self._pending)
        desired[device_class] = candidate
        if old is candidate:
            self._pending.pop(device_class, None)
        else:
            self._pending[device_class] = candidate
        self._save_selection(desired)
        return {"ok": True, "device_class": device_class, "from": old.id if old else None,
                "to": candidate.id, "changed": old is not candidate, "self_test": check,
                "restart_required": old is not candidate, "applied": old is candidate}

    def check_manifest(self, raw: Any) -> dict[str, Any]:
        return validate_manifest(raw, agent_version=self.agent_version)

    def inbox(self) -> list[dict[str, Any]]:
        if not self.inbox_dir or not self.inbox_dir.exists():
            return []
        result = []
        for path in sorted(self.inbox_dir.glob("*.zdp")):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            result.append({"filename": path.name, "size": size})
        return result

    def installed_packages(self) -> list[dict[str, Any]]:
        if not self.releases_dir or not self.releases_dir.exists():
            return []
        result = []
        for path in sorted(self.releases_dir.glob("*/*/install.json")):
            try:
                record = json.loads(path.read_text("utf-8"))
            except Exception:
                continue
            result.append(record)
        return result

    def install_package(self, filename: str) -> dict[str, Any]:
        """验证并原子暂存 inbox 中的 .zdp；不会导入或执行驱动代码。"""
        if not self.inbox_dir or not self.releases_dir or not self.trusted_keys_dir:
            raise ManifestError("驱动存储目录没有配置")
        if Path(filename).name != filename or not filename.endswith(".zdp"):
            raise ManifestError("只能安装驱动收件箱中的 .zdp 文件")
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        package = self.inbox_dir / filename
        if not package.is_file():
            raise ManifestError("驱动收件箱里没有这个包")
        package_size = package.stat().st_size
        if package_size <= 0 or package_size > MAX_PACKAGE_BYTES:
            raise ManifestError("驱动包为空或超过 64 MiB 限制")

        try:
            with zipfile.ZipFile(package) as archive:
                infos = archive.infolist()
                if len(infos) > MAX_ZIP_ENTRIES:
                    raise ManifestError("驱动包文件数量超过限制")
                entry_names = [info.filename for info in infos]
                if len(entry_names) != len(set(entry_names)):
                    raise ManifestError("驱动包包含重复文件名")
                for info in infos:
                    pure = Path(info.filename)
                    if (pure.is_absolute() or ".." in pure.parts or zip_entry_is_symlink(info)):
                        raise ManifestError("驱动包包含不安全路径或符号链接")
                names = {info.filename for info in infos if not info.is_dir()}
                if "manifest.json" not in names:
                    raise ManifestError("驱动包缺少 manifest.json")
                manifest_info = archive.getinfo("manifest.json")
                if manifest_info.file_size > MAX_MANIFEST_BYTES:
                    raise ManifestError("manifest.json 超过 128 KiB")
                try:
                    raw = json.loads(archive.read(manifest_info).decode("utf-8"))
                except Exception as exc:
                    raise ManifestError(f"manifest.json 不是合法 UTF-8 JSON：{exc}") from exc
                manifest = self.check_manifest(raw)
                artifact_name = manifest["artifact"]["file"]
                if artifact_name not in names:
                    raise ManifestError(f"驱动包缺少 {artifact_name}")
                artifact_info = archive.getinfo(artifact_name)
                if artifact_info.file_size <= 0 or artifact_info.file_size > MAX_ARTIFACT_BYTES:
                    raise ManifestError("驱动产物为空或超过 48 MiB 限制")
                artifact = archive.read(artifact_info)
        except zipfile.BadZipFile as exc:
            raise ManifestError("驱动包不是有效 ZIP") from exc

        got = hashlib.sha256(artifact).hexdigest()
        want = str(manifest["artifact"]["sha256"]).lower()
        if got != want:
            raise ManifestError("驱动产物 SHA-256 校验失败")
        trusted, trust_detail = verify_ed25519(manifest, self.trusted_keys_dir)
        if not trusted and not self.allow_untrusted:
            raise ManifestError(trust_detail)

        driver_id = str(manifest["id"])
        version = str(manifest["version"])
        destination = self.releases_dir / driver_id / version
        record = {"id": driver_id, "name": manifest["name"], "version": version,
                  "device_class": manifest["device_class"], "trusted": trusted,
                  "trust": trust_detail, "artifact": artifact_name, "sha256": got,
                  "executable": False, "status": "installed_not_loaded"}
        if destination.exists():
            existing = destination / "install.json"
            if existing.is_file():
                old = json.loads(existing.read_text("utf-8"))
                if old.get("sha256") == got:
                    return {**old, "reused": True}
            raise ManifestError("同版本驱动已存在但内容不同，拒绝覆盖")

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".install-", dir=destination.parent) as temp:
            staged = Path(temp)
            (staged / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
            (staged / artifact_name).write_bytes(artifact)
            (staged / "install.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), "utf-8")
            os.replace(staged, destination)
        return {**record, "reused": False}
