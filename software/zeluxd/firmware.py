"""外设固件包验证、暂存和升级前置检查。

本阶段绝不进入 Bootloader 或写设备；只有驱动将来显式实现刷写接口后，任务状态机
才会从 ``validated`` 进入 ``writing``。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .drivers import (DEVICE_CLASSES, MAX_MANIFEST_BYTES, MAX_PACKAGE_BYTES,
                      MAX_ZIP_ENTRIES, ManifestError, verify_ed25519,
                      zip_entry_is_symlink)
from .protocol import satisfies, version_tuple

MAX_FIRMWARE_BYTES = 48 * 1024 * 1024
_FIRMWARE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,127}$")


class FirmwareError(ValueError):
    pass


def validate_firmware_manifest(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise FirmwareError("固件清单必须是 JSON 对象")
    required = ("schema", "id", "version", "device_class", "models", "driver",
                "image", "signature")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise FirmwareError(f"固件清单缺少：{', '.join(missing)}")
    if raw["schema"] != "zelux.firmware/v1":
        raise FirmwareError("不支持的固件清单版本")
    if not _FIRMWARE_ID.fullmatch(str(raw["id"])):
        raise FirmwareError("固件 id 格式不合法")
    try:
        version_tuple(str(raw["version"]))
    except ValueError as exc:
        raise FirmwareError(str(exc)) from exc
    if raw["device_class"] not in DEVICE_CLASSES:
        raise FirmwareError("device_class 必须是 servo/tof/camera/fan/imu")
    models = raw["models"]
    if not isinstance(models, list) or not models or not all(isinstance(x, str) and x for x in models):
        raise FirmwareError("models 必须是非空型号列表")
    try:
        satisfies("1.0.0", str(raw["driver"]))
    except ValueError as exc:
        raise FirmwareError(f"驱动版本范围不合法：{exc}") from exc
    image = raw["image"]
    if not isinstance(image, dict) or not image.get("file") or not image.get("sha256"):
        raise FirmwareError("image 必须包含 file 和 sha256")
    filename = str(image["file"])
    if Path(filename).name != filename or filename in (".", ".."):
        raise FirmwareError("image.file 只能是包内文件名")
    if not re.fullmatch(r"[a-fA-F0-9]{64}", str(image["sha256"])):
        raise FirmwareError("image.sha256 必须是 64 位十六进制")
    if image.get("size") is not None and (not isinstance(image["size"], int) or image["size"] <= 0):
        raise FirmwareError("image.size 必须是正整数")
    return dict(raw)


class FirmwareManager:
    def __init__(self, state_dir: Path, driver_registry, *, allow_untrusted: bool = False) -> None:
        self.state_dir = Path(state_dir)
        self.driver_registry = driver_registry
        self.allow_untrusted = allow_untrusted
        self.inbox_dir = self.state_dir / "inbox"
        self.releases_dir = self.state_dir / "releases"
        # 驱动和固件共用发布信任根，避免两套密钥生命周期。
        self.trusted_keys_dir = driver_registry.trusted_keys_dir

    def inbox(self) -> list[dict[str, Any]]:
        if not self.inbox_dir.exists():
            return []
        return [{"filename": path.name, "size": path.stat().st_size}
                for path in sorted(self.inbox_dir.glob("*.zfw")) if path.is_file()]

    def installed(self) -> list[dict[str, Any]]:
        if not self.releases_dir.exists():
            return []
        result = []
        for path in sorted(self.releases_dir.glob("*/*/install.json")):
            try:
                result.append(json.loads(path.read_text("utf-8")))
            except Exception:
                continue
        return result

    def state(self) -> dict[str, Any]:
        packages = [self._compatibility(item) for item in self.installed()]
        return {"inbox": self.inbox(), "packages": packages,
                "job": None, "write_enabled": False}

    def _compatibility(self, item: dict[str, Any]) -> dict[str, Any]:
        active = self.driver_registry.get(str(item.get("device_class", "")))
        active_models = set(active.get("detected_models") or active.get("models") or ()) if active else set()
        firmware_models = set(item.get("models") or ())
        model_ok = bool(active_models & firmware_models)
        try:
            driver_ok = bool(active and satisfies(str(active.get("version")), str(item.get("driver"))))
        except ValueError:
            driver_ok = False
        reasons = []
        if not active:
            reasons.append("没有活动驱动")
        if active and not model_ok:
            reasons.append("设备型号不匹配")
        if active and not driver_ok:
            reasons.append("驱动版本不兼容")
        if not item.get("trusted"):
            reasons.append("固件签名不受信任")
        compatible = bool(active and model_ok and driver_ok and item.get("trusted"))
        current = active.get("firmware_version") if active else None
        return {**item, "compatible": compatible, "compatibility_reasons": reasons,
                "detected_models": sorted(active_models), "current_version": current,
                "version_change": f"{current or '未知'} → {item.get('version', '未知')}",
                "risk": "high" if item.get("device_class") == "servo" else "medium"}

    def check_manifest(self, raw: Any) -> dict[str, Any]:
        return validate_firmware_manifest(raw)

    def install_package(self, filename: str) -> dict[str, Any]:
        if Path(filename).name != filename or not filename.endswith(".zfw"):
            raise FirmwareError("只能安装固件收件箱中的 .zfw 文件")
        package = self.inbox_dir / filename
        if not package.is_file():
            raise FirmwareError("固件收件箱里没有这个包")
        size = package.stat().st_size
        if size <= 0 or size > MAX_PACKAGE_BYTES:
            raise FirmwareError("固件包为空或超过 64 MiB 限制")
        try:
            with zipfile.ZipFile(package) as archive:
                infos = archive.infolist()
                if len(infos) > MAX_ZIP_ENTRIES:
                    raise FirmwareError("固件包文件数量超过限制")
                names_list = [info.filename for info in infos]
                if len(names_list) != len(set(names_list)):
                    raise FirmwareError("固件包包含重复文件名")
                for info in infos:
                    path = Path(info.filename)
                    if path.is_absolute() or ".." in path.parts or zip_entry_is_symlink(info):
                        raise FirmwareError("固件包包含不安全路径或符号链接")
                names = {info.filename for info in infos if not info.is_dir()}
                if "manifest.json" not in names:
                    raise FirmwareError("固件包缺少 manifest.json")
                manifest_info = archive.getinfo("manifest.json")
                if manifest_info.file_size > MAX_MANIFEST_BYTES:
                    raise FirmwareError("manifest.json 超过 128 KiB")
                try:
                    manifest = self.check_manifest(
                        json.loads(archive.read(manifest_info).decode("utf-8")))
                except FirmwareError:
                    raise
                except Exception as exc:
                    raise FirmwareError(f"manifest.json 不是合法 UTF-8 JSON：{exc}") from exc
                image_name = manifest["image"]["file"]
                if image_name not in names:
                    raise FirmwareError(f"固件包缺少 {image_name}")
                info = archive.getinfo(image_name)
                if info.file_size <= 0 or info.file_size > MAX_FIRMWARE_BYTES:
                    raise FirmwareError("固件镜像为空或超过 48 MiB 限制")
                image = archive.read(info)
        except zipfile.BadZipFile as exc:
            raise FirmwareError("固件包不是有效 ZIP") from exc

        if manifest["image"].get("size") not in (None, len(image)):
            raise FirmwareError("固件镜像大小与清单不一致")
        digest = hashlib.sha256(image).hexdigest()
        if digest != str(manifest["image"]["sha256"]).lower():
            raise FirmwareError("固件镜像 SHA-256 校验失败")
        trusted, trust_detail = verify_ed25519(manifest, self.trusted_keys_dir)
        if not trusted and not self.allow_untrusted:
            raise FirmwareError(trust_detail.replace("驱动包", "固件包"))

        firmware_id, version = str(manifest["id"]), str(manifest["version"])
        destination = self.releases_dir / firmware_id / version
        record = {"id": firmware_id, "version": version,
                  "device_class": manifest["device_class"], "models": manifest["models"],
                  "driver": manifest["driver"], "trusted": trusted, "trust": trust_detail,
                  "image": image_name, "sha256": digest, "status": "validated"}
        if destination.exists():
            existing = destination / "install.json"
            if existing.is_file():
                old = json.loads(existing.read_text("utf-8"))
                if old.get("sha256") == digest:
                    return {**old, "reused": True}
            raise FirmwareError("同版本固件已存在但内容不同，拒绝覆盖")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".install-", dir=destination.parent) as temp:
            staged = Path(temp)
            (staged / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
            (staged / image_name).write_bytes(image)
            (staged / "install.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), "utf-8")
            os.replace(staged, destination)
        return {**record, "reused": False}

    def precheck(self, firmware_id: str, version: str) -> dict[str, Any]:
        path = self.releases_dir / firmware_id / version
        try:
            manifest = json.loads((path / "manifest.json").read_text("utf-8"))
        except Exception as exc:
            raise FirmwareError("找不到已验证的固件包") from exc
        active = self.driver_registry.get(manifest["device_class"])
        checks = {
            "package_validated": (path / "install.json").is_file(),
            "driver_active": bool(active),
            "driver_version": bool(active and satisfies(active["version"], manifest["driver"])),
            "driver_supports_upgrade": bool(active and active.get("firmware_upgrade_supported")),
            "trusted": bool(json.loads((path / "install.json").read_text("utf-8")).get("trusted")),
        }
        return {"ready": all(checks.values()), "checks": checks,
                "device_class": manifest["device_class"], "driver": active,
                "firmware": {"id": firmware_id, "version": version},
                "write_enabled": False}

