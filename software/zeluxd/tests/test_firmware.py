"""外设固件包只验证和暂存，不执行真实刷写。"""
from __future__ import annotations

import hashlib
import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from zeluxd.firmware import FirmwareError, FirmwareManager, validate_firmware_manifest


class _Registry:
    trusted_keys_dir: Path

    def __init__(self, root: Path, supports_upgrade: bool = False):
        self.trusted_keys_dir = root / "trusted_keys"
        self.supports_upgrade = supports_upgrade

    def get(self, device_class):
        return {"id": "zelux.tof.example", "version": "1.2.0",
                "device_class": device_class,
                "models": ["EXAMPLE-TOF"], "firmware_version": "1.0.0",
                "firmware_upgrade_supported": self.supports_upgrade}


def manifest(image=b"firmware"):
    return {
        "schema": "zelux.firmware/v1", "id": "example.tof", "version": "1.4.0",
        "device_class": "tof", "models": ["EXAMPLE-TOF"], "driver": ">=1.0.0 <2.0.0",
        "image": {"file": "firmware.bin", "size": len(image),
                  "sha256": hashlib.sha256(image).hexdigest()},
        "signature": {"algorithm": "ed25519", "key_id": "dev", "value": "bad"},
    }


def package(path: Path, data: dict, image=b"firmware", extra=None):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(data))
        archive.writestr("firmware.bin", image)
        for name, value in (extra or {}).items():
            archive.writestr(name, value)


class FirmwareTests(unittest.TestCase):
    def test_manifest_valid(self):
        self.assertEqual(validate_firmware_manifest(manifest())["id"], "example.tof")

    def test_manifest_rejects_bad_model_list(self):
        data = manifest()
        data["models"] = []
        with self.assertRaises(FirmwareError):
            validate_firmware_manifest(data)

    def test_dev_install_is_validated_not_written(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inbox = root / "firmware" / "inbox"
            inbox.mkdir(parents=True)
            package(inbox / "example.zfw", manifest())
            manager = FirmwareManager(root / "firmware", _Registry(root), allow_untrusted=True)
            out = manager.install_package("example.zfw")
            self.assertEqual(out["status"], "validated")
            self.assertFalse(out["trusted"])
            self.assertFalse(manager.state()["write_enabled"])

    def test_state_explains_model_version_and_signature_compatibility(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inbox = root / "firmware" / "inbox"
            inbox.mkdir(parents=True)
            package(inbox / "example.zfw", manifest())
            manager = FirmwareManager(root / "firmware", _Registry(root), allow_untrusted=True)
            manager.install_package("example.zfw")
            item = manager.state()["packages"][0]
            self.assertEqual(item["version_change"], "1.0.0 → 1.4.0")
            self.assertFalse(item["compatible"])
            self.assertIn("固件签名不受信任", item["compatibility_reasons"])

    def test_bad_hash_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inbox = root / "firmware" / "inbox"
            inbox.mkdir(parents=True)
            package(inbox / "bad.zfw", manifest(b"expected"), b"actual")
            manager = FirmwareManager(root / "firmware", _Registry(root), allow_untrusted=True)
            with self.assertRaises(FirmwareError):
                manager.install_package("bad.zfw")

    def test_zip_traversal_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inbox = root / "firmware" / "inbox"
            inbox.mkdir(parents=True)
            package(inbox / "bad.zfw", manifest(), extra={"../escape": b"x"})
            manager = FirmwareManager(root / "firmware", _Registry(root), allow_untrusted=True)
            with self.assertRaises(FirmwareError):
                manager.install_package("bad.zfw")

    def test_precheck_requires_trust_and_driver_support(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inbox = root / "firmware" / "inbox"
            inbox.mkdir(parents=True)
            package(inbox / "example.zfw", manifest())
            manager = FirmwareManager(root / "firmware", _Registry(root), allow_untrusted=True)
            manager.install_package("example.zfw")
            out = manager.precheck("example.tof", "1.4.0")
            self.assertFalse(out["ready"])
            self.assertFalse(out["checks"]["trusted"])
            self.assertFalse(out["checks"]["driver_supports_upgrade"])
            self.assertFalse(out["write_enabled"])


if __name__ == "__main__":
    unittest.main()

