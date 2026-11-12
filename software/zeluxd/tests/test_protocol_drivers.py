"""协议协商与内置驱动注册表测试。"""
from __future__ import annotations

import unittest
import hashlib
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from zeluxd.drivers import DriverDescriptor, DriverRegistry, ManifestError, validate_manifest
from zeluxd.protocol import negotiate, satisfies, version_tuple


class ProtocolTests(unittest.TestCase):
    def test_version_parser(self):
        self.assertEqual(version_tuple("1.2.3"), (1, 2, 3))

    def test_range(self):
        self.assertTrue(satisfies("1.1.0", ">=1.0.0 <2.0.0"))
        self.assertFalse(satisfies("2.0.0", ">=1.0.0 <2.0.0"))

    def test_legacy_client_allowed(self):
        self.assertIsNone(negotiate(None))

    def test_incompatible_client(self):
        self.assertEqual(negotiate(">=2.0.0 <3.0.0").code, "protocol.incompatible")


class _Hal:
    def snapshot_servos(self):
        return {"online": 15}

    def snapshot_tof(self):
        return {"state": "ready"}

    def snapshot_fan(self):
        return {"items": [{}, {}]}

    def snapshot_imu(self):
        return {"ok": True}


class DriverRegistryTests(unittest.TestCase):
    @staticmethod
    def package_manifest(artifact: bytes = b"wheel"):
        return {
            "schema": "zelux.driver-manifest/v1", "id": "zelux.tof.example", "name": "Example",
            "version": "1.0.0", "device_class": "tof", "driver_api": ">=1.0.0 <2.0.0",
            "agent": ">=1.0.0 <2.0.0", "entrypoint": "example:Driver",
            "artifact": {"file": "driver.whl", "sha256": hashlib.sha256(artifact).hexdigest()},
            "signature": {"algorithm": "ed25519", "key_id": "dev", "value": "bad"},
        }

    @staticmethod
    def write_package(path: Path, manifest: dict, artifact: bytes = b"wheel", extra=None):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("driver.whl", artifact)
            for name, value in (extra or {}).items():
                archive.writestr(name, value)

    def test_sim_inventory_has_five_device_classes(self):
        registry = DriverRegistry("sim", _Hal())
        state = registry.state()
        self.assertEqual({item["device_class"] for item in state["items"]},
                         {"servo", "tof", "camera", "fan", "imu"})
        self.assertTrue(all(item["active"] for item in state["items"]))

    def test_capabilities_include_driver_identity(self):
        caps = DriverRegistry("sim", _Hal()).capabilities()
        self.assertEqual(caps["tof"]["driver"], "zelux.sim.tof")
        self.assertEqual(caps["tof"]["health"], "ready")

    def test_select_active_driver_is_idempotent_and_persisted(self):
        with TemporaryDirectory() as temp:
            registry = DriverRegistry("sim", _Hal(), state_dir=Path(temp), agent_version="1.0.0")
            out = registry.select("tof", "zelux.sim.tof")
            self.assertFalse(out["changed"])
            self.assertFalse(out["restart_required"])
            self.assertTrue((Path(temp) / "selection.json").exists())

    def test_unknown_driver_cannot_be_selected(self):
        registry = DriverRegistry("sim", _Hal(), agent_version="1.0.0")
        with self.assertRaises(ManifestError):
            registry.select("tof", "zelux.tof.not-installed")

    def test_changed_driver_is_pending_until_restart(self):
        with TemporaryDirectory() as temp:
            registry = DriverRegistry("sim", _Hal(), state_dir=Path(temp), agent_version="1.0.0")
            candidate = DriverDescriptor("zelux.tof.alternate", "Alternate", "1.0.0", "tof",
                                         models=("SIM-TOF-V2",))
            registry._available[candidate.id] = candidate
            out = registry.select("tof", candidate.id)
            self.assertTrue(out["restart_required"])
            self.assertEqual(registry.state()["pending"]["tof"], candidate.id)
            self.assertEqual(registry.get("tof")["id"], "zelux.sim.tof")

    def test_manifest_validation(self):
        manifest = {
            "schema": "zelux.driver-manifest/v1",
            "id": "zelux.tof.vl53l5cx",
            "name": "VL53L5CX",
            "version": "2.1.1",
            "device_class": "tof",
            "driver_api": ">=1.0.0 <2.0.0",
            "agent": ">=1.0.0 <2.0.0",
            "entrypoint": "zelux_tof:Driver",
            "artifact": {"file": "driver.whl", "sha256": "a" * 64},
            "signature": "test-signature",
        }
        self.assertEqual(validate_manifest(manifest, agent_version="1.0.0")["id"], manifest["id"])

    def test_manifest_rejects_path_traversal(self):
        manifest = {
            "schema": "zelux.driver-manifest/v1", "id": "zelux.tof.bad", "name": "bad",
            "version": "1.0.0", "device_class": "tof", "driver_api": ">=1.0.0 <2.0.0",
            "agent": ">=1.0.0 <2.0.0", "entrypoint": "bad:Driver",
            "artifact": {"file": "../driver.whl", "sha256": "a" * 64}, "signature": "x",
        }
        with self.assertRaises(ManifestError):
            validate_manifest(manifest, agent_version="1.0.0")

    def test_dev_install_stages_but_does_not_load_package(self):
        with TemporaryDirectory() as temp:
            state = Path(temp)
            inbox = state / "inbox"
            inbox.mkdir()
            self.write_package(inbox / "example.zdp", self.package_manifest())
            registry = DriverRegistry("sim", _Hal(), state_dir=state,
                                      agent_version="1.0.0", allow_untrusted=True)
            out = registry.install_package("example.zdp")
            self.assertFalse(out["trusted"])
            self.assertFalse(out["executable"])
            self.assertTrue((state / "releases" / "zelux.tof.example" / "1.0.0" /
                             "install.json").exists())
            with self.assertRaises(ManifestError):
                registry.select("tof", "zelux.tof.example")

    def test_production_rejects_untrusted_package(self):
        with TemporaryDirectory() as temp:
            state = Path(temp)
            inbox = state / "inbox"
            inbox.mkdir()
            self.write_package(inbox / "example.zdp", self.package_manifest())
            registry = DriverRegistry("sim", _Hal(), state_dir=state,
                                      agent_version="1.0.0", allow_untrusted=False)
            with self.assertRaises(ManifestError):
                registry.install_package("example.zdp")

    def test_install_rejects_bad_artifact_hash(self):
        with TemporaryDirectory() as temp:
            state = Path(temp)
            inbox = state / "inbox"
            inbox.mkdir()
            self.write_package(inbox / "bad.zdp", self.package_manifest(b"expected"), b"actual")
            registry = DriverRegistry("sim", _Hal(), state_dir=state,
                                      agent_version="1.0.0", allow_untrusted=True)
            with self.assertRaises(ManifestError):
                registry.install_package("bad.zdp")

    def test_install_rejects_zip_traversal(self):
        with TemporaryDirectory() as temp:
            state = Path(temp)
            inbox = state / "inbox"
            inbox.mkdir()
            self.write_package(inbox / "bad.zdp", self.package_manifest(), extra={"../escape": b"x"})
            registry = DriverRegistry("sim", _Hal(), state_dir=state,
                                      agent_version="1.0.0", allow_untrusted=True)
            with self.assertRaises(ManifestError):
                registry.install_package("bad.zdp")


if __name__ == "__main__":
    unittest.main()
