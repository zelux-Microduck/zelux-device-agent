from __future__ import annotations
import copy
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from zeluxd.errors import ZeluxError
from zeluxd.robotd_imu import RobotdImu


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(mode="radxa", robotctl="robotctl", robotd_config=Path("/tmp/robotd.toml"),
                                   robotd_socket=Path("/tmp/robotd.sock"))
        self.catalog = {"schema": "zelux.robotd-imu-management/v1", "revision": "old",
                        "config": {"imu": {"source": "ipc", "model": None}},
                        "models": [{"id": "qingmu-a1", "supported": True, "compatible": True,
                                    "calibration_supported": True},
                                   {"id": "qingmu-b1", "supported": False, "compatible": False,
                                    "calibration_supported": False}]}
        self.status = {"schema": "zelux.robotd-imu-status/v1", "restart": {"safe": True},
                       "health": {"control_loop": {"ticks": 100}, "imu": {"ready": True},
                                  "imu_runtime": {"instance_id": "one", "config_path": str(self.cfg.robotd_config.resolve()),
                                                  "revision": "old", "source": "ipc", "simulated": False}}}
        self.calls = []
        self.failure = False
        self.bridge = RobotdImu(self.cfg, self.run_command)

    async def run_command(self, *args):
        self.calls.append(args)
        if self.failure:
            raise OSError("offline")
        if "imu-restart" in args:
            self.status["health"]["imu_runtime"].update(instance_id="two", revision=self.catalog["revision"], source="imu-dxl")
            return {"accepted": True}
        if "imu-config" in args:
            if "--model" in args:
                self.catalog["revision"] = "new"
                self.catalog["config"]["imu"] = {"source": "imu-dxl", "model": "qingmu-a1"}
            return copy.deepcopy(self.catalog)
        return copy.deepcopy(self.status)

    async def test_save_is_pending_and_does_not_restart(self):
        await self.bridge.refresh()
        result = await self.bridge.select("qingmu-a1", "old")
        self.assertTrue(result["saved"])
        self.assertTrue(result["management"]["pending_restart"])
        self.assertFalse(any("imu-restart" in c for c in self.calls))
        saved_call = next(c for c in self.calls if "--model" in c)
        self.assertEqual(saved_call[:3], ("robotctl", "--robot-socket", str(self.cfg.robotd_socket)))

    async def test_mount_save_uses_robotd_socket_without_restart(self):
        self.catalog["config"]["imu"] = {"source": "imu-dxl", "model": "qingmu-a1"}
        self.status["health"]["imu_runtime"]["source"] = "imu-dxl"
        original = self.bridge._runner
        async def save(*args):
            result = await original(*args)
            if "--mount-correction-wxyz" in args:
                result["revision"] = "corrected"
            return result
        self.bridge._runner = save
        result = await self.bridge.save_mount_correction([1, 0, 0, 0], "old")
        self.assertTrue(result["saved"])
        self.assertTrue(result["management"]["pending_restart"])
        call = next(c for c in self.calls if "--mount-correction-wxyz" in c)
        self.assertEqual(call[:3], ("robotctl", "--robot-socket", str(self.cfg.robotd_socket)))
        self.assertFalse(any("imu-restart" in c for c in self.calls))

    async def test_restart_is_delegated_to_robotd_not_systemctl(self):
        await self.bridge.select("qingmu-a1", "old")
        with patch("zeluxd.robotd_imu.RESTART_CONFIRM_TIMEOUT", 0.01), \
                patch("zeluxd.robotd_imu.RESTART_POLL_INTERVAL", 0.001):
            result = await self.bridge.restart("new", True, True)
        self.assertTrue(result["applied"])
        call = next(c for c in self.calls if "imu-restart" in c)
        self.assertEqual(call[:3], ("robotctl", "--robot-socket", str(self.cfg.robotd_socket)))
        self.assertIn("--expected", call)
        self.assertFalse(any(c[0] == "systemctl" for c in self.calls))

    async def test_unverified_and_unknown_models_never_reach_writer(self):
        for model in ["qingmu-b1", "bad", None]:
            with self.assertRaises(ZeluxError):
                await self.bridge.select(model, "old")
        self.assertFalse(any("--model" in c for c in self.calls))

    async def test_stale_revision_rejected(self):
        with self.assertRaises(ZeluxError):
            await self.bridge.select("qingmu-a1", "stale")
        self.assertEqual(self.catalog["revision"], "old")

    async def test_failure_clears_ready_and_effective_identity(self):
        self.assertTrue((await self.bridge.refresh())["ready"])
        self.failure = True
        view = await self.bridge.refresh()
        self.assertFalse(view["ready"])
        self.assertIsNone(view["runtime"])
        self.assertIsNone(view["pending_restart"])

    async def test_expiry_does_not_report_applied(self):
        await self.bridge.refresh()
        self.bridge._updated = time.monotonic() - 11
        self.assertFalse(self.bridge.view()["available"])
        self.assertIsNone(self.bridge.view()["pending_restart"])

    async def test_missing_runtime_from_old_robotd_is_unknown(self):
        self.status["health"].pop("imu_runtime")
        view = await self.bridge.refresh()
        self.assertFalse(view["available"])
        self.assertIn("robotd", view["error"])

    async def test_wrong_configuration_path_refuses_save(self):
        self.status["health"]["imu_runtime"]["config_path"] = "/wrong.toml"
        with self.assertRaises(ZeluxError):
            await self.bridge.select("qingmu-a1", "old")
        self.assertFalse(any("--model" in c for c in self.calls))

    async def test_simulated_runtime_never_proves_hardware_or_allows_save(self):
        self.status["health"]["imu_runtime"]["simulated"] = True
        self.assertFalse((await self.bridge.refresh())["ready"])
        with self.assertRaises(ZeluxError):
            await self.bridge.select("qingmu-a1", "old")

    async def test_restart_requires_both_confirmations_and_safe_reply(self):
        await self.bridge.select("qingmu-a1", "old")
        for confirm, supported in [(False, True), (True, False), ("yes", True)]:
            with self.assertRaises(ZeluxError):
                await self.bridge.restart("new", confirm, supported)
        self.status["restart"]["safe"] = False
        with self.assertRaises(ZeluxError):
            await self.bridge.restart("new", True, True)
        self.assertFalse(any("imu-restart" in c for c in self.calls))

    async def test_restart_confirms_new_process_and_revision_not_just_command(self):
        await self.bridge.select("qingmu-a1", "old")
        with patch("zeluxd.robotd_imu.RESTART_CONFIRM_TIMEOUT", 0.01), \
                patch("zeluxd.robotd_imu.RESTART_POLL_INTERVAL", 0.001):
            result = await self.bridge.restart("new", True, True)
        self.assertTrue(result["applied"])
        self.assertFalse(result["management"]["pending_restart"])
        self.assertEqual(result["management"]["calibration"], "robotd-managed")

    async def test_restart_mismatch_revision_refused(self):
        with self.assertRaises(ZeluxError):
            await self.bridge.restart("stale", True, True)
        self.assertFalse(any("imu-restart" in c for c in self.calls))

    async def test_missing_instance_never_confirms_runtime(self):
        self.status["health"]["imu_runtime"].pop("instance_id")
        self.assertFalse((await self.bridge.refresh())["available"])

    async def test_service_success_without_new_process_does_not_prove_applied(self):
        await self.bridge.select("qingmu-a1", "old")
        original = self.bridge._runner
        async def no_new_instance(*args):
            if "imu-restart" in args:
                return {"accepted": True}
            return await original(*args)
        self.bridge._runner = no_new_instance
        with patch("zeluxd.robotd_imu.RESTART_CONFIRM_TIMEOUT", 0.01), \
                patch("zeluxd.robotd_imu.RESTART_POLL_INTERVAL", 0.001):
            result = await self.bridge.restart("new", True, True)
        self.assertFalse(result["applied"])
        self.assertTrue(result["management"]["pending_restart"])

    async def test_new_instance_with_wrong_revision_is_not_applied(self):
        await self.bridge.select("qingmu-a1", "old")
        original = self.bridge._runner
        async def wrong_revision(*args):
            result = await original(*args)
            if "imu-restart" in args:
                self.status["health"]["imu_runtime"]["revision"] = "wrong"
            return result
        self.bridge._runner = wrong_revision
        self.assertFalse((await self.bridge.restart("new", True, True))["applied"])

    async def test_zero_ticks_not_ready(self):
        self.status["health"]["control_loop"]["ticks"] = 0
        view = await self.bridge.refresh()
        self.assertFalse(view["ready"])

    async def test_sim_agent_does_not_write(self):
        self.cfg.mode = "sim"
        with self.assertRaises(ZeluxError):
            await self.bridge.select("qingmu-a1", "old")
        self.assertEqual(self.calls, [])

    async def test_bus_source_blocks_old_agent_calibration(self):
        from zeluxd.modules.imu import Module
        hal = SimpleNamespace(snapshot_imu=lambda: {"ok": True, "roll": 99})
        cfg = SimpleNamespace(**vars(self.cfg), state_dir=Path("/nonexistent-imu-test"))
        module = Module(SimpleNamespace(config=cfg, hal=hal))
        module.robotd = self.bridge
        self.catalog["config"]["imu"] = {"source": "imu-dxl", "model": "qingmu-a1"}
        self.status["health"]["imu_runtime"].update(source="imu-dxl", model="qingmu-a1",
                                                        driver_id="zelux.imu.imu-dxl",
                                                        driver_version="0.15.0")
        await self.bridge.refresh()
        state = module.state()["imu"]
        self.assertFalse(state["ok"])
        self.assertTrue(state["device_ready"])
        self.assertEqual(state["source"], "imu-dxl")
        self.assertEqual(state["model"], "qingmu-a1")
        self.assertNotIn("roll", state)
        self.assertFalse(state["calibrated"])
        self.assertEqual(state["management"]["mount_calibration"]["status"], "not_calibrated")
        await self.bridge.select("qingmu-a1", "old")
        for op in ["start", "commit", "bias", "sample", "solve"]:
            with self.assertRaises(ZeluxError):
                await module.cmd(op, {})

    async def test_robotd_managed_imu_not_ready_reports_device_reason(self):
        from zeluxd.modules.imu import Module
        hal = SimpleNamespace(snapshot_imu=lambda: {"ok": True, "roll": 99})
        cfg = SimpleNamespace(**vars(self.cfg), state_dir=Path("/nonexistent-imu-test"))
        module = Module(SimpleNamespace(config=cfg, hal=hal))
        module.robotd = self.bridge
        self.catalog["config"]["imu"] = {"source": "imu-dxl", "model": "qingmu-a1"}
        self.status["health"]["imu_runtime"].update(source="imu-dxl", model="qingmu-a1")
        self.status["health"]["imu"]["ready"] = False
        await self.bridge.refresh()
        state = module.state()["imu"]
        self.assertFalse(state["ok"])
        self.assertFalse(state["device_ready"])
        self.assertIn("IMU", state["device_reason"])
        self.assertNotIn("roll", state)


if __name__ == "__main__":
    unittest.main()
