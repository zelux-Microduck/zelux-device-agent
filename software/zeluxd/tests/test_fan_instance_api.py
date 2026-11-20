"""Fan instance API safety tests.

Tests for instance CRUD validation, driver availability checks, and error handling.
"""
from __future__ import annotations

import unittest
import asyncio
from typing import Any

from zeluxd.errors import ZeluxError
from zeluxd.modules.fan import Module
from zeluxd.hal.radxa_fan import FanHelper, FanRequestError
from unittest.mock import AsyncMock, patch


class FakeHal:
    """Minimal HAL stub for fan module tests."""

    def snapshot_fan(self) -> dict:
        return {"items": [], "inventory": {"instances": []}}

    def snapshot_temp(self) -> float | None:
        return None


class FakeConfig:
    """Minimal config stub."""

    mode = "sim"
    state_dir = None


class FakeContext:
    """Minimal context stub."""

    def __init__(self, hal: FakeHal = None, driver: Any = None) -> None:
        self.hal = hal or FakeHal()
        self.config = FakeConfig()
        self.hub = None
        # If driver provided, inject it into HAL
        if driver is not None:
            self.hal.fan = driver


class FakeFanDriver:
    """Fake async driver for testing CRUD forwarding."""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.updated: list[tuple[str, dict]] = []
        self.deleted: list[str] = []
        self.available = True

    async def create_instance(self, instance: dict) -> dict:
        self.created.append(instance)
        return {"ok": True, "id": instance.get("id"), "created": True}

    async def update_instance(self, instance_id: str, instance: dict) -> dict:
        self.updated.append((instance_id, instance))
        return {"ok": True, "id": instance_id, "updated": True}

    async def delete_instance(self, instance_id: str) -> dict:
        self.deleted.append(instance_id)
        return {"ok": True, "id": instance_id, "deleted": True}

    async def pwm_setup(self, op: str, **_kwargs) -> dict:
        return {"api_version": 1, "ok": True, "pwm_setup": {
            "mapped_pins": [{"id": "radxa-zero3w-pin24-pwm13-m1", "ready": True,
                             "chip": 0, "channel": 0}]}}

    async def refresh(self) -> dict:
        return {"inventory": {"instances": [{"id": "cooling-main", "enabled": True,
            "pwm_pin": "radxa-zero3w-pin24-pwm13-m1", "pwm": {"chip": 0, "channel": 0}}]}}


class FanInstanceApiTests(unittest.TestCase):
    """Test fan instance CRUD API safety."""

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self) -> None:
        self.loop.close()

    def test_enable_disable_forward_stable_identity(self) -> None:
        driver = FakeFanDriver()
        driver.set_enabled = AsyncMock(return_value={"ok": True})
        module = Module(FakeContext(driver=driver))
        for enabled in (True, False):
            result = self.loop.run_until_complete(module.cmd(
                "instance_enable" if enabled else "instance_disable", {"instance_id": "cooling-main"}))
            self.assertTrue(result["ok"])
            driver.set_enabled.assert_awaited_with("cooling-main", enabled)

    def test_service_rejection_keeps_business_code(self) -> None:
        driver = FakeFanDriver()
        driver.create_instance = AsyncMock(side_effect=FanRequestError(
            {"ok": False, "code": "resource_busy", "error": "PWM occupied"}))
        module = Module(FakeContext(driver=driver))
        with self.assertRaises(ZeluxError) as caught:
            self.loop.run_until_complete(module.cmd("instance_create", {"instance": {
                "id": "cooling-main", "pwm_pin": "radxa-zero3w-pin24-pwm13-m1",
                "pwm": {"chip": 0, "channel": 0}}}))
        self.assertEqual(caught.exception.code, "resource_busy")

        reader = AsyncMock()
        reader.readline.return_value = b'{"api_version":1,"ok":null,"items":[],"reason":"no tach"}\n'
        writer = AsyncMock()
        from unittest.mock import Mock
        writer.write = Mock()
        writer.close = Mock()
        with patch("asyncio.open_unix_connection", AsyncMock(return_value=(reader, writer)), create=True):
            result = self.loop.run_until_complete(FanHelper("/unused").call({"op": "selftest"}))
        self.assertIsNone(result["ok"])

    def test_socket_client_preserves_rejection(self) -> None:
        reader = AsyncMock()
        reader.readline.return_value = b'{"api_version":1,"ok":false,"code":"resource_busy","error":"occupied"}\n'
        writer = AsyncMock()
        from unittest.mock import Mock
        writer.write = Mock()
        writer.close = Mock()
        with patch("asyncio.open_unix_connection", AsyncMock(return_value=(reader, writer)), create=True):
            with self.assertRaises(FanRequestError) as caught:
                self.loop.run_until_complete(FanHelper("/unused").call({"op": "create"}))
        self.assertEqual(caught.exception.code, "resource_busy")

    def test_pwm_readiness_forwards_read_only_probe(self) -> None:
        driver = FakeFanDriver()
        driver.pwm_readiness = AsyncMock(return_value={"ok": True, "pwm_readiness": {"state": "overlay_required"}})
        module = Module(FakeContext(driver=driver))
        result = self.loop.run_until_complete(module.cmd("op", {"op": "pwm_readiness"}))
        self.assertEqual(result["pwm_readiness"]["state"], "overlay_required")
        driver.pwm_readiness.assert_awaited_once_with()

    def test_pin_config_raises_unsupported(self) -> None:
        """pin_config should be deprecated with clear error."""
        driver = FakeFanDriver()
        ctx = FakeContext(driver=driver)
        module = Module(ctx)

        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("pin_config", {"instances": [{"id": "0"}]})
            )
        self.assertEqual(exc.exception.code, "unsupported")
        self.assertIn("pin_config 已废弃", exc.exception.message)

    def test_instance_create_forwards_to_driver(self) -> None:
        """create_instance should forward to driver with valid input."""
        driver = FakeFanDriver()
        ctx = FakeContext(driver=driver)
        module = Module(ctx)

        result = self.loop.run_until_complete(
            module.cmd("instance_create", {"instance": {"id": "fan-0",
                "pwm_pin": "radxa-zero3w-pin24-pwm13-m1", "pwm": {"chip": 0, "channel": 0}}})
        )
        self.assertTrue(result["ok"])
        self.assertEqual(len(driver.created), 1)
        self.assertEqual(driver.created[0]["id"], "fan-0")

    def test_create_is_rejected_when_selected_pin_is_not_ready(self) -> None:
        driver = FakeFanDriver()
        driver.pwm_setup = AsyncMock(return_value={"api_version": 1, "ok": True,
            "pwm_setup": {"mapped_pins": [{"id": "radxa-zero3w-pin24-pwm13-m1",
                                           "ready": False, "chip": None, "channel": 0}]}})
        module = Module(FakeContext(driver=driver))
        with self.assertRaises(ZeluxError) as caught:
            self.loop.run_until_complete(module.cmd("instance_create", {"instance": {
                "id": "fan-0", "pwm_pin": "radxa-zero3w-pin24-pwm13-m1",
                "pwm": {"chip": 0, "channel": 0}}}))
        self.assertEqual(caught.exception.code, "not_ready")
        self.assertEqual(driver.created, [])

    def test_create_is_rejected_when_chip_mapping_does_not_match_pin(self) -> None:
        driver = FakeFanDriver()
        module = Module(FakeContext(driver=driver))
        with self.assertRaises(ZeluxError) as caught:
            self.loop.run_until_complete(module.cmd("instance_create", {"instance": {
                "id": "fan-0", "pwm_pin": "radxa-zero3w-pin24-pwm13-m1",
                "pwm": {"chip": 3, "channel": 0}}}))
        self.assertEqual(caught.exception.code, "mapping_mismatch")
        self.assertEqual(driver.created, [])

    def test_create_rejection_is_not_reported_as_success(self) -> None:
        driver = FakeFanDriver()
        driver.create_instance = AsyncMock(return_value={"ok": False, "error": "PWM channel busy"})
        module = Module(FakeContext(driver=driver))

        with self.assertRaises(ZeluxError) as caught:
            self.loop.run_until_complete(
                module.cmd("instance_create", {"instance": {"id": "fan-0",
                    "pwm_pin": "radxa-zero3w-pin24-pwm13-m1", "pwm": {"chip": 0, "channel": 0}}})
            )

        self.assertEqual(caught.exception.code, "hardware")
        self.assertIn("PWM channel busy", caught.exception.message)

    def test_instance_create_requires_dict_instance(self) -> None:
        """create_instance should reject non-dict instance."""
        driver = FakeFanDriver()
        ctx = FakeContext(driver=driver)
        module = Module(ctx)

        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_create", {"instance": "not-a-dict"})
            )
        self.assertEqual(exc.exception.code, "denied")
        self.assertIn("必须是字典", exc.exception.message)

    def test_instance_create_requires_instance_id(self) -> None:
        """create_instance should reject missing/empty id."""
        driver = FakeFanDriver()
        ctx = FakeContext(driver=driver)
        module = Module(ctx)

        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_create", {"instance": {"pin": 33}})
            )
        self.assertEqual(exc.exception.code, "denied")
        self.assertIn("必须包含有效的 id", exc.exception.message)

        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_create", {"instance": {"id": ""}})
            )
        self.assertEqual(exc.exception.code, "denied")

    def test_instance_update_forwards_to_driver(self) -> None:
        """update_instance should forward to driver with valid input."""
        driver = FakeFanDriver()
        ctx = FakeContext(driver=driver)
        module = Module(ctx)

        result = self.loop.run_until_complete(
            module.cmd("instance_update", {
                "instance_id": "fan-0",
                "instance": {"id": "fan-0", "pin": 35, "enabled": True,
                    "pwm_pin": "radxa-zero3w-pin24-pwm13-m1", "pwm": {"chip": 0, "channel": 0}}
            })
        )
        self.assertTrue(result["ok"])
        self.assertEqual(len(driver.updated), 1)
        self.assertEqual(driver.updated[0][0], "fan-0")

    def test_instance_update_requires_instance_id(self) -> None:
        """update_instance should reject missing instance_id."""
        driver = FakeFanDriver()
        ctx = FakeContext(driver=driver)
        module = Module(ctx)

        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_update", {"instance": {"id": "fan-0"}})
            )
        self.assertEqual(exc.exception.code, "denied")
        self.assertIn("instance_id 参数不能为空", exc.exception.message)

    def test_instance_update_requires_dict_instance(self) -> None:
        """update_instance should reject non-dict instance."""
        driver = FakeFanDriver()
        ctx = FakeContext(driver=driver)
        module = Module(ctx)

        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_update", {
                    "instance_id": "fan-0",
                    "instance": None
                })
            )
        self.assertEqual(exc.exception.code, "denied")
        self.assertIn("必须是字典", exc.exception.message)

    def test_instance_update_rejects_id_mismatch(self) -> None:
        """update_instance should reject mismatched instance IDs."""
        driver = FakeFanDriver()
        ctx = FakeContext(driver=driver)
        module = Module(ctx)

        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_update", {
                    "instance_id": "fan-0",
                    "instance": {"id": "fan-1", "pin": 35}
                })
            )
        self.assertEqual(exc.exception.code, "denied")
        self.assertIn("不匹配", exc.exception.message)

    def test_instance_delete_requires_confirm(self) -> None:
        """delete_instance should require explicit confirmation."""
        driver = FakeFanDriver()
        ctx = FakeContext(driver=driver)
        module = Module(ctx)

        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_delete", {"instance_id": "fan-0"})
            )
        self.assertEqual(exc.exception.code, "denied")
        self.assertIn("请先确认", exc.exception.message)

        # With confirm, should succeed
        result = self.loop.run_until_complete(
            module.cmd("instance_delete", {"instance_id": "fan-0", "confirm": True})
        )
        self.assertTrue(result["ok"])
        self.assertEqual(len(driver.deleted), 1)

    def test_instance_delete_requires_instance_id(self) -> None:
        """delete_instance should reject missing instance_id."""
        driver = FakeFanDriver()
        ctx = FakeContext(driver=driver)
        module = Module(ctx)

        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_delete", {"confirm": True})
            )
        self.assertEqual(exc.exception.code, "denied")
        self.assertIn("instance_id 参数不能为空", exc.exception.message)

    def test_absent_driver_returns_zelux_error(self) -> None:
        """All instance CRUD should return ZeluxError, not AttributeError, when driver absent."""
        ctx = FakeContext(driver=None)  # No driver
        module = Module(ctx)

        # All should raise ZeluxError with "unavailable" code
        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_create", {"instance": {"id": "fan-0",
                    "pwm_pin": "radxa-zero3w-pin24-pwm13-m1", "pwm": {"chip": 0, "channel": 0}}})
            )
        self.assertEqual(exc.exception.code, "unavailable")
        self.assertNotIsInstance(exc.exception, AttributeError)

        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_update", {
                    "instance_id": "fan-0",
                    "instance": {"id": "fan-0"}
                })
            )
        self.assertEqual(exc.exception.code, "unavailable")

        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_delete", {"instance_id": "fan-0", "confirm": True})
            )
        self.assertEqual(exc.exception.code, "unavailable")

    def test_driver_missing_crud_method_returns_zelux_error(self) -> None:
        """CRUD should return ZeluxError, not AttributeError, when driver lacks method."""

        class IncompleteDriver:
            """Driver missing CRUD methods."""
            available = True

        driver = IncompleteDriver()
        ctx = FakeContext(driver=driver)
        module = Module(ctx)

        # create_instance missing
        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_create", {"instance": {"id": "fan-0",
                    "pwm_pin": "radxa-zero3w-pin24-pwm13-m1", "pwm": {"chip": 0, "channel": 0}}})
            )
        self.assertEqual(exc.exception.code, "unsupported")
        self.assertIn("create_instance", exc.exception.message)

        # update_instance missing
        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_update", {
                    "instance_id": "fan-0",
                    "instance": {"id": "fan-0"}
                })
            )
        self.assertEqual(exc.exception.code, "unsupported")
        self.assertIn("update_instance", exc.exception.message)

        # delete_instance missing
        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_delete", {"instance_id": "fan-0", "confirm": True})
            )
        self.assertEqual(exc.exception.code, "unsupported")
        self.assertIn("delete_instance", exc.exception.message)

    def test_instance_update_requires_body_id(self) -> None:
        """update_instance should reject missing instance id in body."""
        driver = FakeFanDriver()
        ctx = FakeContext(driver=driver)
        module = Module(ctx)

        # Should reject when instance body has no id field
        with self.assertRaises(ZeluxError) as exc:
            self.loop.run_until_complete(
                module.cmd("instance_update", {
                    "instance_id": "fan-0",
                    "instance": {"pin": 35}
                })
            )
        self.assertEqual(exc.exception.code, "denied")
        self.assertIn("必须包含 id 字段", exc.exception.message)


if __name__ == "__main__":
    unittest.main()
