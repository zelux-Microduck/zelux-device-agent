"""保存的驱动选择必须真正决定 HAL 实例，并能逐设备回退。"""
from __future__ import annotations

import asyncio
import json
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from zeluxd.hal import load_driver_selection
from zeluxd.hal.radxa import DEFAULT_DRIVER_IDS, RadxaHal


class _Part:
    def __init__(self, driver_id: str, fail_start: bool = False) -> None:
        self.driver_id = driver_id
        self.fail_start = fail_start

    async def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("selected driver failed")

    async def stop(self) -> None:
        pass

    def info(self) -> dict:
        return {"driver": self.driver_id, "available": True}


def _ctx():
    config = SimpleNamespace(i2c_bus=3, tof_addr=0x52)
    return SimpleNamespace(config=config, log=logging.getLogger("test.hal"))


class HalSelectionTests(unittest.TestCase):
    def test_selection_file_is_sanitized(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "selection.json"
            path.write_text(json.dumps({"tof": "zelux.tof.vl53l5cx", "bad": "ignored"}), "utf-8")
            self.assertEqual(load_driver_selection(path), {"tof": "zelux.tof.st-vl53l5cx"})

    def test_selected_factory_controls_each_hal_part(self):
        made = []

        def factory(driver_id, *_args, **_kwargs):
            made.append(driver_id)
            return _Part(driver_id)

        hal = RadxaHal(_ctx(), selection={"tof": "zelux.tof.custom"}, factory=factory)
        self.assertEqual(hal.tof.driver_id, "zelux.tof.custom")
        self.assertEqual(hal.servo.driver_id, DEFAULT_DRIVER_IDS["servo"])
        self.assertIn("zelux.tof.custom", made)

    def test_start_failure_falls_back_only_failed_device(self):
        def factory(driver_id, *_args, **_kwargs):
            return _Part(driver_id, fail_start=driver_id == "zelux.tof.broken")

        async def run():
            hal = RadxaHal(_ctx(), selection={"tof": "zelux.tof.broken"}, factory=factory)
            await hal.start()
            try:
                self.assertEqual(hal.tof.driver_id, DEFAULT_DRIVER_IDS["tof"])
                self.assertTrue(hal.driver_report["tof"]["fallback"])
                self.assertIn("启动失败", hal.driver_report["tof"]["error"])
                self.assertFalse(hal.driver_report["servo"]["fallback"])
            finally:
                await hal.stop()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
