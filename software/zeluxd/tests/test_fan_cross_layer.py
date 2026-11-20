"""Exercise the production package factory through Agent and fake fanctl."""
import asyncio
import importlib.machinery
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from zeluxd.modules.fan import Module
from zeluxd.errors import ZeluxError
from zeluxd.hal.radxa_fan import RadxaFan as FallbackFan

try:
    from zelux_drivers import create
    from zelux_drivers.radxa.fan import FanRequestError
except ImportError:
    create = None


@unittest.skipIf(create is None, "independent driver package is not installed")
class CrossLayerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        path = Path(__file__).resolve().parents[4] / "zelux-fan" / "fanctl"
        if not path.exists():
            self.skipTest("sibling zelux-fan source is required for cross-repository tests")
        loader = importlib.machinery.SourceFileLoader("fanctl_cross_layer", str(path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        self.service = importlib.util.module_from_spec(spec)
        loader.exec_module(self.service)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        node = self.root / "pwmchip0" / "pwm0"
        node.mkdir(parents=True)
        for name, value in (("period", "0"), ("duty_cycle", "0"), ("enable", "0")):
            (node / name).write_text(value)
        self.pwm_patch = patch.object(self.service, "PWM_ROOT", self.root)
        self.pwm_patch.start()
        self.addCleanup(self.pwm_patch.stop)
        self.controller = self.service.Controller({"fan": {"inventory_file": str(self.root / "inventory.json")}})
        self.instance = {"id": "fan-mwnrt89v-1", "name": "Cooling", "enabled": True,
                         "pwm": {"chip": 0, "channel": 0}, "period_ns": 40000,
                         "driver_id": "zelux.fan.radxa-pwm", "tach_gpio": None,
                         "pulses_per_rev": 2}

    def agent(self, driver):
        hal = SimpleNamespace(fan=driver, snapshot_fan=driver.snapshot_fan, snapshot_temp=lambda: None)
        config = SimpleNamespace(mode="radxa", state_dir=None)
        return Module(SimpleNamespace(hal=hal, config=config,
                                      hub=SimpleNamespace(log_line=lambda *args, **kwargs: None)))

    async def test_production_driver_stable_id_and_lifecycle(self):
        for driver in (create("zelux.fan.radxa-pwm"), FallbackFan()):
            async def call(payload):
                response = await asyncio.to_thread(self.controller.handle, {"api_version": 1, **payload})
                if response.get("ok") is not True:
                    raise FanRequestError(response)
                return response
            driver.helper.call = call
            module = self.agent(driver)
            result = await module.cmd("op", {"op": "instance_create", "instance": self.instance})
            self.assertTrue(result["ok"])
            self.assertEqual(result["fan"]["inventory"]["instances"], [self.instance])
            await module.cmd("op", {"op": "manual", "instance_id": self.instance["id"], "pct": 30})
            self.assertEqual((self.root / "pwmchip0/pwm0/duty_cycle").read_text(), "12000")
            await module.cmd("op", {"op": "manual", "instance_id": self.instance["id"], "pct": 90})
            self.assertEqual((self.root / "pwmchip0/pwm0/duty_cycle").read_text(), "36000")
            for op in ("instance_disable", "instance_enable"):
                self.assertTrue((await module.cmd("op", {"op": op, "instance_id": self.instance["id"]}))["ok"])
            result = await module.cmd("op", {"op": "instance_delete", "instance_id": self.instance["id"], "confirm": True})
            self.assertTrue(result["ok"])
            self.assertEqual(result["fan"]["inventory"]["instances"], [])

    async def test_package_business_error_translated_by_agent(self):
        driver = create("zelux.fan.radxa-pwm")
        async def reject(payload):
            raise FanRequestError({"code": "resource_busy", "error": "occupied"})
        driver.helper.call = reject
        with self.assertRaises(ZeluxError) as caught:
            await self.agent(driver).cmd("op", {"op": "instance_create", "instance": self.instance})
        self.assertEqual(caught.exception.code, "resource_busy")
