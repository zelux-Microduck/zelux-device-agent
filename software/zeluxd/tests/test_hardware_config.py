from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from zeluxd.config import Config


class HardwareConfigTests(unittest.TestCase):
    def test_fan_and_tof_pin_config_survives_restart(self):
        with TemporaryDirectory() as temp, patch.dict(os.environ, {"ZELUX_STATE": temp}, clear=False):
            cfg = Config()
            channels = [{"name": "A", "pwm": 5, "gpio": 21, "pin": "PIN-A", "tach": None},
                        {"name": "B", "pwm": 6, "gpio": 22, "pin": "PIN-B", "tach": 23}]
            cfg.save_hardware("fan", {"channels": channels})
            cfg.save_hardware("tof", {"i2c_bus": 7, "address": 0x52})
            loaded = Config()
            self.assertEqual(loaded.fan_channels, channels)
            self.assertEqual((loaded.i2c_bus, loaded.tof_addr), (7, 0x52))

    def test_invalid_file_falls_back_to_safe_defaults(self):
        with TemporaryDirectory() as temp, patch.dict(os.environ, {"ZELUX_STATE": temp}, clear=False):
            Path(temp, "hardware.json").write_text(json.dumps({
                "tof": {"i2c_bus": "bad", "address": -1},
                "fan": {"channels": [{"pwm": -1}, {}]},
            }), "utf-8")
            cfg = Config()
            self.assertEqual((cfg.i2c_bus, cfg.tof_addr), (3, 0x52))
            self.assertEqual(cfg.fan_channels[0]["pwm"], 14)


if __name__ == "__main__":
    unittest.main()
