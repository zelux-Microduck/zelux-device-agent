from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from zeluxd.hal.base import JOINT_IDS
from zeluxd.modules.servos import Module, _load_json


class ServoStatusTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.state_dir = Path(temp.name)
        self.hal = SimpleNamespace(snapshot_servos=lambda: {"online": 0, "items": [], "bus": "shared"})
        hub = SimpleNamespace(modules={})
        ctx = SimpleNamespace(
            config=SimpleNamespace(state_dir=self.state_dir),
            hal=self.hal,
            state={},
            hub=hub,
            log=logging.getLogger("test-servo-status"),
        )
        self.module = Module(ctx)

    def test_no_saved_calibration_is_reported_as_pending(self):
        state = self.module.state()["servos"]
        self.assertEqual(state["online"], 0)
        self.assertEqual(state["bus"], "shared")
        self.assertEqual(state["calibration"], {
            "saved_servos": 0, "total_servos": len(JOINT_IDS), "complete": False,
        })

    def test_calibration_progress_counts_only_known_joint_ids(self):
        (self.state_dir / "servo_calib.json").write_text(json.dumps({
            "items": {"10": {"offset": 4}, "11": {"direction": -1}, "999": {"offset": 2}},
        }), "utf-8")
        self.module._calib = _load_json(self.module.calib_path, {})
        self.assertEqual(self.module.state()["servos"]["calibration"]["saved_servos"], 2)

    def test_all_joint_records_are_marked_complete(self):
        (self.state_dir / "servo_calib.json").write_text(json.dumps({
            "items": {str(sid): {"offset": 0} for sid in JOINT_IDS},
        }), "utf-8")
        self.module._calib = _load_json(self.module.calib_path, {})
        self.assertTrue(self.module.state()["servos"]["calibration"]["complete"])


if __name__ == "__main__":
    unittest.main()
