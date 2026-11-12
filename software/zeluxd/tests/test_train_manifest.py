"""策略清单（policy.json）校验的三条红线。

为什么只测这个：录制/回放/打包错了，用户顶多白做一次；
清单错了（维度不够、关节顺序错位），动作会按"别人的关节表"发到 15 颗舵机上，鸭子当场摔。
所以这段校验是整套东西里唯一"必须硬拒且必须指到第几项"的地方。

跑法：`cd software && python -m zeluxd.tests.test_train_manifest`
"""
from __future__ import annotations

import copy
import unittest

from zeluxd.errors import ZeluxError
from zeluxd.hal.base import JOINT_NAMES
from zeluxd.modules.train import validate_manifest


def good_manifest() -> dict:
    """一份真能被 motiond 吃下去的清单（任务名照参考产品写）。"""
    return {
        "task": "Mjlab-VelStand-Rough-Backlash-MicroDuck",
        "obs_dim": 45,
        "act_dim": 15,
        "joint_order": list(JOINT_NAMES),
        "action_scale": [0.25] * len(JOINT_NAMES),
        "action_offset": [0.0] * len(JOINT_NAMES),
        "control_hz": 50,
        "notes": "走路策略，带 3 cm 台阶与齿隙扰动训练",
    }


class ManifestTest(unittest.TestCase):
    # ---------------------------------------------------------------- 工具
    def assertDenied(self, manifest, *fragments) -> ZeluxError:
        with self.assertRaises(ZeluxError) as ctx:
            validate_manifest(manifest)
        err = ctx.exception
        self.assertEqual(err.code, "denied", f"应该是 denied，实际 {err.code}：{err.message}")
        for frag in fragments:
            self.assertIn(frag, err.message, f"错误信息要指出问题，缺 {frag!r}：{err.message}")
        return err

    # ---------------------------------------------------------------- 1. 通过
    def test_manifest_ok(self) -> None:
        out = validate_manifest(good_manifest())
        self.assertEqual(out["task"], "Mjlab-VelStand-Rough-Backlash-MicroDuck")
        self.assertEqual(out["obs_dim"], 45)
        self.assertEqual(out["act_dim"], 15)
        self.assertEqual(out["joint_order"], JOINT_NAMES)
        self.assertEqual(len(out["action_scale"]), 15)
        self.assertIsInstance(out["control_hz"], float)
        self.assertEqual(out["control_hz"], 50.0)

    def test_manifest_ok_with_extra_dims(self) -> None:
        """观测比 15 多（含 IMU / 上一帧动作）是常态，必须放行。"""
        m = good_manifest()
        m["obs_dim"] = 60
        m["act_dim"] = 16
        out = validate_manifest(m)
        self.assertEqual((out["obs_dim"], out["act_dim"]), (60, 16))

    def test_manifest_minimal_notes(self) -> None:
        """notes 可以不写，但不能因此整份清单被判死。"""
        m = good_manifest()
        m.pop("notes")
        self.assertEqual(validate_manifest(m)["notes"], "")

    # ---------------------------------------------------------------- 2. 维度错
    def test_obs_dim_too_small(self) -> None:
        m = good_manifest()
        m["obs_dim"] = 12
        self.assertDenied(m, "第 2 项", "obs_dim", "≥15")

    def test_act_dim_missing(self) -> None:
        m = good_manifest()
        m.pop("act_dim")
        self.assertDenied(m, "第 3 项", "act_dim")

    def test_act_dim_is_string(self) -> None:
        """JSON 里写成 "15" 的太多了，必须当错处理，不能悄悄转 int。"""
        m = good_manifest()
        m["act_dim"] = "15"
        self.assertDenied(m, "第 3 项", "act_dim")

    def test_action_scale_length(self) -> None:
        m = good_manifest()
        m["action_scale"] = [0.25] * 14
        self.assertDenied(m, "第 5 项", "action_scale", "14")

    # ---------------------------------------------------------------- 3. 关节顺序错
    def test_joint_order_swapped(self) -> None:
        m = good_manifest()
        order = list(m["joint_order"])
        order[3], order[4] = order[4], order[3]          # r_knee ↔ r_ankle
        m["joint_order"] = order
        err = self.assertDenied(m, "第 4 项", "joint_order", "第 4 个关节")
        self.assertIn("r_ankle", err.message)

    def test_joint_order_sorted_alphabetically(self) -> None:
        """最常见的坑：导出脚本顺手 sorted() 了一下，名字都对、顺序全错。"""
        m = good_manifest()
        m["joint_order"] = sorted(JOINT_NAMES)
        self.assertDenied(m, "第 4 项", "joint_order")

    def test_joint_order_wrong_count(self) -> None:
        m = good_manifest()
        m["joint_order"] = list(JOINT_NAMES)[:14]
        self.assertDenied(m, "第 4 项", "14")

    # ---------------------------------------------------------------- 其他硬条件
    def test_control_hz_out_of_range(self) -> None:
        for bad in (5, 19.9, 501):
            m = good_manifest()
            m["control_hz"] = bad
            self.assertDenied(m, "第 7 项", "control_hz")

    def test_not_a_dict(self) -> None:
        self.assertDenied([1, 2, 3], "policy.json")          # type: ignore[arg-type]

    def test_task_empty(self) -> None:
        m = good_manifest()
        m["task"] = "   "
        self.assertDenied(m, "第 1 项", "task")

    def test_original_manifest_not_mutated(self) -> None:
        """校验只读：调用方还要拿原清单做 diff 给人看。"""
        m = good_manifest()
        snapshot = copy.deepcopy(m)
        validate_manifest(m)
        self.assertEqual(m, snapshot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
