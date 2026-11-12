"""避障纯逻辑自测：不接 ToF、不开电机，本机（Windows/macOS）就能跑。

跑法：
    cd software && python -m zeluxd.tests.test_avoid

覆盖的是"撞不撞"这件事上最容易错、也最不该靠真机试的几条：
* 8×8 → 三扇区的分位计算（含去掉最上/最下一行）；
* 单个噪点不许把鸭子刹住（这就是不用最小值的原因）；
* stop → clear 必须连续 5 帧（防"抖着抖着往前冲"）；
* 左堵右空 → 向右转（wz 符号必须对，反了就是往墙上撞）；
* 急停已卸力时一帧速度都不许下发；
* 参数越界要夹紧（-5 mm 的停车线会把鸭子钉死）。

顺带锁一条视觉跟随的符号约定（目标在画面右边 → wz 必须为负），
这种"反了就朝反方向追"的错误，真机上要撞一次墙才能发现。
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import types
import unittest
from pathlib import Path

from zeluxd.modules.tof import (CRUISE_VX, TURN_WZ, AvoidFilter, Module, decision,
                                estop_blocks, normalize_cfg, percentile, sector_depths)
from zeluxd.modules.vision import follow_control

COLS = ((0, 1, 2), (3, 4), (5, 6, 7))
ROWS = (1, 2, 3, 4, 5, 6)


# ---------------------------------------------------------------- 造帧
def make_frame(*, l: int | None = None, c: int | None = None, r: int | None = None,
               fill: int = 1500, seq: int = 1) -> dict:
    """按扇区造一帧 8×8：给 None 的扇区保持 fill，给 0 表示无效格。"""
    dist = [fill] * 64
    stat = [5] * 64
    for key, cols in zip(("l", "c", "r"), COLS):
        value = {"l": l, "c": c, "r": r}[key]
        if value is None:
            continue
        for row in ROWS:
            for col in cols:
                i = row * 8 + col
                if value <= 0:
                    dist[i], stat[i] = 0, 255
                else:
                    dist[i], stat[i] = value, 5
    return {"state": "ready", "seq": seq, "distance_mm": dist, "status": stat}


class FakeHal:
    def __init__(self, snap: dict) -> None:
        self.snap = snap
        self.moves: list[tuple[float, float, float]] = []

    def snapshot_tof(self) -> dict:
        return dict(self.snap)

    async def move(self, vx: float, vy: float, wz: float) -> None:
        self.moves.append((vx, vy, wz))


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make_module(self, snap: dict, estop: str = "ready"):
        hal = FakeHal(snap)
        cfg = types.SimpleNamespace(state_dir=self.state_dir, mode="sim",
                                    models_dir=self.state_dir / "models",
                                    shots_dir=self.state_dir / "shots")
        hub = types.SimpleNamespace(estop={"state": estop, "why": ""}, modules={},
                                    emit_event=lambda *a, **k: None,
                                    log_line=lambda *a, **k: None)
        ctx = types.SimpleNamespace(config=cfg, hal=hal, hub=hub, state={},
                                    log=logging.getLogger("zeluxd.test.tof"))
        return Module(ctx), hal


# ---------------------------------------------------------------- 三扇区
class TestSectors(Base):
    def test_percentile_nearest_rank(self):
        self.assertEqual(percentile([10, 20, 30, 40, 50], 0.25), 20)
        self.assertEqual(percentile([7], 0.25), 7)
        self.assertIsNone(percentile([], 0.25))

    def test_sector_quarter_position(self):
        """18 格按行优先填 100..1800：25% 分位 = 第 5 小的 = 500 mm。"""
        frame = make_frame(l=0, c=1200, r=2500, fill=1500)
        n = 0
        for row in ROWS:
            for col in COLS[0]:
                i = row * 8 + col
                frame["distance_mm"][i] = 100 * (n + 1)
                frame["status"][i] = 5
                n += 1
        depths = sector_depths(frame["distance_mm"], frame["status"])
        self.assertEqual(n, 18)
        self.assertEqual(depths["l"], 500, "25% 分位算错了：应取第 5 小的值")
        self.assertEqual(depths["c"], 1200)
        self.assertEqual(depths["r"], 2500)

    def test_noise_rows_skipped(self):
        """最上/最下一行常年打到桌沿和鸭脚：这 16 格必须被排除。"""
        frame = make_frame(fill=1600)
        for col in range(8):
            for row in (0, 7):
                i = row * 8 + col
                frame["distance_mm"][i], frame["status"][i] = 40, 5
        depths = sector_depths(frame["distance_mm"], frame["status"])
        self.assertEqual(depths["c"], 1600)
        self.assertNotEqual(depths["c"], 40)

    def test_invalid_status_ignored(self):
        """status=255 的格子不能当成 0 mm。"""
        frame = make_frame(c=800)
        for row in ROWS:
            for col in COLS[1]:
                i = row * 8 + col
                frame["distance_mm"][i], frame["status"][i] = 0, 255
        frame["status"][1 * 8 + 3] = 255          # 只留一格无效
        depths = sector_depths(frame["distance_mm"], frame["status"])
        self.assertIsNone(depths["c"])
        self.assertEqual(decision(depths)["action"], "stop", "看不见就必须停")


# ---------------------------------------------------------------- 噪声
class TestNoise(Base):
    def test_single_spike_does_not_stop(self):
        """中路 1 格 30 mm 的噪点：25% 分位仍是 1500，不许刹车。"""
        frame = make_frame(fill=1500)
        frame["distance_mm"][3 * 8 + 3] = 30
        depths = sector_depths(frame["distance_mm"], frame["status"])
        self.assertGreater(depths["c"], 220)
        self.assertEqual(decision(depths)["action"], "clear")

    def test_real_wall_does_stop(self):
        """真的贴脸不是噪声：大多数格子都近，就该停。"""
        frame = make_frame(c=150)
        depths = sector_depths(frame["distance_mm"], frame["status"])
        self.assertEqual(decision(depths)["action"], "stop")

    def test_no_reading_stops(self):
        frame = make_frame(l=0, c=0, r=0)
        depths = sector_depths(frame["distance_mm"], frame["status"])
        self.assertEqual(decision(depths)["action"], "stop")


# ---------------------------------------------------------------- 决策
class TestDecision(Base):
    def test_clear_cruises(self):
        d = decision(sector_depths(*_arrays(make_frame(fill=1800))))
        self.assertEqual(d["action"], "clear")
        self.assertAlmostEqual(d["vx"], CRUISE_VX, places=3)

    def test_slow_when_narrow_but_no_side_room(self):
        """中扇区进减速线、两侧也都窄：只能减速直行，不许瞎转。"""
        frame = make_frame(l=500, c=400, r=450)
        d = decision(sector_depths(frame["distance_mm"], frame["status"]))
        self.assertEqual(d["action"], "slow")
        self.assertAlmostEqual(d["vx"], round(CRUISE_VX * 0.4, 3), places=3)
        self.assertEqual(d["wz"], 0.0)

    def test_turn_right_when_left_blocked(self):
        """左堵右空：向右转 = wz 为负（ROS 约定右侧为负角速度）。"""
        frame = make_frame(l=300, c=420, r=2600)
        d = decision(sector_depths(frame["distance_mm"], frame["status"]))
        self.assertEqual(d["action"], "turn")
        self.assertLess(d["wz"], 0)
        self.assertAlmostEqual(d["wz"], -TURN_WZ, places=3)
        self.assertAlmostEqual(d["vx"], round(CRUISE_VX * 0.4, 3), places=3)
        self.assertIn("向右", d["reason"])

    def test_turn_left_when_right_blocked(self):
        frame = make_frame(l=2600, c=420, r=300)
        d = decision(sector_depths(frame["distance_mm"], frame["status"]))
        self.assertEqual(d["action"], "turn")
        self.assertAlmostEqual(d["wz"], TURN_WZ, places=3)

    def test_hold_mode_does_not_creep(self):
        cfg = normalize_cfg({"action": "hold"})
        d = decision(sector_depths(*_arrays(make_frame(fill=2000))), cfg)
        self.assertEqual(d["action"], "clear")
        self.assertEqual(d["vx"], 0.0)

    def test_stop_is_immediate(self):
        """往保守方向不需要防抖：晚 3 帧就是多走 60 cm。"""
        f = AvoidFilter("clear", CRUISE_VX, 0.0)
        out = f.update(decision(sector_depths(*_arrays(make_frame(c=150)))))
        self.assertEqual(out["action"], "stop")


def _arrays(frame: dict) -> tuple[list, list]:
    return frame["distance_mm"], frame["status"]


# ---------------------------------------------------------------- 防抖
class TestFilter(Base):
    def test_recover_from_stop_needs_five_frames(self):
        f = AvoidFilter("stop")
        clear = decision(sector_depths(*_arrays(make_frame(fill=1800))))
        for i in range(4):
            out = f.update(clear)
            self.assertEqual(out["action"], "stop", f"第 {i + 1} 帧就恢复了，太早")
            self.assertEqual(out["pending"], "clear")
        out = f.update(clear)
        self.assertEqual(out["action"], "clear", "连续 5 帧 clear 后必须恢复")

    def test_switch_needs_three_frames(self):
        f = AvoidFilter("clear", CRUISE_VX, 0.0)
        slow = decision(sector_depths(*_arrays(make_frame(l=500, c=400, r=450))))
        for _ in range(2):
            self.assertEqual(f.update(slow)["action"], "clear")
        out = f.update(slow)
        self.assertEqual(out["action"], "slow")
        self.assertTrue(out["settled"])

    def test_pending_frame_keeps_previous_velocity(self):
        """没攒够帧时，下发的仍是上一个动作的速度，不是新动作的。"""
        f = AvoidFilter("clear", CRUISE_VX, 0.0)
        slow = decision(sector_depths(*_arrays(make_frame(l=500, c=400, r=450))))
        out = f.update(slow)
        self.assertEqual(out["action"], "clear")
        self.assertAlmostEqual(out["vx"], CRUISE_VX, places=3)
        self.assertAlmostEqual(out["wz"], 0.0, places=3)

    def test_recover_to_slow_from_stop_not_fast(self):
        """stop → slow 只按普通 3 帧，不必等 5 帧（5 帧只针对 clear）。"""
        f = AvoidFilter("stop")
        slow = decision(sector_depths(*_arrays(make_frame(l=500, c=400, r=450))))
        self.assertEqual(f.update(slow)["action"], "stop")
        self.assertEqual(f.update(slow)["action"], "stop")
        self.assertEqual(f.update(slow)["action"], "slow")


# ---------------------------------------------------------------- 急停
class TestEstop(Base):
    SNAP = make_frame(fill=1800)

    def test_estop_confirmed_sends_nothing(self):
        mod, hal = self.make_module(self.SNAP, estop="confirmed")
        d = asyncio.run(mod._frame_step(self.SNAP))
        self.assertEqual(hal.moves, [], "急停卸力后不许下发任何速度")
        self.assertEqual(d["action"], "clear", "决策照算，只是不下发")
        self.assertIn("急停", mod.state()["tof"]["avoid"]["blocked"])
        self.assertTrue(estop_blocks({"state": "confirmed"}))
        self.assertFalse(estop_blocks({"state": "ready"}))
        self.assertFalse(estop_blocks(None))

    def test_estop_ready_drives(self):
        mod, hal = self.make_module(self.SNAP, estop="ready")
        asyncio.run(mod._frame_step(self.SNAP))
        self.assertEqual(len(hal.moves), 1)
        vx, vy, wz = hal.moves[0]
        self.assertAlmostEqual(vx, CRUISE_VX, places=3)
        self.assertEqual(vy, 0.0)
        self.assertEqual(wz, 0.0)

    def test_stop_sends_zero_three_times(self):
        mod, hal = self.make_module(make_frame(c=150), estop="ready")
        asyncio.run(mod._frame_step(make_frame(c=150)))
        self.assertEqual(len(hal.moves), 3, "停车要连发 3 次零速")
        self.assertTrue(all(m == (0.0, 0.0, 0.0) for m in hal.moves))

    def test_simulate_never_drives(self):
        mod, hal = self.make_module(make_frame(fill=1800), estop="ready")
        res = asyncio.run(mod.cmd("op", {"op": "simulate",
                                        "distances": make_frame(c=150)["distance_mm"]}))
        self.assertTrue(res["ok"])
        self.assertEqual(res["action"], "stop")
        self.assertEqual(hal.moves, [], "演示只算不动")


# ---------------------------------------------------------------- 参数
class TestConfig(Base):
    def test_clamps_out_of_range(self):
        cfg = normalize_cfg({"stop_mm": -5, "slow_mm": 99999, "side_mm": 10,
                             "action": "boom"})
        self.assertEqual(cfg["stop_mm"], 60.0)
        self.assertEqual(cfg["slow_mm"], 3000.0)
        self.assertEqual(cfg["side_mm"], 200.0)
        self.assertEqual(cfg["action"], "forward", "无法识别的动作要回默认值")

    def test_slow_must_be_above_stop(self):
        cfg = normalize_cfg({"stop_mm": 500, "slow_mm": 300})
        self.assertGreater(cfg["slow_mm"], cfg["stop_mm"], "减速线低于停车线等于关掉减速")

    def test_defaults(self):
        cfg = normalize_cfg(None)
        self.assertEqual(cfg, {"stop_mm": 220.0, "slow_mm": 600.0,
                               "side_mm": 900.0, "action": "forward"})

    def test_config_cmd_persists(self):
        mod, _hal = self.make_module(make_frame(fill=1500))
        res = asyncio.run(mod.cmd("op", {"op": "config", "stop_mm": 300, "slow_mm": 900}))
        self.assertTrue(res["ok"])
        self.assertEqual(res["config"]["stop_mm"], 300.0)
        saved = (self.state_dir / "tof.json")
        self.assertTrue(saved.exists(), "参数要能记住，重启后不该回到默认值")


# ---------------------------------------------------------------- 跟随符号
class TestFollowControl(unittest.TestCase):
    def test_dead_zone_keeps_straight(self):
        self.assertEqual(follow_control(0.53, 0.30)["wz"], 0.0)
        self.assertEqual(follow_control(0.47, 0.30)["wz"], 0.0)

    def test_target_on_right_turns_right(self):
        """目标在画面右边 → 右转 → wz < 0。反了就是朝反方向追。"""
        self.assertLess(follow_control(0.85, 0.30)["wz"], 0)
        self.assertGreater(follow_control(0.15, 0.30)["wz"], 0)

    def test_angular_speed_is_capped(self):
        for cx in (0.0, 0.02, 1.0, 0.98):
            self.assertLessEqual(abs(follow_control(cx, 0.30)["wz"]), 1.2)

    def test_distance_bands(self):
        self.assertGreater(follow_control(0.5, 0.10)["vx"], 0)     # 远 → 前进
        self.assertLess(follow_control(0.5, 0.90)["vx"], 0)        # 近 → 后退
        self.assertEqual(follow_control(0.5, 0.40)["vx"], 0.0)     # 适中 → 只转不走


if __name__ == "__main__":
    unittest.main(verbosity=2)
