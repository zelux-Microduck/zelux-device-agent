"""技能库与"技能优先"链路的回归测试。

为什么这些点必须测：

* **命中技能时不许再问大模型**。这是 2 秒指标的根，一旦被改坏（比如有人把 LLM 放到
  匹配前面），产品指标会静默退化，日志里看不出来。
* **一个技能坏了不能带崩技能库**。技能是第三方的，写错 JSON、抛异常是常态。
* **参数缺失要问，不要瞎猜**。"查天气"没给城市就跑 handler，只会得到北京天气然后
  用户以为机器人听错了。
* **卸载后不许再命中**。卸载只是移目录，注册表忘删就会一直命中一个不存在的技能。

跑法：`cd software && python -m zeluxd.tests.test_skills`
"""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from zeluxd.errors import ZeluxError
from zeluxd.modules.ai import Module

REAL_SKILLS = Path(__file__).resolve().parents[2] / "skills"
BUILTIN = {"datetime", "weather", "express", "music", "robot_action", "timer", "system_status"}


# ---------------------------------------------------------------- 假实现
class FakeConfig:
    def __init__(self, root: Path, skills_dir: Path | None = None) -> None:
        self.root = root
        self.state_dir = root / "var"
        self.skills_dir = skills_dir or (root / "skills")
        self.models_dir = self.state_dir / "models"
        self.shots_dir = self.state_dir / "shots"
        self.mode = "sim"
        self.dev = True

    def ensure_dirs(self) -> None:
        for d in (self.state_dir, self.skills_dir, self.models_dir, self.shots_dir):
            d.mkdir(parents=True, exist_ok=True)


class FakeHal:
    backend = "fake"

    def __init__(self) -> None:
        self.moves: list[tuple] = []
        self.poses: list[str] = []
        self.goals: list[dict] = []
        self.said: list[str] = []
        self.relaxed = 0

    async def move(self, vx: float, vy: float, wz: float) -> None:
        self.moves.append((vx, vy, wz))

    async def pose(self, name: str) -> None:
        self.poses.append(name)

    async def servo_read(self, sid: int) -> dict:
        return {"id": sid, "pos": 2048}

    async def servo_torque(self, on: bool, ids=None) -> None:
        self.goals.append({"torque": on, "ids": ids})

    async def servo_sync_goal(self, goals: dict) -> None:
        self.goals.append(dict(goals))

    async def speak(self, text: str) -> bool:
        self.said.append(text)
        return True

    async def relax(self) -> dict:
        self.relaxed += 1
        return {"ok": True}


class FakeHub:
    def __init__(self) -> None:
        self.estop = {"state": "ready", "why": ""}
        self.posture = "holding"
        self.events: list[tuple[str, dict]] = []
        self.logs: list[tuple[str, str]] = []
        self.started_at = time.time()

    def emit_event(self, name: str, data: dict) -> None:
        self.events.append((name, data))

    def log_line(self, msg: str, level: str = "info", cat: str = "zeluxd", **kw) -> None:
        self.logs.append((level, msg))

    def snapshot(self) -> dict:
        return {
            "ts": time.time(),
            "mode": "commissioning",
            "posture": self.posture,
            "estop": dict(self.estop),
            "battery": {"pct": 78, "v": 7.85},
            "cpu": {"temp": 47.2, "load": 0.31},
            "tof": {"state": "ready", "nearest_mm": 210},
            "fan": {"pct": 60, "mode": "auto"},
        }

    def names(self) -> list[str]:
        return [n for n, _ in self.events]


class FakeLLM:
    """只记账：谁在什么时候被调了几次，比断言"回复内容"更能防回归。"""

    def __init__(self, reply: str = "我是一个桌面机器人。你想让我做什么？") -> None:
        self.reply = reply
        self.calls = 0
        self.messages: list[list[dict]] = []

    async def stream(self, messages, on_delta=None, timeout: float = 40.0,
                     model=None, max_tokens=None) -> dict:
        self.calls += 1
        self.messages.append(messages)
        text = self.reply
        for i in range(0, len(text), 4):
            if on_delta:
                res = on_delta(text[i:i + 4])
                if asyncio.iscoroutine(res):
                    await res
        return {"text": text, "first_ms": 6, "ms": 24}

    async def complete(self, messages, timeout: float = 20.0, model=None,
                       max_tokens=None) -> dict:
        self.calls += 1
        return {"text": self.reply, "ms": 9}

    def ensure_ready(self) -> None:
        return None


def make_module(root: Path, skills_dir: Path | None = None):
    cfg = FakeConfig(root, skills_dir)
    cfg.ensure_dirs()
    hal = FakeHal()
    hub = FakeHub()
    ctx = SimpleNamespace(config=cfg, hal=hal, hub=hub,
                          log=logging.getLogger("zeluxd.test"), state={}, version="test")
    return Module(ctx), hal, hub


def write_skill(root: Path, sid: str, meta: dict, code: str) -> Path:
    d = root / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "skill.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    (d / "handler.py").write_text(code, "utf-8")
    return d


def meta(sid: str, triggers: list[str], priority: int = 50, params: list | None = None) -> dict:
    return {"id": sid, "name": sid, "version": "0.0.1", "author": "test",
            "description": f"{sid} 测试技能", "triggers": triggers,
            "params": params or [], "permissions": [], "needs": [], "priority": priority}


ECHO_CODE = '''\
def run(ctx, text, args):
    return {"speech": "命中 echo", "data": dict(args), "display": {}}
'''


class Base(unittest.IsolatedAsyncioTestCase):
    skills_dir: Path | None = None

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.mod, self.hal, self.hub = make_module(self.root, self.skills_dir)
        await self.mod.start()
        self.mod.llm = FakeLLM()

    async def asyncTearDown(self) -> None:
        await self.mod.stop()
        await asyncio.sleep(0)
        self._tmp.cleanup()


# ---------------------------------------------------------------- 内置技能
class BuiltinSkillsTest(Base):
    skills_dir = REAL_SKILLS

    async def test_builtin_seven_all_load(self) -> None:
        """开箱可用：7 个内置技能都要能加载，不能有任何一个报错。"""
        ids = {item["id"] for item in self.mod.registry.list()}
        self.assertEqual(ids, BUILTIN)
        self.assertEqual(self.mod.registry.errors, {})
        self.assertEqual(self.mod.registry.count(), 7)

    async def test_datetime_hits_without_llm(self) -> None:
        res = await self.mod.handle_text("现在几点了")
        self.assertEqual(res["source"], "skill")
        self.assertEqual(res["skill"], "datetime")
        self.assertEqual(self.mod.llm.calls, 0, "命中技能就不许再问大模型")
        self.assertIn("点", res["speech"])
        self.assertNotIn("ai.delta", self.hub.names())

    async def test_status_reads_snapshot_without_hardware(self) -> None:
        res = await self.mod.handle_text("还有多少电")
        self.assertEqual(res["source"], "skill")
        self.assertIn("78", res["speech"])
        self.assertEqual(self.hal.moves, [])
        self.assertEqual(self.hal.poses, [])

    async def test_robot_action_priority_wins(self) -> None:
        """动作类优先级最高：不能被闲聊类抢走。"""
        res = await self.mod.handle_text("点个头")
        self.assertEqual(res["skill"], "robot_action")
        self.assertEqual(self.mod.llm.calls, 0)

    async def test_robot_action_reaches_hal(self) -> None:
        """技能跑在 to_thread 里，动作必须能投递回主循环。

        这里曾经真的坏过：线程里 asyncio.get_event_loop() 拿不到循环，
        动作被静默丢弃，用户只听到"好，点个头"然后什么也没发生。
        """
        await self.mod.handle_text("点个头")
        for _ in range(40):
            if self.hal.goals:
                break
            await asyncio.sleep(0.05)
        self.assertTrue(self.hal.goals, "动作没到 HAL：ctx→事件循环 的线程安全入口断了")
        self.assertEqual(self.hal.goals[0].get("torque"), True, "点头前要先给颈部上力")

    async def test_latency_breakdown_recorded(self) -> None:
        await self.mod.handle_text("现在几点")
        st = self.mod.state()["ai"]
        bd = st["latency_breakdown"]
        for key in ("asr", "skill", "llm_first", "tts_first", "total"):
            self.assertIn(key, bd)
        self.assertIsNotNone(st["latency_ms"])
        self.assertIn("ai.latency", self.hub.names())

    async def test_no_skill_falls_to_llm(self) -> None:
        res = await self.mod.handle_text("你叫什么名字")
        self.assertEqual(res["source"], "llm")
        self.assertEqual(self.mod.llm.calls, 1)
        self.assertIn("机器人", res["speech"])
        self.assertIn("ai.reply", self.hub.names())

    async def test_system_prompt_contains_skill_ids(self) -> None:
        """兜底那一路必须知道有哪些技能，否则永远不会主动调技能。"""
        await self.mod.handle_text("你叫什么名字")
        system = self.mod.llm.messages[0][0]["content"]
        for sid in BUILTIN:
            self.assertIn(sid, system)

    async def test_llm_may_pick_a_skill(self) -> None:
        """正则没命中时，大模型可以用一行 JSON 指定技能，不必自己编答案。"""
        self.mod.llm = FakeLLM('{"skill":"datetime","args":{}}')
        res = await self.mod.handle_text("你觉得现在该干嘛")
        self.assertEqual(res["source"], "skill")
        self.assertEqual(res["skill"], "datetime")

    async def test_unknown_skill_json_is_not_read_aloud(self) -> None:
        """技能名对不上时不能把 {"skill":...} 念给用户听。"""
        self.mod.llm = FakeLLM('{"skill":"ghost","args":{}}')
        res = await self.mod.handle_text("随便问点")
        self.assertEqual(res["source"], "llm")
        self.assertNotIn("{", res["speech"])
        self.assertIn("不会", res["speech"])

    async def test_api_key_masked_everywhere(self) -> None:
        out = await self.mod.cmd("config", {"api_key": "sk-real-secret",
                                            "base_url": "https://example.com/v3",
                                            "model": "demo-model"})
        self.assertEqual(out["api_key"], "sk-***")
        self.assertTrue(out["has_key"])

        raw = (self.mod.config.state_dir / "ai.json").read_text("utf-8")
        self.assertIn("sk-real-secret", raw, "本地文件里要是真的 key，否则重启就用不了")

        self.assertNotIn("sk-real-secret", json.dumps(self.mod.state(), ensure_ascii=False))
        self.assertNotIn("sk-real-secret", json.dumps(self.hub.logs, ensure_ascii=False))
        self.assertNotIn("sk-real-secret", json.dumps(out, ensure_ascii=False))

    async def test_config_keeps_key_when_masked_value_posted_back(self) -> None:
        """前端把打码值原样提交回来时不能把真 key 覆盖成 sk-***。"""
        await self.mod.cmd("config", {"api_key": "sk-one"})
        await self.mod.cmd("config", {"api_key": "sk-***", "model": "m2"})
        self.assertEqual(self.mod.cfg["api_key"], "sk-one")
        self.assertEqual(self.mod.cfg["model"], "m2")

    async def test_nested_keys_masked_and_restorable(self) -> None:
        """云端 ASR 的 key、快递 100 的 key 也是凭据，同样打码、同样能还原。"""
        out = await self.mod.cmd("config", {
            "skills": {"express": {"key": "kd-secret", "customer": "cust"}},
            "asr": {"cloud": {"api_key": "asr-secret", "base_url": "https://x", "model": "w"}},
        })
        self.assertEqual(out["skills"]["express"]["key"], "sk-***")
        self.assertEqual(out["asr"]["cloud"]["api_key"], "sk-***")

        raw = (self.mod.config.state_dir / "ai.json").read_text("utf-8")
        self.assertIn("kd-secret", raw)
        self.assertIn("asr-secret", raw)

        await self.mod.cmd("config", {"skills": out["skills"], "asr": out["asr"]})
        self.assertEqual(self.mod.cfg["skills"]["express"]["key"], "kd-secret")
        self.assertEqual(self.mod.cfg["asr"]["cloud"]["api_key"], "asr-secret")

    async def test_toggle_off_stops_matching(self) -> None:
        await self.mod.cmd("skill_toggle", {"id": "datetime", "on": False})
        res = await self.mod.handle_text("现在几点")
        self.assertEqual(res["source"], "llm")
        self.assertEqual(self.mod.llm.calls, 1)
        self.assertIn("datetime", self.mod.cfg["disabled_skills"])

        await self.mod.cmd("skill_toggle", {"id": "datetime", "on": True})
        res = await self.mod.handle_text("现在几点")
        self.assertEqual(res["source"], "skill")

    async def test_skills_list_reports_metadata(self) -> None:
        out = await self.mod.cmd("skills", {})
        self.assertEqual(out["count"], 7)
        item = next(i for i in out["items"] if i["id"] == "weather")
        self.assertTrue(item["enabled"])
        self.assertIn("network", item["needs"])
        self.assertTrue(item["triggers"])
        # 先按 priority 降序
        priorities = [i["priority"] for i in out["items"]]
        self.assertEqual(priorities, sorted(priorities, reverse=True))


# ---------------------------------------------------------------- 加载器容错
class LoaderTest(Base):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        skills = self.root / "skills"
        skills.mkdir(parents=True)
        write_skill(skills, "good", meta("good", ["打个招呼"]), ECHO_CODE)

        bad = skills / "badjson"
        bad.mkdir()
        (bad / "skill.json").write_text('{"id": "badjson", ', "utf-8")
        (bad / "handler.py").write_text(ECHO_CODE, "utf-8")

        nohandler = skills / "nohandler"
        nohandler.mkdir()
        (nohandler / "skill.json").write_text(
            json.dumps(meta("nohandler", ["你好"]), ensure_ascii=False), "utf-8")

        wrongid = skills / "wrongid"
        wrongid.mkdir()
        (wrongid / "skill.json").write_text(
            json.dumps(meta("other", ["喂"]), ensure_ascii=False), "utf-8")
        (wrongid / "handler.py").write_text(ECHO_CODE, "utf-8")

        self.mod, self.hal, self.hub = make_module(self.root, skills)
        await self.mod.start()
        self.mod.llm = FakeLLM()

    async def asyncTearDown(self) -> None:
        await self.mod.stop()
        await asyncio.sleep(0)
        self._tmp.cleanup()

    async def test_good_skill_loads_despite_broken_neighbours(self) -> None:
        ids = {s["id"] for s in self.mod.registry.list()}
        self.assertEqual(ids, {"good"})
        self.assertIn("badjson", self.mod.registry.errors)
        self.assertIn("nohandler", self.mod.registry.errors)
        self.assertIn("wrongid", self.mod.registry.errors)
        self.assertIn("JSON", self.mod.registry.errors["badjson"])
        self.assertIn("handler.py", self.mod.registry.errors["nohandler"])
        self.assertIn("一致", self.mod.registry.errors["wrongid"])

    async def test_state_counts_failures(self) -> None:
        st = self.mod.state()["ai"]
        self.assertEqual(st["skills"], 1)
        self.assertEqual(st["skills_failed"], 3)

    async def test_good_skill_still_matches(self) -> None:
        res = await self.mod.handle_text("打个招呼")
        self.assertEqual(res["source"], "skill")
        self.assertEqual(res["skill"], "good")


# ---------------------------------------------------------------- 优先级 / 参数
class PriorityAndParamsTest(Base):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        skills = self.root / "skills"
        skills.mkdir(parents=True)
        write_skill(skills, "low", meta("low", ["优先级测试"], priority=10),
                    'def run(ctx, text, args):\n    return {"speech": "low"}\n')
        write_skill(skills, "high", meta("high", ["优先级测试"], priority=900),
                    'def run(ctx, text, args):\n    return {"speech": "high"}\n')
        write_skill(skills, "city", meta(
            "city", ["查天气"], priority=800,
            params=[{"name": "city", "type": "str", "required": True,
                     "description": "城市", "ask": "哪个城市"}]),
            'def run(ctx, text, args):\n    return {"speech": "城市 " + str(args["city"])}\n')
        write_skill(skills, "int_param", meta(
            "int_param", ["亮度"], priority=700,
            params=[{"name": "level", "type": "int", "required": False, "default": 30,
                     "pattern": "(\\d{1,3})"}]),
            'def run(ctx, text, args):\n    return {"speech": "亮度 %d" % args["level"]}\n')
        self.mod, self.hal, self.hub = make_module(self.root, skills)
        await self.mod.start()
        self.mod.llm = FakeLLM()

    async def asyncTearDown(self) -> None:
        await self.mod.stop()
        await asyncio.sleep(0)
        self._tmp.cleanup()

    async def test_high_priority_wins(self) -> None:
        res = await self.mod.handle_text("优先级测试")
        self.assertEqual(res["skill"], "high")
        self.assertEqual(self.mod.llm.calls, 0)

    async def test_missing_required_param_asks_and_skips_handler(self) -> None:
        res = await self.mod.handle_text("查天气")
        self.assertEqual(res["source"], "skill")
        self.assertIn("哪个城市", res["speech"])
        self.assertNotIn("城市 ", res["speech"])
        self.assertTrue(res.get("latency"))

    async def test_param_from_regex_pattern(self) -> None:
        res = await self.mod.handle_text("把亮度调到 80")
        self.assertIn("亮度 80", res["speech"])

    async def test_param_default_used_when_absent(self) -> None:
        res = await self.mod.handle_text("调亮度")
        self.assertIn("亮度 30", res["speech"])

    async def test_disabled_skill_falls_through_to_llm(self) -> None:
        await self.mod.cmd("skill_toggle", {"id": "high", "on": False})
        res = await self.mod.handle_text("优先级测试")
        self.assertEqual(res["source"], "skill")
        self.assertEqual(res["skill"], "low", "停用高优先级的，应该轮到低优先级的")


# ---------------------------------------------------------------- 异常隔离 / 卸载
class IsolationTest(Base):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        skills = self.root / "skills"
        skills.mkdir(parents=True)
        write_skill(skills, "boom", meta("boom", ["炸一下"], priority=600),
                    'def run(ctx, text, args):\n'
                    '    raise RuntimeError("故意炸的")\n')
        write_skill(skills, "speechless", meta("speechless", ["不说话"], priority=600),
                    'def run(ctx, text, args):\n    return {"data": {}}\n')
        write_skill(skills, "fine", meta("fine", ["还好吗"], priority=600),
                    'def run(ctx, text, args):\n    return {"speech": "还好"}\n')
        self.mod, self.hal, self.hub = make_module(self.root, skills)
        await self.mod.start()
        self.mod.llm = FakeLLM()

    async def asyncTearDown(self) -> None:
        await self.mod.stop()
        await asyncio.sleep(0)
        self._tmp.cleanup()

    async def test_exception_isolated(self) -> None:
        res = await self.mod.handle_text("炸一下")
        self.assertEqual(res["skill"], "boom")
        self.assertIn("出错", res["speech"])
        self.assertEqual(self.mod.registry.get("boom").last_error.split(":")[0], "RuntimeError")

    async def test_other_skills_still_work_after_failure(self) -> None:
        await self.mod.handle_text("炸一下")
        res = await self.mod.handle_text("还好吗")
        self.assertEqual(res["speech"], "还好")

    async def test_empty_speech_rejected(self) -> None:
        res = await self.mod.handle_text("不说话")
        self.assertIn("没说话", res["speech"])

    async def test_uninstall_stops_matching(self) -> None:
        out = await self.mod.cmd("skill_uninstall", {"id": "fine"})
        self.assertTrue(out["ok"])
        self.assertTrue(Path(out["moved_to"]).exists(), "卸载要可回滚：目录移到回收站")

        res = await self.mod.handle_text("还好吗")
        self.assertEqual(res["source"], "llm")
        self.assertIsNone(self.mod.registry.get("fine"))

        with self.assertRaises(ZeluxError):
            await self.mod.cmd("skill_uninstall", {"id": "fine"})


# ---------------------------------------------------------------- zip 安装
class InstallTest(Base):
    async def test_install_zip_then_uninstall(self) -> None:
        zpath = self.root / "demo.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("demo/skill.json", json.dumps(
                meta("demo", ["演示一下"]), ensure_ascii=False))
            zf.writestr("demo/handler.py",
                        'def run(ctx, text, args):\n    return {"speech": "装好了"}\n')

        out = await self.mod.cmd("skill_install", {"zip_path": str(zpath)})
        self.assertTrue(out["ok"])
        self.assertEqual(out["id"], "demo")
        self.assertEqual(out["skills"], 1)

        res = await self.mod.handle_text("演示一下")
        self.assertEqual(res["speech"], "装好了")

        # 重装要拒绝，除非明确 force
        with self.assertRaises(ZeluxError):
            await self.mod.cmd("skill_install", {"zip_path": str(zpath)})
        out2 = await self.mod.cmd("skill_install", {"zip_path": str(zpath), "force": True})
        self.assertTrue(out2["ok"])

    async def test_zip_slip_rejected(self) -> None:
        zpath = self.root / "evil.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("demo/skill.json", json.dumps(
                meta("demo", ["演示一下"]), ensure_ascii=False))
            zf.writestr("demo/handler.py", ECHO_CODE)
            zf.writestr("demo/../../etc/passwd", "x")
        with self.assertRaises(ZeluxError):
            await self.mod.cmd("skill_install", {"zip_path": str(zpath)})
        self.assertIsNone(self.mod.registry.get("demo"))

    async def test_bad_zip_rejected(self) -> None:
        zpath = self.root / "empty.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("readme.txt", "no skill here")
        with self.assertRaises(ZeluxError):
            await self.mod.cmd("skill_install", {"zip_path": str(zpath)})


# ---------------------------------------------------------------- 超预算提示
class BudgetHintTest(unittest.TestCase):
    def test_hint_points_at_slowest_stage(self) -> None:
        from zeluxd.modules.ai import _budget_hint

        self.assertIn("首字", _budget_hint({"llm_first": 1500, "skill": 0,
                                            "asr": 0, "tts_first": 1500}))
        self.assertIn("技能", _budget_hint({"llm_first": 0, "skill": 1400,
                                            "asr": 0, "tts_first": 1500}))
        self.assertIn("ASR", _budget_hint({"llm_first": 0, "skill": 0,
                                           "asr": 1200, "tts_first": 1500}))
        self.assertIn("piper", _budget_hint({"llm_first": 0, "skill": 0,
                                             "asr": 0, "tts_first": 1400}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
