"""语音与对话模块：**技能优先，大模型兜底**。

为什么这样分层（2 秒指标的来源）
--------------------------------
RK3566 是 4 核 A55、没有 NPU，本地跑不了像样的语言模型；云端一次往返 600–1200 ms，
再加 ASR 与 TTS，什么话都丢给大模型必然破 2 s。所以：

1. **能确定答案的话不发给大模型**。时间、日期、计时、电量、动作、音乐、天气、快递
   全部做成技能，正则命中即执行（0 次 LLM 往返，本地 1–5 ms 出答案）。
2. 命中不了的才走 LLM，而且**流式**：第一句话一凑齐就送去 TTS。
   用户感知的是"开口时间"（`tts_first`），不是整段生成时间。
3. 每一段耗时都记下来（`t_vad_end / t_asr / t_skill / t_llm_first / t_tts_first`），
   写进 `state.ai.latency_ms` 与 `state.ai.latency_breakdown`，
   超 2000 ms 在日志里给出**具体**的降级建议。产品指标必须可观测、可追。

协议（docs/10-架构与协议.md）：`state()` 返回 `{"ai": {...}}`，命令走 `ai.op`。
契约里 `latency_ms` 是一个数字（开口耗时），这里额外给 `latency_breakdown` 存分段
——新增字段不改协议版本，前端可以只用 `latency_ms`。

安全
----
* `api_key` 只落 `state_dir/ai.json`，任何返回值一律打码 `sk-***`，日志里绝不出现。
* 动作类技能只能通过 `ctx.move/pose/head` 走，模块统一限速，并且每一步都复查急停，
  急停一按（`hub.estop != ready`）立刻停手、回中位。
"""
from __future__ import annotations

import array
import asyncio
import base64
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..errors import ZeluxError, denied, unsupported
from .base import BaseModule

try:  # 有 httpx 就用（异步、连接复用），没有就退回 urllib + to_thread
    import httpx  # type: ignore
except Exception:  # pragma: no cover - 板子上可能没装
    httpx = None  # type: ignore

# ---------------------------------------------------------------- 常量

BUDGET_MS = 2000                     # 产品硬指标：说话结束 → 开口回答
SKILL_TIMEOUT = 3.5                  # 技能执行上限（含它自己的网络调用）
HTTP_TIMEOUT = 3.0                   # 技能 http_get 默认超时（契约要求）
HTTP_CACHE_S = 60.0                  # 技能 http_get 缓存时间
MASK = "sk-***"

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
DEFAULT_MODEL = "doubao-seed-2.0-mini"

DEFAULT_SYSTEM = (
    "你是 ZELUX 机器鸭，一只讲人话的桌面机器人。"
    "回答控制在两句以内、不超过 60 字，因为答案会被读出来。"
    "先给结论，不要客套，不要念参数名。"
)

# 语音链路的采样参数（sherpa-onnx / arecord 用同一套）
ASR_RATE = 16000
CHUNK = 640                          # 16 kHz int16 单声道，20 ms 一帧
VAD_RMS = 460                        # 能量门限：安静房间底噪约 150–300
VAD_SILENCE_FRAMES = 25              # 500 ms 静音算说完了
VAD_MIN_VOICE_FRAMES = 4             # 至少 80 ms 有声，防止咳嗽触发
VAD_MAX_FRAMES = 700                 # 一句最长 14 s，超了强制收尾

WAKE_LABEL = "小鸭小鸭"

_SENT_END = "。！？!?；;\n"
_ID_RE = re.compile(r"^[a-z][a-z0-9_\-]{1,31}$")
_PARAM_RE = re.compile(r"^[a-z_][a-z0-9_]{0,23}$")
_PARAM_TYPES = ("str", "int", "float", "bool")

log = logging.getLogger("zeluxd.ai")


# ---------------------------------------------------------------- 小工具
def _now() -> float:
    return time.perf_counter()


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _which(*names: str) -> str | None:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def _newest(d: Path, *patterns: str) -> Path | None:
    """目录里最新的匹配文件。模型文件常带版本号，取最新的是最省事的约定。"""
    best, best_m = None, -1.0
    if not d.is_dir():
        return None
    for pat in patterns:
        for p in d.glob(pat):
            if not p.is_file():
                continue
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m > best_m:
                best, best_m = p, m
    return best


def _rms(chunk: bytes) -> float:
    """不进 numpy 也算得动：20 ms 只有 320 个采样点，A55 上微秒级。"""
    if len(chunk) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(chunk[: len(chunk) - len(chunk) % 2])
    if not samples:
        return 0.0
    total = 0
    for v in samples:
        total += v * v
    return math.sqrt(total / len(samples))


def _to_float(chunk: bytes):
    """sherpa-onnx 要 [-1,1] 的 float 采样；有 numpy 就走零拷贝。"""
    try:
        import numpy as np  # type: ignore

        return np.frombuffer(chunk, dtype=np.int16).astype("float32") / 32768.0
    except Exception:
        samples = array.array("h")
        samples.frombytes(chunk[: len(chunk) - len(chunk) % 2])
        return [v / 32768.0 for v in samples]


def _looks_json(text: str) -> bool:
    head = text.lstrip()
    return head.startswith("{") or head.startswith("```") or head.startswith("[")


def _split_sentence(buf: str) -> tuple[str, str]:
    """切一句能读的话。太短不切（避免把"好的"单独送 TTS，白白多一次合成）。"""
    for i, ch in enumerate(buf):
        if ch in _SENT_END and i >= 3:
            return buf[: i + 1], buf[i + 1:]
    if len(buf) >= 28:
        cut = max(buf.rfind("，"), buf.rfind(","), buf.rfind("、"))
        if cut >= 8:
            return buf[: cut + 1], buf[cut + 1:]
    return "", buf


def _mask_key(key: str | None) -> str:
    return MASK if key else ""


def _mask_nested(data: dict, path: tuple[str, ...]) -> dict:
    """深拷贝并按路径打码；path 里的 "*" 表示"这一层所有子项"。

    为什么连嵌套的一起打码：云端 ASR 的 key、快递 100 的 key 一样是凭据，
    只保护顶层 api_key 等于没保护。
    """
    out = json.loads(json.dumps(data or {}, ensure_ascii=False))

    def walk(node: Any, keys: list[str]) -> None:
        if not isinstance(node, dict) or not keys:
            return
        head, rest = keys[0], keys[1:]
        if head == "*":
            for v in node.values():
                walk(v, rest)
            return
        if head in node:
            if rest:
                walk(node[head], rest)
            elif node[head]:
                node[head] = MASK

    walk(out, list(path))
    return out


def _restore_masked(new: Any, old: Any) -> Any:
    """把前端回传的 sk-*** 换回真值。不然用户改个模型名就把 key 清空了。"""
    if isinstance(new, dict):
        base = old if isinstance(old, dict) else {}
        return {k: _restore_masked(v, base.get(k)) for k, v in new.items()}
    if new == MASK and old not in (None, "", MASK):
        return old
    return new


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class SkillError(ZeluxError):
    """技能里抛这个，用户会原样听到 message（不要写堆栈、不要写英文）。"""

    def __init__(self, message: str, code: str = "failed"):
        super().__init__(message, code)


# ================================================================ 技能元数据
@dataclass
class Skill:
    id: str
    name: str
    version: str
    author: str
    description: str
    triggers: list[re.Pattern]
    trigger_text: list[str]
    params: list[dict]
    permissions: list[str]
    needs: list[str]
    priority: int
    path: Path
    handler: Callable[..., dict] | None = None
    enabled: bool = True
    error: str | None = None
    calls: int = 0
    last_error: str | None = None
    ctx: Any = None

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "priority": self.priority,
            "enabled": self.enabled,
            "needs": list(self.needs),
            "permissions": list(self.permissions),
            "params": [{"name": p["name"], "type": p["type"], "required": p.get("required", False),
                        "description": p.get("description", "")} for p in self.params],
            "triggers": list(self.trigger_text),
            "calls": self.calls,
            "error": self.error,
            "last_error": self.last_error,
        }


@dataclass
class Route:
    skill: Skill
    args: dict
    matched: str
    missing: list[str] = field(default_factory=list)

    @property
    def question(self) -> str:
        if not self.missing:
            return ""
        names = "、".join(p.get("ask") or p.get("description") or p["name"] for p in self.missing)
        return f"{self.skill.name}还得知道{names}，再说一遍？"


def parse_skill_meta(raw: Any, dir_name: str) -> Skill:
    """校验 skill.json。错就抛 ZeluxError（带第几项），加载器只跳过这一个技能。"""
    if not isinstance(raw, dict):
        raise SkillError("skill.json 必须是一个对象", "denied")

    sid = raw.get("id")
    if not isinstance(sid, str) or not _ID_RE.match(sid):
        raise SkillError("id 只能用 2–32 位小写字母、数字、下划线或减号，且以字母开头", "denied")
    if sid != dir_name:
        raise SkillError(f"id（{sid}）必须和目录名（{dir_name}）一致，否则卸载会删错东西", "denied")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SkillError("name 不能为空", "denied")

    triggers = raw.get("triggers")
    if not isinstance(triggers, list) or not triggers:
        raise SkillError("triggers 至少要写一条正则或关键词", "denied")
    compiled: list[re.Pattern] = []
    texts: list[str] = []
    for i, t in enumerate(triggers):
        if not isinstance(t, str) or not t.strip():
            raise SkillError(f"triggers 第 {i + 1} 条不是字符串", "denied")
        if len(t) > 200:
            raise SkillError(f"triggers 第 {i + 1} 条太长（>200 字符）", "denied")
        try:
            compiled.append(re.compile(t, re.IGNORECASE))
        except re.error as exc:
            raise SkillError(f"triggers 第 {i + 1} 条正则写错了：{exc}", "denied")
        texts.append(t)

    params_raw = raw.get("params") or []
    if not isinstance(params_raw, list):
        raise SkillError("params 必须是数组", "denied")
    params: list[dict] = []
    for i, p in enumerate(params_raw):
        if not isinstance(p, dict):
            raise SkillError(f"params 第 {i + 1} 项必须是对象", "denied")
        pname = p.get("name")
        if not isinstance(pname, str) or not _PARAM_RE.match(pname):
            raise SkillError(f"params 第 {i + 1} 项的 name 不合法（小写字母开头）", "denied")
        ptype = p.get("type", "str")
        if ptype not in _PARAM_TYPES:
            raise SkillError(f"params.{pname}.type 只能是 {'/'.join(_PARAM_TYPES)}", "denied")
        item = {
            "name": pname,
            "type": ptype,
            "required": bool(p.get("required", False)),
            "description": str(p.get("description") or ""),
        }
        if "default" in p:
            item["default"] = p["default"]
        if p.get("pattern"):
            try:
                item["_pattern"] = re.compile(str(p["pattern"]), re.IGNORECASE)
            except re.error as exc:
                raise SkillError(f"params.{pname}.pattern 正则写错了：{exc}", "denied")
        if p.get("ask"):
            item["ask"] = str(p["ask"])
        params.append(item)

    priority = raw.get("priority", 50)
    try:
        priority = int(priority)
    except Exception:
        raise SkillError("priority 必须是整数", "denied")
    priority = max(0, min(1000, priority))

    needs = raw.get("needs") or []
    if not isinstance(needs, list) or not all(isinstance(x, str) for x in needs):
        raise SkillError("needs 必须是字符串数组", "denied")
    perms = raw.get("permissions") or []
    if not isinstance(perms, list) or not all(isinstance(x, str) for x in perms):
        raise SkillError("permissions 必须是字符串数组", "denied")

    return Skill(
        id=sid,
        name=name.strip(),
        version=str(raw.get("version") or "0.0.0"),
        author=str(raw.get("author") or "未署名"),
        description=str(raw.get("description") or ""),
        triggers=compiled,
        trigger_text=texts,
        params=params,
        permissions=list(perms),
        needs=list(needs),
        priority=priority,
        path=Path(),
    )


def _extract_args(skill: Skill, match: re.Match, text: str) -> tuple[dict, list[dict]]:
    """参数取值顺序：正则命名组 → param.pattern → default。缺必填就返回"要问什么"。"""
    groups = match.groupdict() or {}
    args: dict = {}
    missing: list[dict] = []
    for p in skill.params:
        name = p["name"]
        val: Any = None
        if name in groups and groups[name]:
            val = groups[name]
        elif p.get("_pattern") is not None:
            mm = p["_pattern"].search(text)
            if mm:
                val = mm.group(1) if mm.groups() else mm.group(0)
        if val is None and "default" in p:
            val = p["default"]
        if val is None:
            if p["required"]:
                missing.append(p)
            continue
        try:
            if p["type"] == "int":
                val = int(str(val).strip())
            elif p["type"] == "float":
                val = float(str(val).strip())
            elif p["type"] == "bool":
                val = str(val).strip().lower() in ("1", "true", "yes", "on", "是", "要")
            else:
                val = str(val).strip()
        except (TypeError, ValueError):
            if p["required"]:
                missing.append(p)
            continue
        args[name] = val
    return args, missing


# ================================================================ 技能上下文
class SkillContext:
    """交给技能的唯一入口。**全部同步**，因为 handler.run 是普通函数（跑在 to_thread 里）。

    为什么同步：第三方开发者写 `def run(ctx, text, args)` 最省心。要说话、要动，
    内部用 run_coroutine_threadsafe 丢回事件循环，技能不用懂 asyncio。
    """

    def __init__(self, mod: "Module", skill: Skill) -> None:
        self._mod = mod
        self._skill = skill
        self._cache: dict[str, tuple[float, str]] = {}

    # ---------- 说话 ----------
    def speak(self, text: str) -> None:
        """排队播报，不阻塞（技能立刻返回，用户马上听到）。"""
        self._mod.schedule_say(str(text))

    # ---------- 网络 ----------
    def http_get(self, url: str, timeout: float = HTTP_TIMEOUT,
                 headers: dict[str, str] | None = None) -> str:
        return self._fetch(url, headers=headers, timeout=timeout, cache=True)

    def http_post(self, url: str, body: str | None = None,
                  json_body: Any = None, headers: dict[str, str] | None = None,
                  timeout: float = HTTP_TIMEOUT) -> str:
        """快递 100 这类接口只收 POST，所以这里也得给一个（同样有超时）。"""
        data = None
        hdrs = {"User-Agent": "zelux-duck/1.0"}
        if json_body is not None:
            data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        elif body is not None:
            data = body.encode("utf-8")
            hdrs["Content-Type"] = "application/x-www-form-urlencoded"
        hdrs.update(headers or {})
        return self._fetch(url, data=data, headers=hdrs, timeout=timeout, cache=False)

    def _fetch(self, url: str, data: bytes | None = None, headers: dict | None = None,
               timeout: float = HTTP_TIMEOUT, cache: bool = False) -> str:
        if not re.match(r"^https?://", url or ""):
            raise SkillError("这个地址不是 http(s)，我不连")
        if cache:
            hit = self._cache.get(url)
            if hit and time.time() - hit[0] < HTTP_CACHE_S:
                return hit[1]
        hdrs = {"User-Agent": "zelux-duck/1.0"}
        hdrs.update(headers or {})
        req = urllib.request.Request(url, data=data, headers=hdrs,
                                     method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                charset = resp.headers.get_content_charset() or "utf-8"
        except urllib.error.HTTPError as exc:
            raise SkillError(f"接口返回 {exc.code}，可能 key 过期了或者参数不对")
        except (urllib.error.URLError, TimeoutError, OSError):
            raise SkillError("现在没网，或者对方服务器没响应")
        try:
            text = raw.decode(charset, "replace")
        except LookupError:
            text = raw.decode("utf-8", "replace")
        if cache:
            if len(self._cache) > 64:          # 简单上限，别把内存吃满
                self._cache.clear()
            self._cache[url] = (time.time(), text)
        self._mod.note_net(True)
        return text

    # ---------- 状态 / 日志 ----------
    @property
    def state(self) -> dict:
        """机器人当前状态快照（只读，字段见协议 §4：battery/cpu/posture/tof/fan…）"""
        try:
            return self._mod.ctx.hub.snapshot()
        except Exception:
            return {}

    def log(self, msg: str, level: str = "info") -> None:
        self._mod.log_event(f"[{self._skill.id}] {msg}", level=level, cat="ai")

    def emit(self, name: str, data: dict) -> None:
        """给前端推一条事件（计时到点、播放状态…）。"""
        self._mod.emit(name, data)

    @property
    def config(self) -> dict:
        """{"skill": 设置页里这个技能填的东西, "state_dir": ..., "model": ...}"""
        return {
            "skill": self._mod.skill_cfg(self._skill.id),
            "state_dir": str(self._mod.config.state_dir),
            "model": str(self._mod.cfg.get("model") or ""),
        }

    @staticmethod
    def now() -> float:
        return time.time()

    # ---------- 动作（统一限速、统一急停复查） ----------
    def move(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0, ms: int = 400) -> None:
        self._mod.call_motion(self._mod.do_move(vx, vy, wz, ms))

    def pose(self, name: str) -> None:
        self._mod.call_motion(self._mod.do_pose(name))

    def head(self, pitch: float = 0.0, yaw: float = 0.0, ms: int = 300) -> None:
        """头部小幅姿态，度数；结束后自动回中。点头/摇头靠它。"""
        self._mod.call_motion(self._mod.do_head(pitch, yaw, ms))

    def stop_motion(self) -> None:
        """只停运动，不卸力（卸力是急停的活，走 hub.relax）。"""
        self._mod.call_motion(self._mod.do_stop_motion())


# ================================================================ 技能库
class SkillManager:
    """扫描目录、热加载、匹配。一个技能炸了只影响它自己。"""

    def __init__(self, mod: "Module", root: Path, disabled: set[str] | None = None) -> None:
        self.mod = mod
        self.root = Path(root)
        self.skills: dict[str, Skill] = {}
        self.errors: dict[str, str] = {}
        self.disabled = set(disabled or ())

    # ---------------- 加载 ----------------
    def load_all(self) -> dict:
        self.skills.clear()
        self.errors.clear()
        if not self.root.is_dir():
            self.errors["<skills_dir>"] = f"技能目录不存在：{self.root}"
            return {"loaded": 0, "errors": dict(self.errors)}
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or d.name.startswith((".", "_")):
                continue
            try:
                self.load_dir(d)
            except ZeluxError as exc:
                self.errors[d.name] = exc.message
            except Exception as exc:  # 单个技能的意外不许带崩技能库
                self.errors[d.name] = f"{type(exc).__name__}: {exc}"
        return {"loaded": len(self.skills), "errors": dict(self.errors)}

    def load_dir(self, path: Path, apply_disabled: bool = True) -> Skill:
        meta_file = path / "skill.json"
        handler_file = path / "handler.py"
        if not meta_file.is_file():
            raise SkillError("缺 skill.json")
        if not handler_file.is_file():
            raise SkillError("缺 handler.py")
        try:
            raw = json.loads(meta_file.read_text("utf-8"))
        except json.JSONDecodeError as exc:
            raise SkillError(f"skill.json 不是合法 JSON：第 {exc.lineno} 行 {exc.msg}")
        except OSError as exc:
            raise SkillError(f"skill.json 读不出来：{exc}")

        skill = parse_skill_meta(raw, path.name)
        skill.path = path
        skill.handler = self._import_handler(skill, handler_file)
        if apply_disabled and skill.id in self.disabled:
            skill.enabled = False
        skill.ctx = SkillContext(self.mod, skill)
        self.skills[skill.id] = skill
        self.errors.pop(skill.id, None)
        return skill

    def reload(self, sid: str) -> Skill:
        skill = self.skills.get(sid)
        if not skill:
            raise SkillError(f"没有 {sid} 这个技能", "unsupported")
        return self.load_dir(skill.path, apply_disabled=False)

    @staticmethod
    def _import_handler(skill: Skill, handler_file: Path) -> Callable:
        mod_name = f"zelux_skill_{skill.id}"
        sys.modules.pop(mod_name, None)          # 热加载：丢掉旧模块，别拿到上次的函数
        importlib.invalidate_caches()
        spec = importlib.util.spec_from_file_location(mod_name, handler_file)
        if spec is None or spec.loader is None:
            raise SkillError("handler.py 没法当模块加载")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(mod_name, None)
            raise SkillError(f"handler.py 导入失败：{type(exc).__name__}: {exc}")
        run = getattr(module, "run", None)
        if not callable(run):
            raise SkillError("handler.py 里没有 def run(ctx, text, args)")
        return run

    # ---------------- 查询 ----------------
    @property
    def enabled(self) -> list[Skill]:
        return [s for s in self.skills.values() if s.enabled and s.handler]

    def count(self) -> int:
        return len(self.enabled)

    def get(self, sid: str) -> Skill | None:
        return self.skills.get(sid)

    def list(self) -> list[dict]:
        out = [s.public() for s in self.skills.values()]
        out.sort(key=lambda x: (-x["priority"], x["id"]))
        return out

    def set_enabled(self, sid: str, on: bool) -> Skill:
        skill = self.skills.get(sid)
        if not skill:
            raise SkillError(f"没有 {sid} 这个技能", "unsupported")
        skill.enabled = bool(on)
        if on:
            self.disabled.discard(sid)
        else:
            self.disabled.add(sid)
        return skill

    def forget(self, sid: str) -> None:
        self.skills.pop(sid, None)

    # ---------------- 匹配 ----------------
    def match(self, text: str) -> Route | None:
        """priority 高的先来；同一技能内按 triggers 书写顺序。

        为什么按 priority 排而不是按命中长度：动作类（"坐下"）必须压过
        闲聊类，否则大模型的"我能坐吗"会抢走机器人真正的动作。
        """
        if not text:
            return None
        for skill in sorted(self.enabled, key=lambda s: (-s.priority, s.id)):
            for pat in skill.triggers:
                m = pat.search(text)
                if not m:
                    continue
                args, missing = _extract_args(skill, m, text)
                return Route(skill=skill, args=args, matched=m.group(0), missing=missing)
        return None


# ================================================================ LLM 客户端
class LLMClient:
    """OpenAI 兼容 /chat/completions。有 httpx 用 httpx，没有用 urllib（跑在线程里）。"""

    def __init__(self, mod: "Module") -> None:
        self.mod = mod
        self.log = mod.log
        self._open_resp = None

    # ---------- 组装 ----------
    @property
    def cfg(self) -> dict:
        return self.mod.cfg

    def _url(self, path: str = "/chat/completions") -> str:
        return str(self.cfg.get("base_url") or "").rstrip("/") + path

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.cfg.get('api_key', '')}",
                "Content-Type": "application/json"}

    def ensure_ready(self) -> None:
        if not self.cfg.get("api_key"):
            raise denied("还没填 AI 的 API key。去设置页填一下，填完技能和大模型就都能用了")
        if not self.cfg.get("base_url"):
            raise denied("还没填接口地址（base_url）")
        if not self.cfg.get("model"):
            raise denied("还没填模型名")

    def _body(self, messages: list[dict], stream: bool, model: str | None = None,
              max_tokens: int | None = None) -> dict:
        body = {
            "model": model or self.cfg.get("model"),
            "messages": messages,
            "temperature": float(self.cfg.get("temperature", 0.6)),
            "stream": bool(stream),
            "max_tokens": int(max_tokens or self.cfg.get("max_tokens") or 320),
        }
        return body

    @staticmethod
    def _human_error(status: int, text: str) -> str:
        if status in (401, 403):
            return "API key 不对或没权限（401/403），去设置页换一个"
        if status == 404:
            return "接口地址不对（404）：base_url 里通常要带 /v3 或 /api/plan/v3"
        if status == 429:
            return "请求太密或额度用完了（429），等一会儿再试"
        if status >= 500:
            return f"AI 服务自己出错了（{status}），过一会儿再试"
        snippet = (text or "")[:120].replace("\n", " ")
        return f"AI 服务返回 {status}：{snippet}"

    # ---------- 流式 ----------
    async def stream(self, messages: list[dict], on_delta: Callable[[str], Any] | None = None,
                     timeout: float = 40.0, model: str | None = None,
                     max_tokens: int | None = None) -> dict:
        self.ensure_ready()
        t0 = _now()
        body = self._body(messages, True, model=model, max_tokens=max_tokens)
        text, first_ms = "", None

        async def feed(delta: str) -> None:
            nonlocal text, first_ms
            if first_ms is None:
                first_ms = _ms(t0)
            text += delta
            if on_delta:
                try:
                    res = on_delta(delta)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    pass

        if httpx is not None:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=6.0)) as client:
                    async with client.stream("POST", self._url(), json=body,
                                             headers=self._headers()) as resp:
                        if resp.status_code >= 400:
                            raw = (await resp.aread()).decode("utf-8", "replace")
                            raise ZeluxError(self._human_error(resp.status_code, raw), "failed")
                        async for line in resp.aiter_lines():
                            if self.mod._stop_flag.is_set():
                                break
                            delta = _sse_delta(line)
                            if delta:
                                await feed(delta)
            except ZeluxError:
                raise
            except Exception as exc:
                raise ZeluxError(_net_reason(exc), "failed")
        else:
            await self._stream_urllib(body, timeout, feed)

        if not text.strip():
            raise ZeluxError("模型这轮什么都没说，再问一次试试", "failed")
        return {"text": text, "first_ms": first_ms or _ms(t0), "ms": _ms(t0)}

    async def _stream_urllib(self, body: dict, timeout: float, feed: Callable) -> None:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def worker() -> None:
            resp = None
            try:
                req = urllib.request.Request(self._url(), data=json.dumps(body).encode("utf-8"),
                                             headers=self._headers(), method="POST")
                resp = urllib.request.urlopen(req, timeout=timeout)
                self._open_resp = resp
                if resp.status >= 400:
                    raw = resp.read().decode("utf-8", "replace")
                    loop.call_soon_threadsafe(q.put_nowait,
                                              ("error", ZeluxError(self._human_error(resp.status, raw), "failed")))
                    return
                for raw in resp:
                    if self.mod._stop_flag.is_set():
                        break
                    delta = _sse_delta(raw.decode("utf-8", "replace").strip())
                    if delta:
                        loop.call_soon_threadsafe(q.put_nowait, ("delta", delta))
            except urllib.error.HTTPError as exc:
                raw = ""
                try:
                    raw = exc.read().decode("utf-8", "replace")
                except Exception:
                    pass
                loop.call_soon_threadsafe(
                    q.put_nowait, ("error", ZeluxError(self._human_error(exc.code, raw), "failed")))
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, ("error", ZeluxError(_net_reason(exc), "failed")))
            finally:
                self._open_resp = None
                try:
                    resp and resp.close()
                except Exception:
                    pass
                loop.call_soon_threadsafe(q.put_nowait, ("end", None))

        import threading

        threading.Thread(target=worker, name="zelux-llm", daemon=True).start()
        while True:
            kind, payload = await q.get()
            if kind == "delta":
                await feed(payload)
            elif kind == "error":
                raise payload
            else:
                return

    # ---------- 一次性 ----------
    async def complete(self, messages: list[dict], timeout: float = 20.0,
                       model: str | None = None, max_tokens: int | None = None) -> dict:
        self.ensure_ready()
        t0 = _now()
        body = self._body(messages, False, model=model, max_tokens=max_tokens)
        data = await asyncio.to_thread(self._post_json, body, timeout)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except Exception:
            raise ZeluxError("AI 返回的结构不认识（没有 choices）", "failed")
        return {"text": text, "ms": _ms(t0)}

    def _post_json(self, body: dict, timeout: float) -> dict:
        req = urllib.request.Request(self._url(), data=json.dumps(body).encode("utf-8"),
                                     headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raw = ""
            try:
                raw = exc.read().decode("utf-8", "replace")
            except Exception:
                pass
            raise ZeluxError(self._human_error(exc.code, raw), "failed")
        except Exception as exc:
            raise ZeluxError(_net_reason(exc), "failed")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise ZeluxError("AI 返回的不是 JSON，检查 base_url 是不是指向 /chat/completions", "failed")


def _sse_delta(line: str) -> str:
    if not line.startswith("data:"):
        return ""
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return ""
    try:
        obj = json.loads(payload)
        return obj["choices"][0].get("delta", {}).get("content") or ""
    except Exception:
        return ""


def _net_reason(exc: Exception) -> str:
    name = type(exc).__name__
    if "Timeout" in name or "timeout" in str(exc).lower():
        return "AI 服务超时了（网络慢），再问一次或者换更近的接入点"
    return "连不上 AI 服务，检查这台机器有没有网"


# ================================================================ 语音链路
class Pipeline:
    """按可用性组装的语音链路：唤醒 → ASR → 技能/LLM → TTS。

    每一级都允许缺：
    * 唤醒：openwakeword 没有就退回"按住说话"（前端 PTT）+ VAD 自动断句；
    * ASR：sherpa-onnx 本地流式 → 云端 ASR → unsupported（并告诉你怎么装）；
    * TTS：piper（本地，~30 ms 起播）→ edge-tts → 只回文字。
    """

    def __init__(self, mod: "Module") -> None:
        self.mod = mod
        self.log = mod.log
        self._cap: dict | None = None
        self._cap_at = 0.0
        self._q: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self._speaking = False
        self._procs: list[subprocess.Popen] = []
        self._listen: asyncio.Task | None = None
        self._listening = False
        self._finish_now = False
        self._rec_proc = None
        self._recognizer = None
        self._offline = False               # 离线模型要整段喂完再 decode，不能边喂边读
        self._rec_tried = False
        self._stream = None
        self._last_partial = ""
        self._wake = None

    # ---------------- 探测 ----------------
    def capabilities(self, force: bool = False) -> dict:
        if self._cap is not None and not force and time.time() - self._cap_at < 15:
            return self._cap
        cap = {
            "wake": self._probe_wake(),
            "asr": self._probe_asr(),
            "tts": self._probe_tts(),
            "mic": {"mode": "arecord" if _which("arecord") else "none"},
            "player": {"mode": "mpv" if _which("mpv") else
                       ("ffplay" if _which("ffplay") else
                        ("aplay" if _which("aplay") else "none"))},
        }
        self._cap, self._cap_at = cap, time.time()
        return cap

    def _probe_wake(self) -> dict:
        model = _newest(self.mod.config.models_dir / "wake", "*.onnx", "*.tflite")
        if importlib.util.find_spec("openwakeword") and model:
            return {"mode": "openwakeword", "label": WAKE_LABEL, "model": model.name}
        why = ("没装 openwakeword" if not importlib.util.find_spec("openwakeword")
               else "var/models/wake/ 里没有唤醒词模型")
        return {"mode": "ptt", "label": WAKE_LABEL, "why": why,
                "hint": "按住说话照样能用；要免按键就 pip install openwakeword 并把模型放进 var/models/wake/"}

    def _probe_asr(self) -> dict:
        d = self.mod.config.models_dir / "asr"
        tokens = d / "tokens.txt"
        nnet = _newest(d, "*.onnx")
        if importlib.util.find_spec("sherpa_onnx") and tokens.is_file() and nnet is not None:
            return {"mode": "sherpa", "model_dir": str(d)}
        cloud = (self.mod.cfg.get("asr") or {}).get("cloud") or {}
        if cloud.get("base_url") and cloud.get("api_key"):
            return {"mode": "cloud", "base_url": cloud.get("base_url"), "model": cloud.get("model")}
        if not importlib.util.find_spec("sherpa_onnx"):
            why = "没装 sherpa-onnx"
            hint = ("装本地流式识别：pip install sherpa-onnx，然后把 tokens.txt 与 encoder/decoder/joiner "
                    "（或单个 sense-voice 模型）放到 var/models/asr/。也可以填云端 ASR 的地址和 key")
        elif not tokens.is_file() or nnet is None:
            why = "var/models/asr/ 里没有模型（要 tokens.txt + *.onnx）"
            hint = "把 sherpa-onnx 模型解压到 var/models/asr/，或填云端 ASR"
        else:
            why = "本地模型加载失败"
            hint = "看日志里 sherpa 的报错，通常是模型类型对不上（流式/非流式）"
        return {"mode": "none", "why": why, "hint": hint}

    def _probe_tts(self) -> dict:
        model = _newest(self.mod.config.models_dir / "tts", "*.onnx")
        if _which("piper") and model:
            return {"mode": "piper", "model": model.name,
                    "rate": int((self.mod.cfg.get("tts") or {}).get("sample_rate") or 22050)}
        if importlib.util.find_spec("edge_tts") and _which("mpv", "ffplay"):
            return {"mode": "edge", "voice": (self.mod.cfg.get("tts") or {}).get("voice")}
        why = "没装 piper 语音模型" if _which("piper") else "没装 piper"
        return {"mode": "text", "why": why,
                "hint": "装本地合成：下载 piper 与一个中文 .onnx 语音到 var/models/tts/（约 30 ms 起播）；"
                        "或 pip install edge-tts 并装 mpv"}

    # ---------------- 生命周期 ----------------
    def start(self) -> None:
        if self._q is None:
            self._q = asyncio.Queue()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._worker_loop())
        cap = self.capabilities(force=True)
        self.log.info("语音链路：唤醒=%s ASR=%s TTS=%s 麦克风=%s",
                      cap["wake"]["mode"], cap["asr"]["mode"], cap["tts"]["mode"], cap["mic"]["mode"])

    async def stop(self) -> None:
        self.mod._stop_flag.set()
        if self._q is not None:
            while not self._q.empty():
                try:
                    _, fut = self._q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not fut.done():
                    fut.set_result({"engine": "none", "first_ms": None, "skipped": True})
        for p in list(self._procs):
            try:
                p.kill()
            except Exception:
                pass
        self._procs.clear()
        await self._stop_capture()
        if self._worker:
            self._worker.cancel()
            self._worker = None
        if self._stream is not None:
            try:
                self._stream = None
            except Exception:
                pass

    # ---------------- TTS ----------------
    def say_nowait(self, text: str) -> asyncio.Future:
        """排队说话，立刻返回一个 future：播报**真正开始**时 resolve（拿真实起播耗时）。"""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        if self._q is None:
            self._q = asyncio.Queue()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._worker_loop())
        self._q.put_nowait((text or "", fut))
        return fut

    async def say(self, text: str, wait: float = 2.5) -> dict:
        if not (text or "").strip():
            return {"engine": "none", "first_ms": 0}
        fut = self.say_nowait(text)
        try:
            return await asyncio.wait_for(fut, timeout=wait)
        except asyncio.TimeoutError:
            return {"engine": "?", "first_ms": None, "slow": True}

    async def _worker_loop(self) -> None:
        assert self._q is not None
        while True:
            text, fut = await self._q.get()
            if self.mod._stop_flag.is_set():
                if not fut.done():
                    fut.set_result({"engine": "none", "first_ms": None, "skipped": True})
                continue
            self._speaking = True
            started = {"done": False}

            def on_start(engine: str, t0: float) -> None:
                if started["done"]:
                    return
                started["done"] = True
                payload = {"engine": engine, "first_ms": _ms(t0), "text": text}
                loop = self.mod._loop
                if loop is None or loop.is_closed():
                    return
                loop.call_soon_threadsafe(_resolve, fut, payload)

            t0 = _now()
            try:
                mode = self.capabilities()["tts"]["mode"]
                if mode == "piper":
                    await asyncio.to_thread(self._play_piper, text, t0, on_start)
                elif mode == "edge":
                    await self._play_edge(text, t0, on_start)
                else:
                    # 没有本地/在线合成：交给 HAL（真机有蜂鸣/功放时能用），再不行就只回文字
                    ok = False
                    try:
                        ok = bool(await self.mod.hal.speak(text))
                    except Exception:
                        ok = False
                    on_start("hal" if ok else "text", t0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.log.warning("播报失败：%s", exc)
                on_start("error", t0)
            finally:
                self._speaking = False
                if not fut.done():
                    _resolve(fut, {"engine": "none", "first_ms": None})

    def _play_piper(self, text: str, t0: float, on_start) -> None:
        cap = self.capabilities()["tts"]
        model = _newest(self.mod.config.models_dir / "tts", "*.onnx")
        piper = _which("piper")
        aplay = _which("aplay", "paplay")
        if not (model and piper and aplay):
            raise RuntimeError("piper 或 aplay 不见了")
        rate = str(cap.get("rate") or 22050)
        p1 = subprocess.Popen([piper, "--model", str(model), "--output-raw"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
        p2 = subprocess.Popen([aplay, "-q", "-r", rate, "-f", "S16_LE", "-t", "raw", "-"],
                              stdin=p1.stdout, stderr=subprocess.DEVNULL)
        self._procs += [p1, p2]
        try:
            on_start("piper", t0)          # 管道一通就开始出声，这里算起播
            assert p1.stdin is not None
            p1.stdin.write(text.encode("utf-8"))
            p1.stdin.close()
            p2.wait(timeout=90)
        finally:
            for p in (p1, p2):
                try:
                    p.kill()
                except Exception:
                    pass
                if p in self._procs:
                    self._procs.remove(p)

    async def _play_edge(self, text: str, t0: float, on_start) -> None:
        import edge_tts  # 探测阶段已经确认装了就走到这里

        voice = (self.mod.cfg.get("tts") or {}).get("voice") or "zh-CN-XiaoxiaoNeural"
        tmp = Path(tempfile.gettempdir()) / f"zelux-tts-{os.getpid()}.mp3"
        await edge_tts.Communicate(text, voice).save(str(tmp))
        player = self._player_cmd(tmp)
        if not player:
            raise RuntimeError("没有能放 mp3 的播放器（mpv/ffplay）")
        proc = subprocess.Popen(player, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._procs.append(proc)
        try:
            on_start("edge-tts", t0)
            await asyncio.to_thread(proc.wait, 90)
        finally:
            try:
                proc.kill()
            except Exception:
                pass
            if proc in self._procs:
                self._procs.remove(proc)
            try:
                tmp.unlink()
            except OSError:
                pass

    def _player_cmd(self, path: Path) -> list[str] | None:
        if _which("mpv"):
            return ["mpv", "--no-video", "--really-quiet", str(path)]
        if _which("ffplay"):
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
        if _which("aplay") and path.suffix.lower() == ".wav":
            return ["aplay", "-q", str(path)]
        return None

    @property
    def speaking(self) -> bool:
        return self._speaking or bool(self._q and not self._q.empty())

    # ---------------- ASR ----------------
    def _ensure_recognizer(self):
        if self._rec_tried:
            return self._recognizer
        self._rec_tried = True
        try:
            import sherpa_onnx  # type: ignore
        except Exception as exc:
            self.log.info("没装 sherpa-onnx（%s），ASR 走云端", exc)
            return None
        d = self.mod.config.models_dir / "asr"
        tokens = d / "tokens.txt"
        try:
            encoder = _newest(d, "encoder*.onnx")
            decoder = _newest(d, "decoder*.onnx")
            joiner = _newest(d, "joiner*.onnx")
            if encoder and decoder and joiner:
                self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    tokens=str(tokens), encoder=str(encoder), decoder=str(decoder),
                    joiner=str(joiner), num_threads=2, sample_rate=ASR_RATE,
                    feature_dim=80, decoding_method="greedy_search",
                )
            else:
                model = _newest(d, "*.onnx")
                if model is None:
                    return None
                # 单文件模型一般是 sense-voice / paraformer，用离线识别器兜住
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=str(model), tokens=str(tokens), use_itn=True, num_threads=2,
                )
                self._offline = True
        except Exception as exc:
            self.log.warning("sherpa 模型加载失败：%s", exc)
            self._recognizer = None
        return self._recognizer

    async def listen(self, on: bool) -> dict:
        cap = self.capabilities()
        if not on:
            self._finish_now = True
            task = self._listen
            if task and not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=4.0)
                except asyncio.TimeoutError:
                    await self._stop_capture()
            return {"listening": False, "asr": cap["asr"]["mode"]}
        if self._listen and not self._listen.done():
            return {"listening": True, "already": True, "asr": cap["asr"]["mode"]}
        if cap["mic"]["mode"] == "none":
            raise unsupported("没找到麦克风（缺 arecord）。装一下 alsa-utils，或者先用打字对话")
        if cap["asr"]["mode"] == "none":
            raise unsupported(f"{cap['asr'].get('why')}。{cap['asr'].get('hint') or ''}".strip())
        self.mod._stop_flag.clear()
        self._finish_now = False
        self._listening = True
        self._listen = asyncio.create_task(self._capture_loop())
        return {"listening": True, "wake": cap["wake"]["mode"], "asr": cap["asr"]["mode"]}

    @property
    def listening(self) -> bool:
        return self._listening

    async def _stop_capture(self) -> None:
        self._listening = False
        self._finish_now = True
        proc = self._rec_proc
        if proc is not None:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            self._rec_proc = None
        task = self._listen
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
        self._listen = None

    async def _capture_loop(self) -> None:
        """arecord 出 16 kHz raw，20 ms 一帧算能量做 VAD；sherpa 流式边说边出字。

        为什么要 VAD 而不是固定录 3 秒：固定时长要么把用户的话切掉，要么白等 2 秒，
        两条都直接砸掉 2 秒指标。
        """
        cap = self.capabilities()
        mode = cap["asr"]["mode"]
        rec = self._ensure_recognizer() if mode == "sherpa" else None
        if rec is not None and hasattr(rec, "create_stream"):
            self._stream = rec.create_stream()
        elif rec is not None:
            self._stream = None

        wake = None
        if cap["wake"]["mode"] == "openwakeword":
            wake = self._ensure_wake()
        awake = wake is None            # PTT 模式：按下就算已经唤醒

        try:
            proc = await asyncio.create_subprocess_exec(
                "arecord", "-q", "-f", "S16_LE", "-r", str(ASR_RATE), "-c", "1", "-t", "raw", "-",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        except Exception as exc:
            self._listening = False
            self.mod.emit("ai.error", {"error": f"麦克风打不开：{exc}"})
            return
        self._rec_proc = proc
        buf = bytearray()
        voiced = False
        silent = 0
        voiced_frames = 0
        t_voice_start = 0.0
        self._last_partial = ""

        try:
            assert proc.stdout is not None
            while not self.mod._stop_flag.is_set():
                try:
                    chunk = await asyncio.wait_for(proc.stdout.read(CHUNK), timeout=0.6)
                except asyncio.TimeoutError:
                    continue
                if not chunk:
                    break
                if self._finish_now and not voiced:
                    # 松手时一句话都还没说：立刻收工，别让前端的"松手"等满 4 秒
                    break

                if wake is not None and not awake:
                    if wake.feed(chunk):
                        awake = True
                        self.mod.emit("ai.wake", {"label": WAKE_LABEL})
                        self.mod.log_event("听到唤醒词", cat="ai")
                    continue

                loud = _rms(chunk) >= VAD_RMS
                if loud:
                    if not voiced:
                        voiced = True
                        t_voice_start = _now()
                        voiced_frames = 0
                        buf.clear()
                        self._last_partial = ""
                    silent = 0
                    voiced_frames += 1
                elif voiced:
                    silent += 1

                if voiced:
                    buf += chunk
                    if rec is not None:
                        self._feed(rec, chunk)

                end = voiced and (silent >= VAD_SILENCE_FRAMES or self._finish_now
                                  or voiced_frames >= VAD_MAX_FRAMES)
                if not end:
                    continue

                t_vad_end = _now()
                text = self._take_text(rec)
                if not text.strip() and mode == "cloud" and len(buf) > CHUNK * VAD_MIN_VOICE_FRAMES:
                    try:
                        text = await asyncio.to_thread(self._cloud_asr, bytes(buf))
                    except ZeluxError as exc:
                        self.mod.emit("ai.error", {"error": exc.message})
                        text = ""
                self.mod.emit("ai.asr_final", {"text": text})
                buf.clear()
                voiced = False
                silent = 0
                voiced_frames = 0
                if wake is not None:
                    awake = False           # 唤醒词一次只放一句话进去

                if text.strip():
                    asr_ms = int((t_vad_end - t_voice_start) * 1000)
                    try:
                        await self.mod.handle_text(text, source="voice",
                                                   t0=t_vad_end, asr_ms=asr_ms)
                    except ZeluxError as exc:
                        self.mod.emit("ai.error", {"error": exc.message})
                    except Exception:
                        self.log.exception("处理识别结果出错")

                if self._finish_now:        # 按住说话的"松手"只处理一句
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            self.log.exception("采集循环出错")
        finally:
            self._listening = False
            self._finish_now = False
            self._rec_proc = None
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass

    def _ensure_wake(self):
        if self._wake is not None:
            return self._wake
        try:
            from openwakeword.model import Model  # type: ignore
            model = _newest(self.mod.config.models_dir / "wake", "*.onnx", "*.tflite")
            if model is None:
                return None
            self._wake = _WakeWrap(Model(wakeword_models=[str(model)]))
            return self._wake
        except Exception as exc:
            self.log.info("唤醒词不可用：%s", exc)
            return None

    def _feed(self, rec, chunk: bytes) -> None:
        stream = self._stream
        if stream is None:
            return
        try:
            rec.accept_waveform(stream, _to_float(chunk))
            if self._offline:
                return                       # 离线模型只能在句末 decode 一次
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            text = rec.get_result(stream)
            if text and text != self._last_partial:
                self._last_partial = text
                self.mod.emit("ai.asr_partial", {"text": text})
        except Exception:
            self._stream = None

    def _take_text(self, rec) -> str:
        if rec is None or self._stream is None:
            return ""
        try:
            if self._offline and hasattr(rec, "decode_stream"):
                rec.decode_stream(self._stream)
            text = rec.get_result(self._stream) or ""
            if hasattr(rec, "reset"):
                try:
                    rec.reset(self._stream)
                except TypeError:
                    rec.reset()
            return str(text)
        except Exception:
            return ""

    def _cloud_asr(self, pcm: bytes) -> str:
        cloud = (self.mod.cfg.get("asr") or {}).get("cloud") or {}
        base = str(cloud.get("base_url") or "").rstrip("/")
        key = str(cloud.get("api_key") or "")
        if not (base and key):
            raise unsupported("没配云端 ASR")
        wav = _wav_bytes(pcm, ASR_RATE)
        boundary = "----zelux" + hashlib.md5(str(time.time()).encode()).hexdigest()[:10]
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{cloud.get('model') or 'whisper-1'}\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n".encode() + wav + b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        req = urllib.request.Request(base + "/audio/transcriptions", data=b"".join(parts),
                                     headers={"Authorization": f"Bearer {key}",
                                              "Content-Type": f"multipart/form-data; boundary={boundary}"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            raise ZeluxError(f"云端识别返回 {exc.code}，检查 ASR 的 key 和模型名")
        except Exception:
            raise ZeluxError("云端识别连不上（没网？）")
        return str(data.get("text") or "")


class _WakeWrap:
    """openwakeword 的薄封装：只关心"这一帧有没有叫醒我"。"""

    def __init__(self, model) -> None:
        self.model = model
        self.threshold = 0.5

    def feed(self, chunk: bytes) -> bool:
        try:
            import numpy as np  # type: ignore

            arr = np.frombuffer(chunk, dtype=np.int16)
            scores = self.model.predict(arr)
            return any(float(v) >= self.threshold for v in scores.values())
        except Exception:
            return False


def _wav_bytes(pcm: bytes, rate: int) -> bytes:
    import struct

    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


def _close_coro(coro) -> None:
    """没有事件循环可投递时把协程收掉，否则解释器会抱怨 never awaited。"""
    try:
        coro.close()  # type: ignore[attr-defined]
    except Exception:
        pass


def _resolve(fut: asyncio.Future, payload: dict) -> None:
    if not fut.done():
        fut.set_result(payload)


# ================================================================ 模块
class Module(BaseModule):
    name = "ai"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.cfg = _deep_merge(_defaults(), self.load_store())
        self._stop_flag = asyncio.Event()
        self._latency: dict = {}
        self._last_reply = ""
        self._turn = 0
        self._net_ok: bool | None = None
        self._net_at = 0.0
        self._last_move = 0.0
        self._moving = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._motion_futs: set = set()      # call_motion 投递出去的 Future，急停要能撤

        disabled = set(self.cfg.get("disabled_skills") or [])
        self.registry = SkillManager(self, Path(self.config.skills_dir), disabled)
        self.llm = LLMClient(self)
        self.pipeline = Pipeline(self)

    # ---------------- 生命周期 ----------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()   # 技能在线程里跑，必须记住主循环
        self.pipeline.start()
        stats = self.registry.load_all()
        if stats["errors"]:
            for sid, why in stats["errors"].items():
                self.log.warning("技能 %s 跳过：%s", sid, why)
        self.log_event(f"技能库就绪：{stats['loaded']} 个"
                       + (f"，{len(stats['errors'])} 个没加载（看日志）" if stats["errors"] else ""),
                       cat="ai")
        if not self.cfg.get("api_key"):
            self.log_event("还没填 AI 的 key：技能都能用，自由对话要先去设置页填 key", cat="ai")
        self.spawn(self._net_probe_loop())

    async def stop(self) -> None:
        await self.pipeline.stop()
        for fut in list(self._motion_futs):
            fut.cancel()
        self._motion_futs.clear()
        await super().stop()

    async def _net_probe_loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                self._net_ok = await asyncio.to_thread(_probe_net)
                self._net_at = time.time()
            except Exception:
                pass
            await asyncio.sleep(30.0)

    def note_net(self, ok: bool) -> None:
        """技能的网络调用会顺手纠正我们的"有没有网"判断。"""
        self._net_ok = ok

    # ---------------- 状态 ----------------
    def state(self) -> dict:
        cap = self.pipeline.capabilities()
        return {"ai": {
            "ready": bool(self.cfg.get("api_key") and self.cfg.get("base_url")),
            "provider": self.cfg.get("provider"),
            "model": self.cfg.get("model"),
            "listening": self.pipeline.listening,
            "latency_ms": self._latency.get("total"),
            "skills": self.registry.count(),
            "speaking": self.pipeline.speaking,
            "last_reply": (self._last_reply or "")[:120],
            # 新增字段（不改协议版本）：分段耗时 + 链路能力，前端可以只认 latency_ms
            "latency_breakdown": dict(self._latency),
            "budget_ms": BUDGET_MS,
            "wake": cap["wake"]["mode"],
            "asr": cap["asr"]["mode"],
            "tts": cap["tts"]["mode"],
            "skills_failed": len(self.registry.errors),
        }}

    # ---------------- 命令 ----------------
    async def cmd(self, op: str, args: dict) -> dict:
        args = args or {}
        if op == "config":
            return self._op_config(args)
        if op == "test":
            return await self._op_test()
        if op == "chat":
            text = str(args.get("text") or "")
            if not text.strip():
                raise ZeluxError("说点什么吧", "failed")
            return await self.handle_text(text, source="typed")
        if op == "listen":
            return await self.pipeline.listen(bool(args.get("on", True)))
        if op == "stop":
            return await self.stop_all()
        if op == "say":
            text = str(args.get("text") or "")
            if not text.strip():
                raise ZeluxError("要说什么？", "failed")
            res = await self.pipeline.say(text)
            return {"ok": True, **res}
        if op == "skills":
            return {"items": self.registry.list(), "count": self.registry.count(),
                    "errors": dict(self.registry.errors)}
        if op == "skill_toggle":
            sid = str(args.get("id") or "")
            on = bool(args.get("on", True))
            skill = self.registry.set_enabled(sid, on)
            self.cfg["disabled_skills"] = sorted(self.registry.disabled)
            self.save_store(self.cfg)
            self.log_event(f"技能 {skill.name}（{sid}）已{'启用' if on else '停用'}", cat="ai")
            return {"ok": True, "id": sid, "enabled": skill.enabled}
        if op == "skill_install":
            return await self._op_install(args)
        if op == "skill_uninstall":
            return self._op_uninstall(str(args.get("id") or ""))
        if op == "vision_qa":
            return await self.vision_qa(str(args.get("question") or ""),
                                        str(args.get("image_path") or ""))
        raise unsupported(f"ai 不支持 {op}")

    # ---------------- 配置 ----------------
    def public_cfg(self) -> dict:
        key = str(self.cfg.get("api_key") or "")
        # 嵌套的凭据（云端 ASR 的 key、快递 100 的 key）同样不能明文回给前端
        asr = _mask_nested(self.cfg.get("asr") or {}, ("cloud", "api_key"))
        skills = _mask_nested(self.cfg.get("skills") or {}, ("*", "key"))
        out = {
            "provider": self.cfg.get("provider"),
            "base_url": self.cfg.get("base_url"),
            "model": self.cfg.get("model"),
            "api_key": _mask_key(key),          # 永远打码
            "has_key": bool(key),
            "temperature": self.cfg.get("temperature"),
            "vision_model": self.cfg.get("vision_model") or "",
            "system_prompt": self.cfg.get("system_prompt") or "",
            "asr": asr,
            "tts": self.cfg.get("tts"),
            "skills": skills,
            "disabled_skills": sorted(self.registry.disabled),
            "caps": self.pipeline.capabilities(),
        }
        return out

    def _op_config(self, args: dict) -> dict:
        patch: dict = {}
        for k in ("provider", "base_url", "model", "temperature", "vision_model", "system_prompt"):
            if k in args and args[k] is not None:
                patch[k] = args[k]
        if "api_key" in args:
            key = str(args.get("api_key") or "")
            # 传空或传回打码值 = 不改；要清空得明确说 clear_key
            if args.get("clear_key"):
                patch["api_key"] = ""
            elif key and key != MASK:
                patch["api_key"] = key
        for k in ("asr", "tts", "skills"):
            if isinstance(args.get(k), dict):
                # 前端把打码值原样提交回来时必须还原成真值，否则一次保存就把 key 抹了
                patch[k] = _restore_masked(args[k], self.cfg.get(k) or {})
        if not isinstance(patch.get("temperature", 0), (int, float)):
            patch.pop("temperature", None)
        self.cfg = _deep_merge(self.cfg, patch)
        self.save_store(self.cfg)
        self.pipeline.capabilities(force=True)      # 换了 ASR/TTS 配置要重新探测
        self.log_event("AI 配置已保存", cat="ai")
        return self.public_cfg()

    async def _op_test(self) -> dict:
        t0 = _now()
        res = await self.llm.complete(
            [{"role": "user", "content": "用一句话打个招呼，20 字以内。"}], timeout=12.0)
        return {"ok": True, "ms": res["ms"], "reply": res["text"].strip(),
                "model": self.cfg.get("model"), "budget_ms": BUDGET_MS,
                "note": "这一轮没算 ASR 与 TTS，只测模型往返"}

    # ---------------- 技能管理 ----------------
    def skill_cfg(self, sid: str) -> dict:
        return dict((self.cfg.get("skills") or {}).get(sid) or {})

    def _op_uninstall(self, sid: str) -> dict:
        skill = self.registry.get(sid)
        if not skill:
            raise unsupported(f"没有 {sid} 这个技能")
        trash = Path(self.config.state_dir) / "skills_removed"
        trash.mkdir(parents=True, exist_ok=True)
        target = trash / f"{sid}-{int(time.time())}"
        try:
            shutil.move(str(skill.path), str(target))
        except Exception as exc:
            raise ZeluxError(f"删不掉这个技能：{exc}")
        self.registry.forget(sid)
        self.registry.disabled.discard(sid)
        self.cfg["disabled_skills"] = sorted(self.registry.disabled)
        self.save_store(self.cfg)
        self.log_event(f"技能 {sid} 已卸载（目录移到 {target}）", cat="ai")
        return {"ok": True, "id": sid, "moved_to": str(target)}

    async def _op_install(self, args: dict) -> dict:
        zip_path = str(args.get("zip_path") or "")
        url = str(args.get("url") or "")
        force = bool(args.get("force"))
        tmp_file: Path | None = None
        if url:
            if not re.match(r"^https?://", url):
                raise denied("只支持 http/https 的地址")
            tmp_file = Path(tempfile.gettempdir()) / f"zelux-skill-{int(time.time())}.zip"
            try:
                await asyncio.to_thread(self._download, url, tmp_file)
            except ZeluxError:
                raise
            except Exception as exc:
                raise ZeluxError(f"下载失败：{exc}")
            zip_path = str(tmp_file)
        if not zip_path:
            raise ZeluxError("要给 zip_path 或 url", "failed")
        path = Path(zip_path).expanduser()
        if not path.is_file():
            raise ZeluxError(f"找不到这个文件：{path}", "failed")
        if path.stat().st_size > 20 * 1024 * 1024:
            raise denied("技能包太大了（>20 MB），技能不该带这么大的东西")
        try:
            result = self._install_zip(path, force)
        finally:
            if tmp_file:
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
        self.log_event(f"装好了技能：{result['name']}（{result['id']}）", cat="ai")
        return result

    def _download(self, url: str, dest: Path) -> None:
        req = urllib.request.Request(url, headers={"User-Agent": "zelux-duck/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp, open(dest, "wb") as fh:
                size = 0
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 20 * 1024 * 1024:
                        raise ZeluxError("技能包太大了（>20 MB）")
                    fh.write(chunk)
        except ZeluxError:
            raise
        except Exception:
            raise ZeluxError("下载失败：检查地址对不对、这台机器有没有网")

    def _install_zip(self, path: Path, force: bool) -> dict:
        try:
            zf = zipfile.ZipFile(path)
        except zipfile.BadZipFile:
            raise ZeluxError("这不是一个 zip 文件")
        with zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if len(names) > 200:
                raise denied("技能包里的文件太多了（>200 个）")
            # 找 skill.json：允许在根目录，也允许单独套一层目录（压缩软件默认行为）
            metas = [n for n in names if n.endswith("skill.json")]
            if not metas:
                raise ZeluxError("技能包里没有 skill.json")
            meta_name = min(metas, key=lambda n: n.count("/"))
            prefix = meta_name[: -len("skill.json")]
            if any(n.startswith("__MACOSX") for n in names):
                names = [n for n in names if not n.startswith("__MACOSX")]
            total = 0
            for n in names:
                info = zf.getinfo(n)
                total += info.file_size
                if info.file_size > 2 * 1024 * 1024:
                    raise denied(f"{n} 太大了（单文件 >2 MB）")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise denied("技能包里不许有符号链接")
                pure = Path(n)
                if pure.is_absolute() or ".." in pure.parts:
                    raise denied(f"技能包里有越界路径：{n}")
            if total > 5 * 1024 * 1024:
                raise denied("技能包解压后太大了（>5 MB）")
            try:
                raw = json.loads(zf.read(meta_name).decode("utf-8"))
            except Exception as exc:
                raise ZeluxError(f"skill.json 读不出来：{exc}")
            sid = str(raw.get("id") or "")
            if not _ID_RE.match(sid):
                raise ZeluxError("skill.json 里的 id 不合法")
            target = Path(self.config.skills_dir) / sid
            if target.exists() and not force:
                raise denied(f"{sid} 已经装了。要覆盖就传 force:true")
            tmp = Path(self.config.skills_dir) / f".installing-{sid}-{int(time.time())}"
            try:
                tmp.mkdir(parents=True, exist_ok=True)
                for n in names:
                    if not n.startswith(prefix):
                        continue
                    rel = n[len(prefix):]
                    if not rel:
                        continue
                    out = tmp / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(zf.read(n))
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                tmp.rename(target)
            except Exception as exc:
                shutil.rmtree(tmp, ignore_errors=True)
                raise ZeluxError(f"解压失败：{exc}")
        try:
            skill = self.registry.load_dir(target, apply_disabled=False)
        except ZeluxError as exc:
            shutil.rmtree(target, ignore_errors=True)
            raise ZeluxError(f"技能加载不上，已经回滚：{exc.message}")
        self.registry.disabled.discard(skill.id)
        self.cfg["disabled_skills"] = sorted(self.registry.disabled)
        self.save_store(self.cfg)
        return {"ok": True, "id": skill.id, "name": skill.name,
                "version": skill.version, "skills": self.registry.count()}

    # ---------------- 说话/急停 ----------------
    def schedule_say(self, text: str) -> None:
        """给技能和定时器用的线程安全入口。

        为什么不能用 asyncio.get_event_loop()：技能跑在 to_thread 的工作线程里，
        那里根本没有事件循环，取到的会是空值或直接抛异常 —— 播报就被静默丢了。
        所以主循环在建模块时存下来，线程里统一 run_coroutine_threadsafe 丢回去。
        """
        text = (text or "").strip()
        if not text:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            self.log.debug("没有事件循环，丢弃一条播报：%s", text[:20])
            return
        try:
            asyncio.run_coroutine_threadsafe(self.pipeline.say(text), loop)
        except RuntimeError:
            self.log.debug("事件循环已关闭，丢弃一条播报")

    def call_motion(self, coro) -> None:
        """动作统一从这里进：限速在 do_* 里，这里只负责跨线程投递 + 记账。"""
        loop = self._loop
        if loop is None or loop.is_closed():
            _close_coro(coro)
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(self._run_motion(coro), loop)
        except RuntimeError:
            _close_coro(coro)
            return
        self._motion_futs.add(fut)
        fut.add_done_callback(self._motion_futs.discard)

    async def _run_motion(self, coro) -> None:
        """动作在后台跑，出错要**说出来**：技能早就返回了，没人接就没人知道。"""
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except ZeluxError as exc:
            await self.pipeline.say(exc.message)
        except Exception:
            self.log.exception("动作执行失败")
            await self.pipeline.say("动作没做成，检查舵机是不是被占着")

    def _estop_tripped(self) -> bool:
        try:
            return (self.ctx.hub.estop or {}).get("state") != "ready"
        except Exception:
            return False

    async def _wait_turn(self) -> None:
        gap = 0.25 - (time.time() - self._last_move)
        if gap > 0:
            await asyncio.sleep(gap)
        self._last_move = time.time()

    async def do_move(self, vx: float, vy: float, wz: float, ms: int) -> None:
        """限速 + 分片：50 Hz 发一次意图，每片都复查急停，停了立刻归零。"""
        if self._estop_tripped():
            raise ZeluxError("现在是急停状态，先解除急停", "denied")
        if self.ctx.hub.posture in ("absent", "limp"):
            raise ZeluxError("鸭子还没站起来（现在是趴着/卸力），先让它站好", "denied")
        vx = max(-0.25, min(0.25, float(vx)))
        vy = max(-0.2, min(0.2, float(vy)))
        wz = max(-0.7, min(0.7, float(wz)))
        ms = max(100, min(1500, int(ms)))
        await self._wait_turn()
        self._moving = True
        end = time.time() + ms / 1000.0
        try:
            while time.time() < end and not self._stop_flag.is_set():
                if self._estop_tripped():
                    break
                await self.hal.move(vx, vy, wz)
                await asyncio.sleep(0.05)
        finally:
            self._moving = False
            try:
                await self.hal.move(0.0, 0.0, 0.0)
            except Exception:
                pass

    async def do_pose(self, name: str) -> None:
        if name not in ("zero", "stand", "sit", "fold"):
            raise ZeluxError(f"没有 {name} 这个姿势")
        if self._estop_tripped():
            raise ZeluxError("现在是急停状态，先解除急停", "denied")
        await self._wait_turn()
        await self.hal.pose(name)

    async def do_head(self, pitch: float = 0.0, yaw: float = 0.0, ms: int = 300) -> None:
        """点头/摇头：幅度和速度都限死，任何一步发现急停就回中位。

        为什么不用 pose()：pose 只有整机姿势，点头得单独动颈部两颗舵机
        （31 号 neck_tilt / 32 号 head_pan），而且必须能中途放弃。
        """
        if self._estop_tripped():
            raise ZeluxError("现在是急停状态，先解除急停", "denied")
        if self.ctx.hub.posture in ("absent", "limp"):
            raise ZeluxError("鸭子现在是卸力状态，先让它站好", "denied")
        pitch = max(-20.0, min(20.0, float(pitch)))
        yaw = max(-25.0, min(25.0, float(yaw)))
        ms = max(120, min(600, int(ms)))
        ids = {31: pitch, 32: yaw}
        await self._wait_turn()
        base: dict[int, int] = {}
        try:
            await self.hal.servo_torque(True, list(ids))
            for sid in ids:
                try:
                    info = await self.hal.servo_read(sid)
                    base[sid] = int(info.get("pos") or 2048)
                except Exception:
                    base[sid] = 2048
            steps = max(2, int(ms / 40))
            try:
                for i in range(1, steps + 1):
                    if self._estop_tripped() or self._stop_flag.is_set():
                        break
                    k = i / steps
                    goals = {sid: int(base[sid] + deg / 360.0 * 4096 * k) for sid, deg in ids.items()}
                    await self.hal.servo_sync_goal(goals)
                    await asyncio.sleep(ms / steps / 1000.0)
                await asyncio.sleep(0.12)
            finally:
                for i in range(1, 5):          # 回中位也要分几步，别让脑袋一顿
                    if self._estop_tripped():
                        break
                    k = 1 - i / 4
                    await self.hal.servo_sync_goal(
                        {sid: int(base[sid] + deg / 360.0 * 4096 * k) for sid, deg in ids.items()})
                    await asyncio.sleep(0.04)
                await self.hal.servo_sync_goal(base)
        except ZeluxError:
            raise
        except Exception as exc:
            raise ZeluxError(f"脖子没动起来（{type(exc).__name__}），检查舵机总线是不是被占着", "hardware")

    async def do_stop_motion(self) -> None:
        """停下：撤销所有动作任务，速度归零。不卸力。"""
        for fut in list(self._motion_futs):
            fut.cancel()
        self._motion_futs.clear()
        try:
            await self.hal.move(0.0, 0.0, 0.0)
        except Exception:
            pass

    async def stop_all(self) -> dict:
        """急停链路里 AI 这一侧：闭嘴 + 停动作。舵机卸力仍走 hub.relax。"""
        self._stop_flag.set()
        await self.pipeline.stop()
        await self.do_stop_motion()
        self.pipeline.start()               # 采集/播报队列重新可用
        self._stop_flag.clear()             # 不然下一句话会被 worker 当成"正在停"丢掉
        self.log_event("AI 已停止播报与动作", cat="ai")
        return {"ok": True, "speaking": False}

    # ---------------- 一轮对话 ----------------
    async def handle_text(self, text: str, source: str = "typed",
                          t0: float | None = None, asr_ms: int = 0) -> dict:
        """一条文本走完全程：技能优先 → LLM。返回这一轮的结论与耗时。"""
        t0 = t0 or _now()
        text = (text or "").strip()
        if not text:
            raise ZeluxError("没听清，再说一遍？", "failed")
        self._stop_flag.clear()
        self._turn += 1
        turn = self._turn

        try:
            route = self.registry.match(text)
        except Exception:
            self.log.exception("技能匹配炸了（不该发生）")
            route = None

        if route is not None:
            return await self._turn_skill(route, text, t0, asr_ms, turn, source)
        return await self._turn_llm(text, t0, asr_ms, turn, source)

    async def _turn_skill(self, route: Route, text: str, t0: float,
                          asr_ms: int, turn: int, source: str) -> dict:
        skill = route.skill
        blocked = self._needs_reason(skill)
        t_skill = _now()
        if blocked:
            result = {"speech": blocked, "data": None, "skill": skill.id}
        elif route.missing:
            result = {"speech": route.question, "data": None, "skill": skill.id, "ask": True}
        else:
            result = await self._run_skill(skill, text, route.args)
        skill_ms = _ms(t_skill)

        speech = str(result.get("speech") or "").strip() or "嗯，我没得出答案。"
        started = await self.pipeline.say(speech)
        t_tts = _ms(t0)

        bd = {"asr": int(asr_ms), "skill": skill_ms, "llm_first": 0,
              "tts_first": t_tts, "total": t_tts, "budget": BUDGET_MS}
        self._finish_turn(bd, speech, source=f"skill:{skill.id}", turn=turn,
                          detail=f"技能 {skill.id}{'（缺参数）' if result.get('ask') else ''}",
                          engine=started.get("engine"))
        self.emit("ai.reply", {"text": speech, "source": "skill", "skill": skill.id,
                               "data": result.get("data"), "display": result.get("display")})
        return {"ok": True, "speech": speech, "source": "skill", "skill": skill.id,
                "data": result.get("data"), "display": result.get("display"),
                "latency": bd, "engine": started.get("engine")}

    async def _run_skill(self, skill: Skill, text: str, args: dict) -> dict:
        """单独 try：一个技能炸了不影响别的技能，也不影响这一轮回话。"""
        t0 = _now()
        try:
            out = await asyncio.wait_for(
                asyncio.to_thread(self._call_handler, skill, text, args), timeout=SKILL_TIMEOUT)
            skill.last_error = None
            return out
        except asyncio.TimeoutError:
            skill.last_error = f"超时 {SKILL_TIMEOUT}s"
            self.log.warning("技能 %s 超时（%.1fs）", skill.id, SKILL_TIMEOUT)
            return {"speech": f"{skill.name}查得太慢了，等会儿再试", "data": None,
                    "skill": skill.id, "error": "timeout"}
        except ZeluxError as exc:
            skill.last_error = exc.message
            return {"speech": exc.message, "data": None, "skill": skill.id, "error": exc.code}
        except Exception as exc:
            skill.last_error = f"{type(exc).__name__}: {exc}"
            self.log.exception("技能 %s 出错（%.0f ms）", skill.id, _ms(t0))
            return {"speech": f"{skill.name}出错了，我记下来了，等会儿再试", "data": None,
                    "skill": skill.id, "error": "failed"}

    def _call_handler(self, skill: Skill, text: str, args: dict) -> dict:
        if skill.handler is None:
            raise SkillError("这个技能没有 handler")
        skill.calls += 1
        out = skill.handler(skill.ctx, text, args)
        if not isinstance(out, dict):
            raise SkillError(f"{skill.name} 的 run() 要返回 dict")
        if not str(out.get("speech") or "").strip():
            raise SkillError(f"{skill.name}没说话（返回里没有 speech）")
        return out

    def _needs_reason(self, skill: Skill) -> str:
        """needs 里声明了但机器上没有的能力，直接用人话回绝（比让技能崩了好）。"""
        cap = self.pipeline.capabilities()
        for need in skill.needs:
            if need == "audio" and cap["tts"]["mode"] == "text" and cap["player"]["mode"] == "none":
                return "这台机器现在出不了声（没装 TTS，也没有 mpv/ffplay）"
            if need == "mic" and cap["mic"]["mode"] == "none":
                return "没找到麦克风，先装 alsa-utils"
            if need == "network" and self._net_ok is False and time.time() - self._net_at < 120:
                return "现在没网，这个要联网才能查"
        return ""

    async def _turn_llm(self, text: str, t0: float, asr_ms: int,
                        turn: int, source: str) -> dict:
        system = self._system_prompt()
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": text}]
        buf = ""
        head_decided = False
        speak_head = True
        spoken: list[str] = []
        first_speech_ms: int | None = None
        routed = False

        async def on_delta(delta: str) -> None:
            nonlocal buf, head_decided, speak_head, first_speech_ms
            buf += delta
            self.emit("ai.delta", {"text": delta, "turn": turn})
            if not head_decided:
                if len(buf.strip()) < 4:
                    return
                head_decided = True
                speak_head = not _looks_json(buf)
                if not speak_head:
                    return                       # 这一轮是 JSON 路由，不朗读
            if not speak_head:
                return
            while True:
                piece, rest = _split_sentence(buf)
                if not piece:
                    break
                buf = rest
                res = await self.pipeline.say(piece)
                if first_speech_ms is None:
                    first_speech_ms = _ms(t0)
                spoken.append(piece)

        try:
            out = await self.llm.stream(messages, on_delta=on_delta)
        except ZeluxError as exc:
            # 没 key / 没网：也读出来，别让用户对着鸭子等一个不存在的回答
            try:
                await self.pipeline.say(exc.message)
            except Exception:
                pass
            bd = {"asr": int(asr_ms), "skill": 0, "llm_first": _ms(t0),
                  "tts_first": 0, "total": _ms(t0), "budget": BUDGET_MS}
            self._finish_turn(bd, exc.message, source="error", turn=turn, detail="LLM 不可用")
            self.emit("ai.error", {"error": exc.message})
            raise

        reply = out["text"].strip()
        if _looks_json(reply):
            obj = _parse_json_block(reply)
            sid = str((obj or {}).get("skill") or "")
            skill = self.registry.get(sid)
            if skill and skill.enabled:
                args = (obj or {}).get("args") or {}
                if isinstance(args, dict):
                    result = await self._run_skill(skill, text, args)
                    speech = str(result.get("speech") or "").strip()
                    if speech:
                        routed = True
                        started = await self.pipeline.say(speech)
                        bd = {"asr": int(asr_ms), "skill": _ms(t0), "llm_first": out["first_ms"],
                              "tts_first": _ms(t0), "total": _ms(t0), "budget": BUDGET_MS}
                        self._finish_turn(bd, speech, source=f"skill:{skill.id}", turn=turn,
                                          detail="大模型挑了个技能", engine=started.get("engine"))
                        self.emit("ai.reply", {"text": speech, "source": "skill",
                                               "skill": skill.id, "data": result.get("data")})
                        return {"ok": True, "speech": speech, "source": "skill",
                                "skill": skill.id, "latency": bd}

        if speak_head:
            rest = buf.strip()
            if rest:
                await self.pipeline.say(rest)
                spoken.append(rest)
            if first_speech_ms is None:
                first_speech_ms = _ms(t0)

        spoken_text = "".join(spoken).strip()
        if not spoken_text:
            if _looks_json(reply):
                # 模型吐了 JSON 但技能名对不上（比如装了又卸载）：
                # 千万不能把 {"skill":...} 念出来，那是最糟的体验
                spoken_text = "这个我暂时不会，换个说法，或者先给我装个对应的技能。"
            else:
                spoken_text = reply
            await self.pipeline.say(spoken_text)

        tts_first = first_speech_ms if first_speech_ms is not None else _ms(t0)
        bd = {"asr": int(asr_ms), "skill": 0, "llm_first": out["first_ms"],
              "tts_first": tts_first, "total": tts_first, "budget": BUDGET_MS,
              "llm_done": out["ms"]}
        self._finish_turn(bd, spoken_text, source="llm", turn=turn,
                          detail=f"大模型首字 {out['first_ms']} ms")
        self.emit("ai.reply", {"text": spoken_text, "source": "llm"})
        return {"ok": True, "speech": spoken_text, "source": "llm", "latency": bd,
                "model": self.cfg.get("model")}

    def _system_prompt(self) -> str:
        """技能清单只给 id + 名字 + 一句话 + 参数名：prompt 越短首字越快。"""
        base = str(self.cfg.get("system_prompt") or DEFAULT_SYSTEM)
        lines = [base]
        skills = self.registry.enabled
        if not skills:
            return base
        lines.append("需要实时数据或动作时，只输出一行 JSON：{\"skill\":\"<id>\",\"args\":{...}}，不要加别的字。")
        lines.append("可用技能：")
        for s in sorted(skills, key=lambda x: -x.priority)[:16]:
            desc = (s.description or s.name)[:40]
            params = ",".join(p["name"] + ("" if p["required"] else "?") for p in s.params)
            lines.append(f"- {s.id}｜{s.name}｜{desc}｜参数:{params or '无'}")
        return "\n".join(lines)

    # ---------------- 耗时记账 ----------------
    def _finish_turn(self, bd: dict, reply: str, source: str, turn: int,
                     detail: str = "", engine: str | None = None) -> None:
        bd["total"] = int(bd.get("total") or 0)
        self._latency = bd
        self._last_reply = reply
        over = bd["total"] > BUDGET_MS
        msg = (f"一轮 {bd['total']} ms｜ASR {bd['asr']}｜技能 {bd['skill']}｜"
               f"LLM 首字 {bd['llm_first']}｜开口 {bd['tts_first']}（{source}"
               + (f"·{engine}" if engine else "") + "）")
        if over:
            hint = _budget_hint(bd)
            self.log_event(f"{msg} → 超出 {BUDGET_MS} ms 预算：{hint}", level="warning", cat="ai")
            self.log.warning("超预算 %d ms（%s）：%s", bd["total"], detail, hint)
        else:
            self.log.info("%s", msg)
            self.log_event(msg, cat="ai")
        self.emit("ai.latency", {"turn": turn, **bd, "source": source, "over_budget": over})

    # ---------------- 视觉问答（给 vision 模块） ----------------
    async def vision_qa(self, question: str, image_path: str) -> dict:
        self.llm.ensure_ready()
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise ZeluxError(f"没有这张图：{path}", "failed")
        if path.stat().st_size > 8 * 1024 * 1024:
            raise denied("图片太大了（>8 MB），先压一下")
        question = (question or "这张图里有什么？").strip()
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        model = str(self.cfg.get("vision_model") or self.cfg.get("model"))
        messages = [{"role": "system", "content": "你是机器鸭的眼睛，用中文简短回答，先给结论。"},
                    {"role": "user", "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]}]
        res = await self.llm.complete(messages, timeout=25.0, model=model)
        text = res["text"].strip()
        speech = text if text else "我看不出什么。"
        started = await self.pipeline.say(speech)
        self._last_reply = speech
        self.log_event(f"识图问答 {res['ms']} ms：{speech[:40]}", cat="ai")
        return {"ok": True, "reply": speech, "ms": res["ms"], "model": model,
                "engine": started.get("engine")}


# ---------------------------------------------------------------- 辅助
def _defaults() -> dict:
    return {
        "provider": "openai_compatible",
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "api_key": "",
        "temperature": 0.6,
        "max_tokens": 320,
        "vision_model": "",
        "system_prompt": DEFAULT_SYSTEM,
        "asr": {"engine": "auto", "cloud": {"base_url": "", "api_key": "", "model": ""}},
        "tts": {"engine": "auto", "voice": "zh-CN-XiaoxiaoNeural", "sample_rate": 22050},
        "skills": {},
        "disabled_skills": [],
    }


def _probe_net() -> bool:
    """30 秒一次的廉价探活：DNS/UDP 不握手，1 秒超时。"""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.connect(("223.5.5.5", 53))
        s.close()
        return True
    except Exception:
        return False


def _parse_json_block(text: str) -> dict | None:
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?", "", body).strip()
        body = re.sub(r"```$", "", body).strip()
    m = re.search(r"\{.*\}", body, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _budget_hint(bd: dict) -> str:
    """超预算时给出能立刻做的事，而不是"请优化"。"""
    if bd.get("llm_first", 0) > 900:
        return "首字慢：缩短 system prompt、把常见问法写成技能（正则 0 往返）、或换更近的接入点"
    if bd.get("skill", 0) > 900:
        return "技能自己慢：给 http_get 加缓存（已有 60 s）、少查字段、把技能正则排到更前"
    if bd.get("asr", 0) > 900:
        return "ASR 慢：装 sherpa-onnx 本地流式模型（边说边出字），别用云端整段识别"
    if bd.get("tts_first", 0) > 900:
        return "TTS 起播慢：换 piper 本地合成（约 30 ms 起播），别用边下边放的云端语音"
    return "整条链路都慢：先看 ASR 与 LLM 首字，再看是不是板子在跑别的东西"
