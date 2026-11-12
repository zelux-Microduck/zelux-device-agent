"""计时提醒："十分钟后叫我"。

为什么用 threading.Timer 而不是 asyncio：技能跑在 to_thread 里，
线程定时器不会因为事件循环忙（正在播报）而漂移，也不用把 asyncio 概念塞给技能作者。
到点用 ctx.speak（线程安全入口）播报，并推一条 ai.timer 事件给页面。
"""
from __future__ import annotations

import re
import threading
import time

_UNITS = {"秒钟": 1, "秒": 1, "s": 1, "分钟": 60, "分": 60, "min": 60,
          "小时": 3600, "钟头": 3600, "h": 3600}
_CN = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
       "六": 6, "七": 7, "八": 8, "九": 9}

_SPAN_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?|[零一二两三四五六七八九十半]+)\s*(?P<half>个半)?\s*"
    r"(?P<unit>秒钟|秒|分钟|分|钟头|小时|min|s|h)", re.IGNORECASE)

_TIMERS: dict[str, list["_Timer"]] = {}
_LOCK = threading.Lock()


class _Timer:
    def __init__(self, seconds: float, label: str, on_fire) -> None:
        self.seconds = seconds
        self.label = label
        self.due = time.time() + seconds
        self.fired = False
        self.handle = threading.Timer(seconds, on_fire)
        self.handle.daemon = True

    def cancel(self) -> None:
        try:
            self.handle.cancel()
        except Exception:
            pass

    def remain(self) -> int:
        return max(0, int(round(self.due - time.time())))


def run(ctx, text, args) -> dict:
    key = str(ctx.config.get("state_dir") or "var")
    with _LOCK:
        timers = _TIMERS.setdefault(key, [])

    if any(k in text for k in ("取消", "别叫了", "不用了", "停掉")):
        with _LOCK:
            n = len(timers)
            for t in timers:
                t.cancel()
            timers.clear()
        if not n:
            return _said("现在没有在计时。")
        return _said(f"取消了 {n} 个计时。" if n > 1 else "计时取消了。")

    if any(k in text for k in ("还有多久", "还剩", "剩余", "剩多少")):
        with _LOCK:
            alive = [t for t in timers if not t.fired]
        if not alive:
            return _said("现在没有计时。")
        alive.sort(key=lambda t: t.due)
        parts = [f"{_pretty(t.remain())}后" for t in alive[:3]]
        return _said("、".join(parts) + "叫你。", remaining=[t.remain() for t in alive])

    span = str(args.get("span") or "").strip() or _find_span(text)
    m = _SPAN_RE.search(span) or _SPAN_RE.search(text)
    if not m:
        return _said("说个时长，比如「十分钟后叫我」。")

    num = _to_number(m.group("num"))
    if num is None:
        return _said("这个时长我没听懂，换个说法，比如「二十分钟」。")
    if m.group("half"):
        num += 0.5
    seconds = num * _UNITS[m.group("unit").lower()]
    seconds = max(1.0, min(24 * 3600.0, seconds))

    label = f"{_pretty(int(seconds))}的计时"

    def fire() -> None:
        with _LOCK:
            for t in list(timers):
                if t.handle is timer:
                    t.fired = True
                    timers.remove(t)
        ctx.speak(f"时间到了，{label}结束了。")
        ctx.emit("ai.timer", {"label": label, "seconds": int(seconds),
                              "at": time.strftime("%H:%M:%S")})

    timer = _Timer(seconds, label, fire)
    with _LOCK:
        timers.append(timer)
    timer.handle.start()

    at = time.strftime("%H:%M", time.localtime(timer.due))
    return _said(f"好，{_pretty(int(seconds))}后叫你（{at}）。",
                 seconds=int(seconds), due_at=at,
                 count=len([t for t in timers if not t.fired]))


def _to_number(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    if raw == "半":
        return 0.5
    if raw in _CN:
        return float(_CN[raw])
    if "十" in raw:
        head, _, tail = raw.partition("十")
        tens = float(_CN.get(head, 1) if head else 1)
        ones = float(_CN.get(tail, 0) if tail else 0)
        return tens * 10 + ones
    return None


def _find_span(text: str) -> str:
    m = _SPAN_RE.search(text)
    return m.group(0) if m else ""


def _pretty(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds // 3600} 小时"
    if seconds % 60 == 0:
        return f"{seconds // 60} 分钟"
    return f"{seconds} 秒"


def _said(text: str, **data) -> dict:
    return {"speech": text, "data": data, "display": {"kind": "timer", **data}}
