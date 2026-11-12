"""机器人动作：把"点个头""走两步"翻译成 hal 的姿势/速度意图。

两个硬约束：

1. **必须先回话再动**。走路要 600 ms，等它走完再说话，用户以为没听见。
   所以动作丢进后台线程，handler 立刻返回一句话 —— 这是 2 秒指标在动作类技能里的做法。
2. **限速与急停不在这个文件里**。`ctx.move/pose/head` 会统一排队（间隔 ≥250 ms）、
   限幅（vx ≤0.25、wz ≤0.7、头 ≤20°/25°），每一步复查 `hub.estop`，
   急停一按就停手并回中位。技能作者改不了这些，也不该改。
"""
from __future__ import annotations

import threading
import time

from zeluxd.errors import ZeluxError

# 顺序有讲究：更具体的说法放前面，避免"站起来"被"站"之类先命中
ACTIONS = [
    (("停下", "别动", "停住", "不许动"), "stop"),
    (("点个头", "点点头", "点头"), "nod"),
    (("摇个头", "摇摇头", "摇头"), "shake"),
    (("走两步", "往前走", "往前一点", "向前"), "forward"),
    (("后退", "往后退", "退后"), "back"),
    (("转个圈", "转圈", "转一圈"), "spin"),
    (("坐下", "坐下来", "蹲下"), "sit"),
    (("站起来", "起身", "起立"), "stand"),
    (("趴下", "折叠", "收起来"), "fold"),
    (("抬个头", "抬起头", "抬头"), "look_up"),
]


def run(ctx, text, args) -> dict:
    kind = ""
    for keys, name in ACTIONS:
        if any(k in text for k in keys):
            kind = name
            break
    if not kind:
        return _said("我能点头、摇头、走两步、转个圈、坐下、站起来。要哪个？", action="unknown")

    if kind == "stop":
        ctx.stop_motion()
        return _said("停了。", action="stop")

    speech = {
        "nod": "好，点个头。",
        "shake": "好，摇摇头。",
        "forward": "好，往前走两步。",
        "back": "好，往后退一点。",
        "spin": "好，转个圈。",
        "sit": "好，我坐下。",
        "stand": "好，我站起来。",
        "fold": "好，收起来了。",
        "look_up": "好，抬头看看。",
    }[kind]

    # 动作在后台跑：用户先听到回应，动起来才有"听懂"的感觉
    threading.Thread(target=_perform, args=(ctx, kind), daemon=True).start()
    return _said(speech, action=kind)


def _perform(ctx, kind: str) -> None:
    try:
        if kind == "nod":
            ctx.head(pitch=14, ms=220)
            time.sleep(0.36)
            ctx.head(pitch=-10, ms=220)
        elif kind == "shake":
            ctx.head(yaw=18, ms=260)
            time.sleep(0.42)
            ctx.head(yaw=-18, ms=260)
        elif kind == "look_up":
            ctx.head(pitch=10, ms=300)
        elif kind == "forward":
            ctx.move(vx=0.15, ms=600)
            time.sleep(0.35)
            ctx.move(vx=0.15, ms=600)
        elif kind == "back":
            ctx.move(vx=-0.12, ms=500)
        elif kind == "spin":
            ctx.move(wz=0.6, ms=1200)
        elif kind == "sit":
            ctx.pose("sit")
        elif kind == "stand":
            ctx.pose("stand")
        elif kind == "fold":
            ctx.pose("fold")
    except ZeluxError as exc:
        # 站不起来、急停中、总线被占：说人话，别静默失败
        ctx.speak(exc.message)
    except Exception as exc:  # 动作线程不许把进程带崩
        ctx.log(f"动作 {kind} 失败：{type(exc).__name__}: {exc}", level="warning")


def _said(text: str, **data) -> dict:
    return {"speech": text, "data": data, "display": {"kind": "action", **data}}
