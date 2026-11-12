"""手柄模块（页面"手柄"）：实体手柄 + 网页虚拟手柄，谁在动听谁的。

为什么这么写：

* **不让摇杆直接 await 硬件**。sticks 是 20–30 Hz 的流式命令，如果每条都去等 `hal.move()`，
  页面一抖就堆积、hub 的模块锁还会互相顶成 busy。所以 sticks 只更新"目标速度 + 时间戳"，
  由 50 Hz 的 `_speed_loop` 统一出口。顺带把"心跳失联停车"做进同一条环路，只有一个刹车点。
* **无 evdev 依赖**。直接 `os.open` + `struct` 解析 `js_event` / `input_event`，
  镜像里少一个 pip 包，也少一类"库版本不对"的故障。
* **断连 1.5 s 才置 connected:false**。因为无线手柄会瞬断再重连，
  立刻翻成断开会让页面反复闪；这 1.5 s 是"允许它自己回来"的窗口，期间速度已经归零。
* **急停只有一条路**。Start 键不自己卸力，走 `hub.relax()`，与顶栏按钮、空格键同源。
"""
from __future__ import annotations

import asyncio
import math
import os
import struct
import time
from pathlib import Path

from ..errors import denied, hardware
from .base import BaseModule

# ---------- 手感参数（前端要一模一样地画，所以是常量） ----------
DEADZONE = 0.08
EXPO = 1.6
DISCONNECT_S = 1.5     # 掉线多久算真断
HEARTBEAT_S = 0.45     # 多久没收到 sticks 就自动停车
SPEED_HZ = 50.0

# ---------- 速度上限：默认值与硬上限（args.limit 只能在中间调，不能突破硬上限） ----------
LIMIT_VX = 0.45
LIMIT_WZ = 1.2
HARD_VX = 0.6
HARD_WZ = 1.5

# ---------- 动作槽位（与 train 模块一致） ----------
SLOTS = ("walk", "sitstand", "ground_pick", "kick_left", "kick_right", "roulade", "stand")
# 非槽位的"系统动作"，也允许绑键
SYSTEM_ACTIONS = ("sit_toggle", "estop", "servo_restart", "gait_toggle", "none")

DEFAULT_BINDINGS = {
    "a": "ground_pick",
    "x": "roulade",
    "lb": "kick_left",
    "rb": "kick_right",
    "dpad_down": "sit_toggle",
    "dpad_up": "gait_toggle",
    "dpad_right": "servo_restart",
    "start": "estop",
}

# Linux joydev 的按键编号（xpad / ds4 / switch pro 都是这套 HID 顺序）
JS_BUTTONS = {
    0: "a", 1: "b", 2: "x", 3: "y", 4: "lb", 5: "rb",
    6: "select", 7: "start", 8: "guide", 9: "l3", 10: "r3",
    11: "dpad_up", 12: "dpad_down", 13: "dpad_left", 14: "dpad_right",
}
JS_AXES = {0: "lx", 1: "ly", 2: "rx", 3: "ry"}
JS_HAT = {6: "x", 7: "y"}

# evdev 编码
EV_KEY, EV_ABS = 0x01, 0x03
BTN_MAP = {
    0x130: "a", 0x131: "b", 0x133: "y", 0x134: "x",
    0x136: "lb", 0x137: "rb", 0x13A: "select", 0x13B: "start",
    0x13C: "guide", 0x13D: "l3", 0x13E: "r3",
    0x220: "dpad_up", 0x221: "dpad_down", 0x222: "dpad_left", 0x223: "dpad_right",
}
ABS_MAP = {0x00: "lx", 0x01: "ly", 0x03: "rx", 0x04: "ry", 0x10: "hat_x", 0x11: "hat_y"}

# 内核 struct js_event 是 8 字节：time(u32) / value(s16) / type(u8) / number(u8)。
# 头文件里那个额外的 __s32 value 是 union，和 s16 的 value 共用同一块内存，
# 所以"时间/值/类型/编号"四个字段到 value(i32) 的写法是 12 字节——按 12 解析会整体错位。
JS_EVENT = struct.Struct("<IhBB")
TAP_HOLD_S = 3.0                            # 十字上长按 3 s 切走路/轮滑


class Module(BaseModule):
    name = "pad"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        st = self.load_store()
        self._bindings: dict[str, str] = {**DEFAULT_BINDINGS,
                                          **(st.get("bindings") or {})}
        lim = st.get("limits") or {}
        self._limits = {"vx": self._lim(lim.get("vx"), LIMIT_VX, HARD_VX),
                        "wz": self._lim(lim.get("wz"), LIMIT_WZ, HARD_WZ)}
        self._mode = st.get("mode") or "drive"        # drive / walk / action
        self._gait = st.get("gait") or "walk"          # walk / wheel
        self._preferred: str | None = st.get("device")

        # 设备与读取
        self._device: dict | None = None               # {path, name, kind}
        self._reader: asyncio.Task | None = None
        self._candidates: list[dict] = []
        self._lost_at = 0.0
        self._connected = False
        self._source: str | None = None                # device / virtual
        self._virtual_at = 0.0                         # 虚拟手柄最后一次输入（它不该冒充"实体已连接"）

        # 输入状态（state() 只读这几个字段，不做 I/O）
        self._axes = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}
        self._hat = {"x": 0, "y": 0}
        self._pressed: set[str] = set()
        self._seq = -1
        self._dropped = 0
        self._watch = True

        # 速度意图
        self._target = (0.0, 0.0, 0.0)
        self._applied = (0.0, 0.0, 0.0)
        self._dirty = False
        self._moving = False
        self._last_sticks = 0.0
        self._armed = False
        self._wake = asyncio.Event()

        # 长按计时
        self._hold: dict | None = None
        self._hold_task: asyncio.Task | None = None
        self._last_estop = 0.0
        self._note = ""

    # ------------------------------------------------------------------ 生命周期
    async def start(self) -> None:
        self.spawn(self._monitor_loop())
        self.spawn(self._speed_loop())
        self.log.info("手柄模块就绪（绑定 %d 键，虚拟手柄随时可用）", len(self._bindings))

    async def stop(self) -> None:
        if self._hold_task:
            self._hold_task.cancel()
            self._hold_task = None
        await super().stop()
        await self._close_device()
        await self._send_speed(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ 状态
    def state(self) -> dict:
        hold = None
        if self._hold:
            hold = {"key": self._hold["key"],
                    "elapsed_s": round(time.monotonic() - self._hold["since"], 2),
                    "need_s": TAP_HOLD_S}
        age = (time.monotonic() - self._last_sticks) if self._last_sticks else None
        virtual = (time.monotonic() - self._virtual_at) < 5.0 if self._virtual_at else False
        return {"pad": {
            "connected": self._connected,
            "name": (self._device or {}).get("name"),
            "battery": None,          # joydev/evdev 都没有电量，别编
            "source": self._source,
            "virtual": virtual,
            "mode": self._mode,
            "gait": self._gait,
            "armed": self._armed,
            "bindings": dict(self._bindings),
            "axes": {k: round(v, 3) for k, v in self._axes.items()},
            "hat": dict(self._hat),
            "buttons": sorted(self._pressed),
            "hold": hold,
            "seq": self._seq,
            "dropped": self._dropped,
            "sticks_age_ms": round(age * 1000) if age is not None else None,
            "heartbeat_ms": int(HEARTBEAT_S * 1000),
            "limits": {"vx": self._limits["vx"], "wz": self._limits["wz"]},
            "speed": {"vx": round(self._applied[0], 3), "vy": round(self._applied[1], 3),
                      "wz": round(self._applied[2], 3)},
            "deadzone": DEADZONE,
            "expo": EXPO,
            "candidates": self._candidates,
            "path": (self._device or {}).get("path"),
            "hint": self._hint(virtual),
            "note": self._note,
        }}

    def _hint(self, virtual: bool) -> str:
        if not self._connected and not virtual:
            return "插上 USB 手柄，或直接打开网页虚拟手柄拖摇杆"
        if not self._armed:
            return "推一下摇杆就进入驾驶；Start 键随时急停"
        if not self._moving:
            return "摇杆回中了，鸭子站着不动；0.45 秒没指令会自动停车"
        return "推摇杆走，松手即停；Start 键急停"

    # ------------------------------------------------------------------ 命令
    async def cmd(self, op: str, args: dict) -> dict:
        if op == "sticks":
            return self._op_sticks(args)          # 纯内存操作，绝不 await
        if op == "key":
            return await self._op_key(args)
        if op == "watch":
            self._watch = bool(args.get("on", True))
            return {"watching": self._watch}
        if op == "scan":
            return await self._op_scan(args)
        if op == "pair":
            return await self._op_pair(args)
        if op == "forget":
            return await self._op_forget()
        if op == "bind":
            return self._op_bind(args)
        if op == "drive":
            return await self._op_drive(args)
        if op == "stop":
            return await self._op_stop()
        if op == "mode":
            return self._op_mode(args)
        if op == "start":
            return self._op_start()
        if op == "skill":
            return await self._op_skill(args)
        raise denied(f"手柄页没有 {op} 这个操作")

    # ---------------- 摇杆 ----------------
    def _op_sticks(self, args: dict) -> dict:
        seq = args.get("seq")
        if isinstance(seq, float) and seq.is_integer():    # JS 偶尔会把 3 传成 3.0，别为这个拒了
            seq = int(seq)
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise denied("摇杆指令必须带 seq（递增序号），否则丢包时会把旧位置当新位置")
        if seq <= self._seq:
            # 乱序/重放的包：直接丢，别让鸭子的脚来回抽
            self._dropped += 1
            return {"accepted": False, "dropped": True, "seq": self._seq}

        self._seq = seq
        self._source = "virtual"
        self._virtual_at = time.monotonic()
        self._lost_at = 0.0
        if "mode" in args:
            self._mode = str(args["mode"]) if str(args["mode"]) in ("drive", "walk", "action") else self._mode

        # 页面可以按当天场地收紧限速；但只能往小调，硬上限不给破
        lim = args.get("limit")
        if isinstance(lim, dict):
            self._limits["vx"] = self._lim(lim.get("vx"), self._limits["vx"], HARD_VX)
            self._limits["wz"] = self._lim(lim.get("wz"), self._limits["wz"], HARD_WZ)
            self._save()

        lx, ly = self._clamp1(args.get("lx", self._axes["lx"])), self._clamp1(args.get("ly", self._axes["ly"]))
        rx, ry = self._clamp1(args.get("rx", self._axes["rx"])), self._clamp1(args.get("ry", self._axes["ry"]))
        self._axes.update({"lx": lx, "ly": ly, "rx": rx, "ry": ry})
        self._last_sticks = time.monotonic()
        self._armed = True

        # Linux 摇杆往上推是负值，映射成"往前开是正 vx"
        vx = -self._shape(ly) * self._limits["vx"]
        vy = -self._shape(lx) * self._limits["vx"]
        wz = -self._shape(rx) * self._limits["wz"]
        self._set_target(vx, vy, wz)
        return {"accepted": True, "seq": seq,
                "speed": {"vx": round(vx, 3), "vy": round(vy, 3), "wz": round(wz, 3)}}

    def _set_target(self, vx: float, vy: float, wz: float) -> None:
        self._target = (self._cap(vx, self._limits["vx"], HARD_VX),
                        self._cap(vy, self._limits["vx"], HARD_VX),
                        self._cap(wz, self._limits["wz"], HARD_WZ))
        self._dirty = True
        self._wake.set()

    async def _op_drive(self, args: dict) -> dict:
        on = bool(args.get("on", True))
        if not on:
            self._armed = False
            self._set_target(0.0, 0.0, 0.0)
            self._applied = (0.0, 0.0, 0.0)
            await self._send_speed(0.0, 0.0, 0.0)
            self.log_event("已松开油门，鸭子停住", cat="pad")
            return {"armed": False, "speed": {"vx": 0.0, "vy": 0.0, "wz": 0.0}}
        self._armed = True
        self._note = "已进入驾驶"
        self.log_event("进入驾驶模式：推摇杆走", cat="pad")
        return {"armed": True, "mode": self._mode}

    async def _op_stop(self) -> dict:
        """停车 ≠ 卸力：只是把速度归零，鸭子还站着。急停请按 Start。"""
        self._set_target(0.0, 0.0, 0.0)
        self._applied = (0.0, 0.0, 0.0)
        self._armed = False
        await self._send_speed(0.0, 0.0, 0.0)
        return {"stopped": True, "note": "已停车（舵机仍然保持，需要卸力请按 Start）"}

    def _op_mode(self, args: dict) -> dict:
        mode = str(args.get("mode") or "")
        if mode not in ("drive", "walk", "action"):
            raise denied("模式只有三种：drive（摇杆开车）/ walk（走路策略）/ action（动作触发）")
        self._mode = mode
        self._save()
        self.log_event(f"手柄模式：{mode}", cat="pad")
        return {"mode": mode}

    def _op_start(self) -> dict:
        """start = 手柄会话上线（武装），与 Start 键的急停是两件事。"""
        self._armed = True
        return {"armed": True, "note": "手柄已接管，摇杆可用；Start 键是急停"}

    async def _op_skill(self, args: dict) -> dict:
        name = str(args.get("name") or "").strip()
        return await self._request_slot(name, "pad.skill")

    # ---------------- 绑定 ----------------
    def _op_bind(self, args: dict) -> dict:
        button = str(args.get("button") or "").strip().lower()
        slot = str(args.get("slot") or "none").strip().lower()
        known = set(JS_BUTTONS.values()) | {"dpad_up"}
        if button not in known:
            raise denied(f"没有 {button} 这个键，可绑：{'、'.join(sorted(known))}")
        if slot and slot not in SLOTS and slot not in SYSTEM_ACTIONS:
            raise denied(f"没有 {slot} 这个槽位，可绑：{'、'.join(SLOTS + SYSTEM_ACTIONS)}")
        if not slot or slot == "none":
            self._bindings.pop(button, None)
        else:
            self._bindings[button] = slot
        self._save()
        self.log_event(f"键位：{button} → {slot or '（空）'}", cat="pad")
        return {"button": button, "slot": slot or None, "bindings": dict(self._bindings)}

    # ---------------- 设备 ----------------
    async def _op_scan(self, args: dict) -> dict:
        self._candidates = self._scan(include_events=bool(args.get("all", True)))
        names = [c["name"] for c in self._candidates]
        if not names:
            self.log_event("没找到手柄设备（/dev/input 里没有 js*）", level="warning", cat="pad")
        return {"devices": self._candidates, "found": len(self._candidates)}

    async def _op_pair(self, args: dict) -> dict:
        """配对向导：真正配对（蓝牙）在系统层做，这里负责"选中 + 打开 + 报名字"。"""
        pick = args.get("path") or args.get("pick")
        cands = self._scan(include_events=bool(args.get("all", True)))
        self._candidates = cands
        if pick:
            chosen = next((c for c in cands if c["path"] == pick or c["name"] == pick), None)
            if not chosen:
                chosen = {"path": str(pick), "name": Path(str(pick)).name, "kind": "js"
                          if "js" in Path(str(pick)).name else "event"}
        else:
            chosen = cands[0] if cands else None
        if not chosen:
            raise hardware("没插手柄。无线手柄请先用系统蓝牙配好（设置 → 蓝牙），再回来点扫描")
        self._preferred = chosen["path"]
        self._save()
        await self._open_device(chosen)
        steps = [
            "1. 无线手柄：先进『设置 → 蓝牙』配对，名字出现在系统里才算连上",
            "2. 有线手柄：插上 USB，等 1 秒，点『扫描』",
            "3. 在列表里点你的手柄，看到绿色『已连接』再离开本页",
            "4. 去下面的按键绑定，把 A/X/LB/RB 指到动作槽",
        ]
        self.log_event(f"已选手柄：{chosen['name']}（{chosen['path']}）", cat="pad")
        return {"device": chosen, "steps": steps, "devices": cands}

    async def _op_forget(self) -> dict:
        await self._close_device()
        self._preferred = None
        self._connected = False
        self._source = None
        self._axes = {k: 0.0 for k in self._axes}
        self._save()
        self.log_event("已忘记这台手柄，下次扫描重新选", cat="pad")
        return {"forgotten": True}

    # ---------------- 按键 ----------------
    async def _op_key(self, args: dict) -> dict:
        """网页虚拟键盘。down 不传就是"点一下"，传了就按下去/抬起来。"""
        key = str(args.get("key") or "").strip().lower()
        if key not in set(JS_BUTTONS.values()):
            raise denied(f"没有 {key} 这个键")
        down = args.get("down")
        self._source = "virtual"
        self._virtual_at = time.monotonic()
        if down is None:
            await self._press(key)
            await asyncio.sleep(0.06)
            await self._release(key)
            return {"key": key, "tapped": True}
        if down:
            await self._press(key)
        else:
            await self._release(key)
        return {"key": key, "down": bool(down)}

    async def _press(self, key: str) -> None:
        if key in self._pressed:
            return
        self._pressed.add(key)
        action = self._bindings.get(key)
        if not action:
            return
        if action == "estop":
            await self._estop()
        elif action == "servo_restart":
            await self._restart_servos()
        elif action == "gait_toggle":
            self._hold = {"key": key, "since": time.monotonic()}
            self._hold_task = asyncio.create_task(self._hold_gaits(key))
        elif action == "sit_toggle":
            await self._sit_toggle()
        else:
            await self._request_slot(action, f"pad.key.{key}")

    async def _release(self, key: str) -> None:
        self._pressed.discard(key)
        if self._hold and self._hold["key"] == key:
            if self._hold_task:
                self._hold_task.cancel()
                self._hold_task = None
            self._hold = None

    async def _hold_gaits(self, key: str) -> None:
        """十字上长按 3 s：走路 ↔ 轮滑。短按什么都不做，防误触。"""
        try:
            await asyncio.sleep(TAP_HOLD_S)
        except asyncio.CancelledError:
            return
        if key not in self._pressed:
            return
        self._gait = "wheel" if self._gait == "walk" else "walk"
        self._hold = None
        self._save()
        text = "轮滑（足底轮子滑行）" if self._gait == "wheel" else "走路（步态）"
        self.log_event(f"步态切到{text}", cat="pad")
        self.emit("pad.gait", {"gait": self._gait})
        self._note = f"步态：{text}"

    # ---------------- 系统动作 ----------------
    async def _estop(self) -> None:
        self._last_estop = time.time()
        self._set_target(0.0, 0.0, 0.0)
        self._applied = (0.0, 0.0, 0.0)
        self._armed = False
        self.log_event("手柄急停：全部卸力", level="warning", cat="pad")
        await self.ctx.hub.relax()     # 与顶栏红键、空格键同一条路

    async def _restart_servos(self) -> None:
        """十字右：舵机重新上电初始化。会先卸力再接管，鸭子会软一下，人得扶着。"""
        self.log_event("重启舵机总线…", level="warning", cat="pad")
        try:
            await self.hal.servo_release()
            await asyncio.sleep(0.08)
            await self.hal.servo_take()
            await self.hal.servo_torque(True)
        except Exception as exc:
            self.log_event(f"重启舵机失败：{exc}", level="error", cat="pad")
            raise hardware(f"重启舵机失败：{exc}") from exc
        self.log_event("舵机已重新初始化，保持住了；请重新让鸭子站好", cat="pad")

    async def _sit_toggle(self) -> None:
        going_sit = self.ctx.hub.posture != "sitting"
        try:
            await self.hal.pose("sit" if going_sit else "stand")
        except Exception as exc:
            raise hardware(f"换姿势失败：{exc}") from exc
        self.log_event("坐下" if going_sit else "站起来", cat="pad")

    async def _request_slot(self, slot: str, source: str) -> dict:
        if slot not in SLOTS:
            raise denied(f"没有 {slot} 这个动作槽；可用的：{'、'.join(SLOTS)}")
        # 轮滑模式仍用 walk 槽，靠 gait 区分，motiond 侧选不同权重
        posture = self.ctx.hub.posture
        if posture in ("limp", "absent", "homing") and self.config.mode != "sim":
            raise denied("鸭子现在是软的或者在校 0 位，先让它站好（舵机页 → 站姿）再触发动作")
        train = self.ctx.hub.modules.get("train")
        hook = getattr(train, "request_slot", None)
        if hook:
            info = hook(slot, source=source)
        else:
            info = {"slot": slot}
        self.emit("pad.slot", {"slot": slot, "source": source, "gait": self._gait})
        self.log_event(f"触发动作：{slot}", cat="pad")
        return {"requested": slot, "source": source, "gait": self._gait, **info}

    # ------------------------------------------------------------------ 速度环路
    async def _speed_loop(self) -> None:
        """唯一的刹车点：脏了就发速度；心跳超时就发零。"""
        period = 1.0 / SPEED_HZ
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=period)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            now = time.monotonic()

            if self._moving and self._last_sticks and now - self._last_sticks > HEARTBEAT_S:
                # 心跳失联停车：页面卡死、WiFi 掉线、手指滑出浏览器都靠这一条兜住
                self._set_target(0.0, 0.0, 0.0)
                self._applied = (0.0, 0.0, 0.0)
                self._moving = False          # 必须落回去，否则下一轮又进这个分支，空转饿死事件循环
                self._armed = False
                await self._send_speed(0.0, 0.0, 0.0)
                self.log_event("摇杆 0.45 秒没心跳了，自动停车", level="warning", cat="pad")
                self._note = "心跳失联自动停车"
                self._dirty = False
                continue

            if not self._dirty:
                continue
            self._dirty = False
            vx, vy, wz = self._target
            if not self._armed:
                vx = vy = wz = 0.0
            if (vx, vy, wz) == self._applied:
                continue
            self._applied = (vx, vy, wz)
            self._moving = any(abs(v) > 1e-4 for v in self._applied)
            await self._send_speed(vx, vy, wz)

    async def _send_speed(self, vx: float, vy: float, wz: float) -> None:
        try:
            await asyncio.wait_for(self.hal.move(vx, vy, wz), timeout=0.2)
        except asyncio.TimeoutError:
            self.log_event("速度指令超时（实时层没响应）", level="warning", cat="pad")
        except Exception as exc:
            self.log_event(f"速度指令失败：{exc}", level="error", cat="pad")

    # ------------------------------------------------------------------ 热插拔
    async def _monitor_loop(self) -> None:
        while True:
            try:
                if self._device is None:
                    await self._auto_open()
                if self._device is None and self._lost_at and self._connected:
                    if time.monotonic() - self._lost_at > DISCONNECT_S:
                        self._mark_lost()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.log.exception("手柄轮询出错")
            await asyncio.sleep(1.2)

    def _mark_lost(self) -> None:
        self._connected = False
        self._source = None
        self._axes = {k: 0.0 for k in self._axes}
        self._hat = {"x": 0, "y": 0}
        self._pressed.clear()
        self._set_target(0.0, 0.0, 0.0)
        self._note = "手柄断了，摇杆已回中"
        self.log_event("手柄断连超过 1.5 秒，摇杆已回中并停车", level="warning", cat="pad")

    async def _auto_open(self) -> None:
        cands = self._scan(include_events=False)
        self._candidates = cands or self._candidates
        pick = None
        if self._preferred:
            pick = next((c for c in cands if c["path"] == self._preferred), None)
        if not pick and cands:
            pick = cands[0]
        if pick:
            await self._open_device(pick)

    def _scan(self, include_events: bool = False) -> list[dict]:
        root = Path("/dev/input")
        if not root.exists():
            return []
        out: list[dict] = []
        for p in sorted(root.glob("js*")):
            out.append({"path": str(p), "kind": "js", "name": self._name_of(str(p), True)})
        if include_events:
            skip = ("keyboard", "mouse", "power button", "video bus", "consumer control",
                    "lid switch", "sleep button")
            for p in sorted(root.glob("event*")):
                name = self._name_of(str(p), False)
                if any(s in name.lower() for s in skip):
                    continue
                out.append({"path": str(p), "kind": "event", "name": name})
        return out

    def _name_of(self, path: str, is_js: bool) -> str:
        """用 ioctl 问内核要设备名；拿不到就用文件名（不算错）。"""
        try:
            import fcntl

            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            try:
                req = 0x80806A13 if is_js else 0x81004506   # JSIOCGNAME(128) / EVIOCGNAME(256)
                buf = fcntl.ioctl(fd, req, b"\x00" * (128 if is_js else 256))
                name = buf.split(b"\x00", 1)[0].decode("utf-8", "ignore").strip()
                return name or Path(path).name
            finally:
                os.close(fd)
        except Exception:
            return Path(path).name

    async def _open_device(self, dev: dict) -> None:
        await self._close_device()
        self._device = dev                    # 先登记再起任务：读取任务失败时会用它在 _lost 里认门
        try:
            if dev["kind"] == "js":
                self._reader = asyncio.create_task(self._read_js(dev))
            else:
                self._reader = asyncio.create_task(self._read_event(dev))
        except Exception as exc:
            self._device = None
            raise hardware(f"打不开 {dev['path']}：{exc}（可能没有权限，或设备已被别的进程独占）") from exc
        self._connected = True
        self._source = "device"
        self._lost_at = 0.0
        self._note = ""
        self.log_event(f"手柄已连接：{dev['name']}（{dev['path']}）", cat="pad")

    async def _close_device(self) -> None:
        if self._reader:
            self._reader.cancel()
            self._reader = None
        self._device = None

    async def _read_stream(self, path: str, size: int):
        """读一个字符设备：优先 epoll（add_reader），没有就 10 ms 轮询。产出 bytes 块。"""
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        use_reader = hasattr(loop, "add_reader") and os.name != "nt"

        def pump() -> None:
            try:
                data = os.read(fd, size * 64)
            except BlockingIOError:
                return
            except OSError:
                q.put_nowait(None)
                return
            if not data:
                q.put_nowait(None)
                return
            if not q.full():
                q.put_nowait(data)

        try:
            if use_reader:
                loop.add_reader(fd, pump)
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    yield item
            else:
                while True:
                    pump()
                    while not q.empty():
                        item = q.get_nowait()
                        if item is None:
                            return
                        yield item
                    await asyncio.sleep(0.01)
        finally:
            if use_reader:
                try:
                    loop.remove_reader(fd)
                except Exception:
                    pass
            try:
                os.close(fd)
            except Exception:
                pass

    async def _read_js(self, dev: dict) -> None:
        buf = b""
        try:
            async for chunk in self._read_stream(dev["path"], JS_EVENT.size):
                buf += chunk
                while len(buf) >= JS_EVENT.size:
                    raw, buf = buf[:JS_EVENT.size], buf[JS_EVENT.size:]
                    _, value, etype, number = JS_EVENT.unpack(raw)
                    await self._on_js_event(etype, number, value)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log_event(f"读手柄失败：{exc}", level="error", cat="pad")
        finally:
            await self._lost(dev)

    async def _on_js_event(self, etype: int, number: int, value: int) -> None:
        kind = etype & 0x7F          # 0x80 位是"初始值"，同样要收，否则上电瞬间的摇杆位置会丢
        if kind == 0x01:
            btn = JS_BUTTONS.get(number)
            if not btn:
                return
            if value:
                await self._press(btn)
            else:
                await self._release(btn)
        elif kind == 0x02:
            if number in JS_AXES:
                axis = JS_AXES[number]
                self._axes[axis] = self._clamp1(value / 32767.0)
                if self._armed and self._source == "device":
                    await self._on_axis_moved()
            elif number in JS_HAT:
                raw = value if abs(value) <= 1 else value / 32767.0
                self._hat[JS_HAT[number]] = int(round(raw))
                await self._on_hat()

    async def _read_event(self, dev: dict) -> None:
        fmt = "<llHHi" if struct.calcsize("<llHHi") == 24 else "<IIHHi"
        ev = struct.Struct(fmt)
        ranges = self._abs_ranges(dev["path"])
        buf = b""
        try:
            async for chunk in self._read_stream(dev["path"], ev.size):
                buf += chunk
                while len(buf) >= ev.size:
                    raw, buf = buf[:ev.size], buf[ev.size:]
                    _, _, etype, code, value = ev.unpack(raw)
                    if etype == EV_KEY:
                        btn = BTN_MAP.get(code)
                        if btn:
                            if value:
                                await self._press(btn)
                            else:
                                await self._release(btn)
                    elif etype == EV_ABS:
                        await self._on_abs(code, value, ranges)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log_event(f"读手柄失败：{exc}", level="error", cat="pad")
        finally:
            await self._lost(dev)

    async def _on_abs(self, code: int, value: int, ranges: dict) -> None:
        key = ABS_MAP.get(code)
        if not key:
            return
        lo, hi = ranges.get(code, (0, 255))
        span = max(1, hi - lo)
        norm = (value - lo) / span * 2.0 - 1.0
        if key.startswith("hat_"):
            self._hat[key[-1]] = int(round(norm))
            await self._on_hat()
        else:
            self._axes[key] = self._clamp1(norm)
            if self._armed and self._source == "device":
                await self._on_axis_moved()

    def _abs_ranges(self, path: str) -> dict:
        """event 设备的轴量程（0–255 / -32767–32767 / 0–4095 都可能），不查就只能瞎猜。"""
        out: dict[int, tuple[int, int]] = {}
        try:
            import fcntl

            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            try:
                for code in (0x00, 0x01, 0x03, 0x04, 0x10, 0x11):
                    try:
                        buf = fcntl.ioctl(fd, 0x80184540 + code, b"\x00" * 24)
                        _, vmin, vmax, *_ = struct.unpack("<iiiiii", buf)
                        if vmax > vmin:
                            out[code] = (vmin, vmax)
                    except Exception:
                        continue
            finally:
                os.close(fd)
        except Exception:
            pass
        return out

    async def _on_axis_moved(self) -> None:
        self._last_sticks = time.monotonic()
        vx = -self._shape(self._axes["ly"]) * self._limits["vx"]
        vy = -self._shape(self._axes["lx"]) * self._limits["vx"]
        wz = -self._shape(self._axes["rx"]) * self._limits["wz"]
        self._set_target(vx, vy, wz)

    async def _on_hat(self) -> None:
        """十字键在 js 里是 hat 轴，翻成按键按下/抬起，好让长按逻辑与数据通路一致。"""
        x, y = self._hat["x"], self._hat["y"]
        want = set()
        if y < 0:
            want.add("dpad_up")
        if y > 0:
            want.add("dpad_down")
        if x < 0:
            want.add("dpad_left")
        if x > 0:
            want.add("dpad_right")
        held = {b for b in self._pressed if b.startswith("dpad_")}
        for b in want - held:
            await self._press(b)
        for b in held - want:
            await self._release(b)

    async def _lost(self, dev: dict) -> None:
        if (self._device or {}).get("path") != dev["path"]:
            return
        self._device = None
        self._reader = None
        self._lost_at = time.monotonic()
        self.log_event(f"手柄读取中断（{dev['path']}），1.5 秒内回来会自动重连", level="warning", cat="pad")

    # ------------------------------------------------------------------ 小工具
    @staticmethod
    def _shape(v: float) -> float:
        """死区 0.08 + 指数 1.6：手指放在摇杆上不动时鸭子不能抖。"""
        a = abs(v)
        if a <= DEADZONE:
            return 0.0
        x = (a - DEADZONE) / (1.0 - DEADZONE)
        return math.copysign(min(1.0, x) ** EXPO, v)

    @staticmethod
    def _clamp1(v) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return -1.0 if f < -1.0 else 1.0 if f > 1.0 else f

    @staticmethod
    def _cap(v: float, soft: float, hard: float) -> float:
        lim = min(abs(v), soft, hard)
        return math.copysign(lim, v)

    @staticmethod
    def _lim(raw, default: float, hard: float) -> float:
        try:
            v = abs(float(raw))
        except (TypeError, ValueError):
            return default
        if v <= 0:
            return default
        return min(v, hard)

    def _save(self) -> None:
        self.save_store({"bindings": self._bindings, "limits": self._limits,
                         "mode": self._mode, "gait": self._gait, "device": self._preferred,
                         "updated": round(time.time(), 1)})
