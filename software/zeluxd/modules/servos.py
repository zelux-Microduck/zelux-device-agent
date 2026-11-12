"""舵机页（`name = "servos"`）：接管会话、寄存器编辑、0 位微调、官方姿势、草稿落盘。

三条产品语义，写在这里是因为页面和后端全靠它们对齐：

1. **接管是"会话"，不是"请求"**：`bus take` 成功后一直保持，离开页面也不交还
   ——交还/卸力会让鸭子当场趴下，摔一次可能就是几百块的腿。只有显式 `release` 才还。
2. **草稿与标定分离**：任何写（0 位、方向、限位）先只进 `state_dir/servos_draft.json`，
   只有 `commit_draft` 才写 `state_dir/servo_calib.json` 并落到舵机 EEPROM。
   这样标定过程中随时能反悔，不会把现场写坏。
3. **撤销只回一步**：草稿里带上一份快照（`prev`），`undo` 回滚并同步写回舵机。

姿势常量直接抄 microduck 场景关键帧，弧度制；刻度换算 `2048 + rad × 4096/2π`。
15 个值顺序与 `hal/base.py` 的 `JOINT_IDS` 严格一致：右腿 10–14、左腿 20–24、头颈 30–34。
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import time
from pathlib import Path

from ..errors import ZeluxError, denied, hardware, unsupported
from ..hal.base import JOINT_IDS, JOINT_NAMES
from .base import BaseModule

CENTER = 2048
TICKS_PER_RAD = 4096 / (2 * math.pi)

# 寄存器号（与 hal/radxa_servo.py 的表同源；这里只用到 0 位/方向/限位）
R_OFFSET, R_PHASE, R_MIN, R_MAX = 31, 18, 9, 11

try:  # 单一真源：危险寄存器文案在驱动里，页面文案跟着它走
    from ..hal.radxa_servo import DANGER_TEXT as _DANGER
except Exception:  # pragma: no cover - 只在本机缺依赖时兜底
    _DANGER = {}

DANGER: dict[int, str] = dict(_DANGER) or {
    5: "改 ID 后这颗就再也找不到了，只能单独接上、用新 ID 重新扫",
    6: "改了波特率整条总线立刻失联，得单颗用默认波特率救回来",
    8: "应答级别改成不回复，之后所有读取都会超时",
    9: "限位设反会让关节顶死在机构上，堵转几秒就烫手",
    11: "限位设反会让关节顶死在机构上，堵转几秒就烫手",
    18: "方向反了姿势就是镜像的，两条腿会互相打架",
    19: "关掉卸载保护后堵转不卸力，会烧线圈",
    33: "运行模式不对时，位置指令会变成电机一直转",
}

# 官方姿势（弧度）。顺序同 JOINT_IDS。
POSES: dict[str, list[float]] = {
    "zero": [0.0] * 15,
    "stand": [0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491, 0, 0, 0, 0,
              0.0873, 0.4579, 0.0049, -0.453],
    "sit": [0, 0, -0.5236, 1.0472, 0, 0.5, 1.6, 0, 0, 0, 0, 0, 0.5236, -1.0472, 0],
    "fold": [0, 0, 1.57, 1.57, 0, 1, 1, 0, 0, 0, 0, 0, -1.57, -1.57, 0],
}

# 姿势到位速度：太快会甩、太慢用户以为没反应。900 步/秒约 0.5 s 走完半个行程。
POSE_SPEED = 900


def pose_ticks(name: str) -> list[int]:
    """姿势名 → 15 个刻度值。超出 0–4095 的直接夹住（姿势表是理论值，真机有限位）。"""
    rad = POSES.get(name)
    if rad is None:
        raise unsupported(f"没有「{name}」这个姿势，只有 {'/'.join(POSES)}")
    return [max(0, min(4095, int(round(CENTER + r * TICKS_PER_RAD)))) for r in rad]


def _load_json(path: Path, default: dict) -> dict:
    try:
        data = json.loads(path.read_text("utf-8"))
        return data if isinstance(data, dict) else default
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)


class Module(BaseModule):
    name = "servos"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        d = Path(self.config.state_dir)
        self.draft_path = d / "servos_draft.json"
        self.calib_path = d / "servo_calib.json"
        self._draft: dict = _load_json(self.draft_path, {})
        if not self._draft:
            self._draft = {"ts": 0.0, "items": {}, "prev": None, "writes": []}
        self._draft.setdefault("items", {})
        self._draft.setdefault("writes", [])
        self._session: dict | None = None
        self._calib: dict = _load_json(self.calib_path, {})
        if self._calib:
            self.ctx.state["servo_calibrated"] = True
        self._alias()

    # ---------------- 生命周期 ----------------
    async def start(self) -> None:
        hub = getattr(self.ctx, "hub", None)
        locks = getattr(hub, "locks", None)
        if isinstance(locks, dict):
            lock = locks.get(self.name)
            if lock is not None:
                locks.setdefault("servo", lock)

    def _alias(self) -> None:
        """给 `servo` 挂个别名。

        契约 §3.3 的命令前缀是单数（`servo.bus`、`servo.pose`），而模块名/状态键是复数
        `servos`（见 §4 的 `"servos"`）。hub 按模块名派发，两者对不上前端就全打到空处；
        hub 不归我改，所以在这里让同一份实例有两个键。锁在 `start()` 里对齐（那时
        hub 已经给 `servos` 建好锁，共用一把才能保证两条前缀不会并发碰总线）。
        """
        hub = getattr(self.ctx, "hub", None)
        mods = getattr(hub, "modules", None)
        if isinstance(mods, dict):
            mods.setdefault("servo", self)

    def state(self) -> dict:
        snap = dict(self.hal.snapshot_servos() or {})
        snap.setdefault("online", 0)
        snap.setdefault("items", [])
        snap.setdefault("bus", "shared")
        snap["draft"] = self._draft_summary()
        return {"servos": snap, "servo_session": self._session_view()}

    def _session_view(self) -> dict | None:
        if not self._session:
            return None
        snap = self.hal.snapshot_servos() or {}
        items = snap.get("items") or []
        online = snap.get("online") or self._session.get("online", 0)
        return {
            "since": self._session.get("since"),
            "took_ms": self._session.get("took_ms"),
            "online": online,
            "holding": sum(1 for i in items if i.get("torque")),
            "unknown": max(0, len(JOINT_IDS) - int(online or 0)),
            "missing": self._session.get("missing", []),
            "draft_dirty": self._dirty_count() > 0,
        }

    def _draft_summary(self) -> dict:
        return {"dirty": self._dirty_count(), "changed": sorted(
            int(k) for k, v in self._draft["items"].items() if v.get("touched")),
            "undo": self._draft.get("prev") is not None, "ts": self._draft.get("ts", 0)}

    def _dirty_count(self) -> int:
        base = (self._calib.get("items") or {})
        n = 0
        for sid, item in self._draft["items"].items():
            ref = base.get(sid) or {}
            for key in ("offset", "direction", "min", "max"):
                if key in item and item[key] != ref.get(key):
                    n += 1
                    break
        return n

    # ---------------- 命令入口 ----------------
    async def cmd(self, op: str, args: dict) -> dict:
        args = args or {}
        if op == "bus":
            return await self._bus(args)
        if op == "read":
            return await self._read(args)
        if op == "write":
            return await self._write(args)
        if op == "trim":
            return await self._trim(args)
        if op == "pose":
            return await self._pose(args)
        if op == "torque":
            return await self._torque(args)
        if op == "commit_draft":
            return await self._commit(args)
        if op == "export_regs":
            return await self._export(args)
        if op == "stop":
            return await self._stop(args)
        if op == "undo":
            return await self._undo(args)
        if op == "draft":
            return {"draft": self._draft, "calib": self._calib, "summary": self._draft_summary()}
        raise unsupported(f"servos 不支持 {op}")

    # ---------------- 总线会话 ----------------
    async def _bus(self, args: dict) -> dict:
        sub = str(args.get("op", "status"))
        if sub == "take":
            t0 = time.monotonic()
            res = await self.hal.servo_take()
            res = res if isinstance(res, dict) else {}
            took = res.get("took_ms")
            if took is None:      # 模拟 HAL 不报耗时，这里自己量
                took = int((time.monotonic() - t0) * 1000)
            snap = self.hal.snapshot_servos() or {}
            if not snap.get("items") and not res.get("online"):
                # 模拟 HAL 的首帧缓存要等一个采样周期，别让页面第一眼看到 0 颗在线
                await asyncio.sleep(0.06)
                snap = self.hal.snapshot_servos() or {}
            online = int(res.get("online") or snap.get("online") or 0)
            self._session = {"since": time.time(), "took_ms": took, "online": online,
                             "missing": list(res.get("missing") or [])}
            self.ctx.hub.posture = "servo_session"
            self.log_event(f"已接管舵机总线：{online}/15 在线"
                           + (f"，缺 {self._session['missing']}" if self._session["missing"] else ""))
            return {"ok": True, "session": self._session_view(), "took_ms": took,
                    "online": online, "missing": self._session["missing"],
                    "note": "会话会一直保持，离开页面也不交还"}
        if sub == "release":
            await self.hal.servo_release()
            self._session = None
            self.ctx.hub.posture = "holding"
            self.log_event("已交还舵机总线（舵机仍保持住，不会卸力）")
            return {"ok": True, "session": None, "note": "舵机保持在当前位置"}
        if sub not in ("status", "", None):
            raise unsupported(f"bus 只有 take/release/status，收到 {sub}")
        snap = self.hal.snapshot_servos() or {}
        return {"ok": True, "taken": self._session is not None,
                "session": self._session_view(),
                "port": snap.get("port"), "baud": snap.get("baud"),
                "online": snap.get("online", 0), "bus": snap.get("bus"),
                "hal": self.hal.info() if hasattr(self.hal, "info") else {}}

    def _need_session(self) -> None:
        if not self._session:
            raise hardware("还没接管舵机总线：先在页面上点「开始调舵机」")

    # ---------------- 单颗读写 ----------------
    async def _read(self, args: dict) -> dict:
        self._need_session()
        sid = self._sid(args)
        rec = await self.hal.servo_read(sid)
        draft = self._draft["items"].get(str(sid)) or {}
        return {"servo": {**rec, "draft": draft,
                          "name": JOINT_NAMES[JOINT_IDS.index(sid)] if sid in JOINT_IDS else None}}

    async def _write(self, args: dict) -> dict:
        self._need_session()
        sid = self._sid(args)
        try:
            reg = int(args["reg"])
            value = int(args["value"])
        except (KeyError, TypeError, ValueError):
            raise ZeluxError("写寄存器要带 id / reg / value", "denied") from None

        if reg in DANGER and args.get("confirm") is not True:
            # 这里故意把后果写全：让人在点确认之前就知道会坏在哪
            raise denied(f"寄存器 {reg} 是危险寄存器，改了会出事：{DANGER[reg]}。"
                         f"确实要改请带 confirm:true")

        await self.hal.servo_write(sid, reg, value, bool(args.get("eeprom", False)))
        self._touch(sid, reg, value)
        self._save_draft()
        self.log_event(f"{sid} 号 寄存器 {reg} ← {value}")
        return {"ok": True, "id": sid, "reg": reg, "value": value,
                "draft": self._draft["items"].get(str(sid))}

    def _sid(self, args: dict) -> int:
        try:
            sid = int(args["id"])
        except (KeyError, TypeError, ValueError):
            raise ZeluxError("要指名是哪一颗舵机（id）", "denied") from None
        if sid not in JOINT_IDS:
            raise ZeluxError(f"{sid} 不是这台鸭子的关节（只有 {JOINT_IDS}）", "denied")
        return sid

    # ---------------- 0 位微调 ----------------
    async def _trim(self, args: dict) -> dict:
        self._need_session()
        sid = self._sid(args)
        try:
            delta = int(args.get("delta", 0))
        except (TypeError, ValueError):
            delta = 0
        item = self._item(sid)
        if "offset" in item:
            cur = int(item["offset"])
        else:
            # 第一次微调要从舵机里真实的 0 位起步，否则跨会话会越调越偏
            rec = await self.hal.servo_read(sid)
            cur = int(rec.get("offset", 0) or 0)
            item["offset"] = cur
        new = max(-1024, min(1024, cur + delta))
        await self._apply_offset(sid, new)
        self._snapshot()
        item["offset"] = new
        item["touched"] = True
        self._save_draft()
        return {"ok": True, "id": sid, "offset": new, "delta": new - cur,
                "rad": round((new) / TICKS_PER_RAD, 4), "dirty": self._dirty_count()}

    async def _apply_offset(self, sid: int, ticks: int) -> None:
        """有专用方法就用（它会顺手处理补码），没有就退回写寄存器。"""
        setter = getattr(self.hal, "set_offset", None)
        if setter:
            await setter(sid, ticks)
        else:
            await self.hal.servo_write(sid, R_OFFSET, ticks & 0xFFFF, True)

    # ---------------- 官方姿势 ----------------
    async def _pose(self, args: dict) -> dict:
        self._need_session()
        name = str(args.get("name") or args.get("pose") or "")
        ticks = pose_ticks(name)
        goals = dict(zip(JOINT_IDS, ticks))
        # 先加力再下目标：反过来会看到"先塌一下再抬起来"
        await self.hal.servo_torque(True)
        sync = getattr(self.hal, "write_goal_batch", None)
        if sync:
            await sync(goals, POSE_SPEED)
        else:
            await self.hal.servo_sync_goal(goals)
        self.log_event(f"到「{name}」姿势（{POSE_SPEED} 步/秒）")
        return {"ok": True, "pose": name, "goals": {str(k): v for k, v in goals.items()},
                "speed": POSE_SPEED}

    async def _torque(self, args: dict) -> dict:
        self._need_session()
        on = bool(args.get("on", True))
        ids = args.get("ids")
        ids = [int(i) for i in ids] if ids else None
        await self.hal.servo_torque(on, ids)
        self.log_event("全部加力（保持）" if on else "卸力——鸭子会软下来，扶一下",
                       level="info" if on else "warning")
        return {"ok": True, "on": on, "ids": ids or JOINT_IDS}

    # ---------------- 停止 ----------------
    async def _stop(self, args: dict) -> dict:
        self._need_session()
        snap = self.hal.snapshot_servos() or {}
        items = snap.get("items") or []
        if not items:
            items = [await self.hal.servo_read(sid) for sid in JOINT_IDS]
        goals = {int(i["id"]): int(i["pos"]) for i in items if "pos" in i}
        sync = getattr(self.hal, "write_goal_batch", None)
        if sync:
            await sync(goals, 0)          # 速度 0 = 不限速，但目标是"原地"，所以是急停
        else:
            await self.hal.servo_sync_goal(goals)
        self.log_event("就地急停：目标改成当前位置", level="warning")
        return {"ok": True, "frozen": len(goals), "goals": {str(k): v for k, v in goals.items()}}

    # ---------------- 草稿 / 落盘 ----------------
    def _item(self, sid: int) -> dict:
        return self._draft["items"].setdefault(str(sid), {})

    def _snapshot(self) -> None:
        """把当前草稿压进 prev（一份），undo 就靠它。"""
        self._draft["prev"] = {"ts": time.time(), "items": json.loads(json.dumps(self._draft["items"]))}

    def _touch(self, sid: int, reg: int, value: int) -> None:
        self._snapshot()
        item = self._item(sid)
        if reg == R_OFFSET:
            item["offset"] = value - 65536 if value > 32767 else value
        elif reg == R_PHASE:
            item["phase"] = value
            item["direction"] = -1 if value & 0x01 else 1
        elif reg == R_MIN:
            item["min"] = value
        elif reg == R_MAX:
            item["max"] = value
        else:
            # 其它寄存器不进标定文件，但要留在会话日志里，方便回看"刚才改了什么"
            self._draft["writes"].append({"ts": time.time(), "id": sid, "reg": reg, "value": value})
            del self._draft["writes"][:-50]
            return
        item["touched"] = True

    def _save_draft(self) -> None:
        self._draft["ts"] = time.time()
        _write_json(self.draft_path, self._draft)

    async def _undo(self, args: dict) -> dict:
        prev = self._draft.get("prev")
        if not prev:
            raise ZeluxError("没有可撤销的改动（只保留上一份草稿）", "denied")
        restored = prev.get("items") or {}
        # 先把舵机拉回去，再改文件：顺序反了会出现"文件对了但机器没动"
        for sid, item in restored.items():
            try:
                if "offset" in item:
                    await self._apply_offset(int(sid), int(item["offset"]))
                if "direction" in item:
                    setter = getattr(self.hal, "set_direction", None)
                    if setter:
                        await setter(int(sid), int(item["direction"]))
            except ZeluxError as exc:
                self.log_event(f"撤销时 {sid} 号没写回去：{exc.message}", level="warning")
        self._draft["items"] = restored
        self._draft["prev"] = None
        self._save_draft()
        self.log_event("已撤销上一步")
        return {"ok": True, "items": restored}

    async def _commit(self, args: dict) -> dict:
        self._need_session()
        items = {sid: {k: v for k, v in item.items() if k != "touched"}
                 for sid, item in self._draft["items"].items() if item}
        if not items:
            raise ZeluxError("草稿是空的：先调 0 位或方向再落盘", "denied")

        before = (self._calib.get("items") or {})
        diff = {sid: {"before": before.get(sid, {}), "after": item}
                for sid, item in items.items() if before.get(sid) != item}
        if args.get("dry_run"):
            return {"ok": True, "dry_run": True, "diff": diff, "changed": len(diff),
                    "note": "确认无误后再调一次（不带 dry_run）"}

        # 1) 先写硬件：0 位/方向走专用接口，其它（限位）走寄存器
        applied: list[int] = []
        for sid, item in items.items():
            sid_i = int(sid)
            try:
                if "offset" in item:
                    await self._apply_offset(sid_i, int(item["offset"]))
                setter = getattr(self.hal, "set_direction", None)
                if "direction" in item and setter:
                    await setter(sid_i, int(item["direction"]))
                for key, reg in (("min", R_MIN), ("max", R_MAX)):
                    if key in item:
                        await self.hal.servo_write(sid_i, reg, int(item[key]), True)
                applied.append(sid_i)
            except ZeluxError as exc:
                raise hardware(f"{sid} 号写不进去（{exc.message}），先检查这颗的线再重新落盘") from None

        # 2) 再写标定文件（整份替换，带时间戳，方便对比上一版）
        calib = {"ts": time.time(), "items": items, "source": "servos",
                 "count": len(items)}
        prev = _load_json(self.calib_path, {})
        if prev:
            calib["previous"] = {"ts": prev.get("ts"), "items": prev.get("items", {})}
        _write_json(self.calib_path, calib)
        self._calib = calib
        self.ctx.state["servo_calibrated"] = True
        self.ctx.hub.posture = "servo_session"
        self.log_event(f"标定已落盘：{len(items)} 颗（0 位/方向/限位）")
        return {"ok": True, "path": str(self.calib_path), "items": len(items),
                "applied": applied, "diff": diff, "changed": len(diff)}

    # ---------------- 导出 ----------------
    async def _export(self, args: dict) -> dict:
        self._need_session()
        reader = getattr(self.hal, "read_all", None)
        if reader:
            snap = await reader()
        else:
            snap = self.hal.snapshot_servos() or {}
        items = list(snap.get("items") or [])
        if not items:
            items = [await self.hal.servo_read(sid) for sid in JOINT_IDS]

        buf = io.StringIO()
        wr = csv.writer(buf)
        wr.writerow(["id", "name", "pos", "speed", "load", "volt", "temp", "torque",
                     "offset", "direction", "min", "max"])
        for it in items:
            sid = int(it["id"])
            draft = self._draft["items"].get(str(sid), {})
            wr.writerow([sid, JOINT_NAMES[JOINT_IDS.index(sid)] if sid in JOINT_IDS else "",
                         it.get("pos"), it.get("speed", 0), it.get("load"), it.get("volt"),
                         it.get("temp"), int(bool(it.get("torque"))),
                         draft.get("offset", 0), draft.get("direction", 1),
                         draft.get("min", ""), draft.get("max", "")])
        text = buf.getvalue()
        path = Path(self.config.state_dir) / f"servo_regs_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        try:
            path.write_text(text, "utf-8")
        except OSError as exc:
            self.log.warning("导出文件写不进去：%s", exc)
        return {"ok": True, "items": items, "csv": text, "path": str(path),
                "ts": time.time()}
