"""ToF 页 + 自动避障（`name = "tof"`）。

避障算法为什么这么设计（每一条都对应一种真机上会出事的情况）：

1. **8×8 切成 左/中/右 三扇区，并去掉最上、最下一行**：
   最上一行常年打在桌沿/天花板上，最下一行会打到鸭子自己的脚背——两行都会把
   "前方空旷"读成"贴脸"。扇区列划分 0–2 / 3–4 / 5–7，中间只留 2 列：中间窄，
   才不会被左右墙一起拉进来误刹。
2. **每扇区取 25% 分位，不取最小值**：
   64 格里总有几格打到地板缝、桌腿反光或者干脆是噪点。取最小值等于把最坏的那一格
   当结论，结果就是"路过一张纸就急刹"。25% 分位是"这一片里最近的、但不算离群点"的距离，
   单点噪声拿它没办法。
3. **stop/slow/turn 三档 + 侧向绕行**：
   中扇区 < stop_mm 直接停（这是碰撞前的最后 22 cm）；< slow_mm 降到 40% 速度；
   此时若有一侧明显更宽（> side_mm），就转过去而不是硬挤——鸭子是四足，不是扫地机，
   顶着障碍原地磨会打滑。
4. **防抖：普通切换要连续 3 帧（200 ms）一致；从 stop 恢复要连续 5 帧 clear（333 ms）**：
   单帧抖动就让鸭子"停—走—停"，机械结构和电机都比撞一下更受伤；
   而"刚停下就立刻往前冲"是直接撞上去。往更保守的方向（→ stop）反向不需要防抖：
   晚 3 帧才刹车，等于多走 60 cm。

红线：**避障永远不碰急停**。`hub.estop.state == "confirmed"` 时只报状态、不下发速度；
避障的 stop 是"这一帧不动"，不是"卸力"，两者语义不能混。
"""
from __future__ import annotations

import asyncio
import time

from ..errors import ZeluxError
from .base import BaseModule

# ---------------------------------------------------------------- 几何
ROWS = 8
COLS = 8
ZONES = ROWS * COLS
SECTOR_COLS: tuple[tuple[int, ...], ...] = ((0, 1, 2), (3, 4), (5, 6, 7))
SECTOR_ROWS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)   # 去掉最上、最下各一行噪声
SECTOR_Q = 0.25                                     # 25% 分位
VALID_STATUS = (5, 9)
GHOST_MAX_MM = 4000                                 # 与 hal/radxa_tof.py 同口径

# ---------------------------------------------------------------- 决策参数
DEFAULT_CFG: dict = {"stop_mm": 220.0, "slow_mm": 600.0, "side_mm": 900.0, "action": "forward"}
CFG_LIMITS = {"stop_mm": (60.0, 1200.0), "slow_mm": (120.0, 3000.0), "side_mm": (200.0, 4000.0)}
CFG_ACTIONS = ("forward", "hold")

CRUISE_VX = 0.22          # 直行巡航速度 m/s；鸭子走快了四足跟不上，别调太猛
SLOW_GAIN = 0.4           # 减速档 = 巡航 × 0.4
TURN_WZ = 0.8             # 绕行角速度 rad/s
SWITCH_FRAMES = 3         # 普通动作切换所需连续帧数
STOP_RECOVER_FRAMES = 5   # 从 stop 恢复需要连续 clear 帧数
AVOID_HZ = 15             # 与传感器同频

ACTION_CN = {"clear": "直行", "slow": "减速", "turn": "绕行", "stop": "停车"}


# ---------------------------------------------------------------- 纯函数（可单测）
def percentile(vals: list[float], q: float = SECTOR_Q) -> int | None:
    """最近秩分位数：取排序后第 round((n-1)*q) 个。空集合返回 None。

    18 个格子里 25% 分位 = 第 5 近的那格；一个 30 mm 的噪点只是第 1 近，影响不到结论。
    """
    if not vals:
        return None
    xs = sorted(float(v) for v in vals)
    k = int(round((len(xs) - 1) * q))
    return int(xs[max(0, min(len(xs) - 1, k))])


def sector_depths(distance_mm: list[int] | None, status: list[int] | None,
                  q: float = SECTOR_Q) -> dict:
    """8×8 → 三扇区深度（mm）。无效格（status≠5/9、0、超量程哨兵）先剔除再算分位。"""
    dist = list(distance_mm or [])
    stat = list(status or [])
    out: dict[str, int | None] = {}
    for key, cols in zip(("l", "c", "r"), SECTOR_COLS):
        vals: list[int] = []
        for row in SECTOR_ROWS:
            for col in cols:
                i = row * COLS + col
                if i >= len(dist) or i >= len(stat):
                    continue
                d, s = int(dist[i] or 0), int(stat[i])
                if s in VALID_STATUS and 0 < d <= GHOST_MAX_MM:
                    vals.append(d)
        out[key] = percentile(vals, q)
    return out


def _verdict(action: str, vx: float, wz: float, reason: str, depths: dict) -> dict:
    return {"action": action, "vx": round(float(vx), 3), "wz": round(float(wz), 3),
            "reason": reason,
            "sector": {"l": depths.get("l"), "c": depths.get("c"), "r": depths.get("r")}}


def decision(depths: dict, cfg: dict | None = None, cruise_vx: float = CRUISE_VX) -> dict:
    """单帧决策（无状态、无防抖）：给一帧三扇区深度，返回该干什么。"""
    c = cfg or DEFAULT_CFG
    stop_mm, slow_mm, side_mm = c["stop_mm"], c["slow_mm"], c["side_mm"]
    left, mid, right = depths.get("l"), depths.get("c"), depths.get("r")

    if mid is None:
        # 没有有效读数 ≠ 前方空旷。看不见就停，这一条不许为了"别挡着演示"而放宽。
        return _verdict("stop", 0.0, 0.0,
                        "中间扇区没有有效读数（传感器被遮、脏了或者朝天花板），不敢往前走", depths)

    if mid < stop_mm:
        return _verdict("stop", 0.0, 0.0,
                        f"前方 {mid} mm，低于停车线 {int(stop_mm)} mm", depths)

    if mid < slow_mm:
        vx = cruise_vx * SLOW_GAIN
        wide_l, wide_r = int(left or 0), int(right or 0)
        if max(wide_l, wide_r) > side_mm:
            # 向宽的一侧绕：wz 正 = 左转（ROS 约定），所以右侧更宽就给负值
            if wide_r >= wide_l:
                return _verdict("turn", vx, -TURN_WZ,
                                f"前方 {mid} mm 变窄，右侧 {wide_r} mm 更宽，向右绕", depths)
            return _verdict("turn", vx, TURN_WZ,
                            f"前方 {mid} mm 变窄，左侧 {wide_l} mm 更宽，向左绕", depths)
        return _verdict("slow", vx, 0.0,
                        f"前方 {mid} mm，在减速线 {int(slow_mm)} mm 内，降到 40% 速度", depths)

    if c.get("action") == "hold":
        return _verdict("clear", 0.0, 0.0, f"前方 {mid} mm 通畅（只报状态，不前进）", depths)
    return _verdict("clear", cruise_vx, 0.0, f"前方 {mid} mm 通畅", depths)


class AvoidFilter:
    """动作防抖状态机：把单帧 decision 平滑成"能真的下发给电机"的动作。

    只做两件事：动作要连续 N 帧一致才切换；保守方向（→ stop）立即生效。
    """

    def __init__(self, action: str = "clear", vx: float = 0.0, wz: float = 0.0) -> None:
        self.action = action
        self.vx = vx
        self.wz = wz
        self._pending: str | None = None
        self._n = 0

    def reset(self, action: str = "clear") -> None:
        self.action = action
        self.vx = self.wz = 0.0
        self._pending, self._n = None, 0

    def update(self, raw: dict) -> dict:
        target = raw["action"]
        if target == self.action:
            self._pending, self._n = None, 0
            return {**raw, "settled": True, "pending": None, "pending_n": 0, "pending_need": 0}

        need = STOP_RECOVER_FRAMES if (self.action == "stop" and target == "clear") else SWITCH_FRAMES
        if target == "stop":
            need = 1        # 往停车方向不需要防抖：晚 3 帧就是多走 60 cm
        if target == self._pending:
            self._n += 1
        else:
            self._pending, self._n = target, 1

        if self._n >= need:
            self.action, self.vx, self.wz = target, float(raw["vx"]), float(raw["wz"])
            self._pending, self._n = None, 0
            return {**raw, "settled": True, "pending": None, "pending_n": 0, "pending_need": 0}

        # 还没攒够帧：照旧动作执行，但把"为什么看起来没反应"说清楚
        left = need - self._n
        out = dict(raw)
        out.update({"action": self.action, "vx": round(self.vx, 3), "wz": round(self.wz, 3),
                    "settled": False, "pending": target, "pending_n": self._n,
                    "pending_need": need,
                    "reason": f"等 {left} 帧确认（{ACTION_CN[target]}）：{raw['reason']}"
                              f"，当前仍{ACTION_CN[self.action]}"})
        return out


def estop_blocks(estop: dict | None) -> bool:
    """急停已确认卸力时，任何模块都不许下发速度。语义上只有一个字：停。"""
    return (estop or {}).get("state") == "confirmed"


def normalize_cfg(src: dict | None) -> dict:
    """夹紧 + 校验参数。宁可悄声夹紧，也不要让 -5 mm 的停车线把鸭子钉在原地。"""
    out = dict(DEFAULT_CFG)
    if isinstance(src, dict):
        for key, (lo, hi) in CFG_LIMITS.items():
            if key not in src:
                continue
            try:
                v = float(src[key])
            except (TypeError, ValueError):
                continue
            out[key] = float(min(hi, max(lo, v)))
        act = str(src.get("action") or "").strip().lower()
        if act in CFG_ACTIONS:
            out["action"] = act
    if out["slow_mm"] <= out["stop_mm"]:
        # 减速线低于停车线时 slow 分支永远进不去，等于关掉了减速，直接按 2 倍补正
        out["slow_mm"] = min(CFG_LIMITS["slow_mm"][1], out["stop_mm"] * 2.0)
    return out


# ---------------------------------------------------------------- 模块
class Module(BaseModule):
    name = "tof"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        stored = self.load_store() or {}
        self.cfg = normalize_cfg(stored.get("config") if isinstance(stored, dict) else None)
        self._on = False
        self._filter = AvoidFilter()
        self._decision = decision({"l": None, "c": None, "r": None}, self.cfg)
        self._decision["reason"] = "避障没开"
        self._frames = 0
        self._last_seq = -1
        self._blocked = ""
        self._task: asyncio.Task | None = None

    # ---------------- 状态 ----------------
    def state(self) -> dict:
        snap = dict(self.hal.snapshot_tof() or {})
        snap.setdefault("state", "unavailable")
        d = self._decision
        snap["avoid"] = {
            "on": self._on,
            "action": d.get("action", "clear"),
            "vx": d.get("vx", 0.0),
            "wz": d.get("wz", 0.0),
            "reason": d.get("reason", ""),
            "sector": d.get("sector", {"l": None, "c": None, "r": None}),
            "settled": d.get("settled", True),
            "pending": d.get("pending"),
            "config": dict(self.cfg),
            "blocked": self._blocked,
            "frames": self._frames,
            "estop": (self.ctx.hub.estop or {}).get("state") if self.ctx.hub else None,
        }
        return {"tof": snap}

    # ---------------- 命令 ----------------
    async def cmd(self, op: str, args: dict) -> dict:
        args = args or {}
        if op == "op":
            # 契约是 `tof.op` + `{op:"on"}`：动作名在 args 里
            op = str(args.get("op", "") or args.get("mode", ""))
        op = op.strip().lower()

        if op in ("on", "avoid", "avoid_on", "start"):
            # 契约里 `tof.op` 的 `avoid` 就是"打开避障"，也接受 {on:false} 反着用
            if op == "avoid" and not args.get("on", True):
                return await self._stop()
            return await self._start()
        if op in ("off", "avoid_off", "stop"):
            return await self._stop()
        if op == "config":
            return self._config(args)
        if op == "simulate":
            return await self._simulate(args)
        if op in ("status", ""):
            return {"ok": True, "tof": self.state()["tof"]}
        raise ZeluxError(f"tof 不支持 {op}（可用：on/off/config/simulate/status）", "unsupported")

    def _config(self, args: dict) -> dict:
        merged = dict(self.cfg)
        merged.update({k: args[k] for k in list(CFG_LIMITS) + ["action"] if k in args})
        self.cfg = normalize_cfg(merged)
        self.save_store({"config": self.cfg})
        self._filter.reset(self._filter.action)
        self.log_event(f"避障参数已更新：停车 {int(self.cfg['stop_mm'])} mm / "
                       f"减速 {int(self.cfg['slow_mm'])} mm / 侧宽 {int(self.cfg['side_mm'])} mm")
        return {"ok": True, "config": dict(self.cfg),
                "note": "参数立即生效；改回默认值传 stop_mm=220, slow_mm=600, side_mm=900"}

    # ---------------- 开关 ----------------
    async def _start(self) -> dict:
        snap = self.hal.snapshot_tof() or {}
        state = snap.get("state")
        if state in ("unsupported", "unavailable"):
            raise ZeluxError(
                f"ToF 不可用（{snap.get('reason', '没有读数')}）：避障全靠它看路，先修硬件再开",
                "unsupported" if state == "unsupported" else "hardware")
        if self._on:
            return {"ok": True, "avoid": self.state()["tof"]["avoid"], "again": True}
        if state == "stale":
            self.log.warning("ToF 数据已过期（%s），避障先开但只会报停车", snap.get("reason", ""))
        self._filter.reset("clear")
        self._last_seq = -1
        self._blocked = ""
        self._on = True
        self._task = asyncio.create_task(self._loop(), name="tof-avoid")
        self.log_event(f"避障已开：停车 {int(self.cfg['stop_mm'])} mm、减速 {int(self.cfg['slow_mm'])} mm")
        return {"ok": True, "avoid": self.state()["tof"]["avoid"]}

    async def _stop(self) -> dict:
        was = self._on
        self._on = False
        task, self._task = self._task, None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._decision = decision({"l": None, "c": None, "r": None}, self.cfg)
        self._decision["reason"] = "避障没开"
        self._blocked = ""
        if was:
            # 关避障时补一脚零速：否则最后一个 vx 会一直留在实时层的意图里
            if not estop_blocks(getattr(self.ctx.hub, "estop", None)):
                await self._apply_zero(1)
            self.log_event("避障已关闭")
        return {"ok": True, "on": False}

    async def stop(self) -> None:
        self._on = False
        await super().stop()

    # ---------------- 主循环 ----------------
    async def _loop(self) -> None:
        period = 1.0 / AVOID_HZ
        while self._on:
            t0 = time.monotonic()
            snap = self.hal.snapshot_tof() or {}
            if snap.get("state") in ("unsupported", "unavailable"):
                # 传感器中途掉了：停车 + 主动关掉避障。留着"开着但永远停"更让人迷惑。
                await self._apply_zero(3)
                self._on = False
                self._blocked = f"ToF 掉线（{snap.get('reason', '没有读数')}），避障已自动关闭"
                self.log.warning("%s", self._blocked)
                break
            seq = int(snap.get("seq") or 0)
            if seq != self._last_seq:
                self._last_seq = seq
                await self._frame_step(snap)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))

    async def _frame_step(self, snap: dict | None = None, *, drive: bool = True) -> dict:
        """跑一帧：读缓存 → 三扇区 → 决策 → 防抖 → （可能）下发速度。

        单测直接 await 这个方法：它不依赖后台任务，也没有隐藏状态。
        """
        snap = dict(snap if snap is not None else (self.hal.snapshot_tof() or {}))
        depths = sector_depths(snap.get("distance_mm"), snap.get("status"))
        settled = self._filter.update(decision(depths, self.cfg))
        self._decision = settled
        self._frames += 1
        if not drive:
            return settled

        if estop_blocks(getattr(self.ctx.hub, "estop", None)):
            # 急停优先：只报状态。这里连"零速"都不发——卸力状态下再发意图，等于和操作者抢控制权
            self._blocked = "急停已确认卸力：避障只报状态，不下发速度"
            return settled

        self._blocked = ""
        if settled["action"] == "stop":
            await self._apply_zero(3)
        else:
            await self._move(settled["vx"], settled["wz"])
        return settled

    async def _move(self, vx: float, wz: float) -> None:
        try:
            await self.hal.move(float(vx), 0.0, float(wz))
        except Exception as exc:      # 实时层不在、总线被占：不许把 15 Hz 循环打断
            self.log.warning("下发避障速度失败：%s", exc)

    async def _apply_zero(self, times: int = 3) -> None:
        """连发几次零速：实时层是 200 Hz 的意图缓存，只发一次可能被下一帧覆盖掉。"""
        for _ in range(max(1, int(times))):
            await self._move(0.0, 0.0)
            await asyncio.sleep(0.02)

    # ---------------- 演示 / 自测 ----------------
    async def _simulate(self, args: dict) -> dict:
        dist = args.get("distances") or args.get("distance_mm") or []
        if not isinstance(dist, (list, tuple)) or len(dist) != ZONES:
            raise ZeluxError(f"要 64 个数（8×8 行优先），收到 {len(dist) if isinstance(dist, (list, tuple)) else '非数组'} 个",
                             "denied")
        dist = [max(0, int(v or 0)) for v in dist[:ZONES]]
        status = [5 if d > 0 else 255 for d in dist]
        fake = {"state": "ready", "seq": -1, "distance_mm": dist, "status": status}
        # drive=False：演示时只算不动，别让"点一下看看"把鸭子开走
        d = await self._frame_step(fake, drive=False)
        return {"ok": True, **d, "config": dict(self.cfg),
                "note": "只是一次决策，没有下发速度；action 要连续帧稳定才切换"}
