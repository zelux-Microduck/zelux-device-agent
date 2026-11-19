"""IMU 页（`name = "imu"`）：姿态读数 + 五步采集向导。

为什么是"五步"而不是"一键校准"：
只有静止那一段能算出 0 位偏置；但要判断模块**装反了没有**，必须看四个方向的倾斜
是否跟预期同号。所以采集顺序是固定的：平放静止 → 往前 → 往后 → 往左 → 往右。
每步都要求"够稳（峰峰值）+ 够大（≥12°）+ 够多（帧数）"，缺哪条就直说是哪条，
不要给"校准失败"这种没法行动的结论。

判定参数与前端同名同值（前端更严、后端必收，双端一致校验）：
`still_min_n=15, still_max_spread_deg=1.5, level_deg=5, ref_min_n=5, ref_spread_deg=1.5`。
`level_deg` 用在**解算之后**：残差必须 ≤5° 才算这次标定合格。
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from ..errors import ZeluxError, denied, unsupported
from ..imu_mount_calibration import MountCalibration
from ..hal.radxa_imu import wrap180
from ..robotd_imu import RobotdImu
from .base import BaseModule

# 五步向导：key / 页面标题 / 提示。key 与前端一致，改名要同步改前端。
STEPS: list[tuple[str, str, str]] = [
    ("still", "平放静止", "把鸭子放平，手离开，停 3 秒"),
    ("forward", "往前倾", "把鸭子前部抬起 12° 以上"),
    ("back", "往后倾", "把鸭子后部抬起 12° 以上"),
    ("left", "往左倾", "把鸭子左侧抬起 12° 以上"),
    ("right", "往右倾", "把鸭子右侧抬起 12° 以上"),
]

PARAMS = {
    "still_min_n": 15,
    "still_max_spread_deg": 1.5,
    "level_deg": 5,
    "ref_min_n": 5,
    "ref_spread_deg": 1.5,
}

TILT_MIN_DEG = 12.0        # 倾斜步的最小角度：小于这个数分不清"倾斜"和"手抖"
WRONG_WAY_DEG = -3.0       # 反方向超过 3° 就明确说"方向反了"，别让人干等


def _spread(vals: list[float]) -> float:
    return (max(vals) - min(vals)) if vals else 0.0


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


class Module(BaseModule):
    name = "imu"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.params = dict(PARAMS)
        self.robotd = RobotdImu(self.config)
        self.mount_calibration = MountCalibration()
        self._running = False
        self._step = 0
        self._buf: list[tuple[float, float]] = []      # 当前步的 (roll, pitch)
        self._done: dict[str, list[tuple[float, float]]] = {}
        self._why = ""
        self._result: dict | None = None
        self._started = 0.0
        self._calib: dict = self._load_calib()
        self._mount_calibration_revision: str | None = None

    async def start(self) -> None:
        if self.config.mode == "radxa":
            self.spawn(self._poll_robotd())

    async def _poll_robotd(self) -> None:
        while True:
            await self.robotd.refresh()
            await asyncio.sleep(2)

    # ---------------- 状态 ----------------
    def state(self) -> dict:
        management = self._management_view()
        managed = management["calibration"] != "agent"
        if managed:
            # robotd-managed IMU is still a live device. Do not report it as
            # unavailable merely because Agent must not open the shared bus or
            # run the old local-I2C calibration path.
            runtime = management.get("runtime") or {}
            device_ready = bool(management.get("available")
                                and not management.get("simulated")
                                and (management.get("sensor") or {}).get("ready") is True)
            snap = {
                # `ok` continues to mean that Agent has live roll/pitch/yaw
                # samples. robotd owns the shared-bus IMU, so expose its
                # hardware health separately and never unlock the old I2C
                # wizard from that health flag.
                "ok": False,
                "device_ready": device_ready,
                "measurement_source": "robotd",
                "source": runtime.get("source"),
                "model": runtime.get("model"),
                "chip": runtime.get("model"),
                "driver_id": runtime.get("driver_id"),
                "driver_version": runtime.get("driver_version"),
            }
            if not device_ready:
                snap["device_reason"] = (management.get("error")
                                         or ("robotd 正在仿真，不能代表真实 IMU"
                                             if management.get("simulated")
                                             else "robotd 暂未读到有效 IMU 数据"))
        else:
            snap = dict(self.hal.snapshot_imu() or {})
        snap.setdefault("ok", False)
        # 模拟 HAL 不会自己应用标定：这里补一层显示修正，
        # 免得演示时出现"显示已校准、鸭子还歪着"这种自相矛盾的画面。
        if snap.get("ok") and self._calib and not hasattr(self.hal, "apply_calibration"):
            snap["roll"] = round(wrap180(float(snap.get("roll", 0.0))
                                         * self._calib.get("roll_sign", 1)
                                         + self._calib.get("roll_offset", 0.0)), 2)
            snap["pitch"] = round(wrap180(float(snap.get("pitch", 0.0))
                                          * self._calib.get("pitch_sign", 1)
                                          + self._calib.get("pitch_offset", 0.0)), 2)
        snap["management"] = management
        snap["calibrated"] = management.get("mount_calibration", {}).get("status") == "applied" if managed else bool(self._calib)
        snap["wizard"] = self._view()
        snap["wizard"]["enabled"] = not managed
        return {"imu": snap}

    def _management_view(self) -> dict:
        view = self.robotd.view()
        view["mount_calibration_wizard"] = self.mount_calibration.view()
        return view

    def _calib_file(self) -> Path:
        return Path(self.config.state_dir) / "imu_calib.json"

    def _load_calib(self) -> dict:
        try:
            data = json.loads(self._calib_file().read_text("utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _view(self) -> dict:
        steps = []
        for i, (key, label, hint) in enumerate(STEPS):
            done = key in self._done
            steps.append({
                "key": key, "label": label, "hint": hint, "done": done,
                "active": self._running and i == self._step,
                "n": len(self._done[key]) if done else (len(self._buf) if i == self._step else 0),
                "progress": 1.0 if done else (self._progress(key) if i == self._step else 0.0),
            })
        return {"running": self._running, "step": self._step,
                "step_key": STEPS[self._step][0] if self._step < len(STEPS) else None,
                "steps": steps,
                "ready": self._step >= len(STEPS),
                "why": self._why if self._running else "",
                "result": self._result,
                "params": self.params,
                "elapsed_s": round(time.time() - self._started, 1) if self._running else 0.0}

    def _progress(self, key: str) -> float:
        """给页面画进度条：静止看帧数，倾斜看角度差。"""
        if key == "still":
            return min(1.0, len(self._buf) / self.params["still_min_n"])
        axis, sign = self._axis(key)
        seg = self._buf[-self.params["ref_min_n"]:]
        if not seg:
            return 0.0
        delta = (_mean([f[axis] for f in seg]) - self._base(axis)) * sign
        return max(0.0, min(1.0, delta / TILT_MIN_DEG))

    # ---------------- 命令 ----------------
    async def cmd(self, op: str, args: dict) -> dict:
        args = args or {}
        if op == "op":
            # 契约是 `imu.op` + `{op:"start"}`：动作名在 args 里，别把它当未知命令
            op = str(args.get("op", ""))
        if op == "management_status":
            await self.robotd.refresh()
            return {"management": self._management_view()}
        if op == "model_select":
            result = await self.robotd.select(args.get("model"), args.get("expected"))
            self._reset()
            self.mount_calibration.clear()
            self._mount_calibration_revision = None
            result["management"] = self._management_view()
            return result
        if op == "model_restart":
            result = await self.robotd.restart(args.get("expected"), args.get("confirm"), args.get("mechanically_supported"))
            result["management"] = self._management_view()
            return result
        if op == "mount_calibration_start":
            return await self._mount_calibration_start(args)
        if op == "mount_calibration_capture":
            if not self.mount_calibration.view()["running"]:
                raise denied("请先开始机体对齐标定，再按顺序采集姿态")
            sample = await self.robotd.capture_gravity()
            if self.robotd.view().get("catalog", {}).get("revision") != self._mount_calibration_revision:
                self.mount_calibration.clear()
                self._mount_calibration_revision = None
                raise ZeluxError("标定期间 IMU 配置发生变化；本次采集已清除，请刷新后重新开始", "conflict")
            try:
                self.mount_calibration.capture(sample)
            except ValueError as exc:
                raise denied(str(exc)) from exc
            return {"ok": True, "sample": sample, "management": self._management_view()}
        if op == "mount_calibration_solve":
            try:
                result = self.mount_calibration.solve()
            except ValueError as exc:
                raise denied(str(exc)) from exc
            return {"ok": True, "result": result, "management": self._management_view()}
        if op == "mount_calibration_commit":
            result = self.mount_calibration.result
            if not result:
                raise denied("请先完成采集并解算机体对齐")
            if args.get("expected") != self._mount_calibration_revision:
                raise ZeluxError("标定期间 IMU 配置发生变化；请刷新并重新标定", "conflict")
            saved = await self.robotd.save_mount_correction(
                result["correction_wxyz"], args.get("expected"))
            self._mount_calibration_revision = None
            saved["management"] = self._management_view()
            return saved
        if op == "mount_calibration_clear":
            self.mount_calibration.clear()
            self._mount_calibration_revision = None
            return {"ok": True, "management": self._management_view()}
        if op in ("start", "solve", "commit", "bias", "sample") and self.robotd.view()["calibration"] != "agent":
            raise denied("此 IMU 由 robotd 管理，禁止使用 Agent I2C 标定或采样；固定安装方向按型号自动应用")
        if op == "start":
            return await self._start(args)
        if op == "stop":
            return await self._stop()
        if op == "solve":
            return self._solve()
        if op == "commit":
            return await self._commit(args)
        if op == "sim_tilt":
            return await self._sim_tilt(args)
        if op in ("status", ""):
            return {"ok": True, "wizard": self._view(), "imu": self.state()["imu"]}
        if op == "sample":
            return {"imu": self.hal.snapshot_imu(), "recent": self._recent(args)}
        if op == "clear":
            self._reset()
            return {"ok": True, "wizard": self._view()}
        if op == "bias":
            return await self._bias(args)
        raise unsupported(f"imu 不支持 {op}")

    async def _mount_calibration_start(self, args: dict) -> dict:
        await self.robotd.refresh()
        view = self.robotd.view()
        config = view.get("catalog", {}).get("config", {}).get("imu", {})
        runtime = view.get("runtime") or {}
        models = view.get("catalog", {}).get("models", [])
        selected = next((item for item in models if item.get("id") == config.get("model")), {})
        if args.get("confirm") is not True or args.get("mechanically_supported") is not True:
            raise denied("请确认已停止动作、托稳机体并安排现场看护")
        if (not view.get("available") or view.get("simulated") or not view.get("ready")
                or not view.get("restart_safe") or view.get("pending_restart") is not False):
            raise ZeluxError("robotd/IMU 未就绪或当前不允许开始标定", "busy")
        if (config.get("source") != "imu-dxl" or selected.get("calibration_supported") is not True
                or runtime.get("source") != "imu-dxl" or runtime.get("model") != config.get("model")):
            raise denied("当前 robotd 型号目录未开放此 IMU 的机体对齐标定")
        existing = view.get("mount_calibration", {}).get("correction_wxyz")
        revision = view.get("catalog", {}).get("revision")
        self._mount_calibration_revision = revision
        return {"ok": True, "management": self._management_view_with_wizard_start(existing)}

    def _management_view_with_wizard_start(self, existing_correction=None) -> dict:
        self.mount_calibration.start(existing_correction)
        return self._management_view()

    async def _start(self, args: dict) -> dict:
        if not (self.hal.snapshot_imu() or {}).get("ok"):
            why = (self.hal.snapshot_imu() or {}).get("reason", "IMU 没数据")
            raise ZeluxError(f"{why}：先解决硬件，采集才有意义", "hardware")
        self._reset()
        self._running = True
        self._started = time.time()
        self.spawn(self._loop())
        self.log_event("IMU 采集开始：先把鸭子放平别碰")
        return {"ok": True, "wizard": self._view()}

    def _reset(self) -> None:
        self._running = False
        self._step = 0
        self._buf = []
        self._done = {}
        self._why = ""
        self._result = None
        self._started = 0.0

    async def _loop(self) -> None:
        # 20 ms 轮一次缓存就够了：HAL 自己是 100 Hz 采样，这里只是"看"它的最新值
        while self._running:
            if self.robotd.view()["calibration"] != "agent":
                self._reset()
                break
            snap = self.hal.snapshot_imu() or {}
            if snap.get("ok"):
                self._feed(float(snap.get("roll", 0.0)), float(snap.get("pitch", 0.0)))
            await asyncio.sleep(0.02)

    def _feed(self, roll: float, pitch: float) -> None:
        if not self._running:
            return
        self._buf.append((roll, pitch))
        del self._buf[:-400]
        key = STEPS[self._step][0]
        ok, why = self._check(key)
        self._why = why
        if not ok:
            return
        # 只留通过判定的那一小段：整段缓冲里混着"还没倾到位"的帧，
        # 拿它算均值会把 15° 的倾斜平均成 5°，解算出来的偏置就是错的。
        self._done[key] = list(self._buf[-max(15, self.params["ref_min_n"] * 3):])
        self._buf = []
        self._step += 1
        self._why = ""
        self.emit("imu_step", {"step": key, "ok": True, "index": self._step,
                               "next": STEPS[self._step][0] if self._step < len(STEPS) else None})
        if self._step >= len(STEPS):
            self._running = False
            self.emit("imu_step", {"done": True, "steps": len(STEPS)})
            self.log_event("五步采集完成，可以解算了")

    # ---------------- 判定 ----------------
    def _axis(self, key: str) -> tuple[int, int]:
        """(轴上索引 0=roll 1=pitch, 期望的方向符号)。"""
        return {"forward": (1, 1), "back": (1, -1),
                "left": (0, -1), "right": (0, 1)}[key]

    def _base(self, axis: int) -> float:
        still = self._done.get("still")
        if not still:
            return 0.0
        return _mean([f[axis] for f in still])

    def _check(self, key: str) -> tuple[bool, str]:
        if key == "still":
            need = self.params["still_min_n"]
            limit = self.params["still_max_spread_deg"]
            if len(self._buf) < need:
                return False, f"样本 {len(self._buf)}/{need} 帧，稳住别碰"
            seg = self._buf[-need:]
            sr = _spread([f[0] for f in seg])
            sp = _spread([f[1] for f in seg])
            if max(sr, sp) > limit:
                return False, f"还在动：姿态抖了 {max(sr, sp):.1f}°（要求 ≤{limit}°），手离开鸭子"
            return True, ""

        axis, sign = self._axis(key)
        need = self.params["ref_min_n"]
        limit = self.params["ref_spread_deg"]
        if len(self._buf) < need:
            return False, f"样本 {len(self._buf)}/{need} 帧"
        seg = self._buf[-need:]
        vals = [f[axis] for f in seg]
        sp = _spread(vals)
        if sp > limit:
            return False, f"还在动：{ '横滚' if axis == 0 else '俯仰' }抖了 {sp:.1f}°（要求 ≤{limit}°）"
        delta = (_mean(vals) - self._base(axis)) * sign
        word = {"forward": "往前", "back": "往后", "left": "往左", "right": "往右"}[key]
        if delta < WRONG_WAY_DEG:
            back = {"forward": "往后", "back": "往前", "left": "往右", "right": "往左"}[key]
            return False, f"方向反了：这一步要{word}倾，现在在{back}倾"
        if delta < TILT_MIN_DEG:
            return False, f"{word}倾了 {delta:.1f}°，不到 {TILT_MIN_DEG:.0f}°"
        return True, ""

    # ---------------- 停止 ----------------
    async def _stop(self) -> dict:
        self._running = False
        done = [k for k, _, _ in STEPS if k in self._done]
        missing = []
        for i, (key, label, _hint) in enumerate(STEPS):
            if key in self._done:
                continue
            if i == self._step:
                _ok, why = self._check(key)
            else:
                why = "还没开始这一步"
            missing.append({"key": key, "label": label, "why": why})
        self.log_event(f"采集结束：完成 {len(done)}/{len(STEPS)} 步")
        return {"ok": True, "done": done, "missing": missing, "wizard": self._view()}

    # ---------------- 解算 ----------------
    def _solve(self) -> dict:
        if any(k not in self._done for k, _, _ in STEPS):
            missing = [lbl for k, lbl, _ in STEPS if k not in self._done]
            raise ZeluxError(f"还差这几步：{'、'.join(missing)}。先把五步采完再解算", "stale")

        still = self._done["still"]
        half = max(1, len(still) // 2)
        # 用前半段算偏置、后半段验证：自己验自己等于没验
        fit, hold = still[:half], still[half:] or still[-1:]

        base_r, base_p = _mean([f[0] for f in fit]), _mean([f[1] for f in fit])

        # 安装方向：看倾斜时读数往哪边走。走反了就是模块装反（或轴定义相反）
        fwd_p = _mean([f[1] for f in self._done["forward"]]) - base_p
        right_r = _mean([f[0] for f in self._done["right"]]) - base_r
        pitch_sign = -1 if fwd_p < 0 else 1
        roll_sign = -1 if right_r < 0 else 1

        roll_off = -base_r * roll_sign
        pitch_off = -base_p * pitch_sign

        def fix(r: float, p: float) -> tuple[float, float]:
            return r * roll_sign + roll_off, p * pitch_sign + pitch_off

        after = [_mean([fix(*f)[0] for f in hold]), _mean([fix(*f)[1] for f in hold])]
        before = [base_r, base_p]

        checks: list[dict] = []
        for key, label, axis, sign in (
                ("forward", "往前倾", 1, 1), ("back", "往后倾", 1, -1),
                ("left", "往左倾", 0, -1), ("right", "往右倾", 0, 1)):
            raw = _mean([f[axis] for f in self._done[key]])
            calibrated = raw * (roll_sign if axis == 0 else pitch_sign) \
                + (roll_off if axis == 0 else pitch_off)
            val = calibrated * sign
            ok = val >= TILT_MIN_DEG / 2
            checks.append({"step": key, "label": label, "deg": round(val, 1), "ok": ok})

        residual = max(abs(after[0]), abs(after[1]))
        ok = residual <= self.params["level_deg"]
        result = {
            "ok": ok,
            "before": [round(before[0], 1), round(before[1], 1)],
            "after": [round(after[0], 1), round(after[1], 1)],
            "delta_deg": round(max(abs(before[0]), abs(before[1])) - residual, 1),
            "residual_deg": round(residual, 1),
            "level_deg": self.params["level_deg"],
            "offset": {"roll": round(roll_off, 2), "pitch": round(pitch_off, 2)},
            "install": {"roll_sign": roll_sign, "pitch_sign": pitch_sign},
            "checks": checks,
            "samples": {"still": len(still), "ref": {k: len(v) for k, v in self._done.items()
                                                     if k != "still"}},
            "why": "" if ok else f"解算后仍有 {residual:.1f}° 偏角，超过 {self.params['level_deg']}°："
                                 f"把鸭子放平再采一次静止段，或检查模块是不是松了",
        }
        self._result = result
        self.log_event("IMU 解算完成：校准前 %.1f°/%.1f° → 校准后 %.1f°/%.1f°"
                       % (before[0], before[1], after[0], after[1]))
        return result

    # ---------------- 落盘 ----------------
    async def _commit(self, args: dict) -> dict:
        if not self._result:
            raise ZeluxError("还没解算，先点解算看前后对比", "denied")
        if not self._result.get("ok"):
            raise denied(self._result.get("why") or "解算没通过，别急着写文件")
        path = Path(self.config.state_dir) / "imu_calib.json"
        data = {
            "ts": time.time(),
            "roll_offset": self._result["offset"]["roll"],
            "pitch_offset": self._result["offset"]["pitch"],
            "roll_sign": self._result["install"]["roll_sign"],
            "pitch_sign": self._result["install"]["pitch_sign"],
            "residual_deg": self._result["residual_deg"],
            "before": self._result["before"],
            "after": self._result["after"],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(path)
        self._calib = data
        # 立刻对新读数生效；没有这个方法（模拟 HAL）就只落文件，等下次启动加载
        apply = getattr(self.hal, "apply_calibration", None)
        if apply:
            apply(data["roll_offset"], data["pitch_offset"],
                  data["roll_sign"], data["pitch_sign"])
        self.ctx.state["imu_calibrated"] = True
        self.log_event(f"IMU 标定已写入：残差 {data['residual_deg']}°")
        return {"ok": True, "path": str(path), "calib": data,
                "note": "hub.posture 不动，标定不影响在跑的动作"}

    # ---------------- 模拟 / 辅助 ----------------
    async def _sim_tilt(self, args: dict) -> dict:
        hook = getattr(self.hal, "sim_hook", None)
        if not hook:
            raise unsupported("只有模拟模式能模拟倾斜；真机上请用手抬")
        res = await hook("tilt", {"roll": float(args.get("roll", 0.0)),
                                  "pitch": float(args.get("pitch", 0.0))})
        return {"ok": True, "sim": res}

    async def _bias(self, args: dict) -> dict:
        fn = getattr(self.hal, "estimate_bias", None)
        if not fn:
            raise unsupported("这台机器的 HAL 不支持陀螺零偏估计")
        return await fn(float(args.get("seconds", 3.0)))

    def _recent(self, args: dict) -> list:
        fn = getattr(self.hal, "recent", None)
        return fn(int(args.get("n", 60))) if fn else []

    async def stop(self) -> None:
        self._running = False
        await super().stop()
