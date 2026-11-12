"""训练模块（页面"训练"）：录制 → 回放 → 打包 → 导入策略 → 绑槽位 → 自检。

为什么长成这样：

* **录制直接存 JSONL**。50 Hz × 15 维，一段 3 分钟不到 1 MB；JSONL 的好处是采集中途断电，
  前面录的帧还能读（二进制格式断了就是全废）。分析、导出、回放都只读这一个文件。
* **meta.json 最后落盘且原子写**。帧文件是流式追加（不能每帧一次 replace，50 Hz 下等于自杀），
  所以"完整性"由 meta 的 frames/duration 来兜：没有 meta 的目录 = 没录完的会话，导出会拒。
* **回放必须接管总线并要求站好**。回放是"开环发目标位置"，舵机软着的时候发目标等于让它抽搐摔倒，
  所以先 `servo_take` + 检查 posture，并在拒绝时说清楚下一步怎么做。
* **导入只认 sha256 前 12 位做目录名**。同一个模型换个文件名重装不会变成两份；
  清单里的 joint_order 必须与 `JOINT_NAMES` 逐位相等——顺序错了，动作会"照着别人的关节表"发，
  机器会当场摔，所以这是硬拒，不是警告。
* **自检不装样子**。这台机器没装 onnxruntime 就老实说没装（导入照样完成，推理在 motiond 上跑），
  绝不返回一个假的 ok 让人以为权重没问题。
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path

from ..errors import ZeluxError, busy, denied, hardware
from ..hal.base import JOINT_IDS, JOINT_NAMES
from .base import BaseModule

CENTER = 2048
RAD_PER_UNIT = 2.0 * math.pi / 4096.0      # Feetech 0–4095 ↔ 一圈
UNIT_PER_RAD = 1.0 / RAD_PER_UNIT          # 651.9

SLOT_ORDER = ["walk", "sitstand", "ground_pick", "kick_left", "kick_right", "roulade", "stand"]
# 6 个动作槽 + stand 兜底，全部是内置的：模型可换，槽位不可删
BUILTIN_SLOTS = {
    "walk": "builtin-walk",
    "sitstand": "builtin-sitstand",
    "ground_pick": "builtin-ground_pick",
    "kick_left": "builtin-kick_left",
    "kick_right": "builtin-kick_right",
    "roulade": "builtin-roulade",
    "stand": "builtin-stand",
}
BUILTIN_NAMES = {
    "builtin-walk": "内置 · 走路",
    "builtin-sitstand": "内置 · 坐下站起",
    "builtin-ground_pick": "内置 · 地上捡东西",
    "builtin-kick_left": "内置 · 左脚踢球",
    "builtin-kick_right": "内置 · 右脚踢球",
    "builtin-roulade": "内置 · 前滚翻",
    "builtin-stand": "内置 · 站好",
}
SAFE_POSTURES = ("holding", "sitting", "standing", "walking")
DEFAULT_HZ = 50.0
MAX_SESSIONS = 50


# ====================================================================== 清单校验
def validate_manifest(raw) -> dict:
    """校验 policy.json。通过返回规范化后的清单，不通过抛 denied 并指出"第几项"。

    单独抽成纯函数是因为它是这套东西里唯一"错了就摔机"的地方，
    必须能脱离 hub / 硬件被单元测试反复敲（见 tests/test_train_manifest.py）。
    """
    if not isinstance(raw, dict):
        raise denied("policy.json 不是一个 JSON 对象：要一份带 task/obs_dim/… 的清单")

    task = str(raw.get("task") or "").strip()
    if not task:
        raise denied("清单第 1 项 task 不对：要写训练任务名，例如 "
                     "Mjlab-VelStand-Rough-Backlash-MicroDuck")

    obs = raw.get("obs_dim")
    if isinstance(obs, bool) or not isinstance(obs, int) or obs < 15:
        raise denied(f"清单第 2 项 obs_dim 不对：现在给的是 {obs!r}，观测维度必须是 "
                     f"≥15 的整数（15 颗关节）")
    act = raw.get("act_dim")
    if isinstance(act, bool) or not isinstance(act, int) or act < 15:
        raise denied(f"清单第 3 项 act_dim 不对：现在给的是 {act!r}，动作维度必须是 "
                     f"≥15 的整数（15 颗关节）")

    order = raw.get("joint_order")
    if not isinstance(order, list) or len(order) != len(JOINT_NAMES):
        got = len(order) if isinstance(order, list) else type(order).__name__
        raise denied(f"清单第 4 项 joint_order 不对：要正好 {len(JOINT_NAMES)} 个关节名，"
                     f"现在有 {got} 个")
    for i, (have, want) in enumerate(zip([str(x) for x in order], JOINT_NAMES)):
        if have != want:
            raise denied(f"清单第 4 项 joint_order 不对：第 {i + 1} 个关节写的是 "
                         f"{have!r}，应为 {want!r}（顺序必须与 JOINT_NAMES 完全一致，"
                         f"错位会让动作发到别人的关节上）")

    for idx, field in ((5, "action_scale"), (6, "action_offset")):
        seq = raw.get(field)
        if not isinstance(seq, list) or len(seq) != len(JOINT_NAMES):
            got = len(seq) if isinstance(seq, list) else type(seq).__name__
            raise denied(f"清单第 {idx} 项 {field} 不对：要 {len(JOINT_NAMES)} 个数"
                         f"（每个关节一个），现在有 {got} 个")
        for i, v in enumerate(seq):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise denied(f"清单第 {idx} 项 {field} 不对：第 {i + 1} 个值 {v!r} 不是数字")
            if not math.isfinite(float(v)):
                raise denied(f"清单第 {idx} 项 {field} 不对：第 {i + 1} 个值不是有限数")

    hz = raw.get("control_hz")
    if isinstance(hz, bool) or not isinstance(hz, (int, float)):
        raise denied(f"清单第 7 项 control_hz 不对：现在给的是 {hz!r}，要是数字（20–500）")
    if not 20 <= float(hz) <= 500:
        raise denied(f"清单第 7 项 control_hz 不对：{hz} Hz 不在 20–500 之间；"
                     f"低于 20 站不稳，高于 500 这颗总线跑不动")

    return {
        "task": task,
        "obs_dim": int(obs),
        "act_dim": int(act),
        "joint_order": [str(x) for x in order],
        "action_scale": [float(x) for x in raw["action_scale"]],
        "action_offset": [float(x) for x in raw["action_offset"]],
        "control_hz": float(hz),
        "notes": str(raw.get("notes") or ""),
    }


# ====================================================================== 原子写
def _atomic_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    _atomic_bytes(path, text.encode("utf-8"))


def _atomic_json(path: Path, obj) -> None:
    _atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def _read_json(path: Path):
    return json.loads(path.read_text("utf-8"))


def _safe_name(name) -> str:
    """会话名只能是字母数字下划线短横：它会被拼进路径，绝不接受 ../ 之类。"""
    raw = str(name or "").strip()
    if not raw:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", raw):
        raise denied(f"会话名 {raw[:40]!r} 不合法：只允许字母、数字、下划线、短横（≤40）")
    return raw


# ====================================================================== 模块
class Module(BaseModule):
    name = "train"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.dir = ctx.config.state_dir / "train"
        self.models_dir = ctx.config.models_dir
        self.models_path = ctx.config.state_dir / "models.json"
        self._models: dict[str, dict] = {}
        self._slots: dict[str, str] = dict(BUILTIN_SLOTS)
        self._running: str | None = None
        self._rec: dict | None = None
        self._rec_task: asyncio.Task | None = None
        self._job: dict | None = None
        self._job_task: asyncio.Task | None = None
        self._pause = asyncio.Event()
        self._pause.set()                      # set = 正在跑，clear = 暂停
        self._sessions: list[dict] = []
        self._ort: bool | None = None

    # ------------------------------------------------------------------ 生命周期
    async def start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._load_store()
        self._refresh_sessions()
        self._ort = importlib.util.find_spec("onnxruntime") is not None
        self.log.info("训练模块就绪：%d 段录制、%d 个导入模型%s",
                      len(self._sessions), len(self._models),
                      "" if self._ort else "（这台机器没装 onnxruntime，推理在 motiond 上）")

    async def stop(self) -> None:
        if self._job_task:
            self._job_task.cancel()
        if self._rec:
            try:
                await self._finish_record()
            except Exception:
                pass
        await super().stop()

    # ------------------------------------------------------------------ 状态
    def state(self) -> dict:
        rec = self._rec
        now = time.monotonic()
        job = None
        if self._job:
            job = dict(self._job)
            job["elapsed_s"] = round(now - job.get("t0", now), 1)
            job.pop("t0", None)
        return {"train": {
            "recording": bool(rec),
            "session": rec["session"] if rec else None,
            "frames": rec["frames"] if rec else 0,
            "hz": rec["hz"] if rec else DEFAULT_HZ,
            "duration_s": round(now - rec["t0"], 2) if rec else 0.0,
            "missing_samples": rec["missing"] if rec else 0,
            "imu": bool(rec.get("imu")) if rec else False,
            "sessions": list(self._sessions),
            "slots": [{"slot": s, "model": self._slots.get(s), "running": s == self._running,
                       "can_delete": False, "builtin": True,
                       "model_name": self._model_name(self._slots.get(s))}
                      for s in SLOT_ORDER],
            "models": ([{"id": mid, "name": BUILTIN_NAMES[mid], "builtin": True,
                         "task": "builtin", "control_hz": DEFAULT_HZ, "can_delete": False}
                        for mid in BUILTIN_SLOTS.values()]
                       + [dict(v, builtin=False, can_delete=True) for v in self._models.values()]),
            "job": job,
            "busy": bool(self._job_task and not self._job_task.done()),
            "onnxruntime": bool(self._ort),
            "running": self._running,
        }}

    def _model_name(self, mid: str | None) -> str:
        if not mid:
            return "（空）"
        if mid in BUILTIN_NAMES:
            return BUILTIN_NAMES[mid]
        return (self._models.get(mid) or {}).get("name") or mid

    # ------------------------------------------------------------------ 命令
    async def cmd(self, op: str, args: dict) -> dict:
        if op == "record":
            return await self._op_record(args)
        if op == "stop":
            return await self._op_stop()
        if op == "replay":
            return await self._op_replay(args)
        if op == "pause":
            return await self._op_pause(True)
        if op == "resume":
            return await self._op_pause(False)
        if op == "abort":
            return await self._op_abort()
        if op == "export":
            return await self._op_export(args)
        if op == "import":
            return await self._op_import(args)
        if op == "slot":
            return self._op_slot(args)
        if op == "selftest":
            return await self._op_selftest(args)
        if op == "sessions":
            self._refresh_sessions()
            return {"sessions": list(self._sessions)}
        raise denied(f"训练页没有 {op} 这个操作")

    # ---------------- 录制 ----------------
    async def _wait_servos(self, timeout: float = 0.4) -> dict:
        """HAL 的快照是后台刷新的，刚开机就点"录制"会撞上还没填过的缓存。

        等 0.4 秒而不是直接报错：这 0.4 秒是真机上"总线刚上电"的正常抖动。
        """
        snap = self.hal.snapshot_servos()
        waited = 0.0
        while not (snap.get("items") or []) and waited < timeout:
            await asyncio.sleep(0.1)
            waited += 0.1
            snap = self.hal.snapshot_servos()
        return snap

    async def _op_record(self, args: dict) -> dict:
        if self._rec:
            raise busy(f"已经在录 {self._rec['session']} 了，先 stop 结束它")
        if self._job_task and not self._job_task.done():
            raise busy("回放/导入还在跑，等它结束再录")

        snap = await self._wait_servos()
        items = {int(it.get("id", -1)): it for it in (snap.get("items") or [])}
        if not items:
            raise hardware("读不到舵机：检查电源和总线（/dev/ttyS2）后再录")
        missing = [sid for sid in JOINT_IDS if sid not in items]
        if missing:
            raise denied(f"还差 {len(missing)} 颗舵机没上线（ID {missing}）——15 维少一颗，"
                         f"录出来的动作不能用，先修总线")

        session = _safe_name(args.get("session")) or time.strftime("s-%Y%m%d-%H%M%S")
        try:                                  # 同名会话不许覆盖，录一半的东西最容易被误删
            hz = float(args.get("hz", DEFAULT_HZ))
        except (TypeError, ValueError):
            hz = DEFAULT_HZ
        hz = min(200.0, max(10.0, hz))
        sdir = self.dir / session
        sdir.mkdir(parents=True, exist_ok=True)
        path = sdir / "frames.jsonl"
        if path.exists() and not args.get("overwrite"):
            raise denied(f"已经有 {session} 这段录制了：换个名字，或传 overwrite:true 覆盖"
                         f"（旧帧会被清掉，打包好的 .zdm 不受影响）")
        if args.get("overwrite"):
            (sdir / "meta.json").unlink(missing_ok=True)   # 旧 meta 先删，免得"半段新录制"配上"旧时长"
        fh = path.open("w", encoding="utf-8")

        self._rec = {
            "session": session,
            "hz": hz,
            "t0": time.monotonic(),
            "created": round(time.time(), 1),
            "frames": 0,
            "missing": 0,
            "fh": fh,
            "path": path,
            "imu": bool(args.get("imu", True)),
            "bus": snap.get("bus"),
            "backend": (self.hal.info() or {}).get("backend"),
        }
        self._rec_task = asyncio.create_task(self._record_loop())
        self.log_event(f"开始录制 {session}（{hz:g} Hz，15 维 + {'IMU' if self._rec['imu'] else '仅关节'}）",
                       cat="train")
        return {"session": session, "hz": hz, "frames": 0, "path": str(path),
                "note": "在录了：让鸭子把动作做一遍，做完按 stop"}

    async def _record_loop(self) -> None:
        rec = self._rec
        assert rec is not None
        period = 1.0 / rec["hz"]
        t0 = rec["t0"]
        nxt = t0
        prev: list[float] | None = None
        last_flush = t0
        last_emit = t0
        try:
            while True:
                now = time.monotonic()
                if now < nxt:
                    await asyncio.sleep(nxt - now)
                elif now - nxt > 2 * period:
                    nxt = now                 # 落后太多（磁盘/CPU 卡了一下）就别补帧，直接对齐下一秒，
                    self.log_event(           # 否则会一口气写几百帧，时间戳全挤在一起
                        "录制掉拍了，已对齐时间轴", level="warning", cat="train")
                nxt += period
                elapsed = time.monotonic() - t0

                items = {int(it.get("id", -1)): it
                         for it in (self.hal.snapshot_servos().get("items") or [])}
                q: list[float] = []
                for i, sid in enumerate(JOINT_IDS):
                    it = items.get(sid)
                    if it is None or it.get("pos") is None:
                        rec["missing"] += 1
                        q.append(prev[i] if prev else 0.0)   # 掉一颗就保持上一帧，别写 0（0 是极限位）
                        continue
                    q.append(round((float(it["pos"]) - CENTER) * RAD_PER_UNIT, 5))
                prev = q

                frame = {"t": round(elapsed, 4), "q": q}
                if rec["imu"]:
                    imu = self.hal.snapshot_imu() or {}
                    if imu.get("ok"):
                        frame["imu"] = {k: round(float(imu.get(k) or 0.0), 2)
                                        for k in ("roll", "pitch", "yaw", "gz")}
                rec["fh"].write(json.dumps(frame, separators=(",", ":")) + "\n")
                rec["frames"] += 1

                mono = time.monotonic()
                if mono - last_flush > 0.5:
                    rec["fh"].flush()
                    last_flush = mono
                if mono - last_emit > 1.0:
                    last_emit = mono
                    self.emit("train", {"phase": "recording", "session": rec["session"],
                                        "frames": rec["frames"],
                                        "duration_s": round(elapsed, 1)})
        except asyncio.CancelledError:
            raise
        except Exception as exc:              # 磁盘满之类：别把守护进程带崩
            self.log_event(f"录制中断：{exc}", level="error", cat="train")
            if self._job is None:
                self._job = {"kind": "record", "phase": "error", "progress": 0.0,
                             "error": f"录制中断：{exc}", "t0": time.monotonic()}

    async def _finish_record(self) -> dict:
        rec = self._rec
        if not rec:
            raise denied("现在没在录")
        self._rec = None
        if self._rec_task:
            self._rec_task.cancel()
            try:
                await self._rec_task
            except (asyncio.CancelledError, Exception):
                pass
            self._rec_task = None
        fh = rec["fh"]
        try:
            fh.flush()
            os.fsync(fh.fileno())
        except Exception:
            pass
        fh.close()

        duration = round(time.monotonic() - rec["t0"], 2)
        meta = {
            "session": rec["session"],
            "created": rec["created"],
            "duration_s": duration,
            "frames": rec["frames"],
            "hz": rec["hz"],
            "joint_order": list(JOINT_NAMES),
            "joint_ids": list(JOINT_IDS),
            "source": {"kind": "servos", "api": "hal.snapshot_servos",
                       "backend": rec["backend"], "bus": rec["bus"]},
            "imu": rec["imu"],
            "missing_samples": rec["missing"],
            "format": "jsonl: {t: 秒, q: 15 个弧度, imu?: {roll,pitch,yaw,gz}}",
        }
        _atomic_json(self.dir / rec["session"] / "meta.json", meta)
        self._refresh_sessions()
        self.emit("train", {"phase": "recorded", "session": rec["session"],
                            "frames": rec["frames"], "duration_s": duration})
        self.log_event(f"录制结束 {rec['session']}：{rec['frames']} 帧 / {duration:g} 秒"
                       f"{'（有掉帧）' if rec['missing'] else ''}", cat="train")
        return {"session": rec["session"], "frames": rec["frames"], "duration_s": duration,
                "hz": rec["hz"], "missing_samples": rec["missing"],
                "next": "op=export 打成动作包（.zdm），或 op=replay 立刻回放看效果"}

    async def _op_stop(self) -> dict:
        if self._rec:
            return await self._finish_record()
        if self._job_task and not self._job_task.done():
            return await self._op_abort()
        return {"ok": True, "note": "现在既没在录，也没有在跑的任务"}

    # ---------------- 回放 ----------------
    async def _op_replay(self, args: dict) -> dict:
        if self._job_task and not self._job_task.done():
            raise busy("上一个任务还没跑完，先 abort")
        if self._rec:
            raise busy("正在录制，先 stop 再回放（一边录一边放会把动作录脏）")

        session = _safe_name(args.get("session")) or self._latest_session()
        if not session:
            raise denied("还没有任何录制。先 op=record 录一段，再来回放")
        sdir = self.dir / session
        frames_path = sdir / "frames.jsonl"
        if not frames_path.exists():
            raise denied(f"没有 {session} 这段录制的帧文件；先 op=record 录一段")
        meta = {}
        if (sdir / "meta.json").exists():
            try:
                meta = _read_json(sdir / "meta.json")
            except Exception:
                meta = {}
        try:
            speed = float(args.get("speed", 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        speed = min(4.0, max(0.1, speed))

        posture = self.ctx.hub.posture
        if posture not in SAFE_POSTURES and self.config.mode != "sim":
            raise denied("先让鸭子站好、松开策略：现在 posture="
                         f"{posture}。到『舵机』页点『站姿』并保持住，把 motiond 的策略槽清空，"
                         "再回来回放")
        try:
            await self.hal.servo_take()
        except Exception as exc:
            raise hardware(f"接管舵机总线失败：{exc}（有别的进程在占总线？）") from exc

        total = int(meta.get("frames") or self._count_frames(frames_path))
        self._job = {"kind": "replay", "phase": "replaying", "progress": 0.0, "error": None,
                     "session": session, "speed": speed, "frames": 0, "total": total,
                     "t0": time.monotonic()}
        self._pause.set()
        self._job_task = asyncio.create_task(self._replay_loop(session, frames_path, speed, total))
        self.log_event(f"回放 {session}：{total} 帧，{speed:g}× 速度", cat="train")
        return {"session": session, "frames": total, "speed": speed, "phase": "replaying",
                "posture": posture}

    async def _replay_loop(self, session: str, frames_path: Path, speed: float, total: int) -> None:
        job = self._job
        assert job is not None
        base = time.monotonic()
        shift = 0.0
        done = 0
        completed = False
        try:
            with frames_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not self._pause.is_set():        # 暂停：时间轴整体往后挪，别让恢复时"追帧"
                        t_wait = time.monotonic()
                        await self._pause.wait()
                        shift += time.monotonic() - t_wait
                        job["phase"] = "replaying"
                    target = base + float(frame.get("t") or 0.0) / speed + shift
                    delay = target - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    goals = {}
                    for sid, q in zip(JOINT_IDS, frame.get("q") or []):
                        unit = int(round(CENTER + float(q) * UNIT_PER_RAD))
                        goals[sid] = 0 if unit < 0 else 4095 if unit > 4095 else unit
                    await self.hal.servo_sync_goal(goals)
                    done += 1
                    if done % 5 == 0 or done == total:
                        job["frames"] = done
                        job["progress"] = round(done / max(1, total), 3)
            completed = True
            job["phase"] = "done"
            job["progress"] = 1.0
            job["frames"] = done
        except asyncio.CancelledError:
            job["phase"] = "aborted"
            job["error"] = None
            raise
        except Exception as exc:
            job["phase"] = "error"
            job["error"] = f"回放中断：{exc}"
            self.log_event(job["error"], level="error", cat="train")
        finally:
            if completed:
                try:
                    await self.hal.servo_release()   # 交还总线，但舵机继续保持最后姿势
                except Exception:
                    pass
                self.log_event(f"回放完成 {session}：{done} 帧，鸭子停在最后一帧", cat="train")
                self.emit("train", {"phase": "replayed", "session": session, "frames": done})

    async def _op_pause(self, paused: bool) -> dict:
        if not (self._job_task and not self._job_task.done()) or \
                (self._job or {}).get("kind") != "replay":
            raise denied("现在没有在回放，暂停没意义")
        if paused:
            self._pause.clear()
            if self._job:
                self._job["phase"] = "paused"
            self.log_event("回放已暂停", cat="train")
        else:
            self._pause.set()
            if self._job:
                self._job["phase"] = "replaying"
            self.log_event("回放继续", cat="train")
        return {"paused": paused, "phase": (self._job or {}).get("phase")}

    async def _op_abort(self) -> dict:
        if not (self._job_task and not self._job_task.done()):
            if self._rec:
                return await self._finish_record()
            raise denied("现在没有在跑的任务")
        kind = (self._job or {}).get("kind")
        self._pause.set()
        self._job_task.cancel()
        try:
            await self._job_task
        except (asyncio.CancelledError, Exception):
            pass
        self._job_task = None
        if self._job:
            self._job["phase"] = "aborted"
        if kind == "replay":
            try:
                await self.hal.servo_release()   # 交还总线，但舵机继续保持，鸭子不会软下来
            except Exception:
                pass
        self.log_event(f"已中止{kind or '任务'}；舵机保持当前姿势，鸭子不会软下来", level="warning",
                       cat="train")
        return {"aborted": True, "kind": kind}

    # ---------------- 导出 ----------------
    async def _op_export(self, args: dict) -> dict:
        if self._rec:
            raise busy("正在录制，先 stop 再打包（半段录制打出来只有半个动作）")
        session = _safe_name(args.get("session")) or self._latest_session()
        if not session:
            raise denied("还没有录制，导不出动作包")
        sdir = self.dir / session
        frames_path = sdir / "frames.jsonl"
        meta_path = sdir / "meta.json"
        if not frames_path.exists():
            raise denied(f"没有 {session} 的帧文件")
        if not meta_path.exists():
            raise denied(f"{session} 没有 meta.json：这段录制没正常结束，回放确认后再重新录一次")

        self._job = {"kind": "export", "phase": "packing", "progress": 0.0, "error": None,
                     "session": session, "t0": time.monotonic()}
        try:
            meta = _read_json(meta_path)
            points = self._sample_points(frames_path, 160)
            svg = _preview_svg(points, meta)
            frames_text = frames_path.read_text("utf-8")
            pack = sdir / f"{session}.zdm"
            tmp = sdir / f"{session}.zdm.tmp"
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
                z.writestr("frames.jsonl", frames_text)
                z.writestr("preview.svg", svg)
            tmp.replace(pack)                       # 临时文件 + replace：页面永远看不到半个包
        except Exception as exc:
            if self._job:
                self._job["phase"] = "error"
                self._job["error"] = f"打包失败：{exc}"
            raise ZeluxError(f"打包失败：{exc}", "failed") from exc

        self._job["phase"] = "done"
        self._job["progress"] = 1.0
        self._job["file"] = pack.name
        self._refresh_sessions()
        self.log_event(f"动作包已生成：{pack.name}（{pack.stat().st_size // 1024} KB）", cat="train")
        return {"file": pack.name, "path": str(pack), "bytes": pack.stat().st_size,
                "frames": int(meta.get("frames") or 0),
                "contents": ["meta.json", "frames.jsonl", "preview.svg"]}

    def _sample_points(self, path: Path, limit: int) -> list[list[float]]:
        rows: list[list[float]] = []
        every = 1
        total = self._count_frames(path)
        if total > limit:
            every = max(1, total // limit)
        with path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i % every:
                    continue
                try:
                    rows.append([float(x) for x in json.loads(line).get("q") or []])
                except Exception:
                    continue
        return rows

    # ---------------- 导入 ----------------
    async def _op_import(self, args: dict) -> dict:
        name = str(args.get("name") or "").strip() or "导入的策略"
        onnx, manifest_path = self._resolve_inputs(args)
        try:
            raw = _read_json(manifest_path)
        except Exception as exc:
            raise denied(f"policy.json 读不出来：{exc}") from exc
        manifest = validate_manifest(raw)

        data = onnx.read_bytes()
        if not data:
            raise denied(f"{onnx.name} 是空文件")
        digest = hashlib.sha256(data).hexdigest()
        mid = digest[:12]

        self._job = {"kind": "import", "phase": "saving", "progress": 0.2, "error": None,
                     "model_id": mid, "t0": time.monotonic()}
        target = self.models_dir / mid
        reused = target.exists()
        try:
            if not reused:
                tmp = self.models_dir / f".tmp-{mid}"
                if tmp.exists():
                    for p in tmp.iterdir():
                        p.unlink()
                tmp.mkdir(parents=True, exist_ok=True)
                _atomic_bytes(tmp / "policy.onnx", data)
                _atomic_json(tmp / "policy.json", manifest)
                if target.exists():                 # 并发导入同一个模型：谁先到算谁的
                    for p in tmp.iterdir():
                        p.unlink()
                    tmp.rmdir()
                    reused = True
                else:
                    os.replace(tmp, target)         # 整个目录一次性出现，不会出现"半个模型"
        except Exception as exc:
            if self._job:
                self._job["phase"] = "error"
                self._job["error"] = f"写入 models 目录失败：{exc}"
            raise hardware(f"写入 models 目录失败：{exc}") from exc

        entry = {
            "id": mid,
            "name": name,
            "task": manifest["task"],
            "obs_dim": manifest["obs_dim"],
            "act_dim": manifest["act_dim"],
            "control_hz": manifest["control_hz"],
            "notes": manifest["notes"],
            "bytes": len(data),
            "sha256": digest,
            "imported_at": round(time.time(), 1),
            "path": str(target),
        }
        self._models[mid] = entry
        self._save_store()
        if self._job:
            self._job["phase"] = "done"
            self._job["progress"] = 1.0
        self.log_event(f"已导入策略 {name}（{manifest['task']}，{mid}）"
                       f"{'，同一个模型已存在直接复用' if reused else ''}", cat="train")
        self.emit("train", {"phase": "imported", "model_id": mid, "name": name})
        return {"model_id": mid, "name": name, "reused": reused, "task": manifest["task"],
                "control_hz": manifest["control_hz"], "bytes": len(data),
                "next": "op=slot 绑到 walk/sitstand/… 槽位，或 op=selftest 先自检推理"}

    def _resolve_inputs(self, args: dict) -> tuple[Path, Path]:
        """定位 policy.onnx + 同名 json。只允许上传目录里的文件，避免 web 请求变成任意文件读取。"""
        raw = args.get("path")
        uid = args.get("upload_id")
        if raw:
            base = Path(str(raw)).expanduser()
        elif uid:
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", str(uid)):
                raise denied("upload_id 不合法（只允许字母数字和 _.-，≤64）")
            base = self.config.state_dir / "uploads" / str(uid)
        else:
            raise denied("import 需要 path 或 upload_id：先把 policy.onnx 和同名 policy.json 传上来")
        base = base.resolve()

        allowed = [self.config.state_dir.resolve(), Path(tempfile.gettempdir()).resolve(),
                   Path("/tmp").resolve()]
        if not any(base == a or a in base.parents for a in allowed):
            raise denied("只接受上传目录（状态目录/uploads 或临时目录）里的文件，"
                         "不要直接指系统路径")

        if base.is_dir():
            onnx = base / "policy.onnx"
            manifest = base / "policy.json"
            if not onnx.exists():
                cands = sorted(base.glob("*.onnx"))
                if not cands:
                    raise denied(f"{base.name} 里没有 .onnx：要 policy.onnx（或任意名字的 .onnx）")
                if len(cands) > 1:
                    raise denied(f"{base.name} 里有 {len(cands)} 个 .onnx，请用 path 指定是哪一个")
                onnx = cands[0]
                manifest = onnx.with_suffix(".json")
        else:
            if base.suffix.lower() != ".onnx":
                raise denied(f"{base.name} 不是 .onnx")
            onnx = base
            manifest = base.with_suffix(".json")

        if not onnx.exists():
            raise denied(f"找不到模型文件：{onnx}")
        if not manifest.exists():
            raise denied(f"找不到清单 {manifest.name}：policy.onnx 必须配一份同名 json")
        return onnx, manifest

    # ---------------- 槽位 ----------------
    def _op_slot(self, args: dict) -> dict:
        action = str(args.get("action") or "set").strip().lower()
        slot = str(args.get("slot") or "").strip()
        if slot not in SLOT_ORDER:
            raise denied(f"没有 {slot or '（空）'} 这个槽位；只有：{'、'.join(SLOT_ORDER)}")

        if action == "set":
            mid = str(args.get("model_id") or "").strip()
            if not mid:
                raise denied("set 槽位要给 model_id（先用 op=import 导入）")
            if mid not in self._models and mid not in BUILTIN_NAMES:
                raise denied(f"没有 {mid} 这个模型；先 op=import，或从 models 列表里选")
            self._slots[slot] = mid
            self._save_store()
            self.log_event(f"槽位 {slot} → {self._model_name(mid)}", cat="train")
            return {"slot": slot, "model": mid, "model_name": self._model_name(mid),
                    "can_delete": False}

        if action == "clear":
            if slot in BUILTIN_SLOTS:
                # 内置槽位不可删：clear 的语义是"换回内置策略"
                self._slots[slot] = BUILTIN_SLOTS[slot]
                self._save_store()
                self.log_event(f"槽位 {slot} 恢复内置策略", cat="train")
                return {"slot": slot, "model": BUILTIN_SLOTS[slot], "cleared": False,
                        "can_delete": False,
                        "note": "内置槽位删不掉，已换回内置策略（要换自己的模型请用 action=set）"}
            self._slots[slot] = BUILTIN_SLOTS.get(slot, "")
            self._save_store()
            return {"slot": slot, "model": self._slots[slot], "cleared": True}

        raise denied("slot 的 action 只有两种：set / clear")

    def request_slot(self, slot: str, source: str = "pad") -> dict:
        """给 pad 用：把某个槽位设成"当前动作"（真正推理在 motiond）。不走 dispatch 锁。"""
        if slot not in SLOT_ORDER:
            raise denied(f"没有 {slot} 这个动作槽")
        self._running = slot
        mid = self._slots.get(slot)
        self.emit("train.slot", {"slot": slot, "model": mid, "source": source, "running": True})
        return {"slot": slot, "model": mid, "running": slot}

    # ---------------- 自检 ----------------
    async def _op_selftest(self, args: dict) -> dict:
        mid = str(args.get("model_id") or "").strip()
        if not mid:
            raise denied("自检要给 model_id")
        if mid in BUILTIN_NAMES:
            return {"ok": False, "model_id": mid,
                    "error": "内置策略的权重在 motiond 里，页面上没法自检；导入自己的 .onnx 才能在这里验"}
        entry = self._models.get(mid)
        if not entry:
            raise denied(f"没有 {mid} 这个模型")
        if not self._ort:
            return {"ok": False, "model_id": mid,
                    "error": "这台机器没装 onnxruntime，导入仍可完成，推理要在 motiond 上做"}
        path = self.models_dir / mid / "policy.onnx"
        if not path.exists():
            raise denied(f"模型文件不见了：{path}")
        return await asyncio.to_thread(self._run_ort, mid, path, entry)

    def _run_ort(self, mid: str, path: Path, entry: dict) -> dict:
        """全零观测跑一次，量耗时。这里只验"能不能跑通 + 输出维度"，不验策略好坏。"""
        try:
            import numpy as np
            import onnxruntime as ort

            t0 = time.perf_counter()
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            load_ms = (time.perf_counter() - t0) * 1000
            feed = {}
            shape = [1, int(entry.get("obs_dim") or 15)]
            for inp in sess.get_inputs():
                dims = [d if isinstance(d, int) else (shape[i] if i < len(shape) else 1)
                        for i, d in enumerate(inp.shape)]
                feed[inp.name] = np.zeros(dims, dtype=np.float32)
            t1 = time.perf_counter()
            outs = sess.run(None, feed)
            ms = (time.perf_counter() - t1) * 1000
            return {"ok": True, "model_id": mid, "load_ms": round(load_ms, 1),
                    "ms": round(ms, 2), "inputs": {k: list(v.shape) for k, v in feed.items()},
                    "outputs": [list(o.shape) for o in outs],
                    "out_dim": int(outs[0].shape[-1]) if outs and len(outs[0].shape) else 0,
                    "note": "全零观测只为验通路；真实推理在 motiond 的 200 Hz 环里跑"}
        except Exception as exc:
            return {"ok": False, "model_id": mid,
                    "error": f"模型跑不起来：{exc}（清单里的 obs_dim 和导出的图对不上时会这样）"}

    # ------------------------------------------------------------------ 存储
    def _load_store(self) -> None:
        raw = {}
        try:
            raw = _read_json(self.models_path)
        except Exception:
            raw = {}
        models = raw.get("models") or {}
        self._models = {k: v for k, v in models.items() if (self.models_dir / k).exists()}
        slots = raw.get("slots") or {}
        self._slots = {}
        for s in SLOT_ORDER:
            mid = slots.get(s) or BUILTIN_SLOTS[s]
            if mid not in self._models and mid not in BUILTIN_NAMES:
                mid = BUILTIN_SLOTS[s]            # 模型目录被删了：槽位回落，别留一个空槽
            self._slots[s] = mid

    def _save_store(self) -> None:
        _atomic_json(self.models_path, {"models": self._models, "slots": self._slots,
                                        "updated": round(time.time(), 1)})

    def _refresh_sessions(self) -> None:
        if not self.dir.exists():
            self._sessions = []
            return
        out = []
        for d in sorted(self.dir.iterdir()):
            if not d.is_dir():
                continue
            meta = {}
            if (d / "meta.json").exists():
                try:
                    meta = _read_json(d / "meta.json")
                except Exception:
                    meta = {}
            fpath = d / "frames.jsonl"
            frames = meta.get("frames")
            if frames is None:
                frames = self._count_frames(fpath) if fpath.exists() else 0
            out.append({"session": d.name,
                        "frames": int(frames),
                        "duration_s": meta.get("duration_s"),
                        "hz": meta.get("hz", DEFAULT_HZ),
                        "created": meta.get("created"),
                        "bytes": fpath.stat().st_size if fpath.exists() else 0,
                        "packed": (d / f"{d.name}.zdm").exists(),
                        "complete": bool(meta)})
        out.sort(key=lambda s: s.get("created") or 0, reverse=True)
        self._sessions = out[:MAX_SESSIONS]

    @staticmethod
    def _count_frames(path: Path) -> int:
        n = 0
        try:
            with path.open("rb") as fh:
                for line in fh:
                    if line.strip():
                        n += 1
        except Exception:
            return 0
        return n

    def _latest_session(self) -> str:
        if not self._sessions:
            self._refresh_sessions()
        return self._sessions[0]["session"] if self._sessions else ""


# ====================================================================== 缩略图
def _preview_svg(rows: list[list[float]], meta: dict) -> str:
    """动作包里的 preview.svg：15 条关节轨迹叠成一张条带图，不解压也能看出"动作有没有走完"。"""
    w, h = 320, 208
    left, right, top = 12, 308, 40
    n = len(rows)
    session = str(meta.get("session") or "session")
    frames = int(meta.get("frames") or n)
    hz = meta.get("hz") or DEFAULT_HZ
    dur = meta.get("duration_s") or 0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="{session} 关节轨迹">',
        '<rect width="100%" height="100%" fill="#0B0F14" rx="8"/>',
        f'<text x="{left}" y="20" fill="#F6F8FA" font-family="Inter,sans-serif" '
        f'font-size="11" font-weight="600">{_esc(session)}</text>',
        f'<text x="{left}" y="32" fill="#8A94A6" font-family="Inter,sans-serif" font-size="9">'
        f'{frames} 帧 · {float(hz):g} Hz · {float(dur):g} s · 15 关节</text>',
    ]
    if n >= 2 and all(len(r) >= 15 for r in rows):
        band = (h - 56 - top) / 15.0
        for j in range(15):
            vals = [r[j] for r in rows]
            lo, span = min(vals), (max(vals) - min(vals)) or 1.0
            pts = []
            for i, v in enumerate(vals):
                x = left + (right - left) * i / (n - 1)
                y = top + band * j + band * 0.85 - (v - lo) / span * band * 0.7
                pts.append(f"{x:.1f},{y:.1f}")
            color = "#00D8A0" if j < 10 else "#3D7BFF"   # 腿 10 条绿、头颈 5 条蓝
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
                         f'stroke-width="1" stroke-linejoin="round" opacity="0.85"/>')
        parts.append(f'<text x="{left}" y="{h - 6}" fill="#8A94A6" font-family="Inter,sans-serif" '
                     f'font-size="8">腿 r/l 10–34 · 头颈 30–34</text>')
    else:
        parts.append(f'<text x="{left}" y="{h // 2}" fill="#FFB020" font-family="Inter,sans-serif" '
                     f'font-size="10">帧不足，画不出轨迹</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))
