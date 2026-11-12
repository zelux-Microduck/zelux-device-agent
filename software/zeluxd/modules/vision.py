"""视觉页（`name = "vision"`）：拍照 / 录主人 / 跟随 / 识图问答 / 物体检测。

这一页的全部难点都不在代码，而在"别骗人"：

* **识别能力分三档，页面必须说清是哪一档**：insightface 的 512 维人脸 embedding
  （真识别）→ 颜色直方图 + 画面下 60% 区域（弱特征，换衣服就失效）→ 只有名字（啥也没算）。
  所以 enroll 的返回里带 `weak` 与一句人话；把直方图叫"人脸识别"是骗人，会让人真拿它开门。
* **跟随不是"跟着主人"**：没有 re-ID 模型时只能跟"画面里最大的那个人"。
  这一点写在 state.follow_target 里，别让用户以为它认人。
* **没有模型就直说**：`detect` 找不到 `var/models/vision/*.onnx` 时返回 false + 放模型的路径，
  绝不用"置信度 0.87 的人"来编。

所有重活（解码图片、跑 ONNX、抽 embedding、拍照）都走 `asyncio.to_thread`：
hub 的 10 Hz 状态总线就在同一个事件循环里，一次 300 ms 的阻塞会让整页掉帧。
"""
from __future__ import annotations

import asyncio
import base64
import importlib
import math
import time
from pathlib import Path

from ..errors import ZeluxError
from .base import BaseModule

MAX_SHOTS = 50                 # 只留最近 50 张，eMMC 不大，照片会越拍越多
FOLLOW_HZ = 10
LOST_STOP_S = 0.5              # 0.5 秒没检测就停：再往前就是"盲走"
LOST_GIVEUP_S = 3.0            # 3 秒还找不到人就结束跟随，别让它一直对着空气转
DEAD_ZONE = 0.08               # 中心死区 ±8%，否则目标在正中也会左右抖
KP = 1.0
WZ_MAX = 1.2                   # rad/s，鸭子原地转快了会翻
VX_FAR = 0.18                  # 目标远：往前挪
VX_NEAR = -0.12                # 目标近：后退（负速度必须有上限，别一屁股坐地上）
SIZE_FAR = 0.25                # 目标框高度占画面 <25% 算远
SIZE_NEAR = 0.55               # >55% 算太近
DET_CONF = 0.30
FACE_DIR = "faces"
MODEL_SUBDIR = "vision"
MODEL_FILE = "yolov8n.onnx"
MODEL_HINT = f"把 {MODEL_FILE} 放到 var/models/{MODEL_SUBDIR}/ 就自动启用"
DETECT_ERR = "没装检测模型（把 yolov8n.onnx 放到 var/models/vision/ 就自动启用）"

# COCO 80 类：只用来把类别 id 翻成人话，顺序不能改
COCO = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)

FACE_HINT = ("pip install insightface onnxruntime opencv-python-headless"
             "（insightface 首次运行会联网下 buffalo_l 模型，约 300 MB）")
WEAK_NOTE = ("这是弱识别：只记了颜色分布 + 画面下 60% 区域，"
             "换个衣服、换个光线就会认错，不能当人脸识别用")


# ---------------------------------------------------------------- 纯函数（可单测）
def follow_control(cx_ratio: float, h_ratio: float, *, dead: float = DEAD_ZONE,
                   kp: float = KP, wz_max: float = WZ_MAX) -> dict:
    """框中心 → (vx, wz)。P 控制 + 死区 + 限速，纯算术，没有 IO。

    `cx_ratio` = 框中心 x / 画面宽（0..1），`h_ratio` = 框高 / 画面高。
    角速度先把死区减掉再归一化：这样刚出死区时是从 0 平滑加速，
    而不是"啪"一下跳到 0.2 rad/s（后者会让鸭子一顿一顿地扭头）。
    """
    err = float(cx_ratio) - 0.5
    span = max(1e-6, 0.5 - dead)
    if abs(err) <= dead:
        wz = 0.0
    else:
        mag = min(1.0, kp * (abs(err) - dead) / span)
        # 目标在画面右边（err>0）→ 要右转 → wz 为负（ROS 约定：正 = 左转/逆时针）
        wz = -math.copysign(mag * wz_max, err)

    size = float(h_ratio)
    if size < SIZE_FAR:
        vx, zone = VX_FAR, "far"
    elif size > SIZE_NEAR:
        vx, zone = VX_NEAR, "near"
    else:
        vx, zone = 0.0, "ok"
    return {"vx": round(vx, 3), "wz": round(wz, 3), "err": round(err, 3),
            "size": round(size, 3), "zone": zone}


def _safe_name(name: str) -> str:
    """录入的名字要当文件名用，先把路径分隔符挡掉。"""
    s = str(name or "").strip()
    if not s:
        raise ZeluxError("要给主人起个名字（比如「爸爸」）", "denied")
    if len(s) > 24:
        raise ZeluxError("名字太长了（最多 24 个字）", "denied")
    if s.startswith(".") or any(ch in s for ch in '/\\:*?"<>|'):
        raise ZeluxError('名字里不能有 / \\ : * ? " < > | 这些字符', "denied")
    return s


def _have(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- 人脸
class FaceEngine:
    """insightface 的懒加载外壳：第一次调用要加载 ~300 MB 模型，绝不能进事件循环。"""

    def __init__(self) -> None:
        self._app = None
        self.err = ""

    def ready(self) -> bool:
        return self._app is not None

    def embed(self, path: str) -> dict:
        """同步抽 embedding（调用方 to_thread）。返回 {ok, embedding, score, bbox} 或 {ok:False, why}。"""
        try:
            import numpy as np
            from insightface.app import FaceAnalysis
        except ImportError:
            return {"ok": False, "why": "没装 insightface", "hint": FACE_HINT}

        if self._app is None:
            try:
                app = FaceAnalysis(name="buffalo_l")
                app.prepare(ctx_id=-1, det_size=(320, 320))   # ctx_id=-1 = CPU，板子上没 GPU
                self._app = app
            except Exception as exc:      # 首次要联网下模型，公司网/无网必挂
                self.err = f"insightface 加载失败：{exc}"
                return {"ok": False, "why": self.err,
                        "hint": "第一次要联网下 buffalo_l；没网就先把模型拷到 ~/.insightface/models/"}
        try:
            img = _read_bgr(path)
            if img is None:
                return {"ok": False, "why": "读不出这张照片"}
            faces = self._app.get(img)
        except Exception as exc:
            return {"ok": False, "why": f"人脸检测出错：{exc}"}
        if not faces:
            return {"ok": False, "why": "照片里没找到脸：让主人正对镜头、别背光、别太远"}

        best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        emb = getattr(best, "normed_embedding", None)
        if emb is None:
            return {"ok": False, "why": "模型没给 embedding（buffalo_l 才有）"}
        box = getattr(best, "bbox", None)
        return {"ok": True, "embedding": np.asarray(emb, dtype="float32"),
                "score": round(float(getattr(best, "det_score", 0.0)), 3),
                "bbox": [int(v) for v in box] if box is not None else None}


_FACE = FaceEngine()


def _read_bgr(path: str):
    """读一张图成 BGR ndarray：先 cv2（快），退到 PIL。都没有返回 None。"""
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is not None:
            return img
    except ImportError:
        pass
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype="uint8")
        return rgb[:, :, ::-1].copy()      # RGB → BGR，和 cv2 对齐
    except Exception:
        return None


def _weak_feature(path: str) -> dict | None:
    """弱特征：8×8×8 RGB 直方图（取画面下 2/3）+ 归一化的人体框（没有就用整幅下半部）。

    "人体框"在没有检测模型时并不存在，所以这里老实给 None，
    让上层把 `bbox` 报成 null，而不是画一个假框骗人。
    """
    try:
        import numpy as np
    except ImportError:
        return None
    bgr = _read_bgr(path)
    if bgr is None:
        return None
    try:
        h, w = bgr.shape[:2]
        roi = bgr[int(h * 0.33):, :]                 # 下 2/3：人站着的区域
        if roi.size == 0:
            roi = bgr
        q = (roi.reshape(-1, 3) // 32).astype("int32")   # 每通道 8 档
        idx = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
        hist = np.bincount(idx, minlength=512).astype("float32")
        total = float(hist.sum()) or 1.0
        hist /= total
        return {"embedding": hist, "bbox": None, "size": [int(w), int(h)]}
    except Exception:
        return None


# ---------------------------------------------------------------- 检测器
class Detector:
    """YOLOv8 ONNX 的单文件检测器。只在 `var/models/vision/` 有模型时才存在。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._sess = None
        self.err = ""

    def load(self) -> bool:
        if self._sess is not None:
            return True
        try:
            import numpy  # noqa: F401  推理前后处理全靠它，缺了跑不动
        except ImportError:
            self.err = "没装 numpy：pip install numpy"
            return False
        try:
            import onnxruntime as ort
        except ImportError:
            self.err = "没装 onnxruntime：pip install onnxruntime"
            return False
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2      # RK3566 四核，留两个给实时层
            self._sess = ort.InferenceSession(str(self.path), opts,
                                              providers=["CPUExecutionProvider"])
            return True
        except Exception as exc:
            self.err = f"加载 {self.path.name} 失败：{exc}"
            return False

    @property
    def ready(self) -> bool:
        return self._sess is not None

    @property
    def input_size(self) -> int:
        try:
            shape = self._sess.get_inputs()[0].shape
            return int(shape[2]) if int(shape[2]) > 0 else 640
        except Exception:
            return 640

    def detect_file(self, path: str, label: str | None = None, conf: float = DET_CONF) -> list:
        bgr = _read_bgr(path)
        if bgr is None:
            return []
        return self.detect(bgr, label=label, conf=conf)

    def detect(self, bgr, label: str | None = None, conf: float = DET_CONF) -> list:
        """同步推理。返回 [{label, score, box:[x,y,w,h]}]，box 是原图坐标。"""
        try:
            import numpy as np
        except ImportError:
            return []
        if self._sess is None and not self.load():
            return []
        try:
            h, w = bgr.shape[:2]
            size = self.input_size
            scale = min(size / w, size / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            try:
                import cv2

                resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
                canvas = np.full((size, size, 3), 114, dtype="uint8")
                canvas[:nh, :nw] = resized
            except ImportError:
                from PIL import Image

                canvas = np.full((size, size, 3), 114, dtype="uint8")
                im = Image.fromarray(bgr[:, :, ::-1]).resize((nw, nh))
                canvas[:nh, :nw] = np.asarray(im, dtype="uint8")[:, :, ::-1]

            blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype("float32") / 255.0
            out = self._sess.run(None, {self._sess.get_inputs()[0].name: blob})[0]
            pred = np.squeeze(out)
            if pred.ndim != 2:
                return []
            if pred.shape[0] < pred.shape[1]:
                pred = pred.T                      # (84, 8400) → (8400, 84)
            boxes = pred[:, :4]
            scores = pred[:, 4:]
            cls = scores.argmax(axis=1)
            best = scores.max(axis=1)
            keep = best >= conf
            if not keep.any():
                return []
            boxes, cls, best = boxes[keep], cls[keep], best[keep]

            xyxy = np.empty_like(boxes)
            xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
            xyxy /= scale

            results = []
            for i in _nms(xyxy, best, 0.45):
                name = COCO[int(cls[i])] if int(cls[i]) < len(COCO) else str(int(cls[i]))
                if label and name != label:
                    continue
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy[i]]
                results.append({"label": name, "score": round(float(best[i]), 3),
                                "box": [max(0, x1), max(0, y1),
                                        max(0, min(w, x2 - x1)), max(0, min(h, y2 - y1))]})
            results.sort(key=lambda r: r["score"], reverse=True)
            return results[:20]
        except Exception as exc:
            self.err = f"推理失败：{exc}"
            return []


def _nms(boxes, scores, iou_thr: float) -> list[int]:
    """标准 NMS：YOLOv8 输出是"无锚点"的，同一目标会留几重叠框。"""
    import numpy as np

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0)
        order = rest[iou <= iou_thr]
    return keep


# ---------------------------------------------------------------- 模块
class Module(BaseModule):
    name = "vision"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.shots_dir = Path(self.config.shots_dir)
        self.faces_dir = Path(self.config.state_dir) / FACE_DIR
        self.models_dir = Path(self.config.models_dir) / MODEL_SUBDIR
        self._detector: Detector | None = None

        self._following = False
        self._follow_why = ""
        self._follow_target: str | None = None
        self._task: asyncio.Task | None = None
        self._last_shot: str | None = None
        self._cam_err = ""
        self._store = self.load_store() or {}
        self._store.setdefault("owners", {})
        # 依赖探测只做一次：state() 每 100 ms 调一次，每次试 import 等于给状态流加税
        self._has_insight = _have("insightface")

    # ---------------- 状态 ----------------
    def state(self) -> dict:
        owners = self._store.get("owners") or {}
        det = self._detector_path()
        return {"vision": {
            # camera 先乐观报 online：HAL 有拍照能力就说明有摄像头，
            # 真拍不出图时 _capture 会把原因写进 camera_reason 并翻成 offline
            "camera": "offline" if self._cam_err else "online",
            "camera_reason": self._cam_err,
            "following": self._following,
            "follow_reason": self._follow_why,
            "follow_target": self._follow_target,
            "owner": self._current_owner(),
            "owners": sorted(owners),
            "last_shot": self._last_shot,
            "shots": self._count_shots(),
            "model": "vision-llm" if self._ai_ready() else "unset",
            "face_engine": "insightface" if _FACE.ready() else ("idle" if self._has_insight else "missing"),
            "detector": str(det.name) if det else None,
            "detector_ready": bool(det),
            "weak": any(bool(v.get("weak")) for v in owners.values()) if owners else False,
        }}

    def _current_owner(self) -> str | None:
        owners = self._store.get("owners") or {}
        if not owners:
            return None
        latest = max(owners.items(), key=lambda kv: float(kv[1].get("ts") or 0))
        return latest[0]

    def _ai_ready(self) -> bool:
        try:
            return "ai" in (self.ctx.hub.modules or {})
        except Exception:
            return False

    def _count_shots(self) -> int:
        try:
            return sum(1 for _ in self.shots_dir.glob("*.png"))
        except OSError:
            return 0

    def _detector_path(self) -> Path | None:
        try:
            if not self.models_dir.exists():
                return None
            preferred = self.models_dir / MODEL_FILE
            if preferred.exists():
                return preferred
            for p in sorted(self.models_dir.glob("*.onnx")):
                return p
        except OSError:
            return None
        return None

    def _ensure_detector(self) -> Detector | None:
        if self._detector is not None and self._detector.ready:
            return self._detector
        path = self._detector_path()
        if path is None:
            return None
        det = Detector(path)
        if not det.load():
            self.log.warning("检测模型加载失败：%s", det.err)
            self._detector = det
            return None
        self._detector = det
        self.log.info("检测模型已启用：%s", path.name)
        return det

    # ---------------- 生命周期 ----------------
    async def start(self) -> None:
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self._prune()

    async def stop(self) -> None:
        self._following = False
        await super().stop()

    # ---------------- 命令 ----------------
    async def cmd(self, op: str, args: dict) -> dict:
        args = args or {}
        if op == "op":
            op = str(args.get("op", "") or "status")
        op = op.strip().lower()

        if op == "snapshot":
            return await self._snapshot()
        if op == "enroll":
            return await self._enroll(args)
        if op == "follow":
            return await self._follow(args)
        if op == "ask":
            return await self._ask(args)
        if op == "detect":
            return await self._detect(args)
        if op == "shots":
            return {"ok": True, "items": self._shot_list(args)}
        if op in ("status", ""):
            return {"ok": True, "vision": self.state()["vision"]}
        raise ZeluxError(f"vision 不支持 {op}（可用：snapshot/enroll/follow/ask/detect/shots）",
                         "unsupported")

    # ---------------- 拍照 ----------------
    async def _capture(self, prefix: str = "shot") -> tuple[str, Path]:
        ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        name = f"{prefix}-{ts}.png"
        path = self.shots_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ok = await self.hal.camera_snapshot(str(path))
        except Exception as exc:
            self._cam_err = f"拍照失败：{exc}"
            raise ZeluxError(f"{self._cam_err}（检查 CSI 排线，或换 usb 摄像头）", "hardware") from None
        if not ok or not path.exists():
            self._cam_err = "摄像头没出图：检查 CSI 排线插到底、没有别的进程占着 /dev/video0"
            raise ZeluxError(self._cam_err, "hardware")
        self._cam_err = ""
        self._last_shot = name
        self._prune()
        return name, path

    async def _snapshot(self) -> dict:
        name, path = await self._capture("shot")
        size = path.stat().st_size if path.exists() else 0
        self.log_event(f"拍了一张：{name}")
        return {"ok": True, "name": name, "url": f"/api/shots/{name}",
                "bytes": size, "ts": time.time(), "shots": self._count_shots()}

    def _prune(self) -> int:
        """只留最近 50 张。按 mtime 排序，名字里的时间戳不可靠（可能被拷来拷去）。"""
        removed = 0
        try:
            files = sorted(self.shots_dir.glob("*.png"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return 0
        for p in files[MAX_SHOTS:]:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def _shot_list(self, args: dict) -> list:
        limit = max(1, min(MAX_SHOTS, int(args.get("n", 20) or 20)))
        try:
            files = sorted(self.shots_dir.glob("*.png"),
                           key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        except OSError:
            return []
        return [{"name": p.name, "url": f"/api/shots/{p.name}",
                 "bytes": p.stat().st_size, "ts": p.stat().st_mtime} for p in files]

    # ---------------- 录主人 ----------------
    async def _enroll(self, args: dict) -> dict:
        name = _safe_name(args.get("name"))
        _shot_name, path = await self._capture("enroll")

        face = await asyncio.to_thread(_FACE.embed, str(path))
        weak = False
        detail = ""
        emb_path = self.faces_dir / f"{name}.npy"

        if face.get("ok"):
            await asyncio.to_thread(self._save_npy, emb_path, face["embedding"])
            (self.faces_dir / f"{name}.weak.npy").unlink(missing_ok=True)
            detail = (f"人脸 embedding 已存（{face['embedding'].shape[0]} 维，"
                      f"检测置信度 {face['score']}）")
        else:
            weak = True
            feats = await asyncio.to_thread(_weak_feature, str(path))
            if feats is not None:
                await asyncio.to_thread(self._save_npy, self.faces_dir / f"{name}.weak.npy",
                                        feats["embedding"])
                emb_path.unlink(missing_ok=True)
                detail = f"用弱特征替代（{face.get('why', '')}）"
            else:
                # 连 numpy/PIL 都没有：只能记名字。这比"假装录入了"诚实，也让页面能提示装依赖
                detail = ("只记了名字，没算出任何特征："
                          "pip install numpy pillow 或 insightface（见 hint）")

        owners = self._store.setdefault("owners", {})
        owners[name] = {"ts": time.time(), "weak": weak, "shot": _shot_name,
                        "has_feature": (self.faces_dir / f"{name}.weak.npy").exists()
                        or emb_path.exists(),
                        "detail": detail}
        self._save()
        self.log_event(f"录入主人「{name}」" + ("（弱识别）" if weak else "（人脸特征）"))

        out = {"ok": True, "name": name, "weak": weak, "detail": detail,
               "shot": _shot_name, "url": f"/api/shots/{_shot_name}",
               "has_feature": owners[name]["has_feature"]}
        if weak:
            out["warning"] = WEAK_NOTE
            out["hint"] = FACE_HINT
        if not face.get("ok"):
            out["face_why"] = face.get("why", "")
        return out

    @staticmethod
    def _save_npy(path: Path, arr) -> None:
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(arr, dtype="float32"))

    def _save(self) -> None:
        try:
            self.save_store(self._store)
        except OSError as exc:
            self.log.warning("主人名单没存下来：%s", exc)

    # ---------------- 跟随 ----------------
    async def _follow(self, args: dict) -> dict:
        want = args.get("on")
        if want is None:
            want = bool(args.get("follow", True))
        if not want:
            return await self._follow_stop("手动停止")

        if not self._ensure_detector():
            det = self._detector_path()
            if det is None:
                raise ZeluxError(
                    f"跟随需要检测模型：{DETECT_ERR}", "unsupported")
            raise ZeluxError(f"检测模型加载失败：{self._detector.err if self._detector else '未知'}",
                             "unsupported")
        if self._following:
            return {"ok": True, "following": True, "again": True,
                    "follow_target": self._follow_target}

        self._following = True
        self._follow_why = ""
        self._follow_target = str(args.get("label") or "person")
        self._task = asyncio.create_task(self._follow_loop(), name="vision-follow")
        self.log_event(f"视觉跟随开始（跟 {self._follow_target}，"
                       f"死区 ±{int(DEAD_ZONE * 100)}%，0.5 s 丢失自动停）")
        return {"ok": True, "following": True, "follow_target": self._follow_target,
                "note": "只跟画面里最大的那个人；没有 re-ID，换人不会认"}

    async def _follow_stop(self, why: str) -> dict:
        was = self._following
        self._following = False
        task, self._task = self._task, None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._follow_why = why
        if was:
            await self._halt()
            self.log_event(f"视觉跟随结束：{why}")
        return {"ok": True, "following": False, "reason": why}

    async def _halt(self) -> None:
        """跟随的"停"只是零速；绝不碰舵机扭矩（那是 relax 的事）。"""
        try:
            await self.hal.move(0.0, 0.0, 0.0)
        except Exception as exc:
            self.log.warning("跟随停车失败：%s", exc)

    def _tmp_frame(self) -> Path:
        d = Path(self.config.state_dir) / "tmp"
        d.mkdir(parents=True, exist_ok=True)
        return d / "follow.png"

    @staticmethod
    def _frame_size(path: Path) -> tuple[int, int]:
        """画面尺寸：只读文件头，几十微秒，可以放在协程里直接调。"""
        try:
            from PIL import Image

            with Image.open(path) as im:
                return int(im.size[0]), int(im.size[1])
        except Exception:
            pass
        try:
            import cv2

            img = cv2.imread(str(path))
            if img is not None:
                return int(img.shape[1]), int(img.shape[0])
        except Exception:
            pass
        return 640, 480

    async def _follow_loop(self) -> None:
        period = 1.0 / FOLLOW_HZ
        lost_since: float | None = None
        last_det = 0.0
        misses = 0
        det = self._detector
        while self._following:
            if det is None:
                self._following = False
                self._follow_why = "检测模型被卸载了，跟随结束"
                break
            t0 = time.monotonic()
            estop = getattr(self.ctx.hub, "estop", None) or {}
            if estop.get("state") == "confirmed":
                # 被打断就地退出：急停状态下再发速度就是和操作者抢方向盘
                self._following = False
                self._follow_why = "急停已卸力，跟随中止（没有下发任何速度）"
                self.log.warning("%s", self._follow_why)
                break

            path = self._tmp_frame()
            try:
                ok = await self.hal.camera_snapshot(str(path))
            except Exception as exc:
                ok = False
                self._cam_err = f"拍照失败：{exc}"
            if not ok:
                misses += 1
                self._cam_err = self._cam_err or "跟随取帧失败：摄像头没出图"
                if misses >= 10:
                    self._following = False
                    self._follow_why = "连续 10 帧拿不到画面，跟随已停"
                    break
            else:
                self._cam_err = ""
                misses = 0
                boxes = await asyncio.to_thread(
                    det.detect_file, str(path), self._follow_target, DET_CONF)
                if boxes:
                    last_det = time.time()
                    lost_since = None
                    best = max(boxes, key=lambda b: b["box"][2] * b["box"][3])
                    x, _y, bw, bh = best["box"]
                    fw, fh = self._frame_size(path)
                    cmd = follow_control((x + bw / 2) / max(1, fw), bh / max(1, fh))
                    try:
                        await self.hal.move(cmd["vx"], 0.0, cmd["wz"])
                    except Exception as exc:
                        self.log.warning("跟随时下发速度失败：%s", exc)
                    self._follow_why = (f"目标 {best['score']} · "
                                        f"偏差 {int(cmd['err'] * 100)}% · 距离档 {cmd['zone']}")
                else:
                    lost_since = lost_since or time.time()
                    gap = time.time() - lost_since
                    if gap >= LOST_STOP_S:
                        await self._halt()     # 0.5 秒没看到人就停，绝不"按记忆往前冲"
                        self._follow_why = f"目标丢失 {gap:.1f} s，已停车"
                    if gap >= LOST_GIVEUP_S or time.time() - last_det >= LOST_GIVEUP_S:
                        self._following = False
                        self._follow_why = "丢失超过 3 秒，跟随结束"
                        await self._halt()
                        break
            await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))

    # ---------------- 识图问答 ----------------
    async def _ask(self, args: dict) -> dict:
        question = str(args.get("question") or "").strip()
        if not question:
            raise ZeluxError("要问什么？比如「这道题怎么算」", "denied")

        # 允许复用已经拍好的照片（页面里点"就这张"）
        name = str(args.get("image") or "").strip()
        if name:
            if "/" in name or ".." in name:
                raise ZeluxError("照片名字不合法", "denied")
            path = self.shots_dir / name
            if not path.exists():
                raise ZeluxError(f"没有这张照片：{name}", "denied")
            self._last_shot = name
        else:
            name, path = await self._capture("ask")

        # ai 模块的 vision_qa(question, image_path) 自己读盘（它要压到 8 MB 以内），
        # 所以这里给路径为主；image_b64 只作为备用，省得以后换接口还要改这边。
        raw = await asyncio.to_thread(path.read_bytes)
        payload = {"question": question, "image_path": str(path), "shot": name,
                   "image_b64": base64.b64encode(raw).decode("ascii"),
                   "mime": "image/png", "bytes": len(raw)}

        ai = (self.ctx.hub.modules or {}).get("ai")
        if ai is None or not hasattr(ai, "cmd"):
            return {"ok": False, "code": "unsupported", "shot": name,
                    "url": f"/api/shots/{name}",
                    "error": ("AI 模块还没加载：先在设置里配好大模型 Key（ai 模块起来后就能识图问答）")}
        try:
            result = await ai.cmd("vision_qa", payload)
        except ZeluxError as exc:
            if exc.code != "unsupported":
                raise
            return {"ok": False, "code": "unsupported", "shot": name,
                    "url": f"/api/shots/{name}",
                    "error": ("AI 模块还没实现识图问答（vision_qa）。"
                              "在 modules/ai.py 的 cmd 里加一个 vision_qa 分支："
                              "把 image_b64 转成 image_url 发给多模态模型即可"),
                    "hint": "临时办法：拿这张照片去大模型 App 里问，答案一样"}
        except Exception as exc:
            return {"ok": False, "code": "failed", "shot": name,
                    "error": f"识图问答失败：{exc}"}

        out = {"ok": True, "shot": name, "url": f"/api/shots/{name}"}
        if isinstance(result, dict):
            out.update(result)
            out.setdefault("ok", True)
            out["shot"] = name
            # ai 模块回的是 {reply, ms}，前端统一读 answer，这里补齐
            if "answer" not in out and out.get("reply"):
                out["answer"] = out["reply"]
        else:
            out["answer"] = str(result)
        return out

    # ---------------- 物体检测 ----------------
    async def _detect(self, args: dict) -> dict:
        det = self._ensure_detector()
        if det is None:
            path = self._detector_path()
            if path is None:
                # 不编框：没模型就说没模型，并给出放模型的路径
                return {"ok": False, "error": DETECT_ERR,
                        "model_dir": str(self.models_dir)}
            why = (self._detector.err if self._detector is not None else "") or "未知"
            return {"ok": False, "error": f"模型加载失败：{why}",
                    "model_dir": str(self.models_dir),
                    "hint": "pip install onnxruntime numpy"}

        name = str(args.get("image") or "").strip()
        if name:
            if "/" in name or ".." in name:
                raise ZeluxError("照片名字不合法", "denied")
            path = self.shots_dir / name
            if not path.exists():
                raise ZeluxError(f"没有这张照片：{name}", "denied")
        else:
            name, path = await self._capture("detect")

        label = str(args.get("label") or "").strip() or None
        boxes = await asyncio.to_thread(det.detect_file, str(path), label, DET_CONF)
        return {"ok": True, "shot": name, "url": f"/api/shots/{name}",
                "label": label, "model": det.path.name, "count": len(boxes),
                "boxes": boxes,
                "note": "" if boxes else "没检到目标（换角度/靠近一点/降低阈值再试）"}
