"""Guided full-frame alignment from robotd-owned gravity samples.

The correction is a proper 3-D rotation, not independent roll/pitch signs or offsets.
All samples arrive through robotd's state subscription; this module never reads hardware.
"""
from __future__ import annotations

import math


STEPS = (
    ("still", "机体立正", "将机器人扶稳在正常站立姿态，保持静止后采集"),
    ("forward", "前倾", "沿机器人前方缓慢倾斜，保持后采集"),
    ("back", "后倾", "沿机器人后方缓慢倾斜，保持后采集"),
    ("left", "左倾", "沿机器人左侧缓慢倾斜，保持后采集"),
    ("right", "右倾", "沿机器人右侧缓慢倾斜，保持后采集"),
)
MIN_TILT_DEG = 12.0


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _unit(v):
    norm = math.sqrt(_dot(v, v))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("重力变化不足或方向不可解")
    return [x / norm for x in v]


def _angle(a, b):
    a, b = _unit(a), _unit(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(a, b)))))


def _matrix_to_quaternion(m):
    # Scalar-first quaternion for a right-handed orthonormal 3x3 matrix.
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (m[2][1] - m[1][2]) / s,
             (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(max(0.0, 1.0 + m[0][0] - m[1][1] - m[2][2])) * 2.0
        q = [(m[2][1] - m[1][2]) / s, 0.25 * s,
             (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    elif m[1][1] > m[2][2]:
        s = math.sqrt(max(0.0, 1.0 + m[1][1] - m[0][0] - m[2][2])) * 2.0
        q = [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s,
             0.25 * s, (m[1][2] + m[2][1]) / s]
    else:
        s = math.sqrt(max(0.0, 1.0 + m[2][2] - m[0][0] - m[1][1])) * 2.0
        q = [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s,
             (m[1][2] + m[2][1]) / s, 0.25 * s]
    norm = math.sqrt(_dot(q, q))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("标定旋转无效")
    q = [x / norm for x in q]
    return q if q[0] >= 0 else [-x for x in q]


def _normalise_quaternion(q):
    if not isinstance(q, (list, tuple)) or len(q) != 4:
        raise ValueError("当前机体对齐修正无效，请刷新配置后重试")
    values = [float(x) for x in q]
    norm = math.sqrt(_dot(values, values))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("当前机体对齐修正无效，请刷新配置后重试")
    values = [x / norm for x in values]
    return values if values[0] >= 0 else [-x for x in values]


def _multiply_quaternions(left, right):
    """Compose scalar-first rotations in application order: left * right."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return _normalise_quaternion([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])


def solve_correction(samples, existing_correction=None):
    """Return the full correction from the model's nominal frame to the body frame.

    `samples` are robotd's published projected-gravity vectors, already expressed through
    the currently active model and calibration transforms. The solved rotation therefore
    maps the current output frame to the desired body frame; compose it with the existing
    correction so recalibration does not discard or double-apply the prior transform.
    """
    upright = _unit(samples["still"])
    axes = {}
    for key in ("forward", "back", "left", "right"):
        gravity = _unit(samples[key])
        angle = _angle(upright, gravity)
        if angle < MIN_TILT_DEG or angle > 65.0:
            raise ValueError(f"{key} 倾斜幅度 {angle:.1f}° 不在 {MIN_TILT_DEG:.0f}°–65° 范围内")
        axes[key] = _unit(_cross(gravity, upright))

    # Expected right-handed body frame: forward pitch is +Y; left roll is -X.
    measured_y = _unit([axes["forward"][i] - axes["back"][i] for i in range(3)])
    measured_neg_x = _unit([axes["left"][i] - axes["right"][i] for i in range(3)])
    if _dot(axes["forward"], axes["back"]) > -0.75:
        raise ValueError("前倾与后倾没有形成相反方向，请按页面提示重新采集")
    if _dot(axes["left"], axes["right"]) > -0.75:
        raise ValueError("左倾与右倾没有形成相反方向，请按页面提示重新采集")
    orthogonality = abs(_dot(measured_y, measured_neg_x))
    if orthogonality > 0.35:
        raise ValueError("前后倾与左右倾的旋转轴不够正交，请保持机体沿单一方向倾斜后重采")

    # Orthogonalise the two measured body directions and complete the frame with a cross product.
    projection = _dot(measured_neg_x, measured_y)
    measured_neg_x = _unit([measured_neg_x[i] - projection * measured_y[i] for i in range(3)])
    measured_z = _unit(_cross(measured_y, measured_neg_x))
    source = (measured_y, measured_neg_x, measured_z)
    target = ([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    matrix = [[sum(target[k][r] * source[k][c] for k in range(3))
               for c in range(3)] for r in range(3)]
    delta_correction = _matrix_to_quaternion(matrix)
    existing = _normalise_quaternion(existing_correction or [1.0, 0.0, 0.0, 0.0])
    correction = _multiply_quaternions(delta_correction, existing)

    checks = []
    for key, expected in (("forward", target[0]), ("back", [-x for x in target[0]]),
                          ("left", target[1]), ("right", [-x for x in target[1]])):
        axis = axes[key]
        mapped = [sum(matrix[r][c] * axis[c] for c in range(3)) for r in range(3)]
        error = _angle(mapped, expected)
        checks.append({"step": key, "error_deg": round(error, 2), "ok": error <= 25.0})
    if not all(check["ok"] for check in checks):
        raise ValueError("四个方向的采样不能由同一个三维旋转解释，请按顺序重新采集")
    return {"correction_wxyz": [round(x, 9) for x in correction],
            "delta_correction_wxyz": [round(x, 9) for x in delta_correction],
            "checks": checks,
            "max_axis_error_deg": max(item["error_deg"] for item in checks),
            "method": "guided-gravity-axes-v1"}


class MountCalibration:
    def __init__(self):
        self.running = False
        self.step = 0
        self.samples = {}
        self.result = None
        self.error = ""
        self.existing_correction = None

    def start(self, existing_correction=None):
        self.running = True
        self.step = 0
        self.samples = {}
        self.result = None
        self.error = ""
        self.existing_correction = existing_correction
        return self.view()

    def capture(self, result):
        if not self.running or self.step >= len(STEPS):
            raise ValueError("请先开始机体对齐标定")
        key = STEPS[self.step][0]
        gravity = _unit(result["gravity"])
        if key == "still":
            self.samples[key] = gravity
        else:
            upright = self.samples.get("still")
            if upright is None:
                raise ValueError("请先采集机体立正姿态")
            angle = _angle(upright, gravity)
            if angle < MIN_TILT_DEG:
                raise ValueError(f"当前倾斜 {angle:.1f}°，至少需要 {MIN_TILT_DEG:.0f}°")
            if angle > 65.0:
                raise ValueError("倾斜超过 65°，请回到较小且稳定的角度重新采集")
            self.samples[key] = gravity
        self.step += 1
        self.error = ""
        if self.step == len(STEPS):
            self.running = False
        return self.view()

    def solve(self):
        if self.step != len(STEPS):
            raise ValueError("请先完成立正、前后倾和左右倾五个姿态采集")
        self.result = solve_correction(self.samples, self.existing_correction)
        return self.result

    def clear(self):
        self.__init__()
        return self.view()

    def view(self):
        steps = []
        for index, (key, label, hint) in enumerate(STEPS):
            steps.append({"key": key, "label": label, "hint": hint,
                          "done": key in self.samples,
                          "active": self.running and index == self.step})
        return {"running": self.running, "step": self.step,
                "step_key": STEPS[self.step][0] if self.step < len(STEPS) else None,
                "steps": steps, "ready": self.step == len(STEPS),
                "result": self.result, "why": self.error}
