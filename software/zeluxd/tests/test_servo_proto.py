"""纯逻辑自测：不需要舵机、串口、IMU，本机（含 Windows）就能跑。

跑法：
    cd software && python -m zeluxd.tests.test_servo_proto

覆盖什么：
* 帧格式与校验和（含 SYNC_WRITE 的长度/地址/长度字段，算错一位真机就是全瞎）；
* 半双工回显：假总线默认回显，协议层必须把回显滤掉还能正常收发；
* 速度/负载/0 位的编码（16 位补码、bit15 方向位）；
* 姿势弧度 → 刻度；
* IMU 的滤波数学与五步向导判定（含"往前倾了 6°，不到 12°"这类人话原因）。
"""
from __future__ import annotations

import logging
import math
import struct
import tempfile
import types
from pathlib import Path

from zeluxd.hal.base import JOINT_IDS
from zeluxd.hal.radxa_imu import (accel_to_roll_pitch, complementary, spread,
                                  wrap180, yaw_step)
from zeluxd.hal.radxa_servo import (MAX_SPEED, FakeLink, ServoProto, checksum,
                                    decode_load, decode_speed, encode_reg,
                                    encode_speed, frame, rad_to_tick,
                                    sync_write_frame, tick_to_rad)
from zeluxd.modules.imu import Module as ImuModule
from zeluxd.modules.servos import POSES, pose_ticks


# ---------------------------------------------------------------- 帧格式
def test_checksum_property():
    body = bytes((10, 4, 2, 56, 8))
    assert (sum(body) + checksum(body)) & 0xFF == 0xFF


def test_read_frame_bytes():
    f = frame(10, 2, bytes((56, 8)))
    # FF FF id len instr addr n chk
    assert f[:5] == bytes((0xFF, 0xFF, 10, 4, 2))
    assert f[5:7] == bytes((56, 8))
    assert f[7] == checksum(bytes((10, 4, 2, 56, 8)))
    assert len(f) == 8


def test_write_frame_bytes():
    f = frame(20, 3, bytes((40, 1)))
    assert f[2:5] == bytes((20, 4, 3))       # len = 参数 2 + instr/chk 2
    assert len(f) == 4 + f[3]


def test_sync_write_layout():
    items = [(sid, struct.pack("<H", 2048) + encode_speed(900)) for sid in JOINT_IDS]
    f = sync_write_frame(41, 4, items)
    assert f[:5] == bytes((0xFF, 0xFF, 0xFE, 81, 0x83)), "SYNC_WRITE 头/长度不对"
    assert f[5:9] == struct.pack("<HH", 41, 4), "必须是 2 字节地址 + 2 字节长度"
    assert len(f) == 4 + f[3] == 85
    assert [f[9 + 5 * i] for i in range(15)] == JOINT_IDS
    assert f[-1] == checksum(f[2:-1])


def test_sync_write_rejects_bad_width():
    try:
        sync_write_frame(41, 4, [(10, b"\x00\x01")])
    except ValueError:
        return
    raise AssertionError("宽度对不上应该直接报错，而不是发一帧废数据")


# ---------------------------------------------------------------- 编解码
def test_speed_field():
    assert encode_speed(0) == b"\x00\x00"
    assert encode_speed(900) == struct.pack("<H", 900)
    assert encode_speed(-300) == struct.pack("<H", 0x8000 | 300)
    assert encode_speed(99999) == struct.pack("<H", MAX_SPEED)
    assert decode_speed(900) == 900
    assert decode_speed(0x8000 | 300) == -300


def test_load_decode():
    assert decode_load(0) == 0.0
    assert decode_load(120) == 12.0
    assert decode_load(0x0400 | 120) == -12.0


def test_offset_twos_complement():
    assert encode_reg(-100, 2) == struct.pack("<H", 0xFF9C)
    assert encode_reg(31, 2) == struct.pack("<H", 31)
    assert encode_reg(0x1FF, 1) == b"\xff"      # 1 字节寄存器要截断，不能溢出两字节


def test_tick_math():
    assert rad_to_tick(0.0) == 2048
    assert abs(rad_to_tick(1.0) - 2700) <= 1
    assert rad_to_tick(99.0) == 4095            # 越界夹住
    assert rad_to_tick(-99.0) == 0
    assert abs(tick_to_rad(2048)) < 1e-9


# ---------------------------------------------------------------- 假总线往返
def test_fake_bus_roundtrip():
    link = FakeLink(echo=True)                  # 回显打开：模拟真实 TTL 转接板
    proto = ServoProto(link, retries=2, timeout=0.03)
    assert proto.ping(10)
    rec = proto.read_servo(10)
    assert rec["pos"] == 2048
    assert rec["volt"] == 7.8 and rec["temp"] == 32
    assert rec["torque"] is False

    # 写目标位置：现值不动，等"走到"才动
    assert proto.write(10, 41, struct.pack("<H", 3000))
    assert proto.read_servo(10)["pos"] == 2048
    link.settle()
    assert proto.read_servo(10)["pos"] == 3000


def test_fake_bus_sync_write_and_torque():
    link = FakeLink(echo=True)
    proto = ServoProto(link, retries=1, timeout=0.03)
    proto.sync_write(41, 4, [(10, struct.pack("<H", 2500) + encode_speed(-200))])
    link.settle()
    rec = proto.read_servo(10)
    assert rec["pos"] == 2500
    assert link.regs[10][43] == 0x8000 | 200, "负速度要落在 bit15 上"

    proto.sync_write(40, 1, [(sid, b"\x01") for sid in JOINT_IDS])
    assert proto.read_servo(10)["torque"] is True
    proto.sync_write(40, 1, [(sid, b"\x00") for sid in JOINT_IDS])
    assert proto.read_servo(10)["torque"] is False


def test_fake_bus_offset_and_missing():
    link = FakeLink(echo=True)
    proto = ServoProto(link, retries=1, timeout=0.02)
    proto.write(10, 31, encode_reg(-100, 2))
    assert proto.read(10, 31, 2) == struct.pack("<H", 0xFF9C)
    assert not proto.ping(99), "不在总线上的 ID 必须判为离线"


def test_echo_is_filtered():
    """回显开启时读到的必须还是应答，不是自己发出去的那一帧。"""
    for n in (1, 2, 8):
        link = FakeLink(echo=True)
        proto = ServoProto(link, retries=1, timeout=0.03)
        data = proto.read(10, 56, n)
        assert data is not None and len(data) == n, f"读 {n} 字节被回显带偏了"


def test_no_response_when_mute():
    link = FakeLink(echo=False)
    link.regs[10][8] = 0                        # 应答级别 0 = 不回话
    proto = ServoProto(link, retries=1, timeout=0.01)
    assert not proto.ping(10)


# ---------------------------------------------------------------- 姿势
def test_poses():
    assert len(POSES) == 4
    assert pose_ticks("zero") == [2048] * 15
    stand = pose_ticks("stand")
    assert len(stand) == 15
    assert stand[1] == 1991 and stand[2] == 1749 and stand[4] == 2343
    sit = pose_ticks("sit")
    assert sit[3] == 2731 and sit[6] == 3091
    fold = pose_ticks("fold")
    assert fold[2] == 3071 and fold[12] == 1025
    for name in POSES:
        assert all(0 <= t <= 4095 for t in pose_ticks(name))
    try:
        pose_ticks("dance")
    except Exception as exc:
        assert getattr(exc, "code", "") == "unsupported"
    else:
        raise AssertionError("不认识的姿势应该抛 unsupported")


# ---------------------------------------------------------------- IMU 数学
def test_attitude_math():
    g = 9.80665
    r, p = accel_to_roll_pitch(0.0, 0.0, g)
    assert abs(r) < 1e-6 and abs(p) < 1e-6
    # 前倾 30°：ax = -g·sin30，az = g·cos30
    _r, p = accel_to_roll_pitch(-g * 0.5, 0.0, g * math.cos(math.radians(30)))
    assert abs(p - 30.0) < 0.1
    # 右侧抬高 30°：ay = g·sin30
    r, _p = accel_to_roll_pitch(0.0, g * 0.5, g * math.cos(math.radians(30)))
    assert abs(r - 30.0) < 0.1


def test_complementary_and_yaw():
    r, p = complementary(0.0, 0.0, 10.0, -10.0, 0.0, 0.0, 0.01)
    assert 0.0 < r < 10.0 and -10.0 < p < 0.0      # 缓慢被加速度计拉住
    for _ in range(500):
        r, p = complementary(r, p, 10.0, -10.0, 0.0, 0.0, 0.01)
    assert abs(r - 10.0) < 0.1 and abs(p + 10.0) < 0.1
    assert wrap180(190.0) == -170.0
    assert wrap180(-190.0) == 170.0
    assert abs(yaw_step(179.0, 20.0, 1.0) + 161.0) < 1e-9
    assert spread([1.0, 3.0, 2.0]) == 2.0


# ---------------------------------------------------------------- 向导
def _imu_module() -> ImuModule:
    hub = types.SimpleNamespace(log_line=lambda *a, **k: None,
                                emit_event=lambda *a, **k: None)
    cfg = types.SimpleNamespace(state_dir=Path(tempfile.mkdtemp(prefix="zelux-test-")))
    ctx = types.SimpleNamespace(config=cfg, log=logging.getLogger("test"), state={},
                                hub=hub, hal=None)
    mod = ImuModule(ctx)
    mod.log_event = lambda *a, **k: None        # 测试里不往 hub 推
    return mod


def _frames(n: int, roll: float, pitch: float, jitter: float = 0.2):
    return [(roll + (i % 3 - 1) * jitter * 0.3, pitch + (i % 2) * jitter * 0.3)
            for i in range(n)]


def test_wizard_tilt_message():
    mod = _imu_module()
    mod._done = {"still": _frames(20, 2.0, -3.0)}
    mod._buf = _frames(6, 2.0, 3.0)             # 相对静止段只往前 6°
    ok, why = mod._check("forward")
    assert not ok and "往前倾了 6" in why and "12" in why, why
    # 反方向要直说
    mod._buf = _frames(6, 2.0, -20.0)
    ok, why = mod._check("forward")
    assert not ok and "方向反了" in why, why


def test_wizard_still_message():
    mod = _imu_module()
    mod._buf = _frames(15, 0.0, 0.0, jitter=8.0)   # 抖得厉害
    ok, why = mod._check("still")
    assert not ok and "还在动" in why, why
    mod._buf = _frames(15, 0.0, 0.0, jitter=0.2)
    assert mod._check("still")[0]


def test_wizard_solve_normal():
    mod = _imu_module()
    mod._done = {
        "still": _frames(20, 2.0, -3.0),
        "forward": _frames(6, 2.0, 12.0),       # pitch 抬 15°
        "back": _frames(6, 2.0, -18.0),
        "left": _frames(6, -13.0, -3.0),        # roll 往左 -15°
        "right": _frames(6, 17.0, -3.0),
    }
    res = mod._solve()
    assert res["ok"], res
    assert res["install"] == {"roll_sign": 1, "pitch_sign": 1}
    assert abs(res["offset"]["roll"] + 2.0) < 0.6
    assert abs(res["offset"]["pitch"] - 3.0) < 0.6
    assert max(abs(res["after"][0]), abs(res["after"][1])) <= 1.0
    assert res["delta_deg"] > 0
    assert all(c["ok"] for c in res["checks"]), res["checks"]


def test_wizard_solve_reversed_mount():
    """模块装反了（俯仰轴反号）：解算要自己发现并纠正。"""
    mod = _imu_module()
    mod._done = {
        "still": _frames(20, 0.0, 0.0),
        "forward": _frames(6, 0.0, -15.0),      # 往前倾时 pitch 反而变负
        "back": _frames(6, 0.0, 15.0),
        "left": _frames(6, 15.0, 0.0),          # 往左倾时 roll 变正
        "right": _frames(6, -15.0, 0.0),
    }
    res = mod._solve()
    assert res["install"] == {"roll_sign": -1, "pitch_sign": -1}
    assert res["ok"], res
    for c in res["checks"]:
        assert c["ok"], c


def test_wizard_solve_needs_all_steps():
    mod = _imu_module()
    mod._done = {"still": _frames(20, 0.0, 0.0)}
    try:
        mod._solve()
    except Exception as exc:
        assert getattr(exc, "code", "") == "stale", exc
        assert "往前倾" in str(exc)
    else:
        raise AssertionError("缺步骤必须拒绝解算")


# ---------------------------------------------------------------- 跑
def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - 自测要把意外也报出来
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过" + ("" if not failed else f"，{failed} 个失败"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
