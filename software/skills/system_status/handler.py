"""自检问答：电量、温度、风扇、前方障碍。

数据全部来自 `ctx.state`（HAL 已经缓存好的快照），**不碰硬件**，
所以这个问题 0 ms 就能答，也不会和舵机总线抢资源。
"""
from __future__ import annotations


def run(ctx, text, args) -> dict:
    st = ctx.state or {}
    bat = st.get("battery") or {}
    cpu = st.get("cpu") or {}
    fan = st.get("fan") or {}
    tof = st.get("tof") or {}

    ask_battery = any(k in text for k in ("电", "电池", "续航"))
    ask_temp = any(k in text for k in ("温度", "热", "烫", "风扇"))
    ask_dist = any(k in text for k in ("障碍", "前面", "离墙", "多远", "距离"))
    everything = not (ask_battery or ask_temp or ask_dist)

    parts: list[str] = []
    data: dict = {}

    if ask_battery or everything:
        pct = _num(bat.get("pct"))
        volt = _num(bat.get("v"))
        if pct is not None:
            data["battery_pct"] = pct
            data["battery_v"] = volt
            seg = f"电量 {int(pct)}%"
            if volt is not None:
                seg += f"（{volt:.2f} 伏）"
            if pct <= 15:
                seg += "，快没电了，赶紧充"
            parts.append(seg)

    if ask_temp or everything:
        temp = _num(cpu.get("temp"))
        if temp is not None:
            data["cpu_temp"] = temp
            seg = f"主板 {int(round(temp))} 度"
            pct_fan = _num(fan.get("pct"))
            if pct_fan is not None:
                seg += f"，风扇 {int(pct_fan)}%"
                data["fan_pct"] = pct_fan
            parts.append(seg)

    if ask_dist:
        near = _num(tof.get("nearest_mm"))
        if near:
            data["nearest_mm"] = near
            parts.append(f"前面最近 {int(near)} 毫米")
        elif tof.get("state") and tof.get("state") != "ready":
            parts.append("测距现在没数据")

    if everything:
        posture = _posture(st.get("posture"))
        if posture:
            data["posture"] = st.get("posture")
            parts.append(f"现在是{posture}")

    if not parts:
        return {"speech": "传感器这会儿没数据，可能还没接上。",
                "data": data, "display": {"kind": "status", "empty": True}}

    speech = "；".join(parts) + "。"
    return {"speech": speech, "data": data, "display": {"kind": "status", **data}}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _posture(v) -> str:
    return {"holding": "站着保持", "walking": "走着", "sitting": "坐着",
            "limp": "卸力趴着", "homing": "在回零位", "absent": "没上电",
            "servo_session": "在调舵机"}.get(str(v or ""), "")
