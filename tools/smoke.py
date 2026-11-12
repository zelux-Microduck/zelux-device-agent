"""本机联调脚本：连 WS、收状态、跑一组命令，验证协议。

    python tools/smoke.py [口令] [地址]
"""
import asyncio
import json
import sys
import urllib.request

BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
PW = sys.argv[1] if len(sys.argv) > 1 else "123456"


def login() -> str:
    req = urllib.request.Request(BASE + "/api/login", method="POST",
                                 data=json.dumps({"password": PW}).encode(),
                                 headers={"Content-Type": "application/json", "Origin": BASE})
    with urllib.request.urlopen(req, timeout=8) as resp:
        cookie = resp.headers.get("set-cookie", "").split(";")[0]
    print("登录 OK", cookie[:24], "...")
    return cookie


async def main() -> None:
    import websockets

    cookie = login()
    url = BASE.replace("http", "ws") + "/ws"
    async with websockets.connect(url, additional_headers={"Cookie": cookie}) as ws:
        hello = json.loads(await ws.recv())
        print("hello:", hello.get("t"), hello.get("version"), hello.get("caps", {}).get("modules"))
        print("protocol:", hello.get("protocol"), "drivers:", sorted((hello.get("capabilities") or {}).keys()))
        # 第二条不一定是 state（可能是日志事件），要等到 state 为止
        state = {}
        for _ in range(20):
            m = json.loads(await ws.recv())
            if m.get("t") == "state":
                state = m.get("data") or {}
                break
        print("state 顶层字段:", sorted(state.keys()))

        seq = 0

        async def cmd(name, args=None, timeout=15):
            nonlocal seq
            seq += 1
            await ws.send(json.dumps({"t": "cmd", "id": f"t{seq}", "name": name, "args": args or {}}))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout))
                if msg.get("t") == "result" and msg.get("id") == f"t{seq}":
                    return msg

        checks = [
            ("servos.bus", {"op": "take"}),
            ("servos.read", {"id": 10}),
            ("servos.pose", {"name": "stand"}),
            ("imu.sample", {}),
            ("tof.on", {"on": True}),
            ("tof.simulate", {"distances": [300] * 64}),
            ("fan.op", {"op": "auto"}),
            ("vision.snapshot", {}),
            ("ai.skills", {}),
            ("train.sessions", {}),
            ("sys.time", {}),
            ("ota.check", {}),
            ("relax", {}),
        ]
        # 这些在"没配"时本来就该失败：升级源要现场配，不算缺陷。
        expected_fail = {"ota.check": "没配升级源时必然失败"}
        ok = 0
        for name, args in checks:
            r = await cmd(name, args)
            soft = name in expected_fail and not r.get("ok")
            flag = "OK " if r.get("ok") else ("SKIP" if soft else "ERR")
            detail = "" if r.get("ok") else f"  ← {r.get('error')} ({r.get('code')})"
            print(f"{flag} {name:22s}{detail}")
            ok += 1 if (r.get("ok") or soft) else 0
        print(f"\n{ok}/{len(checks)} 条命令通过（SKIP = 预期内失败）")


asyncio.run(main())
