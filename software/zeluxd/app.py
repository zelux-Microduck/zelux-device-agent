"""HTTP + WebSocket 入口。

路由很少：静态文件、登录/会话/口令、诊断、日志，其余全走 /ws。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .auth import Auth
from .config import CONFIG
from .errors import ZeluxError
from .hub import Hub, system_info
from .protocol import (DRIVER_API_VERSION, MOTION_PROTOCOL_VERSION,
                       PROTOCOL_SUPPORTED, PROTOCOL_VERSION, negotiate)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
log = logging.getLogger("zeluxd.app")

auth = Auth(CONFIG)
hub = Hub(CONFIG)
COOKIE = "zelux_sid"


def _hello() -> dict:
    capabilities = hub.driver_registry.capabilities() if hub.driver_registry else {}
    return {
        "t": "hello",
        # v1 兼容字段
        "duck": hub.duck,
        "version": __version__,
        "caps": {"sim": CONFIG.mode == "sim", "modules": sorted(hub.modules)},
        # v1.1 结构化协商字段
        "device": {"id": hub.duck.get("label", "zelux-duck"),
                   "model": "microduck-sim" if CONFIG.mode == "sim" else "microduck-radxa-zero3w",
                   "hw_revision": None},
        "versions": {"agent": __version__, "protocol": PROTOCOL_VERSION,
                     "driver_api": DRIVER_API_VERSION,
                     "motion_protocol": MOTION_PROTOCOL_VERSION},
        "protocol": {"selected": PROTOCOL_VERSION, "supported": PROTOCOL_SUPPORTED},
        "capabilities": capabilities,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    await hub.boot()
    yield
    await hub.shutdown()


app = FastAPI(title="ZELUX Duck Studio", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)


# ------------------------------------------------------------------ 工具
def _same_origin(request: Request) -> bool:
    if CONFIG.dev:
        return True
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        return False
    host = request.headers.get("host", "")
    return origin.split("://")[-1].split("/")[0] == host


def _csrf_guard(request: Request) -> JSONResponse | None:
    token = request.cookies.get(COOKIE)
    if not auth.csrf_ok(token, request.headers.get("x-csrf")):
        return JSONResponse({"ok": False, "error": "会话过期了，请重新登录"}, status_code=403)
    return None


# ------------------------------------------------------------------ 会话
@app.get("/api/session")
async def api_session(request: Request):
    token = request.cookies.get(COOKIE)
    s = auth.session(token)
    if not s:
        return JSONResponse({"ok": False, "has_password": auth.has_password}, status_code=401)
    return {"ok": True, "csrf": s["csrf"], "duck": hub.duck, "version": __version__,
            "mode": hub.mode, "backend": CONFIG.mode,
            "protocol": {"version": PROTOCOL_VERSION, "supported": PROTOCOL_SUPPORTED}}


@app.post("/api/login")
async def api_login(request: Request):
    if not _same_origin(request):
        return JSONResponse({"ok": False, "error": "请求来源不对"}, status_code=403)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not auth.verify(str(body.get("password", ""))):
        return JSONResponse({"ok": False, "error": "口令不对"}, status_code=401)
    token, csrf = auth.new_session()
    resp = JSONResponse({"ok": True, "csrf": csrf, "duck": hub.duck, "version": __version__})
    resp.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=7 * 24 * 3600, path="/")
    hub.log_line(f"页面登录（{request.client.host if request.client else '?'}）", cat="zeluxd")
    return resp


@app.post("/api/logout")
async def api_logout(request: Request):
    auth.drop(request.cookies.get(COOKIE))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE, path="/")
    return resp


@app.post("/api/password")
async def api_password(request: Request):
    if err := _csrf_guard(request):
        return err
    body = await request.json()
    old = str(body.get("old", ""))
    new = str(body.get("new", ""))
    if auth.has_password and not auth.verify(old):
        return JSONResponse({"ok": False, "error": "当前口令不对"}, status_code=400)
    if len(new) < 4:
        return JSONResponse({"ok": False, "error": "新口令至少 4 位"}, status_code=400)
    auth.set_password(new)
    token, csrf = auth.new_session()
    resp = JSONResponse({"ok": True, "csrf": csrf})
    resp.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=7 * 24 * 3600, path="/")
    hub.log_line("口令已修改", cat="zeluxd")
    return resp


# ------------------------------------------------------------------ 诊断 / 日志
@app.get("/api/diag")
async def api_diag(request: Request):
    if not auth.session(request.cookies.get(COOKIE)):
        return JSONResponse({"ok": False}, status_code=401)
    return {"ok": True, "version": __version__, "mode": CONFIG.mode,
            "sys": system_info(), "hal": hub.hal.info() if hub.hal else {},
            "modules": sorted(hub.modules), "uptime_s": round(time.time() - hub.started_at),
            "clients": len(hub.clients),
            "state": hub.snapshot()}


@app.get("/api/logs")
async def api_logs(request: Request, n: int = 200):
    if not auth.session(request.cookies.get(COOKIE)):
        return JSONResponse({"ok": False}, status_code=401)
    items = list(hub.log_bus.items)[-n:]
    return {"ok": True, "items": items}


@app.get("/api/health")
async def api_health():
    return {"ok": True, "version": __version__, "mode": CONFIG.mode}


@app.get("/api/shots/{name}")
async def api_shot(request: Request, name: str):
    if not auth.session(request.cookies.get(COOKIE)):
        return JSONResponse({"ok": False}, status_code=401)
    if "/" in name or ".." in name:
        return JSONResponse({"ok": False, "error": "名字不合法"}, status_code=400)
    path = CONFIG.shots_dir / name
    if not path.exists():
        return JSONResponse({"ok": False, "error": "没有这张照片"}, status_code=404)
    return FileResponse(path)


# ------------------------------------------------------------------ WebSocket
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    token = ws.cookies.get(COOKIE)
    if not auth.session(token):
        await ws.close(code=4401)
        return
    if not CONFIG.dev:
        origin = ws.headers.get("origin", "")
        host = ws.headers.get("host", "")
        if origin and origin.split("://")[-1].split("/")[0] != host:
            await ws.close(code=4403)
            return
    await ws.accept()
    client = __import__("zeluxd.hub", fromlist=["Client"]).Client(ws, ws.client.host if ws.client else "?")
    hub.clients.add(client)
    hub.log_line(f"页面连上（{client.peer}，现在 {len(hub.clients)} 个）", cat="zeluxd")
    tasks: set[asyncio.Task] = set()
    try:
        await ws.send_json(_hello())
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            kind = msg.get("t")
            if kind == "client_hello":
                protocol = msg.get("protocol") or {}
                error = negotiate(protocol.get("supported"))
                if error:
                    await ws.send_json({"t": "error", "error": {
                        "code": error.code, "message": error.message,
                        "supported": error.supported,
                    }})
                    await ws.close(code=4406, reason="protocol incompatible")
                    return
                await ws.send_json(_hello())
            elif kind == "hb":
                client.visible = bool(msg.get("visible", True))
                client.route = str(msg.get("route", "/"))
                client.focus = msg.get("focus") or None
                client.last_hb = time.monotonic()
                await ws.send_json({"t": "hb"})
            elif kind == "cmd":
                task = asyncio.create_task(_run_cmd(client, msg))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.info("页面断了：%s", exc)
    finally:
        hub.clients.discard(client)
        hub.log_line(f"页面断开（{client.peer}，还剩 {len(hub.clients)} 个）", cat="zeluxd")
        for t in tasks:
            t.cancel()


async def _run_cmd(client, msg: dict) -> None:
    cid = msg.get("id")
    name = str(msg.get("name", ""))
    args = msg.get("args") or {}
    try:
        data = await asyncio.wait_for(hub.dispatch(name, args), timeout=float(msg.get("timeout", 30)))
        payload = {"t": "result", "id": cid, "ok": True, "data": data}
    except ZeluxError as exc:
        payload = {"t": "result", "id": cid, **exc.payload()}
    except asyncio.TimeoutError:
        payload = {"t": "result", "id": cid, "ok": False, "error": "等太久了，操作可能还在跑", "code": "timeout"}
    except Exception as exc:
        log.exception("命令 %s 出错", name)
        payload = {"t": "result", "id": cid, "ok": False, "error": f"{name} 出错：{exc}", "code": "failed"}
    try:
        await client.ws.send_json(payload)
    except Exception:
        pass


# ------------------------------------------------------------------ 静态前端
WEB = CONFIG.web_dir
if WEB.exists():
    @app.middleware("http")
    async def no_cache_html(request: Request, call_next):
        resp = await call_next(request)
        if request.url.path.endswith((".html", ".js", ".css", ".json")) or request.url.path == "/":
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
else:  # 前端还没拷过来时给一句人话
    @app.get("/")
    async def missing():
        return JSONResponse({"ok": False, "error": f"前端目录没找到：{WEB}"}, status_code=500)
