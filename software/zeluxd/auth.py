"""口令与会话。

设计取舍：
* 只存 PBKDF2 哈希（sha256 / 200k 轮），不存明文；
* 会话是服务端内存 + 签名 cookie（进程重启要重登，安全 > 便利，镜像里开机自动登录不是这里的事）；
* 所有改动类请求要带 `X-CSRF`，并且校验 Origin —— 防"同一局域网里别的网站偷偷 POST"。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path

ITER = 200_000
TTL = 7 * 24 * 3600


class Auth:
    def __init__(self, config) -> None:
        self.config = config
        self.path = Path(config.state_dir) / "auth.json"
        self.sessions: dict[str, dict] = {}
        self.fails = {"n": 0, "until": 0.0}
        self.no_password = False
        self._load()
        # 环境变量优先级最高：本机演示/评审时不想每次登录就用它。
        # 注意这里只影响"要不要口令"，不改已存的口令哈希。
        if self.config.allow_no_password:
            self.no_password = True

    # ---------------- 口令 ----------------
    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text("utf-8"))
                self._salt = bytes.fromhex(data["salt"])
                self._hash = bytes.fromhex(data["hash"])
                self.no_password = bool(data.get("no_password"))
                return
            except Exception:
                pass
        self.set_password(self.config.initial_password or "zelux")

    def set_password(self, password: str) -> None:
        self._salt = secrets.token_bytes(16)
        self._hash = hashlib.pbkdf2_hmac("sha256", password.encode(), self._salt, ITER)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"salt": self._salt.hex(), "hash": self._hash.hex(),
                                         "iter": ITER, "no_password": self.no_password}), "utf-8")
        self.sessions.clear()

    def verify(self, password: str) -> bool:
        if self.no_password:
            return True
        if time.time() < self.fails["until"]:
            return False
        calc = hashlib.pbkdf2_hmac("sha256", (password or "").encode(), self._salt, ITER)
        ok = hmac.compare_digest(calc, self._hash)
        if ok:
            self.fails = {"n": 0, "until": 0.0}
        else:
            self.fails["n"] += 1
            if self.fails["n"] >= 5:
                self.fails = {"n": 0, "until": time.time() + 30}
        return ok

    @property
    def has_password(self) -> bool:
        return not self.no_password

    # ---------------- 会话 ----------------
    def new_session(self) -> tuple[str, str]:
        token = secrets.token_urlsafe(24)
        csrf = secrets.token_urlsafe(24)
        self.sessions[token] = {"csrf": csrf, "exp": time.time() + TTL}
        return token, csrf

    def session(self, token: str | None) -> dict | None:
        if self.no_password and not token:
            token = "anonymous"
            self.sessions.setdefault(token, {"csrf": "dev", "exp": time.time() + TTL})
        if not token:
            return None
        s = self.sessions.get(token)
        if not s:
            return None
        if s["exp"] < time.time():
            self.sessions.pop(token, None)
            return None
        s["exp"] = time.time() + TTL          # 滑动续期
        return s

    def drop(self, token: str | None) -> None:
        if token:
            self.sessions.pop(token, None)

    def csrf_ok(self, token: str | None, given: str | None) -> bool:
        if self.no_password:
            return True
        s = self.session(token)
        return bool(s and given and hmac.compare_digest(s["csrf"], given))
