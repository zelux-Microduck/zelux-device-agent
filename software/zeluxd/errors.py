"""错误约定：所有面向用户的原因都必须是"人话"，并且带 code 供前端分支。"""
from __future__ import annotations


class ZeluxError(Exception):
    """业务错误 → {ok:false, error, code}"""

    def __init__(self, message: str, code: str = "failed", **extra):
        super().__init__(message)
        self.message = message
        self.code = code
        self.extra = extra

    def payload(self) -> dict:
        out = {"ok": False, "error": self.message, "code": self.code}
        out.update(self.extra)
        return out


def busy(what: str = "上一个操作还没做完") -> ZeluxError:
    return ZeluxError(what, "busy")


def denied(why: str) -> ZeluxError:
    return ZeluxError(why, "denied")


def unsupported(why: str = "这台机器不支持这个功能") -> ZeluxError:
    return ZeluxError(why, "unsupported")


def hardware(why: str) -> ZeluxError:
    return ZeluxError(why, "hardware")
