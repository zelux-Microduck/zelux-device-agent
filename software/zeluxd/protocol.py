"""Studio-Agent 协议版本与握手协商。

v1 客户端不会发送 client_hello，服务端继续接受；新客户端可以在连接后声明
支持范围。这里刻意不依赖第三方 semver 包，设备最小安装仍只需 FastAPI。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


PROTOCOL_VERSION = "1.4.0"
PROTOCOL_SUPPORTED = ">=1.0.0 <2.0.0"
DRIVER_API_VERSION = "1.0.0"
DRIVER_API_SUPPORTED = ">=1.0.0 <2.0.0"
MOTION_PROTOCOL_VERSION = "0.1.0"

_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")
_BOUND = re.compile(r"^(>=|>|<=|<|=)?(\d+\.\d+\.\d+)$")


def version_tuple(value: str) -> tuple[int, int, int]:
    match = _VERSION.match(str(value).strip())
    if not match:
        raise ValueError(f"无效版本号：{value}")
    return tuple(int(part) for part in match.groups())


def satisfies(version: str, expression: str) -> bool:
    """检查简单 SemVer 范围，例如 ``>=1.0.0 <2.0.0``。"""
    current = version_tuple(version)
    for token in str(expression).split():
        match = _BOUND.match(token)
        if not match:
            raise ValueError(f"不支持的版本范围：{expression}")
        op, raw = match.groups()
        bound = version_tuple(raw)
        if op in (None, "=") and current != bound:
            return False
        if op == ">=" and current < bound:
            return False
        if op == ">" and current <= bound:
            return False
        if op == "<=" and current > bound:
            return False
        if op == "<" and current >= bound:
            return False
    return True


def ranges_overlap(client_range: str, server_version: str = PROTOCOL_VERSION) -> bool:
    """当前协商选择服务端版本；后续多版本编解码器可在此扩展。"""
    try:
        return satisfies(server_version, client_range)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ProtocolError:
    code: str
    message: str
    supported: str = PROTOCOL_SUPPORTED


def negotiate(client_range: str | None) -> ProtocolError | None:
    if not client_range:  # v1 旧客户端
        return None
    if ranges_overlap(client_range):
        return None
    return ProtocolError(
        code="protocol.incompatible",
        message=f"Studio 支持 {client_range}，设备支持 {PROTOCOL_SUPPORTED}",
    )

