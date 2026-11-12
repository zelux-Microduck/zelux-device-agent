"""快递查询：快递100 的实时查询接口。

为什么是这个接口：国内覆盖率最高，一次 POST 就能拿到轨迹，不用轮询订阅。
没有 key 时**不要假装查到了**，直接告诉用户去设置页填 —— 静默失败最难排查。

设置页（ai.json 的 skills.express）需要两个字段：
    {"key": "快递100 的授权 key", "customer": "快递100 的 customer"}
"""
from __future__ import annotations

import hashlib
import json
import urllib.parse

from zeluxd.errors import ZeluxError

URL = "https://poll.kuaidi100.com/poll/query.do"

# 单号前缀 → 快递100 的公司编码。猜不准就让用户报公司名。
PREFIX = [
    ("SF", "shunfeng"), ("YT", "yuantong"), ("ZTO", "zhongtong"),
    ("STO", "shentong"), ("YD", "yunda"), ("JD", "jd"),
    ("JT", "jtexpress"), ("HTKY", "huitongkuaidi"), ("EMS", "ems"),
    ("YZ", "youzhengguonei"), ("DB", "debangwuliu"), ("ZJS", "zhaijisong"),
]

NAME2CODE = {
    "顺丰": "shunfeng", "圆通": "yuantong", "中通": "zhongtong", "申通": "shentong",
    "韵达": "yunda", "京东": "jd", "极兔": "jtexpress", "百世": "huitongkuaidi",
    "邮政": "youzhengguonei", "EMS": "ems", "德邦": "debangwuliu", "宅急送": "zhaijisong",
}

STATE = {"在途": "在路上", "揽收": "已揽收", "派件": "正在派送", "签收": "已签收",
         "退签": "已退回", "疑难": "出了点问题", "转投": "转投其他快递"}


def run(ctx, text, args) -> dict:
    cfg = ctx.config.get("skill") or {}
    key = str(cfg.get("key") or "").strip()
    customer = str(cfg.get("customer") or "").strip()
    num = str(args.get("num") or "").strip()

    if not key or not customer:
        return {
            "speech": "还没配快递查询的 key，去设置页填（快递100 的 key 和 customer）。",
            "data": {"configured": False},
            "display": {"kind": "express", "need_config": ["key", "customer"]},
        }

    com = str(args.get("com") or "").strip() or _guess_company(num, text)
    if not com:
        raise ZeluxError("认不出这是哪家快递，跟我说公司名（比如顺丰）")

    param = json.dumps({"com": com, "num": num, "resultv2": "1", "phone": ""},
                       ensure_ascii=False, separators=(",", ":"))
    sign = hashlib.md5((param + key + customer).encode("utf-8")).hexdigest().upper()
    body = urllib.parse.urlencode({"customer": customer, "sign": sign, "param": param})

    try:
        raw = ctx.http_post(URL, body=body, timeout=3.0)
        data = json.loads(raw)
    except ZeluxError as exc:
        raise ZeluxError(f"快递查询没成功：{exc.message}")
    except json.JSONDecodeError:
        raise ZeluxError("快递接口回了看不懂的内容，等会儿再试")

    msg = str(data.get("message") or "")
    if msg and msg not in ("ok", "OK"):
        raise ZeluxError(_human_msg(msg, num))

    items = data.get("data") or []
    if not items:
        return {"speech": f"{num} 还没有轨迹，可能刚下单还没揽收。",
                "data": {"num": num, "com": com, "items": []},
                "display": {"kind": "express", "num": num, "com": com, "items": []}}

    # 接口返回的是倒序（最新在前），取两条够说清楚了
    latest = items[0]
    when = str(latest.get("ftime") or latest.get("time") or "")
    what = str(latest.get("context") or "").strip()
    status = str(data.get("state") or "")
    speech = f"{num} 最新：{_shorten(what)}"
    if when:
        speech += f"（{_shorten(when, 16)}）"
    if status in STATE:
        speech += f"，{STATE[status]}"
    speech += "。"

    return {
        "speech": speech,
        "data": {"num": num, "com": com, "state": STATE.get(status, status),
                 "items": [{"time": i.get("ftime") or i.get("time"),
                            "context": i.get("context")} for i in items[:5]]},
        "display": {"kind": "express", "num": num, "com": com,
                    "items": [{"time": i.get("ftime") or i.get("time"),
                               "context": i.get("context")} for i in items[:5]]},
    }


def _guess_company(num: str, text: str) -> str:
    for name, code in NAME2CODE.items():
        if name in text:
            return code
    up = num.upper()
    for prefix, code in PREFIX:
        if up.startswith(prefix):
            return code
    return ""


def _human_msg(msg: str, num: str) -> str:
    table = {
        "查询无结果": f"{num} 查不到，检查单号对不对",
        "单号不存在": f"{num} 这个单号不存在",
        "快递公司编码错误": "快递公司编码不对",
        "参数错误": "单号或公司填得不对",
        "访问次数达到限制": "查询太频繁了，等一分钟再试",
        "签名错误": "key 和 customer 对不上，去设置页核对一下",
    }
    return table.get(msg, f"查询失败：{msg}")


def _shorten(text: str, limit: int = 44) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"
