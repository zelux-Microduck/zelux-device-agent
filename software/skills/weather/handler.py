"""天气：走 wttr.in 的 JSON 接口，不需要 key。

为什么选 wttr.in：免费、不用注册、支持中文城市名，装完就能用。
要换和风天气只需要改 `_fetch_wttr` 一个函数（返回同样的 dict）。
"""
from __future__ import annotations

import json
import urllib.parse

from zeluxd.errors import ZeluxError

# wttr.in 的 condition code（WW 码）到中文。只翻常见的，剩下的原样报英文。
COND = {
    "113": "晴", "116": "多云", "119": "阴", "122": "阴天",
    "143": "雾", "248": "雾", "260": "冻雾",
    "176": "阵雨", "263": "毛毛雨", "266": "毛毛雨", "293": "小雨", "296": "小雨",
    "299": "中雨", "302": "中雨", "305": "大雨", "308": "大雨",
    "311": "冻雨", "314": "冻雨", "353": "小阵雨", "356": "阵雨", "359": "暴雨",
    "200": "雷阵雨", "386": "雷阵雨", "389": "雷雨",
    "179": "阵雪", "182": "雨夹雪", "185": "冻毛毛雨", "317": "雨夹雪", "320": "雨夹雪",
    "323": "小雪", "326": "小雪", "329": "中雪", "332": "中雪", "335": "大雪", "338": "大雪",
    "350": "冰粒", "362": "小阵雪", "365": "阵雪", "368": "小阵雪", "371": "大雪",
    "374": "冰粒", "377": "冰粒", "392": "雷雪", "395": "雷雪",
}

DEFAULT_CITY = "北京"

# "今天天气"这种问法，正则会把"今天"当成城市抓出来。这些词直接丢掉用默认城市，
# 否则会拿"今天"去查天气，用户会觉得鸭子听不懂人话。
NOT_CITY = {"今天", "明天", "现在", "这里", "那里", "这边", "那边",
            "外面", "多少", "什么", "怎么", "咋样", "怎么样"}


def run(ctx, text, args) -> dict:
    cfg = ctx.config.get("skill") or {}
    city = str(args.get("city") or "").strip("的了吗呢 ")
    if city in NOT_CITY:
        city = ""
    city = city or str(cfg.get("city") or DEFAULT_CITY).strip() or DEFAULT_CITY

    try:
        raw = ctx.http_get("https://wttr.in/" + urllib.parse.quote(city) + "?format=j1&lang=zh-cn",
                           timeout=2.5)
        data = json.loads(raw)
    except ZeluxError as exc:
        raise ZeluxError(f"查不到{city}的天气：{exc.message}")
    except json.JSONDecodeError:
        raise ZeluxError("天气接口返回了看不懂的东西，等会儿再试")

    cur = (data.get("current_condition") or [{}])[0]
    if not cur:
        raise ZeluxError(f"没查到 {city} 这个城市，换个名字试试（比如「北京」）")

    today = (data.get("weather") or [{}])[0]
    code = str(cur.get("weatherCode") or "")
    cond = COND.get(code) or (cur.get("weatherDesc") or [{}])[0].get("value") or "说不清"
    temp = _num(cur.get("temp_C"))
    feels = _num(cur.get("FeelsLikeC"))
    hum = _num(cur.get("humidity"))
    wind = _num(cur.get("windspeedKmph"))
    lo, hi = _num(today.get("mintempC")), _num(today.get("maxtempC"))

    pieces = [f"{city}现在{cond}，{temp} 度"]
    if feels is not None and temp is not None and abs(feels - temp) >= 3:
        pieces.append(f"体感 {feels} 度")
    if lo is not None and hi is not None:
        pieces.append(f"今天 {lo} 到 {hi} 度")
    speech = "，".join(pieces) + "。"

    rain = _rain_chance(today)
    if rain is not None and rain >= 50 and "雨" not in cond:
        speech += f"有 {rain}% 的可能下雨，记得带伞。"

    return {
        "speech": speech,
        "data": {"city": city, "cond": cond, "temp": temp, "feels": feels,
                 "min": lo, "max": hi, "humidity": hum, "wind_kmph": wind,
                 "rain_chance": rain, "source": "wttr.in"},
        "display": {"kind": "weather", "city": city, "cond": cond,
                    "temp": temp, "min": lo, "max": hi, "rain_chance": rain},
    }


def _num(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _rain_chance(today: dict):
    hourly = today.get("hourly") or []
    best = None
    for h in hourly:
        v = _num(h.get("chanceofrain"))
        if v is None:
            continue
        best = v if best is None else max(best, v)
    return best
