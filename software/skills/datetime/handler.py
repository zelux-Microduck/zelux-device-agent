"""日期时间：纯本地，1 ms 出答案，不联网也不调模型。

为什么必须是技能：模型不知道"现在"，每次问时间都要一次往返，还可能编日期。
这里读的是板子的系统时钟，也是整机唯一真相。
"""
from __future__ import annotations

import datetime as dt

# 农历大节只内置到 2032 年。超出范围就直说"我这张表没到那年"，
# 不猜 —— 猜错日期比承认不知道更糟。
LUNAR = {
    "春节": {2024: "02-10", 2025: "01-29", 2026: "02-17", 2027: "02-06",
             2028: "01-26", 2029: "02-13", 2030: "02-03", 2031: "01-23",
             2032: "02-11"},
    "端午": {2024: "06-10", 2025: "05-31", 2026: "06-19", 2027: "06-09",
             2028: "05-28", 2029: "06-16", 2030: "06-05", 2031: "06-24",
             2032: "06-12"},
    "中秋": {2024: "09-17", 2025: "10-06", 2026: "09-25", 2027: "09-15",
             2028: "10-03", 2029: "09-22", 2030: "09-12", 2031: "10-01",
             2032: "09-19"},
}

# 公历节日（只放大家都过得这几个）
SOLAR = {"01-01": "元旦", "05-01": "劳动节", "06-01": "儿童节",
         "10-01": "国庆节", "12-25": "圣诞节"}

WEEK = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


def _speech(text: str, **data) -> dict:
    return {"speech": text, "data": data, "display": {"kind": "datetime", **data}}


def run(ctx, text, args) -> dict:
    now = dt.datetime.now()

    if any(k in text for k in ("节日", "过年", "春节", "中秋", "端午", "还有几天", "放假")):
        return _festival(now)
    if any(k in text for k in ("星期", "周几", "礼拜")):
        return _speech(f"今天{WEEK[now.weekday()]}。", date=now.strftime("%Y-%m-%d"),
                       week=WEEK[now.weekday()])
    if any(k in text for k in ("几点", "时间", "什么时候")):
        return _clock(now)
    return _date(now)


def _clock(now: dt.datetime) -> dict:
    h, m = now.hour, now.minute
    hour = h if h <= 12 else h - 12
    part = "上午" if h < 12 else ("中午" if h == 12 else ("下午" if h < 18 else "晚上"))
    speech = f"现在是{part}{hour if hour else 12}点{m:02d}分。"
    return _speech(speech, time=now.strftime("%H:%M"), epoch=int(now.timestamp()))


def _date(now: dt.datetime) -> dict:
    speech = f"今天是 {now.year} 年 {now.month} 月 {now.day} 日，{WEEK[now.weekday()]}。"
    return _speech(speech, date=now.strftime("%Y-%m-%d"), week=WEEK[now.weekday()])


def _festival(now: dt.datetime) -> dict:
    today = now.date()

    named = [name for name in ("春节", "端午", "中秋")
             if LUNAR[name].get(now.year) == today.strftime("%m-%d")]
    if today.strftime("%m-%d") in SOLAR:
        named.append(SOLAR[today.strftime("%m-%d")])
    if named:
        return _speech(f"今天就是{'、'.join(named)}。", today=named)

    # 每个节日只取"今天之后最近的那一次"。不能遇到第一个未来日期就 break：
    # 表的年份是书写顺序，2024/2025 已经过去，先撞上的是 2028 而不是 2027。
    best: dict[str, dt.date] = {}
    for name, table in LUNAR.items():
        for year, mmdd in table.items():
            month, day = (int(x) for x in mmdd.split("-"))
            try:
                d = dt.date(year, month, day)
            except ValueError:
                continue
            if d >= today and (name not in best or d < best[name]):
                best[name] = d
    upcoming: list[tuple[dt.date, str]] = [(d, name) for name, d in best.items()]
    for mmdd, name in SOLAR.items():
        month, day = (int(x) for x in mmdd.split("-"))
        d = dt.date(now.year, month, day)
        if d < today:
            d = dt.date(now.year + 1, month, day)
        upcoming.append((d, name))

    if not upcoming:
        return _speech("我这张农历表只到 2032 年，再往后就不敢乱说了。")

    upcoming.sort()
    d, name = upcoming[0]
    days = (d - today).days
    if days == 0:
        return _speech(f"今天就是{name}。", festival=name)
    speech = f"离{name}还有 {days} 天（{d.month} 月 {d.day} 日）。"
    return _speech(speech, festival=name, days=days, date=d.strftime("%Y-%m-%d"))
