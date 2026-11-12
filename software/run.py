#!/usr/bin/env python3
"""开发/真机统一入口：`python run.py`（在 software/ 目录下）。

    ZELUX_MODE=sim   python run.py     # 本机模拟，没有硬件也能开全部页面
    ZELUX_MODE=radxa python run.py     # 板子上跑真机
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

from zeluxd.config import CONFIG  # noqa: E402

if __name__ == "__main__":
    dev = CONFIG.dev
    # 注意：不要在这里传 ws_ping_interval 之类的新参数，老版本 uvicorn 不认，会直接起不来。
    uvicorn.run("zeluxd.app:app", host=CONFIG.host, port=CONFIG.port,
                reload=dev, reload_dirs=[str(Path(__file__).parent / "zeluxd")] if dev else None,
                log_level="info", access_log=False)
