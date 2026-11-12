"""放音乐：扫 var/music，用 mpv/ffplay/aplay 播，维护一个队列。

为什么自己管队列而不调 mpris/音乐服务：板子上没有桌面环境，装一个播放服务
比装 mpv 还重。这里只要"放/停/下一首/音量"四件事。

为什么用 mpv 的 IPC socket：暂停和音量要**立刻生效**，杀掉进程重开一首
会让用户听到一声爆音、还丢进度。ffplay/aplay 没有 IPC，就退回信号方案。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

EXT = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus")

_PLAYERS: dict[str, "_Player"] = {}
_LOCK = threading.Lock()


def _which(*names: str) -> str | None:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


class _Player:
    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.dir = Path(ctx.config["state_dir"]) / "music"
        self.volume = 70
        self.paused = False
        self.stopped = True
        self.proc: subprocess.Popen | None = None
        self.track: Path | None = None
        self.queue: list[Path] = []
        self.index = -1
        self._lock = threading.RLock()
        if _which("mpv"):
            self.mode = "mpv"
        elif _which("ffplay"):
            self.mode = "ffplay"
        elif _which("aplay"):
            self.mode = "aplay"
        else:
            self.mode = "none"
        run_dir = Path(ctx.config["state_dir"]) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.sock = run_dir / f"mpv-{os.getpid()}.sock"

    # ---------------- 列表 ----------------
    def scan(self) -> list[Path]:
        if not self.dir.is_dir():
            return []
        files = [p for p in sorted(self.dir.iterdir())
                 if p.is_file() and p.suffix.lower() in EXT]
        self.queue = files
        return files

    def find(self, what: str) -> int:
        if not what:
            return -1
        key = what.strip().lower()
        for i, p in enumerate(self.queue):
            if key in p.stem.lower():
                return i
        return -1

    # ---------------- 播放 ----------------
    def play(self, index: int | None = None, what: str = "") -> str:
        with self._lock:
            if not self.scan():
                raise _NoMusic()
            if self.mode == "none":
                raise _NoPlayer()
            if what:
                hit = self.find(what)
                if hit >= 0:
                    index = hit
            if index is None:
                index = (self.index + 1) % len(self.queue) if self.index >= 0 else 0
            index %= len(self.queue)
            self._spawn(index)
            self.stopped = False
            self.paused = False
            self.index = index
            self.track = self.queue[index]
            return self.track.stem

    def next(self) -> str:
        with self._lock:
            if not self.queue:
                self.scan()
            if not self.queue:
                raise _NoMusic()
            return self.play((self.index + 1) % len(self.queue))

    def prev(self) -> str:
        with self._lock:
            if not self.queue:
                self.scan()
            if not self.queue:
                raise _NoMusic()
            return self.play((self.index - 1) % len(self.queue))

    def _spawn(self, index: int) -> None:
        self._kill()
        path = self.queue[index]
        if self.mode == "mpv":
            try:
                self.sock.unlink()
            except OSError:
                pass
            cmd = ["mpv", "--no-video", "--really-quiet", f"--volume={self.volume}",
                   f"--input-ipc-server={self.sock}", str(path)]
        elif self.mode == "ffplay":
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                   "-volume", str(self.volume), str(path)]
        else:
            if path.suffix.lower() != ".wav":
                raise _NoPlayer()
            cmd = ["aplay", "-q", str(path)]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._watch, args=(self.proc,), daemon=True).start()

    def _watch(self, proc: subprocess.Popen) -> None:
        """一首放完自动下一首。被用户切掉的不算（proc 已经换人了）。"""
        try:
            proc.wait()
        except Exception:
            pass
        with self._lock:
            if self.proc is not proc or self.stopped or self.paused:
                return
            self.proc = None
            nxt = (self.index + 1) % len(self.queue) if self.queue else -1
            if nxt < 0:
                return
            try:
                self._spawn(nxt)
                self.index = nxt
                self.track = self.queue[nxt]
            except Exception:
                pass

    def _kill(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # ---------------- 控制 ----------------
    def _ipc(self, *command) -> bool:
        if self.mode != "mpv":
            return False
        for _ in range(2):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(0.4)
                    s.connect(str(self.sock))
                    s.sendall((json.dumps({"command": list(command)}) + "\n").encode())
                return True
            except Exception:
                time.sleep(0.08)
        return False

    def set_pause(self, on: bool) -> bool:
        with self._lock:
            if self.proc is None:
                return False
            if self._ipc("set_property", "pause", on):
                self.paused = on
                return True
            if self.mode in ("ffplay", "aplay"):
                try:
                    self.proc.send_signal(signal.SIGSTOP if on else signal.SIGCONT)
                    self.paused = on
                    return True
                except Exception:
                    return False
            return False

    def set_volume(self, value: int) -> bool:
        value = max(0, min(100, int(value)))
        self.volume = value
        with self._lock:
            if self.proc is None:
                return True
            if self._ipc("set_property", "volume", value):
                return True
            if self.mode == "ffplay" and self.track is not None:
                # ffplay 的音量只能在启动时给，重开一首代价最小
                self._spawn(self.index)
                return True
            return False

    def stop(self) -> None:
        with self._lock:
            self.stopped = True
            self.paused = False
            self._kill()


class _NoMusic(Exception):
    pass


class _NoPlayer(Exception):
    pass


def _get(ctx) -> _Player:
    key = str(ctx.config.get("state_dir") or "var")
    with _LOCK:
        player = _PLAYERS.get(key)
        if player is None:
            player = _PLAYERS[key] = _Player(ctx)
        return player


def run(ctx, text, args) -> dict:
    player = _get(ctx)
    name = str(getattr(player.track, "stem", "") or "")
    count = len(player.scan())

    try:
        # 顺序有讲究：先判"别放了/暂停"，再判"下一首"，最后才是"放一首"
        if any(k in text for k in ("别放了", "关掉音乐", "停止播放", "安静", "别唱了")):
            player.stop()
            return _said("好，不放了。")

        if "暂停" in text or "停一下" in text:
            if player.set_pause(True):
                return _said("暂停了。", playing=name, paused=True)
            return _said("现在没有在放歌。")

        if any(k in text for k in ("继续", "接着放", "接着播", "恢复")):
            if player.set_pause(False):
                return _said("接着放。", playing=name, paused=False)
            return _said("没有暂停中的歌。")

        if any(k in text for k in ("音量", "声音大", "声音小", "小声", "大声")):
            return _volume(ctx, player, text)

        if any(k in text for k in ("下一首", "换一首", "换歌", "下一曲")):
            title = player.next()
            return _playing(title, count, player)

        if any(k in text for k in ("上一首", "上一曲", "前一首")):
            title = player.prev()
            return _playing(title, count, player)

        if any(k in text for k in ("什么歌", "在放什么", "这歌")):
            if not name:
                return _said("现在没放歌。")
            return _said(f"在放《{name}》。", playing=name)

        if any(k in text for k in ("歌单", "有什么歌", "列表", "都有什么")):
            if not player.queue:
                return _said("音乐目录还是空的：把 mp3 或 wav 拷到 var/music/ 再叫我。")
            names = "、".join(p.stem for p in player.queue[:6])
            more = f" 等 {len(player.queue)} 首" if len(player.queue) > 6 else f"共 {len(player.queue)} 首"
            return {"speech": f"有：{names}{more}。",
                    "data": {"songs": [p.name for p in player.queue], "count": len(player.queue)},
                    "display": {"kind": "music", "songs": [p.name for p in player.queue]}}

        what = str(args.get("what") or "").strip()
        if what in ("", "歌", "音乐", "首歌"):
            what = ""
        title = player.play(what=what)
        if what and player.find(what) < 0:
            return _playing(title, count, player, prefix=f"没找到《{what}》，先放")
        return _playing(title, count, player)
    except _NoMusic:
        return _said("音乐目录还是空的：把 mp3 或 wav 拷到 var/music/ 再叫我。")
    except _NoPlayer:
        return _said("这台机器上没有能放歌的播放器，装一个 mpv 就行。")


def _volume(ctx, player: _Player, text: str) -> dict:
    m = re.search(r"(\d{1,3})", text)
    if m:
        target = int(m.group(1))
    elif any(k in text for k in ("大", "大声")):
        target = player.volume + 10
    elif any(k in text for k in ("小", "小声")):
        target = player.volume - 10
    else:
        return _said(f"音量是 {player.volume}。", volume=player.volume)
    player.set_volume(target)
    return _said(f"音量 {player.volume}。", volume=player.volume)


def _playing(title: str, count: int, player: _Player, prefix: str = "") -> dict:
    speech = f"{prefix}《{title}》。" if prefix else f"放《{title}》。"
    if count > 1:
        speech += f"第 {player.index + 1} 首，共 {count} 首。"
    return {"speech": speech,
            "data": {"track": title, "index": player.index + 1, "count": count,
                     "engine": player.mode},
            "display": {"kind": "music", "track": title, "index": player.index + 1,
                        "count": count}}


def _said(text: str, **data) -> dict:
    return {"speech": text, "data": data, "display": {"kind": "music", **data}}
