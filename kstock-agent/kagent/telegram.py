"""텔레그램 봇 알림. bot_token/chat_id 가 없으면 조용히 건너뛴다."""
from __future__ import annotations

import io
import sys

import requests


class TeeStdout(io.TextIOBase):
    """콘솔에도 찍고 버퍼에도 모으는 stdout 대체용."""

    def __init__(self, buffer: io.StringIO, console=None):
        self._buffer = buffer
        self._console = console or sys.__stdout__

    def write(self, s: str) -> int:
        try:
            self._console.write(s)
        except UnicodeEncodeError:
            enc = getattr(self._console, "encoding", None) or "utf-8"
            self._console.write(s.encode(enc, errors="replace").decode(enc, errors="replace"))
        self._buffer.write(s)
        return len(s)

    def flush(self) -> None:
        self._console.flush()

API_ROOT = "https://api.telegram.org/bot{token}/{method}"
CHUNK = 3800  # 텔레그램 4096자 제한에 여유를 둔 값


def _post(token: str, method: str, **payload) -> dict:
    url = API_ROOT.format(token=token, method=method)
    r = requests.post(url, data=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def _chunks(text: str, size: int = CHUNK) -> list[str]:
    lines = text.split("\n")
    out, cur = [], ""
    for ln in lines:
        if len(cur) + len(ln) + 1 > size:
            out.append(cur)
            cur = ln
        else:
            cur = f"{cur}\n{ln}" if cur else ln
    if cur:
        out.append(cur)
    return out or [text[:size]]


def send_message(cfg, text: str, title: str | None = None) -> bool:
    """cfg.telegram_bot_token / cfg.telegram_chat_id 가 비어있으면 False 를 반환하고 건너뛴다."""
    token, chat_id = cfg.telegram_bot_token, cfg.telegram_chat_id
    if not token or not chat_id:
        print("[텔레그램 건너뜀] bot_token/chat_id 미설정")
        return False
    body = f"{title}\n\n{text}" if title else text
    try:
        for part in _chunks(body):
            _post(token, "sendMessage", chat_id=chat_id, text=part,
                  disable_web_page_preview=True)
        return True
    except requests.RequestException as e:
        print(f"[텔레그램 발송 실패] {e}")
        return False


def send_document(cfg, path, caption: str | None = None) -> bool:
    token, chat_id = cfg.telegram_bot_token, cfg.telegram_chat_id
    if not token or not chat_id:
        return False
    try:
        url = API_ROOT.format(token=token, method="sendDocument")
        with open(path, "rb") as f:
            r = requests.post(
                url, data={"chat_id": chat_id, "caption": caption or ""},
                files={"document": f}, timeout=30,
            )
        r.raise_for_status()
        return True
    except (requests.RequestException, OSError) as e:
        print(f"[텔레그램 파일 발송 실패] {e}")
        return False
