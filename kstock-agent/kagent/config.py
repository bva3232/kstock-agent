"""설정 로더. config.yaml 또는 환경변수에서 값을 읽는다."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "dart": {"api_key": ""},
    "telegram": {"bot_token": "", "chat_id": ""},
    "account": {
        "seed_krw": 1_000_000,
        "max_positions": 4,
        "risk_per_trade": 0.02,
        "stop_loss_atr_mult": 2.0,
        "roundtrip_cost_pct": 0.30,
    },
    "universe": {
        "markets": ["KOSPI", "KOSDAQ"],
        "min_market_cap_krw": 100_000_000_000,
        "min_avg_turnover_krw": 1_000_000_000,
        "exclude_keywords": ["스팩", "리츠"],
        "exclude_preferred": True,
        "max_pbr": 8.0,
    },
    "screen": {"lookback_days": 90, "top_n": 10},
    "paths": {"cache_dir": "./.cache", "output_dir": "./reports"},
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, data: dict):
        self._d = data
        for key in ("cache_dir", "output_dir"):
            Path(self._d["paths"][key]).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        data = DEFAULTS
        p = Path(path)
        if p.exists():
            data = _deep_merge(data, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
        env_key = os.getenv("DART_API_KEY")
        if env_key:
            data["dart"]["api_key"] = env_key
        tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if tg_token:
            data["telegram"]["bot_token"] = tg_token
        tg_chat = os.getenv("TELEGRAM_CHAT_ID")
        if tg_chat:
            data["telegram"]["chat_id"] = tg_chat
        return cls(data)

    def __getitem__(self, key: str):
        return self._d[key]

    @property
    def cache_dir(self) -> Path:
        return Path(self._d["paths"]["cache_dir"])

    @property
    def output_dir(self) -> Path:
        return Path(self._d["paths"]["output_dir"])

    @property
    def dart_key(self) -> str:
        return self._d["dart"]["api_key"]

    @property
    def telegram_bot_token(self) -> str:
        return self._d["telegram"]["bot_token"]

    @property
    def telegram_chat_id(self) -> str:
        return self._d["telegram"]["chat_id"]
