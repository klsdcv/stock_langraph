import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from config.settings import CACHE_DIR


def _cache_path(key: str) -> Path:
    """캐시 키로부터 파일 경로 생성."""
    safe_key = hashlib.md5(key.encode()).hexdigest()
    return Path(CACHE_DIR) / f"{safe_key}.json"


def _read_cache(path: Path, ttl_hours: float) -> Any | None:
    """캐시 파일 읽기. TTL 내이면 데이터 반환, 아니면 None."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        age_hours = (time.time() - cached["timestamp"]) / 3600
        if age_hours <= ttl_hours:
            return cached["data"]
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _read_cache_expired(path: Path) -> Any | None:
    """만료 무시하고 캐시 데이터 반환 (Graceful Degradation)."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        return cached["data"]
    except (json.JSONDecodeError, KeyError):
        return None


def _write_cache(path: Path, data: Any) -> None:
    """캐시 파일 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.time(), "data": data}, f, ensure_ascii=False)


def cached_api_call(
    key: str,
    fetcher: Callable[[], Any],
    ttl_hours: float = 1.0,
) -> Any:
    """모든 외부 API 호출의 단일 진입점.

    캐시 확인 → miss 시 API 호출 → 캐시 저장.
    API 실패 시 만료된 캐시라도 반환. 캐시도 없으면 None.
    """
    path = _cache_path(key)

    # 캐시 히트
    cached = _read_cache(path, ttl_hours)
    if cached is not None:
        return cached

    # API 호출
    try:
        data = fetcher()
        if data is not None:
            _write_cache(path, data)
        return data
    except Exception:
        # Graceful Degradation: 만료된 캐시라도 반환
        expired = _read_cache_expired(path)
        if expired is not None:
            return expired
        return None
