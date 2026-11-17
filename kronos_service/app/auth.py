"""Authentication: API keys (hash + rate limit + daily quota) and admin JWT."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request
from sqlalchemy import func, select

from .db_models import ApiKey, RequestLog


# --------------------------------------------------------------------- keys
def generate_api_key() -> tuple[str, str, str]:
    """Return (plaintext, sha256_hex, display_prefix)."""
    raw = f"kron-{secrets.token_hex(16)}"
    return raw, hash_key(raw), raw[:12]


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class KeyInfo:
    id: int
    key_hash: str
    key_prefix: str
    name: str
    enabled: bool
    rate_limit_per_min: int
    daily_quota: int
    expires_at: datetime | None


class KeyCache:
    """In-memory mirror of api_keys for O(1) lookup on the hot path."""

    def __init__(self, session_factory):
        self._session_factory = session_factory
        self._lock = threading.Lock()
        self._by_hash: dict[str, KeyInfo] = {}

    def reload(self) -> None:
        session = self._session_factory()
        try:
            rows = session.query(ApiKey).all()
            mapping = {
                r.key_hash: KeyInfo(
                    id=r.id,
                    key_hash=r.key_hash,
                    key_prefix=r.key_prefix,
                    name=r.name,
                    enabled=bool(r.enabled and r.revoked_at is None),
                    rate_limit_per_min=r.rate_limit_per_min,
                    daily_quota=r.daily_quota,
                    expires_at=r.expires_at,
                )
                for r in rows
            }
            with self._lock:
                self._by_hash = mapping
        finally:
            session.close()

    def refresh_key(self, key_id: int) -> None:
        session = self._session_factory()
        try:
            row = session.get(ApiKey, key_id)
            with self._lock:
                if row is None:
                    self._by_hash = {h: k for h, k in self._by_hash.items() if k.id != key_id}
                    return
                self._by_hash.pop(row.key_hash, None)
                self._by_hash[row.key_hash] = KeyInfo(
                    id=row.id,
                    key_hash=row.key_hash,
                    key_prefix=row.key_prefix,
                    name=row.name,
                    enabled=bool(row.enabled and row.revoked_at is None),
                    rate_limit_per_min=row.rate_limit_per_min,
                    daily_quota=row.daily_quota,
                    expires_at=row.expires_at,
                )
        finally:
            session.close()

    def lookup(self, key_hash: str) -> KeyInfo | None:
        with self._lock:
            return self._by_hash.get(key_hash)


# --------------------------------------------------------------- rate limits
class RateLimiter:
    """Sliding-window per-minute limiter (process-local, single worker)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._hits: dict[int, deque[float]] = defaultdict(deque)

    def allow(self, key_id: int, limit_per_min: int) -> bool:
        if limit_per_min <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key_id]
            while hits and now - hits[0] > 60.0:
                hits.popleft()
            if len(hits) >= limit_per_min:
                return False
            hits.append(now)
            return True

    def reset(self, key_id: int | None = None) -> None:
        with self._lock:
            if key_id is None:
                self._hits.clear()
            else:
                self._hits.pop(key_id, None)


class DailyQuota:
    """Per-key daily call quota; lazily seeded from request_logs."""

    def __init__(self, session_factory):
        self._session_factory = session_factory
        self._lock = threading.Lock()
        self._counters: dict[tuple[int, date], int] = {}

    def _today(self) -> date:
        return datetime.now(timezone.utc).date()

    def check(self, key_id: int, quota: int) -> bool:
        if quota <= 0:
            return True
        day = self._today()
        with self._lock:
            slot = (key_id, day)
            if slot not in self._counters:
                self._counters[slot] = self._seed(key_id, day)
            return self._counters[slot] < quota

    def record(self, key_id: int) -> None:
        day = self._today()
        with self._lock:
            self._counters[(key_id, day)] = self._counters.get((key_id, day), 0) + 1
            # prune yesterday's slots
            stale = [s for s in self._counters if s[1] != day]
            for s in stale:
                del self._counters[s]

    def _seed(self, key_id: int, day: date) -> int:
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        session = self._session_factory()
        try:
            count = session.execute(
                select(func.count(RequestLog.id)).where(
                    RequestLog.api_key_id == key_id, RequestLog.ts >= start
                )
            ).scalar_one()
            return int(count or 0)
        finally:
            session.close()


# ------------------------------------------------------------------ admin JWT
class AdminAuth:
    def __init__(self, username: str, password: str, secret: str, expire_hours: int):
        self.username = username
        self.password = password
        self.secret = secret
        self.expire_hours = expire_hours
        self._fails: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    # -- login guard ------------------------------------------------------
    def _guard_key(self, ip: str, username: str) -> str:
        return f"{ip}|{username}"

    def is_locked(self, ip: str, username: str) -> int:
        """Return remaining lock seconds if locked, else 0."""
        key = self._guard_key(ip, username)
        with self._lock:
            entry = self._fails.get(key)
            if entry and entry[0] >= 5:
                remaining = 900.0 - (time.monotonic() - entry[1])
                if remaining > 0:
                    return int(remaining) + 1
                del self._fails[key]
        return 0

    def check_credentials(self, ip: str, username: str, password: str) -> bool:
        key = self._guard_key(ip, username)
        with self._lock:
            entry = self._fails.get(key)
            if entry and entry[0] >= 5 and time.monotonic() - entry[1] < 900:
                raise HTTPException(status_code=429, detail="尝试次数过多，请 15 分钟后重试")
        ok = hmac.compare_digest(username.encode(), self.username.encode()) and hmac.compare_digest(
            password.encode(), self.password.encode()
        )
        with self._lock:
            if ok:
                self._fails.pop(key, None)
            else:
                fails, _ = self._fails.get(key, (0, 0.0))
                self._fails[key] = (fails + 1, time.monotonic())
        return ok

    # -- tokens ------------------------------------------------------------
    def issue_token(self) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": self.username,
            "role": "admin",
            "iat": now,
            "exp": now + timedelta(hours=self.expire_hours),
        }
        return jwt.encode(payload, self.secret, algorithm="HS256")

    def verify_token(self, token: str) -> dict[str, Any]:
        try:
            payload = jwt.decode(token, self.secret, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="无效凭证")
        if payload.get("role") != "admin":
            raise HTTPException(status_code=401, detail="无效凭证")
        return payload


# ------------------------------------------------------------------ fastapi deps
def _state(request: Request):
    return request.app.state


def extract_api_key(request: Request) -> str | None:
    header = request.headers.get("x-api-key")
    if header:
        return header.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer kron-"):
        return auth[7:].strip()
    return None


async def require_api_key(request: Request) -> KeyInfo:
    state = _state(request)
    raw = extract_api_key(request)
    if not raw:
        raise HTTPException(status_code=401, detail="缺少 X-API-Key 请求头")
    info = state.key_cache.lookup(hash_key(raw))
    if info is None:
        raise HTTPException(status_code=401, detail="API 密钥无效")
    if not info.enabled:
        raise HTTPException(status_code=403, detail="API 密钥已被禁用或吊销")
    if info.expires_at is not None:
        exp = info.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp < datetime.now(timezone.utc):
            raise HTTPException(status_code=403, detail="API 密钥已过期")
    if not state.rate_limiter.allow(info.id, info.rate_limit_per_min):
        raise HTTPException(status_code=429, detail="请求频率超出该密钥限制")
    if not state.daily_quota.check(info.id, info.daily_quota):
        raise HTTPException(status_code=429, detail="已达到该密钥当日调用配额")
    return info


async def require_admin(request: Request) -> dict[str, Any]:
    state = _state(request)
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    return state.admin_auth.verify_token(auth[7:].strip())
