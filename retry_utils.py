#!/usr/bin/env python3
"""HTTP retry utilities for transient errors (e.g., Railway cold start 502).

Provides request_with_retry which wraps requests.request with:
- Exponential backoff with jitter
- Retry on 502/503/504/429 and common network errors
- Optional respect for Retry-After header
- Optional callback on each retry to update UI/logs
"""
from __future__ import annotations

import os
import random
import time
from typing import Any, Callable, Dict, Iterable, Optional

import requests


def _to_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _to_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def request_with_retry(
    method: str,
    url: str,
    *,
    session: Optional[requests.Session] = None,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    data: Any = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    stream: bool = False,
    allow_statuses: Iterable[int] = (200, 201, 202, 204),
    retry_on_statuses: Iterable[int] = (429, 502, 503, 504),
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None,
    max_backoff: Optional[float] = None,
    on_retry: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> requests.Response:
    """Perform an HTTP request with retries for transient failures.

    - method/url/params/json/data/headers/timeout/stream are passed to requests.
    - allow_statuses: statuses considered success (default: typical 2xx)
    - retry_on_statuses: statuses considered transient and retried
    - max_retries: total attempts (including the first) defaults from env HTTP_MAX_RETRIES (6)
    - backoff_factor: base seconds for exponential backoff (default from env HTTP_BACKOFF_FACTOR=1.0)
    - max_backoff: cap for backoff seconds (default env HTTP_MAX_BACKOFF=30)
    - on_retry: callback receiving context {attempt, wait, status, error, url, method}
    """

    # env-tunable defaults
    if max_retries is None:
        max_retries = _to_int("HTTP_MAX_RETRIES", 6)
    if backoff_factor is None:
        backoff_factor = _to_float("HTTP_BACKOFF_FACTOR", 1.0)
    if max_backoff is None:
        max_backoff = _to_float("HTTP_MAX_BACKOFF", 30.0)

    sess = session or requests.Session()

    last_exc: Optional[BaseException] = None
    attempt = 1
    while True:
        try:
            resp = sess.request(
                method=method,
                url=url,
                params=params,
                json=json,
                data=data,
                headers=headers,
                timeout=timeout,
                stream=stream,
            )
            # success
            if resp.status_code in allow_statuses or (200 <= resp.status_code < 300):
                return resp

            # Retry-able HTTP status
            if resp.status_code in retry_on_statuses and attempt < max_retries:
                # Retry-After support
                wait = None
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        wait = float(ra)
                    except Exception:
                        wait = None
                if wait is None:
                    wait = min(max_backoff, backoff_factor * (2 ** (attempt - 1)))
                    # add jitter 0.5x~1.0x
                    wait *= 0.5 + random.random() * 0.5
                if on_retry:
                    on_retry(
                        {
                            "attempt": attempt,
                            "wait": wait,
                            "status": resp.status_code,
                            "error": None,
                            "url": url,
                            "method": method,
                        }
                    )
                time.sleep(wait)
                attempt += 1
                continue

            # Not retry-able or retries exhausted -> raise
            resp.raise_for_status()
            return resp  # safety, though raise_for_status() above should raise
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            last_exc = e
            if attempt >= max_retries:
                raise
            wait = min(max_backoff, backoff_factor * (2 ** (attempt - 1)))
            wait *= 0.5 + random.random() * 0.5
            if on_retry:
                on_retry(
                    {
                        "attempt": attempt,
                        "wait": wait,
                        "status": None,
                        "error": e,
                        "url": url,
                        "method": method,
                    }
                )
            time.sleep(wait)
            attempt += 1
            continue
        except Exception as e:
            # Non-transient error: do not retry by default
            last_exc = e
            raise

    # Should not reach here
    if last_exc:
        raise last_exc
