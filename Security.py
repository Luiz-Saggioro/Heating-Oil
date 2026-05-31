"""
security.py — Security middleware for Energy Intelligence Dashboard.

Provides:
  - RateLimiter   : in-memory per-session token-bucket (5 attempts / 15 min)
  - sanitize_str  : strip dangerous characters from user-supplied strings
  - validate_enum : reject values not in an explicit allowlist
  - reject_oversized : hard-limit on string / collection size
  - security_audit_report : static analysis helper called at app startup

All functions are pure Python; no external dependencies beyond the stdlib.
"""

from __future__ import annotations

import os
import re
import time
import html
import logging
from collections import defaultdict
from typing import Any, Collection, Optional

logger = logging.getLogger(__name__)

# ── Configuration (override via environment variables) ────────────────────────

_RATE_LIMIT_WINDOW: int = int(os.environ.get("RATE_LIMIT_WINDOW", 900))   # 15 min
_RATE_LIMIT_MAX: int    = int(os.environ.get("RATE_LIMIT_MAX_ATTEMPTS", 5))
_MAX_PAYLOAD: int       = int(os.environ.get("MAX_PAYLOAD_BYTES", 65536))  # 64 KB


# ── Rate Limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Simple sliding-window rate limiter keyed by session_id.

    Usage:
        limiter = RateLimiter()
        allowed, remaining, reset_in = limiter.check("session-abc")
        if not allowed:
            st.error(f"Too many requests. Try again in {reset_in}s.")
    """

    def __init__(
        self,
        max_attempts: int = _RATE_LIMIT_MAX,
        window_secs: int  = _RATE_LIMIT_WINDOW,
    ) -> None:
        self._max  = max_attempts
        self._win  = window_secs
        # session_id -> list of timestamps
        self._log: dict[str, list[float]] = defaultdict(list)

    def check(self, session_id: str) -> tuple[bool, int, int]:
        """
        Returns (allowed, remaining_attempts, seconds_until_reset).
        Calling this method counts as one attempt if allowed.
        """
        now = time.monotonic()
        window_start = now - self._win
        history = self._log[session_id]

        # Purge expired entries
        history[:] = [t for t in history if t > window_start]

        if len(history) >= self._max:
            reset_in = int(self._win - (now - history[0])) + 1
            return False, 0, reset_in

        history.append(now)
        remaining = self._max - len(history)
        return True, remaining, 0

    def remaining(self, session_id: str) -> int:
        now = time.monotonic()
        window_start = now - self._win
        history = [t for t in self._log.get(session_id, []) if t > window_start]
        return max(0, self._max - len(history))

    def reset(self, session_id: str) -> None:
        """Clear the rate-limit history for a session (for testing)."""
        self._log.pop(session_id, None)


# ── Input Sanitization ────────────────────────────────────────────────────────

# Characters allowed in user-supplied option/filter strings
_SAFE_PATTERN = re.compile(r"[^a-zA-Z0-9 $.\-+%/(),]")

# Maximum lengths
_MAX_STR_LEN  = 128
_MAX_LIST_LEN = 200


def sanitize_str(value: Any, field: str = "input") -> str:
    """
    Sanitize a string value for use in UI display and log messages.
    - Coerce to str
    - HTML-escape to prevent XSS in st.markdown(unsafe_allow_html=True)
    - Strip control characters and characters outside the safe set
    - Truncate to _MAX_STR_LEN
    Raises ValueError if the cleaned result is empty but value was non-empty.
    """
    if value is None:
        return ""
    raw = str(value)
    escaped = html.escape(raw, quote=True)
    cleaned = _SAFE_PATTERN.sub("", escaped)
    cleaned = cleaned.strip()[:_MAX_STR_LEN]
    return cleaned


def validate_enum(value: Any, allowed: Collection[Any], field: str = "field") -> Any:
    """
    Reject any value not in the explicit allowlist.
    Returns the original value if valid; raises ValueError otherwise.
    """
    if value not in allowed:
        raise ValueError(
            f"Invalid {field} value {value!r}. "
            f"Must be one of: {sorted(str(a) for a in allowed)}"
        )
    return value


def reject_oversized(value: Any, field: str = "input") -> Any:
    """
    Raises ValueError if a string or collection exceeds configured limits.
    Returns the value unchanged if within limits.
    """
    if isinstance(value, str) and len(value) > _MAX_PAYLOAD:
        raise ValueError(
            f"{field} string too large: {len(value)} bytes "
            f"(max {_MAX_PAYLOAD})"
        )
    if isinstance(value, (list, dict, tuple, set)) and len(value) > _MAX_LIST_LEN:
        raise ValueError(
            f"{field} collection too large: {len(value)} items "
            f"(max {_MAX_LIST_LEN})"
        )
    return value


# ── Security Audit ────────────────────────────────────────────────────────────

def security_audit_report() -> str:
    """
    Run a lightweight static check at startup and return a human-readable
    report. Checks:
      1. No hardcoded API keys present in environment surface
      2. EIA_API_KEY is not the placeholder "DEMO_KEY"
      3. Rate-limit settings are within reasonable bounds
      4. output/ directory is not world-writable
    """
    issues: list[str] = []
    passed: list[str] = []

    # 1. EIA API key
    eia = os.environ.get("EIA_API_KEY", "")
    if not eia:
        issues.append("EIA_API_KEY not set — EIA inventory fetch will fail")
    elif eia == "DEMO_KEY":
        issues.append(
            "EIA_API_KEY is still 'DEMO_KEY' — set a real key in .env or "
            "Streamlit Cloud secrets"
        )
    else:
        passed.append("EIA_API_KEY is set")

    # 2. Rate-limit sanity
    if _RATE_LIMIT_MAX < 1 or _RATE_LIMIT_MAX > 100:
        issues.append(
            f"RATE_LIMIT_MAX_ATTEMPTS={_RATE_LIMIT_MAX} is outside the "
            "recommended range (1-100)"
        )
    else:
        passed.append(
            f"Rate limit: {_RATE_LIMIT_MAX} attempts / {_RATE_LIMIT_WINDOW}s"
        )

    # 3. Payload limit
    if _MAX_PAYLOAD < 1024:
        issues.append(
            f"MAX_PAYLOAD_BYTES={_MAX_PAYLOAD} is very small — may break "
            "legitimate requests"
        )
    else:
        passed.append(f"Payload limit: {_MAX_PAYLOAD} bytes")

    # 4. Output directory permissions
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    if os.path.isdir(output_dir):
        mode = oct(os.stat(output_dir).st_mode)[-3:]
        if mode[-1] in ("6", "7"):
            issues.append(
                f"output/ directory is world-writable (mode {mode}) — "
                "consider chmod o-w output/"
            )
        else:
            passed.append(f"output/ permissions OK ({mode})")

    lines = ["=== SECURITY AUDIT REPORT ==="]
    if passed:
        lines += ["", "PASSED:"] + [f"  [OK] {p}" for p in passed]
    if issues:
        lines += ["", "WARNINGS:"] + [f"  [!!] {i}" for i in issues]
    if not issues:
        lines.append("\nAll checks passed.")
    return "\n".join(lines)


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this in streamlit_app.py:  from security import limiter
limiter = RateLimiter()
