"""Audit logging system with date-based rotation."""
import json
import os
import sys
import threading
import time
from datetime import datetime

from .config import CONFIG

_audit_lock = threading.Lock()


def get_audit_log_path(log_dir: str = None, dt: datetime = None) -> str:
    """Return the audit log file path for the given date (default today)."""
    if not log_dir:
        log_dir = CONFIG.get("audit_log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    if dt is None:
        dt = datetime.now()
    date_str = dt.strftime("%Y-%m-%d")
    return os.path.join(log_dir, f"audit_{date_str}.log")


def record_audit_log(
    client_ip: str,
    model: str,
    request_data,
    response_data,
    log_dir: str = None,
) -> bool:
    """Record an audit log entry containing client_ip, model, request, and response.

    Rotates daily into audit_YYYY-MM-DD.log.
    Returns True if logged, False if audit logging is disabled or on error.
    """
    if not CONFIG.get("audit_log", False):
        return False

    # Normalize request_data
    if isinstance(request_data, bytes):
        try:
            request_data = json.loads(request_data.decode("utf-8"))
        except Exception:
            request_data = request_data.decode("utf-8", errors="replace")

    # Normalize response_data
    if isinstance(response_data, bytes):
        try:
            response_data = json.loads(response_data.decode("utf-8"))
        except Exception:
            response_data = response_data.decode("utf-8", errors="replace")

    now = datetime.now()
    entry = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "client_ip": client_ip or "-",
        "model": model or "-",
        "request": request_data,
        "response": response_data,
    }

    try:
        log_path = get_audit_log_path(log_dir, now)
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with _audit_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        return True
    except Exception as e:
        if CONFIG.get("log_requests", True):
            sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] Audit log error: {e}\n")
            sys.stderr.flush()
        return False
