"""Configuration management."""
import json
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = {
    "port": 8081,
    "host": "0.0.0.0",
    "retry_attempts": 3,
    "retry_delay_sec": 2,
    "request_timeout_sec": 180,
    "default_model": "gemini-3.8-flash",
    "log_requests": True,
    "cookie_file": None,
    "proxy": None,
    "api_keys": [],
    "temporary_chats": False,
    "audit_log": False,
    "rate_limit": None,
    "rate_limit_jitter": False,
}

CONFIG = dict(DEFAULT_CONFIG)


def resolve_cookie_file(cookie_file: str = None, base_dir: str = None) -> str:
    """Resolve cookie file path (relative to base_dir or package directory)."""
    target = cookie_file or CONFIG.get("cookie_file")
    if not target:
        return None
    if os.path.isabs(target):
        return target
    base = base_dir or PACKAGE_DIR
    return os.path.join(base, target)


def load_config(path: str = None):
    """Load config from JSON file."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            CONFIG.update(json.load(f))
    return CONFIG


def find_config() -> str:
    """Search for config.json in gemini_web2api directory."""
    pkg_config = os.path.join(PACKAGE_DIR, "config.json")
    if os.path.exists(pkg_config):
        return pkg_config
    return None
