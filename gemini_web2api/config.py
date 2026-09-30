"""Configuration management."""
import json
import os
import tempfile

DEFAULT_CONFIG = {
    "port": 8081,
    "host": "0.0.0.0",
    "retry_attempts": 3,
    "retry_delay_sec": 2,
    "request_timeout_sec": 180,
    "gemini_bl": "boq_assistant-bard-web-server_20260716.08_p0",
    "auth_user": None,
    "xsrf_token": None,
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

AUTH_SYNC_KEYS = ("gemini_bl", "auth_user", "xsrf_token")


def sync_cookie_auth_to_config(config_path: str, cookie_file: str = None) -> bool:
    """Sync auth parameters (gemini_bl, auth_user, xsrf_token) from cookie_file into config.json.

    Returns True if config.json was updated, False otherwise.
    """
    if not config_path or not os.path.exists(config_path):
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception:
        return False

    if not isinstance(config_data, dict):
        return False

    target_cookie_file = cookie_file or config_data.get("cookie_file")
    if not target_cookie_file:
        return False

    if not os.path.isabs(target_cookie_file):
        if not os.path.exists(target_cookie_file):
            candidate = os.path.join(os.path.dirname(config_path), target_cookie_file)
            if os.path.exists(candidate):
                target_cookie_file = candidate
            else:
                return False
    elif not os.path.exists(target_cookie_file):
        return False

    try:
        with open(target_cookie_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content.startswith("{"):
            return False
        auth_data = json.loads(content)
    except Exception:
        return False

    if not isinstance(auth_data, dict):
        return False

    changed = False
    for key in AUTH_SYNC_KEYS:
        val = auth_data.get(key)
        if val is not None and str(val).strip() != "":
            val_str = str(val).strip() if key == "auth_user" else str(val)
            if config_data.get(key) != val_str:
                config_data[key] = val_str
                changed = True

    if cookie_file and config_data.get("cookie_file") != cookie_file:
        config_data["cookie_file"] = cookie_file
        changed = True

    if changed:
        try:
            config_dir = os.path.dirname(os.path.abspath(config_path)) or "."
            temp_file = tempfile.NamedTemporaryFile("w", dir=config_dir, delete=False, encoding="utf-8")
            json.dump(config_data, temp_file, indent=2, ensure_ascii=False)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()

            try:
                orig_mode = os.stat(config_path).st_mode & 0o777
                os.chmod(temp_file.name, orig_mode)
            except Exception:
                pass

            os.replace(temp_file.name, config_path)
            print(f"[config] Synced auth parameters (gemini_bl, auth_user, xsrf_token) from {target_cookie_file} to {config_path}")
            return True
        except Exception:
            if "temp_file" in locals() and os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except OSError:
                    pass
            return False

    return False


def load_config(path: str = None):
    """Load config from JSON file."""
    if path and os.path.exists(path):
        with open(path) as f:
            CONFIG.update(json.load(f))
    return CONFIG


def find_config():
    """Search for config file in standard locations."""
    for p in ["./config.json", os.path.expanduser("~/.config/gemini-web2api/config.json")]:
        if os.path.exists(p):
            return p
    return None
