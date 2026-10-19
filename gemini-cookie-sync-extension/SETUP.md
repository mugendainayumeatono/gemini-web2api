# Gemini Cookie Sync Setup

Short guide for extracting fresh Gemini auth data and applying it to `gemini-web2api`.

## What this extension exports

The extension reads the current signed-in Gemini session and exports:

- Google session cookies
- `SAPISID`
- `SNlM0e` (`xsrf_token`)
- `cfb2h` (`gemini_bl`)
- `auth_user`

It saves them locally as `gemini-auth.json`.

## Install and export

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select the `gemini-cookie-sync-extension` folder
5. Open [https://gemini.google.com/app](https://gemini.google.com/app)
6. Sign in and refresh the page
7. Open the extension and click **Inspect session**
8. Confirm the session looks ready
9. Click **Export gemini-auth.json**

Expected ready state:

```text
XSRF / SNlM0e: present
gemini_bl / cfb2h: present
Session and XSRF are ready for export.
```

## Apply it in `gemini-web2api`

Move the exported file into `gemini_web2api`:

```bash
cd /path/to/gemini-web2api

WIN_HOME=$(wslpath "$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath(\"UserProfile\")' | tr -d '\r')")
cp "$WIN_HOME/Downloads/gemini-auth.json" ./gemini_web2api/gemini-auth.json
chmod 600 ./gemini_web2api/gemini-auth.json
```

Ensure `gemini_web2api/config.json` sets `"cookie_file"`:

```json
{
  "cookie_file": "gemini-auth.json"
}
```

The server reads `cookie`, `sapisid`, `auth_user`, `xsrf_token`, and `gemini_bl` directly from `gemini-auth.json` (with automatic mtime-based hot-reloading).

## Restart and test

```bash
systemctl --user restart gemini-proxy
```

```bash
curl -sS http://127.0.0.1:10012/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "gemini-3.1-pro",
    "messages": [
      {
        "role": "user",
        "content": "Reply exactly with: authenticated-ok"
      }
    ]
  }' | jq
```

## Keep it secret

`gemini-auth.json` contains a real Google session. Do not share it, print it, or commit it to Git.
