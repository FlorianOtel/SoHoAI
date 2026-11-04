"""
rag_engine/opencode_client.py — minimal HTTP client for the OpenCode 2.x server.

Used by scanner.scan_opencode_sessions() (session discovery) and
ingest._parse_opencode_session() (session content).

OpenCode 2.x requires HTTP Basic auth (user "opencode", password from the
server's OPENCODE_SERVER_PASSWORD). Its JSON API lives under /api/...; the
legacy unprefixed routes (/session/{id}, /session/{id}/message) are served by
the web app's catch-all and answer 200 with HTML. get_json() therefore checks
the content type, so a wrong route fails loudly instead of parsing as "empty".
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request

# Default location of the OpenCode server's environment file on the ingest host.
# It is the server's own source of truth, so a password rotation is followed
# automatically. Override with opencode.password_env_file in SoHoAI-config.yaml.
DEFAULT_PASSWORD_ENV_FILE = "~/.config/opencode/opencode-server.env"

# OpenCode 2.x Basic-auth user name (fixed by the server).
_AUTH_USER = "opencode"


def load_password(password_env_file: str | None) -> str:
    """
    Read OPENCODE_SERVER_PASSWORD from a KEY=VALUE environment file.

    Raises RuntimeError with a clear message if the file or the key is missing,
    so a misconfiguration is visible in the ingestion log.
    """
    path = os.path.expanduser(password_env_file or DEFAULT_PASSWORD_ENV_FILE)
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError as exc:
        raise RuntimeError(
            f"OpenCode password file not readable: {path} ({exc}); "
            "set opencode.password_env_file in SoHoAI-config.yaml"
        ) from exc

    for line in lines:
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "OPENCODE_SERVER_PASSWORD":
            # Tolerate optional surrounding quotes, as systemd EnvironmentFile does.
            return value.strip().strip('"').strip("'")

    raise RuntimeError(f"OPENCODE_SERVER_PASSWORD not found in {path}")


def get_json(
    api_url: str,
    path: str,
    password: str,
    params: dict[str, str] | None = None,
    timeout: float = 30,
):
    """
    GET {api_url}{path}?{params} with Basic auth and return the decoded JSON.

    Raises urllib.error.HTTPError on HTTP errors (e.g. 401 on a wrong password),
    and ValueError if the server answers with something other than JSON (the
    2.x web app catch-all returns HTML with status 200 for unknown routes).
    """
    url = f"{api_url}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    token = base64.b64encode(f"{_AUTH_USER}:{password}".encode()).decode()
    request = urllib.request.Request(url, headers={"Authorization": f"Basic {token}"})

    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ValueError(
                f"OpenCode answered {content_type or 'no content type'} (not JSON) for {path} "
                "— wrong route for OpenCode 2.x?"
            )
        return json.load(response)
