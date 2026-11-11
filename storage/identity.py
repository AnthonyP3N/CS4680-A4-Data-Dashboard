"""Who owns a workspace.

AskData needs to know which saved conversations belong to the person in front
of it. There are three modes, chosen with the ASKDATA_AUTH setting:

  local   One owner for the whole installation. Right for an app running on
          your own computer. Never use it on a public address: every visitor
          would share the same conversations.
  link    Each browser gets a long random token in the page address (?u=...).
          Bookmarking that address returns you to your conversations. The token
          is a secret: anyone who has the link can read them. No sign-in needed.
  login   Streamlit's built-in sign-in (OpenID Connect). The owner is the
          signed-in account. Needs an identity provider configured in secrets.

When ASKDATA_AUTH isn't set, the mode is "local" on localhost and "link"
anywhere else, so a public deployment never defaults to one shared owner.

Only a hash of the token or account id is stored, so the metadata database
alone doesn't reveal anyone's link or email address.
"""

from __future__ import annotations

import hashlib
import re
import secrets

MODES = ("local", "link", "login")
LOCAL_OWNER = "local"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}
_TOKEN = re.compile(r"^[A-Za-z0-9_\-]{24,128}$")


def default_mode(host: str | None) -> str:
    """"local" only when the app is clearly being used on the same machine; otherwise "link"."""
    if not host:
        return "link"
    name = host.strip().lower()
    name = name.split("://", 1)[-1].split("/", 1)[0]
    if name.startswith("["):                     # [::1]:8501
        name = name.split("]", 1)[0] + "]"
    else:
        name = name.rsplit(":", 1)[0] if name.count(":") == 1 else name
    return "local" if name in _LOCAL_HOSTS else "link"


def resolve_mode(configured: str | None, host: str | None) -> str:
    mode = (configured or "").strip().lower()
    return mode if mode in MODES else default_mode(host)


def new_token() -> str:
    return secrets.token_urlsafe(32)


def valid_token(token: str | None) -> bool:
    return bool(token) and bool(_TOKEN.match(token))


def _digest(kind: str, value: str) -> str:
    return f"{kind}:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:40]}"


def owner_from_token(token: str) -> str:
    if not valid_token(token):
        raise ValueError("That isn't a valid AskData link.")
    return _digest("link", token)


def owner_from_login(account_id: str) -> str:
    """`account_id` is the provider's stable subject id (or, failing that, the email address)."""
    account_id = (account_id or "").strip().lower()
    if not account_id:
        raise ValueError("The sign-in did not provide an account id.")
    return _digest("user", account_id)
