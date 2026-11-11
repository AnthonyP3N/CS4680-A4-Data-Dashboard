"""LLM integration: Google Gemini, plus any OpenAI-compatible API (Groq, OpenAI, OpenRouter).

Handles:
- Structured JSON output (Gemini: enforced response_schema; others: JSON mode +
  format instructions), parsed and checked by parse_agent_json
- Client-side rate limiting (sliding window) to stay inside free-tier quotas
- Retries with exponential backoff + jitter on 429 / 5xx / network errors,
  honouring the server's suggested retry delay, inside a fixed time budget so a
  struggling provider can't keep the user waiting for minutes
- Switching to the next model when one is overloaded, retired or unavailable,
  and asking the provider which models the key can use when the built-in list runs out
- Falling back to a second provider (FallbackClient) when the first stays overloaded
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

from .prompts import JSON_FORMAT_INSTRUCTIONS, RESPONSE_SCHEMA

REQUIRED_KEYS = {"reasoning", "intent", "sql", "explanation", "chart_type", "x", "y",
                 "color", "chart_title", "follow_ups"}
VALID_INTENTS = {"query", "modify", "clarify", "unsupported"}

# Provider presets. Model lists are tried in order; if all are unavailable the
# client asks the provider for its model list. Override with <PROVIDER>_MODEL.
PROVIDERS = {
    "gemini": {
        "name": "Gemini", "key": "GEMINI_API_KEY", "signup": "aistudio.google.com/apikey",
        "models": ["gemini-flash-latest", "gemini-2.5-flash", "gemini-flash-lite-latest", "gemini-3-flash-preview"],
    },
    "groq": {
        "name": "Groq", "key": "GROQ_API_KEY", "signup": "console.groq.com/keys",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
    },
    "openai": {
        "name": "OpenAI", "key": "OPENAI_API_KEY", "signup": "platform.openai.com/api-keys",
        "base_url": None,
        "models": ["gpt-4.1-mini", "gpt-4o-mini"],
    },
    "openrouter": {
        "name": "OpenRouter", "key": "OPENROUTER_API_KEY", "signup": "openrouter.ai/keys",
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["openai/gpt-oss-120b:free", "meta-llama/llama-3.3-70b-instruct:free"],
    },
}


class LLMError(Exception):
    """Raised when the LLM cannot produce a usable response."""


class TransientLLMError(LLMError):
    """The provider is overloaded, rate-limited or unreachable right now; another provider may work."""


@dataclass
class LLMResponse:
    data: dict
    raw_text: str
    model: str
    attempts: int
    latency_ms: float
    usage: dict = field(default_factory=dict)


class LLMClient(Protocol):
    model: str

    def generate(self, system: str, turns: list[tuple[str, str]]) -> LLMResponse: ...


# --------------------------------------------------------------------------- #
# Output parsing (shared by all clients)
# --------------------------------------------------------------------------- #
def parse_agent_json(text: str) -> dict:
    """Parse and normalise the model's JSON. Raises LLMError when unusable."""
    if not text:
        raise LLMError("The model returned an empty response.")
    cleaned = text.strip()
    # Tolerate ```json fences or text around the object if a model ignores JSON mode
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.S)
    if fence:
        cleaned = fence.group(1)
    elif not cleaned.startswith("{") and "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.index("{"): cleaned.rindex("}") + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMError(f"The model's response was not valid JSON ({e.msg}).") from None
    if not isinstance(data, dict):
        raise LLMError("The model's response was not a JSON object.")

    missing = REQUIRED_KEYS - data.keys()
    for k in missing:
        data[k] = [] if k == "follow_ups" else ""
    data["intent"] = str(data.get("intent", "")).strip().lower()
    if data["intent"] not in VALID_INTENTS:
        raise LLMError(f"Unknown intent '{data['intent']}' in model response.")
    for k in REQUIRED_KEYS - {"follow_ups"}:
        data[k] = "" if data[k] is None else str(data[k]).strip()
    if not isinstance(data["follow_ups"], list):
        data["follow_ups"] = []
    data["follow_ups"] = [str(f) for f in data["follow_ups"]][:3]
    return data


# --------------------------------------------------------------------------- #
# Rate limiter
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Allow at most `max_calls` per `period` seconds; blocks (up to max_wait) otherwise."""

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls, self.period = max_calls, period
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self, max_wait: float = 30.0) -> float:
        waited = 0.0
        while True:
            with self.lock:
                now = time.monotonic()
                while self.calls and now - self.calls[0] > self.period:
                    self.calls.popleft()
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return waited
                wait = self.period - (now - self.calls[0]) + 0.05
            if waited + wait > max_wait:
                raise LLMError(
                    f"Rate limit: more than {self.max_calls} requests in a minute. "
                    f"Please wait about {int(wait)} seconds and try again."
                )
            time.sleep(wait)
            waited += wait


# --------------------------------------------------------------------------- #
# Shared retry / fallback loop
# --------------------------------------------------------------------------- #
class _BaseClient:
    """Subclasses implement _call, _discover_models and (optionally) _normalise_error."""

    provider = "LLM"
    TRANSIENT_CODES = {408, 409, 429, 500, 502, 503, 504, 529}

    def __init__(self, models: list[str], model: str | None, *, requests_per_minute: int,
                 temperature: float, time_budget_s: float, max_attempts: int):
        self.models = ([model] + [m for m in models if m != model]) if model else list(models)
        self.model = self.models[0]
        self.temperature = temperature
        self.time_budget_s = time_budget_s
        self.max_attempts = max_attempts
        self.limiter = RateLimiter(requests_per_minute)
        self._discovered = False

    # -- to implement ------------------------------------------------------ #
    def _call(self, system: str, turns: list[tuple[str, str]]) -> tuple[str, dict]:
        raise NotImplementedError

    def _discover_models(self) -> list[str]:
        return []

    def _adapt(self, err: Exception) -> bool:
        """Adjust request settings after a 400 (e.g. drop an unsupported option). True = retry now."""
        return False

    # -- helpers ----------------------------------------------------------- #
    @property
    def label(self) -> str:
        return f"{self.provider} · {self.model}"

    @staticmethod
    def _code(err: Exception) -> int | None:
        for attr in ("status_code", "code"):
            v = getattr(err, attr, None)
            if isinstance(v, int):
                return v
        return None

    @staticmethod
    def _message(err: Exception | None) -> str:
        if err is None:
            return "unknown error"
        msg = getattr(err, "message", None) or str(err)
        code = _BaseClient._code(err)
        s = f"{code}: {msg}" if code and not str(msg).startswith(str(code)) else str(msg)
        return s if len(s) < 300 else s[:300] + "..."

    @staticmethod
    def _retry_delay_hint(err: Exception) -> float | None:
        m = re.search(r"retry(?:Delay)?\W+(?:in\s+|after\s+)?(\d+(?:\.\d+)?)\s*s", str(err), re.I)
        return float(m.group(1)) if m else None

    def _is_missing_model(self, err: Exception, code: int | None) -> bool:
        if code == 404:
            return True
        msg = str(err).lower()
        return code == 400 and "model" in msg and any(
            w in msg for w in ("decommissioned", "does not exist", "not found", "not a valid model", "no longer supported"))

    def _next_model(self) -> bool:
        """Move to the next model (wrapping around). False if there is only one."""
        if len(self.models) < 2:
            return False
        self.model = self.models[(self.models.index(self.model) + 1) % len(self.models)]
        return True

    # -- main entry -------------------------------------------------------- #
    def generate(self, system: str, turns: list[tuple[str, str]]) -> LLMResponse:
        t0 = time.monotonic()
        deadline = t0 + self.time_budget_s
        attempts = bad_outputs = 0
        tried_this_call: set[str] = set()
        last_err: Exception | None = None
        last_transient_code: int | None = None

        while attempts < self.max_attempts and time.monotonic() < deadline:
            attempts += 1
            tried_this_call.add(self.model)
            self.limiter.acquire(max_wait=max(1.0, min(30.0, deadline - time.monotonic())))
            try:
                text, usage = self._call(system, turns)
                return LLMResponse(data=parse_agent_json(text), raw_text=text, model=self.label,
                                   attempts=attempts, latency_ms=(time.monotonic() - t0) * 1000, usage=usage)
            except TransientLLMError:
                raise
            except LLMError as e:
                # Unusable output: the model is non-deterministic, so just ask again
                last_err = e
                bad_outputs += 1
                if bad_outputs >= 3:
                    raise
                continue
            except Exception as e:  # provider SDK errors, timeouts, network errors
                last_err = e
                code = self._code(e)

                if self._is_missing_model(e, code):
                    if self.model == self.models[-1] and not self._discovered:
                        self._discovered = True
                        self.models += [m for m in self._discover_models() if m not in self.models]
                    if self.model != self.models[-1]:
                        bad = self.model
                        self._next_model()
                        self.models.remove(bad)   # don't come back to it
                        attempts -= 1
                        continue
                    raise LLMError(
                        f"None of the {self.provider} models this app tried are available to your key "
                        f"(last tried: {self.model}). Set {self.provider.upper()}_MODEL in secrets to a model "
                        "your account can use."
                    ) from None
                if code in (401, 403):
                    raise LLMError(f"{self.provider} rejected the API key. Check it in your secrets.") from None
                if code == 400 or code == 422:
                    if self._adapt(e):
                        attempts -= 1
                        continue
                    raise LLMError(f"{self.provider} rejected the request: {self._message(e)}") from None
                if code is not None and code not in self.TRANSIENT_CODES:
                    raise LLMError(f"{self.provider} request failed: {self._message(e)}") from None

                # Transient: overloaded (5xx), rate limited (429), timeout / network (code None).
                # Another model often has spare capacity, so switch first and back off only
                # once every model has been tried in this call.
                last_transient_code = code
                switched = self._next_model()
                if switched and self.model not in tried_this_call:
                    continue
                delay = self._retry_delay_hint(e) or min(2 ** (attempts - 1), 8)
                remaining = deadline - time.monotonic()
                if delay > remaining:
                    break
                time.sleep(delay + random.uniform(0, 0.5))

        waited = time.monotonic() - t0
        if last_transient_code == 429:
            raise TransientLLMError(
                f"{self.provider}'s rate limit or daily quota was reached. Wait a minute and try again.")
        if last_transient_code is not None or (last_err is not None and self._code(last_err) is None):
            raise TransientLLMError(
                f"{self.provider} is overloaded or not responding right now (tried for {waited:.0f}s). "
                f"Try again in a minute. Last error: {self._message(last_err)}")
        raise LLMError(f"{self.provider} request failed after {attempts} attempts: {self._message(last_err)}")


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
class GeminiClient(_BaseClient):
    provider = "Gemini"

    def __init__(self, api_key: str, model: str | None = None, *, requests_per_minute: int = 8,
                 temperature: float = 0.1, timeout_s: float = 25, time_budget_s: float = 45,
                 max_attempts: int = 6):
        if not api_key:
            raise LLMError("No Gemini API key configured. Set GEMINI_API_KEY in Streamlit secrets or the environment.")
        from google import genai
        from google.genai import types

        super().__init__(PROVIDERS["gemini"]["models"], model, requests_per_minute=requests_per_minute,
                         temperature=temperature, time_budget_s=time_budget_s, max_attempts=max_attempts)
        self._types = types
        self._schema = types.Schema.model_validate(RESPONSE_SCHEMA)  # fails fast if the schema is malformed
        # v1beta carries the current models and full structured-output support;
        # some SDK versions default to v1, where newer model names return 404.
        # The SDK's own retries are off (attempts=1): this class owns retrying.
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            api_version="v1beta", timeout=int(timeout_s * 1000),
            retry_options=types.HttpRetryOptions(attempts=1)))

    def _call(self, system, turns):
        t = self._types
        config = t.GenerateContentConfig(system_instruction=system, temperature=self.temperature,
                                         response_mime_type="application/json", response_schema=self._schema)
        contents = [t.Content(role=role, parts=[t.Part(text=text)]) for role, text in turns]
        resp = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        text = resp.text or ""
        if not text:
            reason = ""
            try:
                reason = str(resp.candidates[0].finish_reason)
            except (AttributeError, IndexError, TypeError):
                pass
            raise LLMError(f"Empty response from the model {reason}".strip())
        usage = {}
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage = {"prompt_tokens": um.prompt_token_count, "output_tokens": um.candidates_token_count,
                     "total_tokens": um.total_token_count}
        return text, usage

    def _discover_models(self):
        skip = ("image", "tts", "audio", "live", "embedding", "robotics", "computer", "veo", "imagen", "learnlm")
        found = []
        try:
            for m in self.client.models.list():
                name = (m.name or "").removeprefix("models/")
                actions = getattr(m, "supported_actions", None) or []
                if "generateContent" in actions and "gemini" in name and not any(s in name for s in skip):
                    found.append(name)
        except Exception:
            return []
        return sorted(found, key=lambda n: ("flash" not in n, "lite" in n, "preview" in n or "exp" in n, n))[:5]


# --------------------------------------------------------------------------- #
# OpenAI-compatible APIs: Groq, OpenAI, OpenRouter (and anything else that speaks the same protocol)
# --------------------------------------------------------------------------- #
class OpenAICompatibleClient(_BaseClient):
    def __init__(self, provider_id: str, api_key: str, model: str | None = None, *,
                 base_url: str | None = None, requests_per_minute: int = 8, temperature: float = 0.1,
                 timeout_s: float = 25, time_budget_s: float = 45, max_attempts: int = 6):
        preset = PROVIDERS.get(provider_id, {"name": provider_id, "models": [], "key": "API_KEY"})
        if not api_key:
            raise LLMError(f"No {preset['name']} API key configured. Set {preset['key']} in secrets.")
        models = preset["models"] or ([model] if model else [])
        if not models:
            raise LLMError(f"Set {provider_id.upper()}_MODEL to the model you want to use.")
        from openai import OpenAI

        super().__init__(models, model, requests_per_minute=requests_per_minute, temperature=temperature,
                         time_budget_s=time_budget_s, max_attempts=max_attempts)
        self.provider = preset["name"]
        self.client = OpenAI(api_key=api_key, base_url=base_url or preset.get("base_url"),
                             timeout=timeout_s, max_retries=0)   # this class owns retrying
        self.json_mode = True
        self.send_temperature = True

    def _call(self, system, turns):
        messages = [{"role": "system", "content": system + "\n\n" + JSON_FORMAT_INSTRUCTIONS}]
        messages += [{"role": "assistant" if role == "model" else "user", "content": text} for role, text in turns]
        kwargs = {"model": self.model, "messages": messages}
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.send_temperature:
            kwargs["temperature"] = self.temperature
        resp = self.client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        if not text:
            raise LLMError("Empty response from the model.")
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens,
                     "total_tokens": resp.usage.total_tokens}
        return text, usage

    def _adapt(self, err):
        msg = str(err).lower()
        if self.json_mode and ("response_format" in msg or "json" in msg):
            self.json_mode = False          # model doesn't support JSON mode: rely on instructions + parser
            return True
        if self.send_temperature and "temperature" in msg:
            self.send_temperature = False   # some reasoning models only accept the default temperature
            return True
        return False

    def _discover_models(self):
        skip = ("audio", "realtime", "tts", "whisper", "transcribe", "embed", "image", "dall-e", "moderation",
                "search", "guard", "orpheus", "vision")
        found = []
        try:
            for m in self.client.models.list():
                mid = getattr(m, "id", "") or ""
                if mid and not any(s in mid.lower() for s in skip):
                    found.append(mid)
        except Exception:
            return []
        return sorted(found, key=lambda n: ("free" not in n and self.provider == "OpenRouter",
                                            "mini" not in n and "gpt-oss" not in n, n))[:5]


# --------------------------------------------------------------------------- #
# Fallback across providers
# --------------------------------------------------------------------------- #
class FallbackClient:
    """Try providers in order; move to the next one only when the current one is overloaded or rate-limited."""

    def __init__(self, clients: list[_BaseClient]):
        if not clients:
            raise LLMError("No LLM provider configured.")
        self.clients = clients
        self.active = clients[0]

    @property
    def model(self) -> str:
        names = [c.label for c in self.clients]
        return names[0] if len(names) == 1 else f"{self.active.label} (fallback: " + \
            ", ".join(c.provider for c in self.clients if c is not self.active) + ")"

    def generate(self, system, turns):
        errors = []
        for c in self.clients:
            try:
                resp = c.generate(system, turns)
                self.active = c
                return resp
            except TransientLLMError as e:
                errors.append(str(e))
                continue
        if len(errors) == 1:
            raise TransientLLMError(errors[0])
        raise TransientLLMError("All configured providers are busy right now. " + " | ".join(errors))


CUSTOM_PROVIDER = "custom"


def check_public_https_url(url: str) -> str:
    """Accept a user-supplied API address only if it is https and every address it resolves to is public.

    This stops a visitor pointing the server at itself or its private network
    (localhost, 10.x, 192.168.x, cloud metadata at 169.254.169.254 and so on).
    Returns the URL without a trailing slash; raises LLMError otherwise.
    """
    import ipaddress
    import socket
    from urllib.parse import urlsplit

    url = (url or "").strip()
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise LLMError("The address must start with https:// and include a host name.")
    if parts.username or parts.password:
        raise LLMError("Don't put a user name or password in the address; enter the key in the API key box.")
    if parts.query or parts.fragment:
        raise LLMError("The address can't contain ? or #. Use the provider's base URL, such as https://host/v1.")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        raise LLMError(f"Couldn't find the host {parts.hostname}. Check the address.") from None
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if not address.is_global:
            raise LLMError("That address points to a private or local network, which isn't allowed.")
    return url.rstrip("/")


def build_custom_client(api_key: str, base_url: str, model: str, name: str = "", *,
                        requests_per_minute: int = 8) -> FallbackClient:
    """A client for any OpenAI-compatible API the user names (Together, Mistral, DeepSeek, a company gateway...)."""
    if not (model or "").strip():
        raise LLMError("Enter the model name your provider uses (for example meta-llama/Llama-3.3-70B-Instruct-Turbo).")
    url = check_public_https_url(base_url)
    client = OpenAICompatibleClient(CUSTOM_PROVIDER, api_key, model.strip(), base_url=url,
                                    requests_per_minute=requests_per_minute)
    from urllib.parse import urlsplit
    client.provider = (name or "").strip()[:40] or urlsplit(url).hostname
    client.client._client.follow_redirects = False     # a redirect must not lead somewhere the check didn't see
    return FallbackClient([client])


def build_client(get_secret) -> FallbackClient:
    """Create the LLM client from settings.

    LLM_PROVIDER picks providers in order, e.g. "groq" or "groq,gemini" (the second is a fallback).
    Without it, every provider whose API key is set is used, in the order of PROVIDERS.
    """
    rpm = int(get_secret("REQUESTS_PER_MINUTE", 8) or 8)
    wanted = str(get_secret("LLM_PROVIDER", "") or "").lower().replace(" ", "")
    order = [p for p in wanted.split(",") if p] or [p for p in PROVIDERS if get_secret(PROVIDERS[p]["key"], "")]
    if not order:
        keys = ", ".join(p["key"] for p in PROVIDERS.values())
        raise LLMError(f"No API key found. Add one of these to your secrets: {keys}.")

    clients = []
    for i, pid in enumerate(order):
        if pid not in PROVIDERS:
            raise LLMError(f"Unknown LLM_PROVIDER '{pid}'. Use one of: {', '.join(PROVIDERS)}.")
        key = get_secret(PROVIDERS[pid]["key"], "")
        model = get_secret(f"{pid.upper()}_MODEL") or None
        # Earlier providers in a chain get a shorter time budget so the fallback starts sooner
        budget = 45 if i == len(order) - 1 else 20
        if pid == "gemini":
            clients.append(GeminiClient(key, model, requests_per_minute=rpm, time_budget_s=budget))
        else:
            clients.append(OpenAICompatibleClient(pid, key, model, requests_per_minute=rpm, time_budget_s=budget))
    return FallbackClient(clients)
