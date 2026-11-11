"""Offline tests for the LLM clients: provider errors are simulated, no API keys needed."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from agent import llm as L

GOOD = json.dumps({"reasoning": "r", "intent": "query", "sql": "SELECT 1", "explanation": "e",
                   "chart_type": "metric", "x": "", "y": "", "color": "", "chart_title": "", "follow_ups": []})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(L.time, "sleep", lambda s: slept.append(s))
    return slept


class FakeAPIError(Exception):
    def __init__(self, code, message="boom"):
        super().__init__(f"{code} {message}")
        self.status_code, self.message = code, message


def openai_client(script, models=None, provider="groq", **kw):
    """OpenAICompatibleClient whose HTTP calls come from `script` (dict: model -> list of results)."""
    c = L.OpenAICompatibleClient(provider, "fake-key", requests_per_minute=100, **kw)
    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        queue = script.get(kwargs["model"], [FakeAPIError(404, "model not found")])
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return NS(choices=[NS(message=NS(content=item))], usage=None)

    c.client = NS(chat=NS(completions=NS(create=create)),
                  models=NS(list=lambda: [NS(id=m) for m in (models or [])]))
    return c, seen


def test_groq_success_sends_json_mode_and_converts_roles():
    c, seen = openai_client({"openai/gpt-oss-120b": [GOOD]})
    r = c.generate("system prompt", [("user", "q1"), ("model", "a1"), ("user", "q2")])
    assert r.data["sql"] == "SELECT 1" and r.model == "Groq · openai/gpt-oss-120b"
    msgs = seen[0]["messages"]
    assert msgs[0]["role"] == "system" and "Output format" in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]
    assert seen[0]["response_format"] == {"type": "json_object"}


def test_overloaded_model_switches_to_next_model_without_waiting(no_sleep):
    c, seen = openai_client({"openai/gpt-oss-120b": [FakeAPIError(503, "over capacity")],
                             "openai/gpt-oss-20b": [GOOD]})
    r = c.generate("s", [("user", "q")])
    assert r.model.endswith("gpt-oss-20b") and no_sleep == []


def test_overloaded_everywhere_gives_up_within_budget(no_sleep):
    err = [FakeAPIError(503, "over capacity")]
    c, seen = openai_client({m: err for m in L.PROVIDERS["groq"]["models"]}, time_budget_s=45)
    with pytest.raises(L.TransientLLMError, match="overloaded"):
        c.generate("s", [("user", "q")])
    assert len(seen) <= 6 and sum(no_sleep) < 45


def test_decommissioned_model_falls_back_and_discovers(no_sleep):
    gone = [FakeAPIError(400, "The model `x` has been decommissioned")]
    script = {m: gone for m in L.PROVIDERS["groq"]["models"]}
    script["qwen/qwen-new"] = [GOOD]
    c, seen = openai_client(script, models=["whisper-large-v3", "qwen/qwen-new"])
    assert c.generate("s", [("user", "q")]).model.endswith("qwen/qwen-new")


def test_unsupported_json_mode_and_temperature_are_dropped():
    c, seen = openai_client({"openai/gpt-oss-120b": [
        FakeAPIError(400, "response_format json_object is not supported"),
        FakeAPIError(400, "Unsupported value: temperature"),
        GOOD]})
    c.generate("s", [("user", "q")])
    assert "response_format" not in seen[-1] and "temperature" not in seen[-1]


def test_bad_key_is_not_retried():
    c, seen = openai_client({"openai/gpt-oss-120b": [FakeAPIError(401, "invalid api key")]})
    with pytest.raises(L.LLMError, match="rejected the API key"):
        c.generate("s", [("user", "q")])
    assert len(seen) == 1


def test_text_around_json_is_tolerated():
    c, _ = openai_client({"openai/gpt-oss-120b": ["Here you go:\n" + GOOD + "\nHope that helps"]})
    assert c.generate("s", [("user", "q")]).data["intent"] == "query"


def test_fallback_client_moves_to_second_provider_only_when_busy():
    busy = [FakeAPIError(503, "busy")]
    first, _ = openai_client({m: busy for m in L.PROVIDERS["groq"]["models"]}, time_budget_s=5)
    second, _ = openai_client({"openai/gpt-oss-120b:free": [GOOD]}, provider="openrouter")
    chain = L.FallbackClient([first, second])
    r = chain.generate("s", [("user", "q")])
    assert r.model.startswith("OpenRouter") and chain.active is second

    bad_key, _ = openai_client({"openai/gpt-oss-120b": [FakeAPIError(401)]})
    with pytest.raises(L.LLMError, match="API key"):   # config errors are shown, not hidden by fallback
        L.FallbackClient([bad_key, second]).generate("s", [("user", "q")])


def test_build_client_picks_providers_from_secrets():
    secrets = {"GROQ_API_KEY": "g", "GEMINI_API_KEY": "m"}
    get = lambda k, d=None: secrets.get(k, d)
    c = L.build_client(get)
    assert [x.provider for x in c.clients] == ["Gemini", "Groq"]
    secrets["LLM_PROVIDER"] = "groq"
    assert [x.provider for x in L.build_client(get).clients] == ["Groq"]
    with pytest.raises(L.LLMError, match="No API key"):
        L.build_client(lambda k, d=None: d)


def test_gemini_uses_v1beta_and_no_sdk_retries():
    c = L.GeminiClient("fake")
    opts = c.client._api_client._http_options
    assert opts.api_version == "v1beta" and opts.retry_options.attempts == 1


# --------------------------------------------------------------------------- #
# "Enter your own" provider addresses
# --------------------------------------------------------------------------- #
def fake_dns(monkeypatch, address):
    import socket
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, **kw: [(2, 1, 6, "", (address, port))])


def test_public_https_address_is_accepted_and_redirects_are_off(monkeypatch):
    fake_dns(monkeypatch, "104.18.2.161")
    assert L.check_public_https_url("https://api.together.xyz/v1/") == "https://api.together.xyz/v1"
    client = L.build_custom_client("sk-together-0123456789", "https://api.together.xyz/v1", "meta-llama/Llama-3",
                                   "Together")
    inner = client.clients[0]
    assert client.model == "Together · meta-llama/Llama-3" and inner.models == ["meta-llama/Llama-3"]
    assert str(inner.client.base_url).rstrip("/") == "https://api.together.xyz/v1"
    assert inner.client._client.follow_redirects is False
    with pytest.raises(L.LLMError, match="model name"):
        L.build_custom_client("sk-together-0123456789", "https://api.together.xyz/v1", "  ")


@pytest.mark.parametrize("address", ["127.0.0.1", "10.1.2.3", "192.168.0.10", "169.254.169.254", "::1", "fd00::1",
                                     "100.64.0.1"])
def test_a_public_name_that_resolves_to_a_private_address_is_refused(monkeypatch, address):
    fake_dns(monkeypatch, address)
    with pytest.raises(L.LLMError, match="private or local network"):
        L.check_public_https_url("https://looks-public.example.com/v1")
