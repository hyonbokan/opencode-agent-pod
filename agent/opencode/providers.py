"""Translate our model identifiers and provider keys into what opencode expects.

opencode addresses models as ``provider/model`` and reads each provider's key from its own
environment variable. This module resolves a model id to its opencode form and copies the
matching key into the variable opencode looks for. New providers are added as a row here.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from llm import ReasoningEffort
from llm.models import get_provider_for_model, supports_xhigh_effort
from llm.types import Provider

# Our provider enum -> opencode's provider segment. Key injection is handled separately by
# build_daemon_env (via _KEY_ALIAS), so only the opencode name is needed here.
_OPENCODE_PROVIDER: dict[Provider, str] = {
    Provider.ANTHROPIC: "anthropic",
    Provider.OPENAI: "openai",
    Provider.GEMINI: "google",
    Provider.GROK: "xai",
}

# opencode's built-in (models.dev catalog) provider segments. A model on any other segment is a
# custom OpenAI-compatible provider declared in config; we don't know its reasoning-effort support.
_CATALOG_PROVIDERS: frozenset[str] = frozenset({"anthropic", "openai", "google", "xai"})

# Env var carrying a JSON array of custom OpenAI-compatible providers (local vLLM etc.), e.g.
#   AGENT_CUSTOM_PROVIDERS='[{"name":"vllm","base_url":"http://host:8000/v1","models":["llama-bgp"]}]'
_CUSTOM_PROVIDERS_ENV = "AGENT_CUSTOM_PROVIDERS"

# When this env var holds a running proxy's base URL, the daemon environment carries no real provider
# key: every provider is pointed at ``<proxy>/<provider>`` with a dummy key instead, so a shell
# command the daemon spawns cannot read one.
KEY_PROXY_ENV = "AGENT_KEY_PROXY_URL"

# The placeholder key the daemon and the commands it spawns see instead of a real one. It is
# worthless, but opencode still wants *a* key set so a provider isn't skipped as unconfigured.
DUMMY_PROXY_KEY = "opencode-agent-pod-proxied-key"

# opencode catalog provider segment -> the env var(s) that may carry its real key, most-specific
# first.
CATALOG_KEY_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY"),
    "xai": ("XAI_API_KEY",),
}

# Every env var that carries a real provider key — stripped from the daemon env in proxied mode.
_PROVIDER_KEY_VARS: tuple[str, ...] = tuple(
    dict.fromkeys(var for vars_ in CATALOG_KEY_VARS.values() for var in vars_)
)


class CustomProvider(BaseModel):
    """A custom OpenAI-compatible provider the pod can address — a model served at an arbitrary
    ``/v1`` base URL (a local vLLM endpoint), outside opencode's models.dev catalog.

    Its models are addressed as ``name/model``. The key is taken from ``api_key_env`` (an env var
    name, preferred so secrets stay in the environment) or the literal ``api_key``, if either is set.
    """

    name: str
    base_url: str
    models: list[str]
    api_key: str | None = None
    api_key_env: str | None = None
    npm: str = "@ai-sdk/openai-compatible"


def load_custom_providers(env: Mapping[str, str]) -> list[CustomProvider]:
    """Custom providers declared in the environment (AGENT_CUSTOM_PROVIDERS), empty if unset."""
    raw = env.get(_CUSTOM_PROVIDERS_ENV, "").strip()
    if not raw:
        return []
    return [CustomProvider.model_validate(entry) for entry in json.loads(raw)]


def build_provider_config(
    providers: list[CustomProvider], env: Mapping[str, str]
) -> dict[str, Any] | None:
    """The opencode ``{"provider": {...}}`` config that registers each custom provider, or None if
    there are none. Each provider's key is resolved from its env var (falling back to the literal)."""
    if not providers:
        return None
    provider_map: dict[str, Any] = {}
    for p in providers:
        options: dict[str, Any] = {"baseURL": p.base_url}
        key = env.get(p.api_key_env) if p.api_key_env else None
        key = key or p.api_key
        if key:
            options["apiKey"] = key
        provider_map[p.name] = {
            "npm": p.npm,
            "name": p.name,
            "options": options,
            "models": {model: {"name": model} for model in p.models},
        }
    return {"provider": provider_map}


# Providers whose key we hold under a different name than opencode reads it from. Only Gemini
# differs — we set GEMINI_API_KEY, opencode's Google client wants GOOGLE_GENERATIVE_AI_API_KEY;
# Anthropic/OpenAI/xAI already use the name opencode expects.
_KEY_ALIAS: dict[str, str] = {"GOOGLE_GENERATIVE_AI_API_KEY": "GEMINI_API_KEY"}

_OPENAI_SNAPSHOT_DATE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _strip_openai_snapshot(provider: str, model_id: str) -> str:
    """Drop the trailing ``-YYYY-MM-DD`` snapshot date from an OpenAI model id.

    opencode's OpenAI model list is keyed by the undated id, so the snapshot suffix is stripped;
    other providers are returned unchanged.
    """
    return _OPENAI_SNAPSHOT_DATE.sub("", model_id) if provider == "openai" else model_id


def to_opencode_variant(effort: ReasoningEffort | None, opencode_model: str) -> str | None:
    """Return the opencode ``--variant`` for a reasoning effort on a model, or None for the default.

    opencode selects reasoning effort through a per-model variant. Anthropic exposes only ``high``
    and ``max``, so lower efforts have no distinct variant and run at the model default. OpenAI,
    Google, and xAI expose ``low``/``medium``/``high`` directly, and MAX clamps to the strongest
    effort the model accepts — ``xhigh`` on the OpenAI models that offer it, otherwise ``high`` —
    because a variant a catalog model does not recognize is silently ignored rather than rejected.
    Custom (non-catalog) providers get no variant: we don't know their effort support, and a strict
    OpenAI-compatible endpoint may reject an unexpected parameter rather than ignore it.
    """
    if effort is None:
        return None
    if provider_of(opencode_model) not in _CATALOG_PROVIDERS:
        return None
    if provider_of(opencode_model) == "anthropic":
        # opencode has no low/medium variants for Anthropic models
        return {ReasoningEffort.HIGH: "high", ReasoningEffort.MAX: "max"}.get(effort)
    if effort is ReasoningEffort.MAX:
        model_id = opencode_model.split("/", 1)[-1]
        if provider_of(opencode_model) == "openai" and supports_xhigh_effort(model_id):
            return "xhigh"
        return "high"
    return effort.value


def to_opencode_model(model: str) -> str:
    """Return the ``provider/model`` id opencode should run for one of our model identifiers.

    An already-qualified id keeps its provider; a bare id is resolved to its provider, and one that
    no provider recognizes raises rather than defaulting to a provider that may not own it. Either
    way an OpenAI snapshot date is stripped so the id matches opencode's model list.
    """
    if "/" in model:
        provider_seg, _, model_id = model.partition("/")
        return f"{provider_seg}/{_strip_openai_snapshot(provider_seg, model_id)}"
    provider = get_provider_for_model(model)
    name = _OPENCODE_PROVIDER.get(provider) if provider is not None else None
    if name is None:
        raise ValueError(
            f"Cannot resolve a provider for model {model!r}; pass it as 'provider/model'."
        )
    return f"{name}/{_strip_openai_snapshot(name, model)}"


def provider_of(opencode_model: str) -> str:
    """The provider segment of an opencode ``provider/model`` id."""
    return opencode_model.split("/", 1)[0]


def _proxied_provider_config(env: Mapping[str, str], proxy_url: str) -> dict[str, Any] | None:
    """Build opencode provider config that points every keyed provider at the proxy with a dummy key.

    A catalog provider is included only when a real key for it is present; a custom provider is always
    included. Each entry overrides only the provider's base URL and key, so the daemon never holds a
    real one. Returns None when there is nothing to route.
    """
    base = proxy_url.rstrip("/")
    provider_map: dict[str, Any] = {}
    for seg, key_vars in CATALOG_KEY_VARS.items():
        if any(env.get(var) for var in key_vars):
            provider_map[seg] = {"options": {"baseURL": f"{base}/{seg}", "apiKey": DUMMY_PROXY_KEY}}
    for p in load_custom_providers(env):
        provider_map[p.name] = {
            "npm": p.npm,
            "name": p.name,
            "options": {"baseURL": f"{base}/{p.name}", "apiKey": DUMMY_PROXY_KEY},
            "models": {model: {"name": model} for model in p.models},
        }
    return {"provider": provider_map} if provider_map else None


def _build_proxied_daemon_env(env: dict[str, str], proxy_url: str) -> dict[str, str]:
    """Build the daemon environment with every real provider key removed and each provider routed
    through the proxy.

    A shell command the daemon spawns inherits this environment, so with the keys gone it cannot read
    one. An explicit OPENCODE_CONFIG_CONTENT is left untouched.
    """
    out = dict(env)
    for var in _PROVIDER_KEY_VARS:
        out.pop(var, None)
    if not out.get("OPENCODE_CONFIG_CONTENT"):
        config = _proxied_provider_config(env, proxy_url)
        if config is not None:
            out["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    return out


def build_daemon_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build the environment for an ``opencode serve`` daemon.

    When ``AGENT_KEY_PROXY_URL`` is set, every real provider key is stripped and each provider is
    pointed at that proxy with a dummy key, so no key reaches the daemon or the commands it spawns.
    Otherwise a single daemon serves any provider directly: each aliased key is copied across when the
    variable opencode reads is unset but the source key is held (GEMINI_API_KEY →
    GOOGLE_GENERATIVE_AI_API_KEY), a key absent everywhere is left absent rather than blanked, and any
    custom OpenAI-compatible providers declared in AGENT_CUSTOM_PROVIDERS are compiled into
    OPENCODE_CONFIG_CONTENT. An explicit OPENCODE_CONFIG_CONTENT is left untouched.
    """
    env = dict(base_env if base_env is not None else os.environ)
    proxy_url = env.get(KEY_PROXY_ENV)
    if proxy_url:
        return _build_proxied_daemon_env(env, proxy_url)
    for var, alias in _KEY_ALIAS.items():
        if not env.get(var) and env.get(alias):
            env[var] = env[alias]
    provider_config = build_provider_config(load_custom_providers(env), env)
    if provider_config is not None and not env.get("OPENCODE_CONFIG_CONTENT"):
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(provider_config)
    return env
