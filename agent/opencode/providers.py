"""Translate our model identifiers and provider keys into what opencode expects.

opencode addresses models as ``provider/model`` and reads each provider's key from its own
environment variable. This module resolves a model id to its opencode form and copies the
matching key into the variable opencode looks for. New providers are added as a row here.
"""

from __future__ import annotations

import os
import re

from llm import ThinkingEffort
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


def to_opencode_variant(effort: ThinkingEffort | None, opencode_model: str) -> str | None:
    """Return the opencode ``--variant`` for a reasoning effort on a model, or None for the default.

    opencode selects reasoning effort through a per-model variant. Anthropic exposes only ``high``
    and ``max``, so lower efforts have no distinct variant and run at the model default. OpenAI,
    Google, and xAI expose ``low``/``medium``/``high`` directly, and MAX clamps to the strongest
    effort the model accepts — ``xhigh`` on the OpenAI models that offer it, otherwise ``high`` —
    because a variant a model does not recognize is silently ignored rather than rejected.
    """
    if effort is None:
        return None
    if provider_of(opencode_model) == "anthropic":
        # opencode has no low/medium variants for Anthropic models
        return {ThinkingEffort.HIGH: "high", ThinkingEffort.MAX: "max"}.get(effort)
    if effort is ThinkingEffort.MAX:
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


def build_daemon_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build the environment for an ``opencode serve`` daemon, applying every provider key alias.

    A single daemon serves any provider's model over its lifetime, so it copies each aliased key
    across when the variable opencode reads is unset but the source key is held (Gemini's
    GEMINI_API_KEY → GOOGLE_GENERATIVE_AI_API_KEY). A key absent everywhere is left absent, never
    blanked.
    """
    env = dict(base_env if base_env is not None else os.environ)
    for var, alias in _KEY_ALIAS.items():
        if not env.get(var) and env.get(alias):
            env[var] = env[alias]
    return env
