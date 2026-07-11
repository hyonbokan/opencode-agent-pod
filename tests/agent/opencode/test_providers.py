"""Unit tests for opencode provider/model mapping and daemon environment construction.
No opencode invocation, no network — pure mapping."""

from __future__ import annotations

import json

import pytest

from agent.opencode.providers import (
    CustomProvider,
    build_daemon_env,
    build_provider_config,
    load_custom_providers,
    provider_of,
    to_opencode_model,
    to_opencode_variant,
)
from llm import ReasoningEffort
from llm.models import AnthropicModels, GeminiModels, GrokModels, OpenAIModels


@pytest.mark.parametrize(
    "given,expected",
    [
        # Bare id -> provider resolved from the registry, mapped to opencode's segment. Gemini
        # and Grok exercise the segment remap (enum says gemini/grok, opencode wants google/xai).
        (AnthropicModels.CLAUDE_SONNET_4_6, f"anthropic/{AnthropicModels.CLAUDE_SONNET_4_6}"),
        (GeminiModels.GEMINI_3_1_PRO, f"google/{GeminiModels.GEMINI_3_1_PRO}"),
        (GrokModels.GROK_4_1_FAST_REASONING, f"xai/{GrokModels.GROK_4_1_FAST_REASONING}"),
        # OpenAI dated snapshots are stripped to the base id opencode's registry knows.
        (OpenAIModels.GPT_5_NANO, "openai/gpt-5-nano"),
        (OpenAIModels.GPT_5_5, "openai/gpt-5.5"),
        ("gpt-5.4-mini-2026-03-17", "openai/gpt-5.4-mini"),
        # Anthropic's compact -YYYYMMDD date is a different shape: left intact (registered as-is).
        (AnthropicModels.CLAUDE_HAIKU_4_5, f"anthropic/{AnthropicModels.CLAUDE_HAIKU_4_5}"),
        # Already-qualified ids pass through untouched (the escape hatch for unlisted models).
        ("anthropic/claude-haiku-4-5", "anthropic/claude-haiku-4-5"),
        ("openrouter/some-new-model", "openrouter/some-new-model"),
    ],
)
def test_to_opencode_model(given, expected):
    assert to_opencode_model(given) == expected


def test_to_opencode_model_rejects_unknown_bare_id():
    """A bare id no provider owns must fail fast, not silently route to a default provider."""
    with pytest.raises(ValueError):
        to_opencode_model("totally-made-up-model")


def test_provider_of():
    assert provider_of("anthropic/claude-sonnet-4-6") == "anthropic"
    assert provider_of("openai/gpt-5") == "openai"


def test_build_daemon_env_aliases_every_provider_key():
    # A daemon serves any provider, so the Gemini alias is applied regardless of which model runs.
    env = build_daemon_env({"GEMINI_API_KEY": "gk", "OPENAI_API_KEY": "ok"})
    assert env["GOOGLE_GENERATIVE_AI_API_KEY"] == "gk"
    assert env["OPENAI_API_KEY"] == "ok"


def test_build_daemon_env_prefers_direct_key_and_invents_nothing():
    env = build_daemon_env({"GOOGLE_GENERATIVE_AI_API_KEY": "direct", "GEMINI_API_KEY": "source"})
    assert env["GOOGLE_GENERATIVE_AI_API_KEY"] == "direct"  # direct key wins, not overwritten
    # No source anywhere -> the aliased var is not conjured into existence.
    assert "GOOGLE_GENERATIVE_AI_API_KEY" not in build_daemon_env({"OPENAI_API_KEY": "ok"})


@pytest.mark.parametrize(
    "effort,model,expected",
    [
        # Anthropic exposes only high/max; lower efforts have no variant and run at the default.
        (ReasoningEffort.LOW, "anthropic/claude-haiku-4-5", None),
        (ReasoningEffort.MEDIUM, "anthropic/claude-haiku-4-5", None),
        (ReasoningEffort.HIGH, "anthropic/claude-haiku-4-5", "high"),
        (ReasoningEffort.MAX, "anthropic/claude-haiku-4-5", "max"),
        # OpenAI exposes low/medium/high directly; MAX clamps to xhigh only where the model offers
        # it (gpt-5.4*/gpt-5.5*), otherwise high — sending an unsupported variant is silently ignored.
        (ReasoningEffort.LOW, "openai/gpt-5-nano", "low"),
        (ReasoningEffort.MEDIUM, "openai/gpt-5-nano", "medium"),
        (ReasoningEffort.HIGH, "openai/gpt-5-nano", "high"),
        (ReasoningEffort.MAX, "openai/gpt-5-nano", "high"),
        (ReasoningEffort.MAX, "openai/gpt-5.5", "xhigh"),
        (ReasoningEffort.MAX, "openai/gpt-5.4-mini", "xhigh"),
        # Google exposes low/medium/high; it has no xhigh/max, so MAX clamps to high.
        (ReasoningEffort.MEDIUM, "google/gemini-3.1-pro", "medium"),
        (ReasoningEffort.MAX, "google/gemini-3.1-pro", "high"),
        # xAI has no xhigh, so MAX clamps to high.
        (ReasoningEffort.MAX, "xai/grok-4-1-fast-reasoning", "high"),
        # No effort requested -> no variant, on any provider.
        (None, "openai/gpt-5-nano", None),
        (None, "anthropic/claude-haiku-4-5", None),
        # Custom (non-catalog) providers get no variant regardless of effort — we don't know their
        # effort support, and a strict OpenAI-compatible endpoint may reject an unexpected param.
        (ReasoningEffort.MEDIUM, "vllm/llama-bgp", None),
        (ReasoningEffort.MAX, "vllm/llama-bgp", None),
    ],
)
def test_to_opencode_variant(effort, model, expected):
    assert to_opencode_variant(effort, model) == expected


def test_load_custom_providers_parses_env_json_and_defaults_empty():
    assert load_custom_providers({}) == []
    assert load_custom_providers({"AGENT_CUSTOM_PROVIDERS": "  "}) == []
    providers = load_custom_providers(
        {
            "AGENT_CUSTOM_PROVIDERS": json.dumps(
                [{"name": "vllm", "base_url": "http://h:8000/v1", "models": ["llama-bgp"]}]
            )
        }
    )
    assert len(providers) == 1
    assert providers[0].name == "vllm"
    assert providers[0].npm == "@ai-sdk/openai-compatible"  # default


def test_build_provider_config_shapes_opencode_config_and_resolves_key_from_env():
    providers = [
        CustomProvider(
            name="vllm",
            base_url="http://h:8000/v1",
            models=["llama-bgp", "gemma"],
            api_key_env="VLLM_KEY",
        )
    ]
    config = build_provider_config(providers, {"VLLM_KEY": "secret"})
    assert config is not None
    entry = config["provider"]["vllm"]
    assert entry["npm"] == "@ai-sdk/openai-compatible"
    assert entry["options"] == {"baseURL": "http://h:8000/v1", "apiKey": "secret"}
    assert set(entry["models"]) == {"llama-bgp", "gemma"}


def test_build_provider_config_omits_apikey_when_none_and_returns_none_when_empty():
    assert build_provider_config([], {}) is None
    # No env var value and no literal -> no apiKey emitted (a keyless local endpoint is valid).
    config = build_provider_config(
        [CustomProvider(name="vllm", base_url="http://h/v1", models=["m"], api_key_env="MISSING")],
        {},
    )
    assert config is not None
    assert "apiKey" not in config["provider"]["vllm"]["options"]


def test_build_daemon_env_injects_custom_provider_config():
    env = build_daemon_env(
        {
            "AGENT_CUSTOM_PROVIDERS": json.dumps(
                [{"name": "vllm", "base_url": "http://h:8000/v1", "models": ["llama-bgp"]}]
            )
        }
    )
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert config["provider"]["vllm"]["options"]["baseURL"] == "http://h:8000/v1"
    assert "llama-bgp" in config["provider"]["vllm"]["models"]


def test_build_daemon_env_leaves_config_content_alone_when_no_custom_providers_or_when_explicit():
    # No custom providers declared -> the daemon config content is not conjured into existence.
    assert "OPENCODE_CONFIG_CONTENT" not in build_daemon_env({"OPENAI_API_KEY": "ok"})
    # An explicit config content is authoritative and never overwritten by the compiled one.
    env = build_daemon_env(
        {
            "OPENCODE_CONFIG_CONTENT": '{"explicit": true}',
            "AGENT_CUSTOM_PROVIDERS": json.dumps(
                [{"name": "vllm", "base_url": "http://h/v1", "models": ["m"]}]
            ),
        }
    )
    assert env["OPENCODE_CONFIG_CONTENT"] == '{"explicit": true}'
