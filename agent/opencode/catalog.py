"""Known model ids per provider, for resolving a bare model id to its provider."""

from enum import StrEnum


class Provider(StrEnum):
    """Model providers a bare model id can resolve to."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROK = "grok"


# https://platform.openai.com/docs/models
class OpenAIModels:
    """OpenAI model name constants."""

    O3_MINI = "o3-mini"
    O3 = "o3-2025-04-16"
    O3_PRO = "o3-pro-2025-06-10"
    O4_MINI = "o4-mini-2025-04-16"
    GPT_4_1 = "gpt-4.1-2025-04-14"
    GPT_4_1_MINI = "gpt-4.1-mini-2025-04-14"
    GPT_4_1_NANO = "gpt-4.1-nano-2025-04-14"
    GPT_5 = "gpt-5-2025-08-07"
    GPT_5_MINI = "gpt-5-mini-2025-08-07"
    GPT_5_NANO = "gpt-5-nano-2025-08-07"
    GPT_5_1 = "gpt-5.1-2025-11-13"
    GPT_5_2 = "gpt-5.2-2025-12-11"
    GPT_5_4 = "gpt-5.4-2026-03-05"
    GPT_5_4_MINI = "gpt-5.4-mini-2026-03-17"
    GPT_5_4_NANO = "gpt-5.4-nano-2026-03-17"
    GPT_5_5 = "gpt-5.5-2026-04-23"


# https://docs.anthropic.com/en/docs/about-claude/models/overview
class AnthropicModels:
    """Anthropic model name constants."""

    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5"
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CLAUDE_OPUS_4_1 = "claude-opus-4-1-20250805"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5-20251001"
    CLAUDE_OPUS_4_5 = "claude-opus-4-5-20251101"
    CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"
    CLAUDE_OPUS_4_6 = "claude-opus-4-6"
    CLAUDE_OPUS_4_7 = "claude-opus-4-7"
    CLAUDE_OPUS_4_8 = "claude-opus-4-8"
    CLAUDE_SONNET_5 = "claude-sonnet-5"


# https://ai.google.dev/gemini-api/docs/models
class GeminiModels:
    """Google Gemini model name constants."""

    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_3_FLASH = "gemini-3-flash-preview"  # Preview
    GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite"
    GEMINI_3_1_PRO = "gemini-3.1-pro-preview"  # Preview
    GEMINI_3_5_FLASH = "gemini-3.5-flash"


# https://docs.x.ai/docs/models
class GrokModels:
    """xAI Grok model name constants."""

    GROK_4 = "grok-4-0709"
    GROK_4_1_FAST_REASONING = "grok-4-1-fast-reasoning"
    GROK_4_1_FAST_NON_REASONING = "grok-4-1-fast-non-reasoning"


def _get_model_constants(cls: type) -> list[str]:
    """Get all model constants from a model class."""
    return [
        getattr(cls, attr)
        for attr in dir(cls)
        if not attr.startswith("_") and isinstance(getattr(cls, attr), str)
    ]


# Supported models by provider
SUPPORTED_MODELS: dict[Provider, list[str]] = {
    Provider.OPENAI: _get_model_constants(OpenAIModels),
    Provider.ANTHROPIC: _get_model_constants(AnthropicModels),
    Provider.GEMINI: _get_model_constants(GeminiModels),
    Provider.GROK: _get_model_constants(GrokModels),
}


def get_provider_for_model(model: str) -> Provider | None:
    """Get the provider for a specific model."""
    for provider, models in SUPPORTED_MODELS.items():
        if model in models:
            return provider
    return None


def supports_xhigh_effort(model: str) -> bool:
    """Whether an OpenAI model accepts the custom ``xhigh`` reasoning effort; others cap at ``high``.

    Only gpt-5.4* and gpt-5.5* support it. The set is enumerated by family prefix and must be
    extended as new OpenAI families ship.
    """
    return model.startswith("gpt-5.4") or model.startswith("gpt-5.5")
