"""Type definitions for LLM providers and model capabilities."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel


class Provider(StrEnum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROK = "grok"


class Message(TypedDict, total=False):
    """A chat message with role and content."""

    role: Literal["system", "user", "assistant", "developer"]
    content: str | list[dict]


class Tool(BaseModel):
    """Tool definition for function calling.

    Matches the Anthropic tool format which is then converted per-provider.
    """

    name: str
    description: str
    input_schema: Any = None


class ToolCall(BaseModel):
    """Provider-agnostic tool-call request. `input` is always a parsed dict."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolHandler:
    """Pairs a Tool definition with its async handler. `timeout` is per-call."""

    tool: Tool
    handler: Callable[..., Awaitable[str]]
    timeout: float = 30.0


# Type alias for streaming callback
OnChunkCallback = Callable[[str], Awaitable[None]]


class VerboseLevel(StrEnum):
    """Verbosity levels for OpenAI Responses API text output.

    Matches OpenAI `text.verbosity`: low | medium | high.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ThinkingEffort(StrEnum):
    """Thinking effort levels for models that support configurable thinking.

    Maps to OpenAI/Grok ``reasoning.effort`` and Anthropic ``output_config.effort``.
    MAX maps to ``xhigh`` for OpenAI and ``max`` for Anthropic.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"
