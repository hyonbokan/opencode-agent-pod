"""The request contract for ``POST /agent/run`` and the response-schema pass-through.

The request is deliberately domain-free: a model, a prompt, a tool allow-list, an optional JSON
Schema for structured output, caps, and a workspace pointer. No field means anything domain-specific
(no "timerange", no "declaration") — that lives in the caller.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from llm import ReasoningEffort


class Workspace(BaseModel):
    """A pointer to the caller's external storage, read into an ephemeral working dir per run.

    ``mode`` is read-only: the pod reads the source, runs against a throwaway copy, and never writes
    back to the caller's storage.
    """

    source: str
    mode: Literal["ro"] = "ro"


class RunRequest(BaseModel):
    """One autonomous run: everything the pod needs arrives here; it holds nothing between requests."""

    model: str
    prompt: str
    system_prompt: str | None = None
    tools: list[str] = Field(default_factory=list)
    response_schema: dict[str, Any] | None = None
    reasoning_effort: ReasoningEffort | None = None
    max_budget_usd: float | None = None
    workspace: Workspace | None = None


def response_model_from_schema(schema: dict[str, Any]) -> type[BaseModel]:
    """Wrap a caller-supplied JSON Schema as a response model the engine can request.

    The engine asks a response model for ``model_json_schema()`` to build opencode's structured-output
    request and calls ``model_validate()`` on the reply. This pass-through returns the caller's schema
    verbatim and validates permissively (the schema is opencode's contract, not re-checked here), so
    any JSON Schema the caller sends round-trips without being modeled field by field.
    """

    class RequestedSchema(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return schema

    return RequestedSchema
