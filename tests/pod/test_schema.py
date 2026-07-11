"""The request contract and the response-schema pass-through."""

from __future__ import annotations

from pod.schema import RunRequest, Workspace, response_model_from_schema


def test_run_request_defaults_to_a_toolless_run_with_no_workspace():
    req = RunRequest(model="anthropic/claude-haiku-4-5", prompt="hi")
    assert req.tools == []  # empty = tool-less, not the engine's default tool set
    assert req.workspace is None
    assert req.response_schema is None
    assert req.reasoning_effort is None


def test_workspace_mode_is_read_only():
    ws = Workspace(source="file:///data/run-42")
    assert ws.mode == "ro"


def test_response_model_from_schema_returns_the_schema_verbatim():
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
    model = response_model_from_schema(schema)
    assert model.model_json_schema() == schema


def test_response_model_from_schema_round_trips_any_object_permissively():
    # The schema is opencode's contract, so validation keeps whatever the model returned rather than
    # re-checking it field by field — extras are preserved so the caller gets its object back.
    model = response_model_from_schema({"type": "object"})
    instance = model.model_validate({"count": 3, "note": "ok"})
    assert instance.model_dump(mode="json") == {"count": 3, "note": "ok"}
