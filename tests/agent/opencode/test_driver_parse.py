"""Unit tests for build_timeline — the pure reconstruction of a per-step trace timeline from decoded
event dicts. No subprocess."""

from __future__ import annotations

from typing import Any

from agent.opencode.driver import build_timeline


def _line(obj: dict) -> dict:
    return obj


def test_build_timeline_groups_events_by_message_id():
    """The timeline groups events into steps by messageID, keeping first-seen order, and captures
    each step's timing, tokens, cost, text, and tool calls for trace recreation."""
    lines = [
        _line({"type": "step_start", "timestamp": 100, "part": {"messageID": "m1"}}),
        _line(
            {
                "type": "tool_use",
                "part": {
                    "messageID": "m1",
                    "tool": "opencode-agent_probe",
                    "state": {
                        "input": {"contract": "Vault"},
                        "output": "ok",
                        "time": {"start": 110, "end": 120},
                    },
                },
            }
        ),
        _line(
            {
                "type": "step_finish",
                "timestamp": 130,
                "part": {
                    "messageID": "m1",
                    "cost": 0.011,
                    "tokens": {
                        "total": 8578,
                        "input": 3,
                        "output": 59,
                        "reasoning": 0,
                        "cache": {"write": 8516, "read": 0},
                    },
                },
            }
        ),
        _line({"type": "step_start", "timestamp": 200, "part": {"messageID": "m2"}}),
        _line({"type": "text", "part": {"messageID": "m2", "text": "done"}}),
        _line(
            {"type": "step_finish", "timestamp": 230, "part": {"messageID": "m2", "cost": 0.001}}
        ),
    ]
    steps = build_timeline(lines)
    assert [s.message_id for s in steps] == ["m1", "m2"]

    s1 = steps[0]
    assert s1.start_ms == 100
    assert s1.end_ms == 130
    assert abs(s1.cost_usd - 0.011) < 1e-9
    assert s1.tokens == {
        "input": 3,
        "output": 59,
        "reasoning": 0,
        "cache_read": 0,
        "cache_write": 8516,
        "total": 8578,
    }
    assert len(s1.tools) == 1
    tool = s1.tools[0]
    assert tool.name == "opencode-agent_probe"
    assert tool.input == {"contract": "Vault"}
    assert tool.output == "ok"
    assert (tool.start_ms, tool.end_ms) == (110, 120)

    s2 = steps[1]
    assert s2.text == "done"
    assert s2.tools == []


def test_build_timeline_skips_events_without_message_id():
    """Events with no messageID (blank/malformed/unlinked) never open a phantom step."""
    lines: list[Any] = [
        "not json",
        _line({"type": "text", "part": {"text": "orphan"}}),
        _line({"type": "step_finish", "part": {"messageID": "m1", "cost": 0.02}}),
    ]
    steps = build_timeline(lines)
    assert [s.message_id for s in steps] == ["m1"]
    assert abs(steps[0].cost_usd - 0.02) < 1e-9
