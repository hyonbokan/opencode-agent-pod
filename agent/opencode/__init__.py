"""opencode engine for the agent.

Drives the agentic loop through the opencode ``serve`` daemon behind the public OpencodeRunner /
OpencodeResult interface. The package holds the daemon lifecycle and pool, the HTTP session driver,
provider/model mapping, and the per-session permission ruleset; tools are served over an MCP server
and structured output is enforced by the request schema.
"""
