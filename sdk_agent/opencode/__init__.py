"""opencode engine for the SDK agent.

Drives the agentic loop through the opencode ``serve`` daemon instead of the Anthropic Claude Agent
SDK, while preserving the public SDKAgentRunner / SDKAgentResult interface. The package holds the
daemon lifecycle and pool, the HTTP session driver, provider/model mapping, and the per-session
permission ruleset; analysis tools are served over the Python MCP server and structured output is
enforced by the request schema, so no plugin source is generated.
"""
