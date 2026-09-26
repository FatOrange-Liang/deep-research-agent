from dataclasses import dataclass, field
from typing import Any, Literal


MessageRole = Literal[
    "system",
    "user",
    "assistant",
    "tool",
]


@dataclass
class ToolCall:
    """
    A structured request from the LLM to execute a tool.

    Example:
        ToolCall(
            id="call_001",
            name="calculator",
            arguments={"expression": "12 * 8"},
        )
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """
    Provider-independent chat message used inside the agent.

    The agent should work with this class rather than directly
    depending on OpenAI, Anthropic, or other provider formats.
    """

    role: MessageRole
    content: str | None = None

    # Used mainly by tool messages.
    name: str | None = None
    tool_call_id: str | None = None

    # Used when an assistant message requests one or more tools.
    tool_calls: list[ToolCall] = field(default_factory=list)

    def __post_init__(self) -> None:
        valid_roles = {
            "system",
            "user",
            "assistant",
            "tool",
        }

        if self.role not in valid_roles:
            raise ValueError(
                f"Invalid message role: {self.role}"
            )

        if self.role == "tool" and not self.tool_call_id:
            raise ValueError(
                "Tool messages must provide 'tool_call_id'."
            )


@dataclass
class LLMResponse:
    """
    Normalized response returned by every LLM backend.

    A response may contain:
    - normal text
    - tool calls
    - or both
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def is_final_answer(self) -> bool:
        return (
            self.content is not None
            and len(self.tool_calls) == 0
        )