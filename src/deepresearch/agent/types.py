from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from deepresearch.llm import (
    Message,
    ToolCall,
)


if TYPE_CHECKING:
    from deepresearch.research.state import (
        ResearchState,
    )


StopReason = Literal[
    "completed",
    "max_steps",
]


@dataclass
class ToolObservation:
    """
    Result returned by one tool execution.
    """

    tool_call_id: str

    tool_name: str

    content: str

    is_error: bool = False


@dataclass
class AgentStep:
    """
    One reasoning / action step executed by the agent.
    """

    step_number: int

    assistant_content: str | None = None

    tool_calls: list[ToolCall] = field(
        default_factory=list
    )

    observations: list[ToolObservation] = field(
        default_factory=list
    )


@dataclass
class AgentResult:
    """
    Final result returned by the agent runtime.
    """

    answer: str | None

    stop_reason: StopReason

    messages: list[Message]

    steps: list[AgentStep]

    research_state: ResearchState | None = None

    @property
    def completed(self) -> bool:
        return (
            self.stop_reason
            == "completed"
        )