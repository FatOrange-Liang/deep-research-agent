from dataclasses import dataclass, field
from typing import Literal

from deepresearch.llm import Message, ToolCall


StopReason = Literal[
    "completed",
    "max_steps",
]


@dataclass
class ToolObservation:
    """
    Result produced after executing one tool call.
    """

    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False


@dataclass
class AgentStep:
    """
    One reasoning/action step in an agent trajectory.
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
    Final result returned by an agent run.
    """

    answer: str | None

    stop_reason: StopReason

    messages: list[Message]

    steps: list[AgentStep]

    @property
    def completed(self) -> bool:
        return self.stop_reason == "completed"