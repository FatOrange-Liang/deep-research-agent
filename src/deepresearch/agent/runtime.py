from deepresearch.llm import (
    BaseLLM,
    Message,
)

from deepresearch.tools import ToolRegistry

from .prompts import DEFAULT_SYSTEM_PROMPT
from .types import (
    AgentResult,
    AgentStep,
    ToolObservation,
)


class Agent:
    """
    Minimal autonomous agent runtime.

    The runtime repeatedly:

        LLM
         ↓
      Tool Call
         ↓
      Tool Execution
         ↓
      Observation
         ↓
        LLM

    until the model returns a final answer or the maximum
    number of steps is reached.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 8,
    ) -> None:

        if max_steps <= 0:
            raise ValueError(
                "'max_steps' must be greater than zero."
            )

        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(
        self,
        task: str,
    ) -> AgentResult:

        if not isinstance(task, str):
            raise TypeError(
                "'task' must be a string."
            )

        task = task.strip()

        if not task:
            raise ValueError(
                "'task' cannot be empty."
            )

        messages = [
            Message(
                role="system",
                content=self.system_prompt,
            ),
            Message(
                role="user",
                content=task,
            ),
        ]

        steps: list[AgentStep] = []

        for step_number in range(
            1,
            self.max_steps + 1,
        ):

            response = self.llm.chat(
                messages=messages,
                tools=self.tools.schemas(),
            )

            assistant_message = Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            )

            messages.append(
                assistant_message
            )

            step = AgentStep(
                step_number=step_number,
                assistant_content=response.content,
                tool_calls=list(
                    response.tool_calls
                ),
            )

            # -----------------------------------------
            # Case 1:
            # LLM requests one or more tool calls
            # -----------------------------------------

            if response.has_tool_calls:

                for tool_call in response.tool_calls:

                    observation = (
                        self._execute_tool_call(
                            tool_call.name,
                            tool_call.arguments,
                        )
                    )

                    tool_observation = (
                        ToolObservation(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            content=observation[0],
                            is_error=observation[1],
                        )
                    )

                    step.observations.append(
                        tool_observation
                    )

                    messages.append(
                        Message(
                            role="tool",
                            content=observation[0],
                            name=tool_call.name,
                            tool_call_id=tool_call.id,
                        )
                    )

                steps.append(step)

                continue

            # -----------------------------------------
            # Case 2:
            # LLM returns final answer
            # -----------------------------------------

            if response.content is not None:

                steps.append(step)

                return AgentResult(
                    answer=response.content,
                    stop_reason="completed",
                    messages=messages,
                    steps=steps,
                )

            # -----------------------------------------
            # Case 3:
            # Invalid empty response
            # -----------------------------------------

            raise RuntimeError(
                "LLM returned neither content "
                "nor tool calls."
            )

        return AgentResult(
            answer=None,
            stop_reason="max_steps",
            messages=messages,
            steps=steps,
        )

    def _execute_tool_call(
        self,
        tool_name: str,
        arguments: dict,
    ) -> tuple[str, bool]:
        """
        Execute one tool call.

        Tool failures are converted into observations rather
        than crashing the whole agent runtime.

        Returns:
            (content, is_error)
        """

        try:

            result = self.tools.execute(
                tool_name,
                **arguments,
            )

            return result, False

        except Exception as exc:

            error_message = (
                f"TOOL_ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            return error_message, True