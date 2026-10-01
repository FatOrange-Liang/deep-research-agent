
from deepresearch.llm import (
    BaseLLM,
    Message,
    ToolCall,
)

from deepresearch.tools import ToolRegistry

from deepresearch.research.state import (
    ResearchState,
)

from deepresearch.research.controller import (
    ResearchController,
)

from .prompts import DEFAULT_SYSTEM_PROMPT

from .types import (
    AgentResult,
    AgentStep,
    ToolObservation,
)


class Agent:
    """
    Autonomous agent runtime with optional
    research completion control.

    The runtime repeatedly executes:

        LLM
         |
         v
      Tool Call
         |
         v
      Tool Execution
         |
         v
      Observation
         |
         v
    ResearchState
         |
         v
        LLM

    If a ResearchController is provided, the agent
    cannot finish until the configured evidence
    requirements have been satisfied.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 8,
        research_controller: ResearchController | None = None,
    ) -> None:

        if max_steps <= 0:
            raise ValueError(
                "'max_steps' must be greater than zero."
            )

        self.llm = llm

        self.tools = tools

        self.system_prompt = system_prompt

        self.max_steps = max_steps

        # Optional research completion gate.
        # None means normal Agent behavior.
        self.research_controller = research_controller

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

        # -----------------------------------------
        # Initialize research state
        # -----------------------------------------

        research_state = ResearchState(
            task=task
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

        # -----------------------------------------
        # Main Agent Loop
        # -----------------------------------------

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

                    observation_content, is_error = (
                        self._execute_tool_call(
                            tool_call
                        )
                    )

                    observation = ToolObservation(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content=observation_content,
                        is_error=is_error,
                    )

                    step.observations.append(
                        observation
                    )

                    tool_message = Message(
                        role="tool",
                        name=tool_call.name,
                        tool_call_id=tool_call.id,
                        content=observation_content,
                    )

                    # Maintain conversation history.
                    messages.append(
                        tool_message
                    )

                    # Maintain research state in real time.
                    # Failed tool calls must not become evidence.
                    if not is_error:

                        research_state.ingest_message(
                            tool_message
                        )

                steps.append(
                    step
                )

                continue

            # -----------------------------------------
            # Case 2:
            # LLM attempts to return a final answer
            # -----------------------------------------

            if response.content is not None:

                # -------------------------------------
                # Optional Research Completion Gate
                # -------------------------------------

                if self.research_controller is not None:

                    control = (
                        self.research_controller.evaluate(
                            research_state
                        )
                    )

                    # Evidence requirements are not met.
                    # Reject premature completion.
                    if not control.can_finish:

                        if not control.instruction:

                            raise RuntimeError(
                                "Research controller blocked "
                                "completion but returned "
                                "no instruction."
                            )

                        # Record this attempted answer as
                        # an actual Agent step.
                        steps.append(
                            step
                        )

                        # Feed controller guidance back
                        # into the next LLM iteration.
                        #
                        # Use a user-role feedback message
                        # rather than introducing another
                        # system message mid-conversation.
                        messages.append(
                            Message(
                                role="user",
                                content=(
                                    "Research controller feedback:\n\n"
                                    f"{control.instruction}\n\n"
                                    "Continue the research task. "
                                    "Do not repeat the rejected "
                                    "final answer."
                                ),
                            )
                        )

                        # Do not return AgentResult here.
                        # Continue to the next LLM step.
                        continue

                # -------------------------------------
                # Completion is allowed
                # -------------------------------------

                steps.append(
                    step
                )

                return AgentResult(
                    answer=response.content,
                    stop_reason="completed",
                    messages=messages,
                    steps=steps,
                    research_state=research_state,
                )

            # -----------------------------------------
            # Case 3:
            # Invalid empty LLM response
            # -----------------------------------------

            raise RuntimeError(
                "LLM returned neither content "
                "nor tool calls."
            )

        # -----------------------------------------
        # Maximum step limit reached
        # -----------------------------------------

        return AgentResult(
            answer=None,
            stop_reason="max_steps",
            messages=messages,
            steps=steps,
            research_state=research_state,
        )

    def _execute_tool_call(
        self,
        tool_call: ToolCall,
    ) -> tuple[str, bool]:
        """
        Execute one tool call.

        Tool failures are converted into observations
        instead of crashing the entire agent.
        """

        try:

            content = self.tools.execute(
                tool_call.name,
                **tool_call.arguments,
            )

            return content, False

        except Exception as exc:

            error_message = (
                f"TOOL_ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            return error_message, True
