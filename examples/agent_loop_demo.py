from deepresearch.agent import Agent

from deepresearch.llm import (
    LLMResponse,
    MockLLM,
    ToolCall,
)

from deepresearch.tools import (
    CalculatorTool,
    ToolRegistry,
)


def main() -> None:

    registry = ToolRegistry()

    registry.register(
        CalculatorTool()
    )

    llm = MockLLM(
        responses=[
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call_001",
                        name="calculator",
                        arguments={
                            "expression":
                            "28374 * 928",
                        },
                    )
                ]
            ),
            LLMResponse(
                content=(
                    "28374 × 928 = 26331072"
                )
            ),
        ]
    )

    agent = Agent(
        llm=llm,
        tools=registry,
    )

    result = agent.run(
        "Calculate 28374 * 928"
    )

    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(result.answer)

    print()

    print("=" * 60)
    print("AGENT TRAJECTORY")
    print("=" * 60)

    for step in result.steps:

        print(
            f"\nStep {step.step_number}"
        )

        if step.assistant_content:

            print(
                "Assistant:",
                step.assistant_content,
            )

        for tool_call in step.tool_calls:

            print(
                f"Tool call: "
                f"{tool_call.name}"
            )

            print(
                f"Arguments: "
                f"{tool_call.arguments}"
            )

        for observation in (
            step.observations
        ):

            print(
                f"Observation: "
                f"{observation.content}"
            )

    print()

    print(
        "Stop reason:",
        result.stop_reason,
    )


if __name__ == "__main__":
    main()