from deepresearch.agent import Agent

from deepresearch.config import (
    load_settings,
)

from deepresearch.llm import (
    OpenAICompatibleLLM,
)

from deepresearch.tools import (
    CalculatorTool,
    ToolRegistry,
)


def main() -> None:

    # ----------------------------------
    # Configuration
    # ----------------------------------

    settings = load_settings()

    # ----------------------------------
    # Real LLM
    # ----------------------------------

    llm = OpenAICompatibleLLM(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    # ----------------------------------
    # Tools
    # ----------------------------------

    registry = ToolRegistry()

    registry.register(
        CalculatorTool()
    )

    # ----------------------------------
    # Agent
    # ----------------------------------

    agent = Agent(
        llm=llm,
        tools=registry,
        max_steps=8,
    )

    # ----------------------------------
    # User task
    # ----------------------------------

    # task = (
    #     "请准确计算 28374 乘以 928，"
    #     "并告诉我结果。"
    # )
    task = (
        "你好，请用一句话介绍你自己。"
    )

    print("=" * 70)
    print("TASK")
    print("=" * 70)

    print(task)

    result = agent.run(task)

    # ----------------------------------
    # Final answer
    # ----------------------------------

    print()
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(result.answer)

    # ----------------------------------
    # Trace
    # ----------------------------------

    print()
    print("=" * 70)
    print("AGENT TRAJECTORY")
    print("=" * 70)

    for step in result.steps:

        print(
            f"\nStep {step.step_number}"
        )

        if step.assistant_content:

            print(
                "Assistant:",
                step.assistant_content,
            )

        for tool_call in (
            step.tool_calls
        ):

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

            print(
                f"Error: "
                f"{observation.is_error}"
            )

    print()

    print(
        "Stop reason:",
        result.stop_reason,
    )


if __name__ == "__main__":
    main()