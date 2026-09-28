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
    WebSearchTool,
)

from deepresearch.research import (
    collect_sources,
    validate_citations,
)

import json


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

    if not settings.tavily_api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not configured."
        )

    registry.register(
        WebSearchTool(
            api_key=settings.tavily_api_key,
        )
    )

    print(
        "Registered tools:",
        registry.names(),
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

    # task = (
    #     "你好，请用一句话介绍你自己。"
    # )

    task = (
        "请搜索网页获取最新资料，"
        "告诉我 LangGraph 主要是用来做什么的，"
        "并简要总结。"
    )

    print("=" * 70)
    print("TASK")
    print("=" * 70)

    print(task)

    print(
        json.dumps(
            registry.schemas(),
            ensure_ascii=False,
            indent=2,
        )
    )

    result = agent.run(task)

    sources = collect_sources(
        result.messages
    )

    validation = validate_citations(
        result.answer or "",
        sources,
    )

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

    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in sources.values():
        print(
            f"[{source.source_id}] "
            f"{source.title}"
        )
        print(
            f"URL: {source.url}"
        )
        print()


    print("=" * 70)
    print("CITATION VALIDATION")
    print("=" * 70)

    print(
        "Cited:",
        validation.cited_source_ids,
    )

    print(
        "Valid:",
        validation.valid_source_ids,
    )

    print(
        "Invalid:",
        validation.invalid_source_ids,
    )

    print(
        "Citation validation passed:",
        validation.is_valid,
    )


if __name__ == "__main__":
    main()