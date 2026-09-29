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
    WebPageReaderTool,
    WebSearchTool,
)

from deepresearch.research import (
    CitationGuard,
    ResearchState,
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

    registry.register(
        WebPageReaderTool()
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

    # task = (
    #     "请搜索网页获取最新资料，"
    #     "告诉我 LangGraph 主要是用来做什么的，"
    #     "并简要总结。"
    # )

    task = (
        "请搜索 LangGraph 的官方资料，"
        "打开并阅读至少一个最相关的官方网页，"
        "然后告诉我 LangGraph 的核心定位、"
        "状态持久化和 human-in-the-loop "
        "分别是怎么实现的。请提供引用。"
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

    research_state = (
        ResearchState.from_messages(
            task=task,
            messages=result.messages,
        )
    )

    citation_guard = CitationGuard(
        llm=llm,
        max_repair_attempts=2,
    )

    guard_result = citation_guard.check(
        answer=result.answer or "",
        state=research_state,
    )

    research_state = (
        result.research_state
    )

    if research_state is None:
        raise RuntimeError(
            "Agent did not return research state."
        )

    citation_guard = CitationGuard(
        llm=llm,
        max_repair_attempts=2,
    )

    guard_result = citation_guard.check(
        answer=result.answer or "",
        state=research_state,
    )


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

        for tool_call in step.tool_calls:

            print(
                f"Tool call: "
                f"{tool_call.name}"
            )

            print(
                f"Arguments: "
                f"{tool_call.arguments}"
            )

        for observation in step.observations:

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


    # ----------------------------------
    # Final answer
    # ----------------------------------

    print()
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(
        guard_result.answer
    )


    # ----------------------------------
    # Sources
    # ----------------------------------

    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in (
        guard_result.sources.values()
    ):

        print(
            f"[{source.source_id}] "
            f"{source.title}"
        )

        print(
            f"URL: {source.url}"
        )

        print()


    # ----------------------------------
    # Citation guard
    # ----------------------------------

    print("=" * 70)
    print("CITATION GUARD")
    print("=" * 70)

    print(
        "Passed:",
        guard_result.passed,
    )

    print(
        "Repaired:",
        guard_result.repaired,
    )

    print(
        "Repair attempts:",
        guard_result.attempts,
    )

    print(
        "Cited:",
        guard_result.validation.cited_source_ids,
    )

    print(
        "Invalid:",
        guard_result.validation.invalid_source_ids,
    )

    print(
        "Malformed:",
        guard_result
        .validation
        .malformed_source_ids,
    )

    print()
    print("=" * 70)
    print("RESEARCH STATE")
    print("=" * 70)

    print(
        "Search queries:",
        research_state.search_queries,
    )

    print(
        "Sources:",
        research_state.source_count,
    )

    print(
        "Full pages read:",
        research_state.read_source_count,
    )

    print(
        "Unread sources:",
        research_state.unread_source_count,
    )

    print(
        "Tool observations:",
        research_state.tool_observation_count,
    )


if __name__ == "__main__":
    main()