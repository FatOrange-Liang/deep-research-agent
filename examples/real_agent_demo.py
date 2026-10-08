
import json

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
    EvidencePolicy,
    EvidencePolicyConfig,
    ResearchController,
    ResearchVerifier,
)


def main() -> None:

    # =========================================
    # 1. Configuration
    # =========================================

    settings = load_settings()

    # =========================================
    # 2. Real LLM
    # =========================================

    llm = OpenAICompatibleLLM(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        max_completion_tokens=2048,
    )

    # =========================================
    # 3. Tool Registry
    # =========================================

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

    # =========================================
    # 4. Deep Research Evidence Policy
    # =========================================

    policy = EvidencePolicy(
        EvidencePolicyConfig(
            min_sources=3,
            min_full_pages=1,
            min_search_queries=1,
            max_recommended_reads=2,

            # Source Authority Policy
            required_hosts=(
                "docs.langchain.com",
            ),

            min_required_host_full_pages=1,
        )
    )

    controller = ResearchController(
        policy=policy
    )

    # =========================================
    # 5. Initialize Research Agent
    # =========================================

    agent = Agent(
        llm=llm,
        tools=registry,
        research_controller=controller,
        max_steps=12,
    )

    # =========================================
    # 6. Research Task
    # =========================================

    task = (
        "请搜索 LangGraph 的官方资料，"
        "打开并阅读至少一个最相关的官方网页，"
        "然后告诉我 LangGraph 的核心定位、"
        "状态持久化和 human-in-the-loop "
        "分别是怎么实现的。请提供引用。"
    )

    print()
    print("=" * 70)
    print("TASK")
    print("=" * 70)

    print(task)

    print()
    print("=" * 70)
    print("AVAILABLE TOOLS")
    print("=" * 70)

    print(
        json.dumps(
            registry.schemas(),
            ensure_ascii=False,
            indent=2,
        )
    )

    # =========================================
    # 7. Execute Agent
    # =========================================

    # This was missing in the previous version.
    # Agent.run() returns AgentResult.

    result = agent.run(task)

    # ResearchState is now maintained by
    # Agent Runtime during tool execution.
    #
    # Do not reconstruct it from messages.

    research_state = result.research_state

    if research_state is None:
        raise RuntimeError(
            "Agent did not return research state."
        )

    # =========================================
    # 8. Agent Trajectory
    # =========================================

    print()
    print("=" * 70)
    print("AGENT TRAJECTORY")
    print("=" * 70)

    for step in result.steps:

        print()
        print(
            f"Step {step.step_number}"
        )

        # -------------------------------------
        # Assistant output
        # -------------------------------------

        if step.assistant_content:

            print(
                "Assistant:",
                step.assistant_content,
            )

        # -------------------------------------
        # Tool calls
        # -------------------------------------

        for tool_call in step.tool_calls:

            print(
                "Tool call:",
                tool_call.name,
            )

            print(
                "Arguments:",
                tool_call.arguments,
            )

        # -------------------------------------
        # Tool observations
        # -------------------------------------

        for observation in step.observations:

            # Limit terminal log length.
            # This does NOT modify the actual
            # evidence stored in ResearchState.

            content_preview = (
                observation.content
            )

            if len(content_preview) > 1500:

                content_preview = (
                    content_preview[:1500]
                    + "\n...[LOG TRUNCATED]"
                )

            print(
                "Observation:",
                content_preview,
            )

            print(
                "Error:",
                observation.is_error,
            )

    print()
    print(
        "Stop reason:",
        result.stop_reason,
    )

    # =========================================
    # 9. Research State Summary
    # =========================================

    print()
    print("=" * 70)
    print("RESEARCH STATE")
    print("=" * 70)

    print(
        "Search queries:",
        research_state.search_queries,
    )

    print(
        "Sources discovered:",
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

    # =========================================
    # 10. Research Completion Check
    # =========================================

    # If max_steps was reached before the
    # evidence requirements were satisfied,
    # do not publish an incomplete final answer.

    if not result.completed:

        print()
        print("=" * 70)
        print("RESEARCH INCOMPLETE")
        print("=" * 70)

        print(
            "The agent stopped before completing "
            "the research task."
        )

        print(
            "Stop reason:",
            result.stop_reason,
        )

        return

    # Guard against an empty final answer.

    if not result.answer or not result.answer.strip():

        print()
        print(
            "ERROR: Agent completed but returned "
            "an empty final answer."
        )

        return

    # =========================================
    # 11. Citation Guard
    # =========================================

    # Only execute citation validation after
    # research completion has been confirmed.

    citation_guard = CitationGuard(
        llm=llm,
        max_repair_attempts=2,
    )

    guard_result = citation_guard.check(
        answer=result.answer,
        state=research_state,
    )

    # =========================================
    # 12. Citation Validation Report
    # =========================================

    print()
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
        guard_result.validation.malformed_source_ids,
    )

    # =========================================
    # 13. Sources
    # =========================================

    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in guard_result.sources.values():

        print(
            f"[{source.source_id}] "
            f"{source.title}"
        )

        print(
            f"URL: {source.url}"
        )

        print()

    # =========================================
    # 14. Citation Gate
    # =========================================

    # A failed citation guard means the answer
    # is not ready to be presented as validated.

    if not guard_result.passed:

        print()
        print("=" * 70)
        print("CITATION VALIDATION FAILED")
        print("=" * 70)

        print(
            "The candidate answer did not pass "
            "citation validation."
        )

        print()
        print("Candidate answer for debugging:")
        print(
            guard_result.answer
        )

        return

    
    # =========================================
    # Research Verification Pipeline
    # =========================================

    print()
    print("=" * 70)
    print("RESEARCH VERIFICATION REPORT")
    print("=" * 70)

    # Important:
    # This is report-only verification.
    #
    # Structural verification is diagnostic.
    # It does not block publication of an answer
    # that already passed CitationGuard.
    #
    # Verify the repaired answer, not the
    # original pre-repair Agent output.

    verification_report = None

    try:

        verifier = ResearchVerifier()

        verification_report = verifier.verify(
            answer=guard_result.answer,
            state=research_state,
        )

    except Exception as exc:

        print(
            "Verification pipeline error:",
            f"{type(exc).__name__}: {exc}",
        )

        print(
            "Structural verification is unavailable. "
            "This does not imply that the answer "
            "has passed structural verification."
        )

    if verification_report is not None:

        report = verification_report

        print()
        print(
            "Citation IDs valid:",
            report.citations_valid,
        )

        print(
            "Extracted claims:",
            report.claim_count,
        )

        print(
            "Evidence candidates:",
            report.candidate_count,
        )

        print(
            "Citation coverage:",
            f"{report.citation_coverage:.2%}",
        )

        print(
            "Anchor coverage:",
            f"{report.anchor_coverage:.2%}",
        )

        print(
            "Structural checks passed:",
            report.structural_checks_passed,
        )

        print(
            "Claims without retrieved evidence:",
            report.missing_candidate_claim_ids,
        )

        # -------------------------------------
        # Per-claim inspection
        # -------------------------------------

        print()
        print("-" * 70)
        print("CLAIM-LEVEL DETAILS")
        print("-" * 70)

        for entry in report.manifest.entries:

            claim = entry.claim

            print()
            print(
                f"[{claim.claim_id}] "
                f"{claim.text}"
            )

            print(
                "Citations:",
                claim.cited_source_ids,
            )

            print(
                "Anchored:",
                entry.anchored,
            )

            # Show the evidence candidates retrieved
            # from the already stored source text.

            candidates = report.candidates.get(
                claim.claim_id,
                (),
            )

            if not candidates:

                print(
                    "Evidence candidates: NONE"
                )

            for index, candidate in enumerate(
                candidates[:2],
                start=1,
            ):

                print()
                print(
                    f"Candidate {index}:"
                )

                print(
                    "Source:",
                    candidate.source_id,
                )

                print(
                    "Lexical score:",
                    round(
                        candidate.lexical_score,
                        4,
                    ),
                )

                print(
                    "Evidence quote:",
                    candidate.quote[:350],
                )

            # Print the deterministic anchor
            # verification result.

            for check in entry.anchor_checks:

                print(
                    "Anchor status:",
                    check.status,
                )

        print()
        print("-" * 70)

        print(
            "Note: Anchor coverage measures "
            "traceable text matches, not "
            "semantic factual correctness."
        )


    # =========================================
    # 15. Final Answer
    # =========================================

    print()
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(
        guard_result.answer
    )


if __name__ == "__main__":
    main()
