from __future__ import annotations

import argparse
import json
import time

from deepresearch.agent import Agent
from deepresearch.config import load_settings
from deepresearch.llm import OpenAICompatibleLLM
from deepresearch.tools import (
    CalculatorTool,
    ToolRegistry,
    WebPageReaderTool,
    WebSearchTool,
)
from deepresearch.research import (
    CitationGuard,
    EvidenceCandidateRetriever,
    EvidencePolicy,
    EvidencePolicyConfig,
    HybridEvidenceRetriever,
    MultilingualE5Embedder,
    MultilingualEvidenceReranker,
    ResearchController,
    ResearchVerifier,
)


DEFAULT_TASK = (
    "请搜索 LangGraph 的官方资料，打开并阅读至少一个最相关的官方网页，"
    "然后告诉我 LangGraph 的核心定位、状态持久化和 human-in-the-loop "
    "分别是怎么实现的。请提供引用。"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real Deep Research Agent and compare "
            "evidence-retrieval modes during post-answer verification."
        )
    )

    parser.add_argument(
        "--retrieval",
        choices=(
            "lexical",
            "hybrid",
            "hybrid-rerank",
        ),
        default="lexical",
        help=(
            "Evidence retrieval used by ResearchVerifier. "
            "'hybrid-rerank' enables Lexical + E5 + RRF + BGE reranking."
        ),
    )

    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Research task sent to the Agent.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for E5/BGE verification models, e.g. cpu or cuda.",
    )

    parser.add_argument(
        "--e5-model",
        default="intfloat/multilingual-e5-small",
    )

    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
    )

    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=10,
        help="Hybrid RRF candidate depth before optional semantic reranking.",
    )

    parser.add_argument(
        "--output-top-k",
        type=int,
        default=3,
        help="Final evidence candidates retained per claim.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
    )

    return parser


def build_verifier(
    args: argparse.Namespace,
) -> ResearchVerifier:
    """
    Build the post-answer ResearchVerifier.

    Important:
    - Agent research itself is unchanged.
    - Expensive retrieval models are constructed only after
      the Agent has completed and CitationGuard has passed.
    - BGE weights remain lazy-loaded until reranking is actually used.
    """

    if args.retrieval == "lexical":
        return ResearchVerifier(
            retriever=EvidenceCandidateRetriever(
                top_k_per_claim=args.output_top_k,
            )
        )

    print()
    print(
        "Loading multilingual E5 for verification..."
    )

    embedder = MultilingualE5Embedder(
        model_name=args.e5_model,
        device=args.device,
    )

    reranker = None

    if args.retrieval == "hybrid-rerank":
        reranker = MultilingualEvidenceReranker(
            model_name=args.reranker_model,
            device=args.device,
            batch_size=8,
        )

    hybrid_retriever = HybridEvidenceRetriever(
        embedder=embedder,
        reranker=reranker,
        candidate_top_k=args.candidate_top_k,
        output_top_k=args.output_top_k,
        rrf_k=args.rrf_k,
    )

    return ResearchVerifier(
        retriever=hybrid_retriever,
    )


def print_candidate_details(
    *,
    index: int,
    candidate,
) -> None:
    print()
    print(
        f"Candidate {index}:"
    )

    print(
        "Type:",
        type(candidate).__name__,
    )

    print(
        "Source:",
        candidate.source_id,
    )

    if hasattr(
        candidate,
        "lexical_score",
    ):
        print(
            "Lexical score:",
            round(
                float(
                    candidate.lexical_score
                ),
                4,
            ),
        )

    if hasattr(
        candidate,
        "semantic_score",
    ):
        print(
            "Dense semantic score:",
            round(
                float(
                    candidate.semantic_score
                ),
                4,
            ),
        )

    if hasattr(
        candidate,
        "lexical_rank",
    ):
        print(
            "Lexical rank:",
            candidate.lexical_rank,
        )

    if hasattr(
        candidate,
        "dense_rank",
    ):
        print(
            "Dense rank:",
            candidate.dense_rank,
        )

    if hasattr(
        candidate,
        "rrf_rank",
    ):
        print(
            "RRF rank:",
            candidate.rrf_rank,
        )

    if hasattr(
        candidate,
        "rrf_score",
    ):
        print(
            "RRF score:",
            round(
                float(
                    candidate.rrf_score
                ),
                6,
            ),
        )

    if (
        hasattr(
            candidate,
            "reranker_score",
        )
        and candidate.reranker_score
        is not None
    ):
        print(
            "Reranker score:",
            round(
                float(
                    candidate.reranker_score
                ),
                6,
            ),
        )

    print(
        "Evidence quote:",
        candidate.quote[:500],
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.candidate_top_k < 1:
        parser.error("--candidate-top-k must be positive.")

    if args.output_top_k < 1:
        parser.error("--output-top-k must be positive.")

    if args.output_top_k > args.candidate_top_k:
        parser.error(
            "--output-top-k cannot exceed --candidate-top-k."
        )

    if args.rrf_k < 1:
        parser.error("--rrf-k must be positive.")

    if args.max_steps < 1:
        parser.error("--max-steps must be positive.")

    settings = load_settings()

    llm = OpenAICompatibleLLM(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        max_completion_tokens=2048,
    )

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
        WebPageReaderTool(
            timeout=30.0,
            max_chars=80_000,
            trust_env=False,
        )
    )

    print(
        "Registered tools:",
        registry.names(),
    )

    print(
        "Verification retrieval mode:",
        args.retrieval,
    )

    policy = EvidencePolicy(
        EvidencePolicyConfig(
            min_sources=3,
            min_full_pages=1,
            min_search_queries=1,
            max_recommended_reads=2,
            required_hosts=(
                "docs.langchain.com",
            ),
            min_required_host_full_pages=1,
        )
    )

    controller = ResearchController(
        policy=policy
    )

    agent = Agent(
        llm=llm,
        tools=registry,
        research_controller=controller,
        max_steps=args.max_steps,
    )

    task = args.task

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

    result = agent.run(task)

    research_state = result.research_state

    if research_state is None:
        raise RuntimeError(
            "Agent did not return research state."
        )

    print()
    print("=" * 70)
    print("AGENT TRAJECTORY")
    print("=" * 70)

    for step in result.steps:
        print()
        print(
            f"Step {step.step_number}"
        )

        if step.assistant_content:
            print(
                "Assistant:",
                step.assistant_content,
            )

        for tool_call in step.tool_calls:
            print(
                "Tool call:",
                tool_call.name,
            )
            print(
                "Arguments:",
                tool_call.arguments,
            )

        for observation in step.observations:
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

    if not result.answer or not result.answer.strip():
        print()
        print(
            "ERROR: Agent completed but returned "
            "an empty final answer."
        )
        return

    citation_guard = CitationGuard(
        llm=llm,
        max_repair_attempts=2,
    )

    guard_result = citation_guard.check(
        answer=result.answer,
        state=research_state,
    )

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
        print(
            "Candidate answer for debugging:"
        )
        print(
            guard_result.answer
        )

        return

    print()
    print("=" * 70)
    print("RESEARCH VERIFICATION REPORT")
    print("=" * 70)

    print(
        "Retrieval mode:",
        args.retrieval,
    )

    verification_report = None

    try:
        verification_started = (
            time.perf_counter()
        )

        verifier = build_verifier(
            args
        )

        verification_report = (
            verifier.verify(
                answer=guard_result.answer,
                state=research_state,
            )
        )

        verification_seconds = (
            time.perf_counter()
            - verification_started
        )

        print(
            "Verification time:",
            f"{verification_seconds:.2f}s",
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

            candidates = (
                report.candidates.get(
                    claim.claim_id,
                    (),
                )
            )

            if not candidates:
                print(
                    "Evidence candidates: NONE"
                )

            for index, candidate in enumerate(
                candidates[
                    :args.output_top_k
                ],
                start=1,
            ):
                print_candidate_details(
                    index=index,
                    candidate=candidate,
                )

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

        if args.retrieval == "hybrid-rerank":
            print(
                "Note: Reranker scores are semantic "
                "relevance scores, not entailment "
                "probabilities."
            )

    print()
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(
        guard_result.answer
    )


if __name__ == "__main__":
    main()
