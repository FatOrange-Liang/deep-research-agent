from __future__ import annotations

from dataclasses import dataclass

from deepresearch.llm import (
    BaseLLM,
    Message,
)

from .citations import (
    CitationValidation,
    validate_citations,
)

from .sources import Source
from .state import ResearchState


@dataclass(frozen=True)
class CitationGuardResult:
    """
    Final result produced by the citation guard.
    """

    answer: str

    validation: CitationValidation

    sources: dict[str, Source]

    passed: bool

    repaired: bool

    attempts: int


class CitationGuard:
    """
    Validate citations against structured research state.

    The guard may ask the LLM to repair malformed,
    missing, or hallucinated citation IDs.

    It never performs new research.
    """

    def __init__(
        self,
        *,
        llm: BaseLLM,
        max_repair_attempts: int = 2,
        require_citations_when_sources_exist: bool = True,
    ) -> None:

        if max_repair_attempts < 0:
            raise ValueError(
                "'max_repair_attempts' cannot be negative."
            )

        self.llm = llm

        self.max_repair_attempts = (
            max_repair_attempts
        )

        self.require_citations_when_sources_exist = (
            require_citations_when_sources_exist
        )

    def check(
        self,
        *,
        answer: str,
        state: ResearchState,
    ) -> CitationGuardResult:
        """
        Validate and, when necessary, repair a candidate answer.
        """

        sources = state.citation_sources

        validation = validate_citations(
            answer,
            sources,
        )

        if self._passes(
            validation,
            sources,
        ):
            return CitationGuardResult(
                answer=answer,
                validation=validation,
                sources=sources,
                passed=True,
                repaired=False,
                attempts=0,
            )

        current_answer = answer

        for attempt in range(
            1,
            self.max_repair_attempts + 1,
        ):

            repair_prompt = (
                self._build_repair_prompt(
                    answer=current_answer,
                    validation=validation,
                    sources=sources,
                )
            )

            response = self.llm.chat(
                messages=[
                    Message(
                        role="user",
                        content=repair_prompt,
                    )
                ],
                tools=None,
            )

            if response.has_tool_calls:
                raise RuntimeError(
                    "Citation repair must not "
                    "invoke tools."
                )

            if not response.content:
                raise RuntimeError(
                    "Citation repair returned "
                    "an empty answer."
                )

            current_answer = (
                response.content
            )

            validation = validate_citations(
                current_answer,
                sources,
            )

            if self._passes(
                validation,
                sources,
            ):
                return CitationGuardResult(
                    answer=current_answer,
                    validation=validation,
                    sources=sources,
                    passed=True,
                    repaired=True,
                    attempts=attempt,
                )

        return CitationGuardResult(
            answer=current_answer,
            validation=validation,
            sources=sources,
            passed=False,
            repaired=True,
            attempts=(
                self.max_repair_attempts
            ),
        )

    def _passes(
        self,
        validation: CitationValidation,
        sources: dict[str, Source],
    ) -> bool:

        if not validation.is_valid:
            return False

        if (
            self.require_citations_when_sources_exist
            and sources
            and not validation.cited_source_ids
        ):
            return False

        return True

    @staticmethod
    def _build_repair_prompt(
        *,
        answer: str,
        validation: CitationValidation,
        sources: dict[str, Source],
    ) -> str:

        source_blocks = []

        for source in sources.values():

            excerpt = (
                source.content[:600]
                .replace("\n", " ")
                .strip()
            )

            source_blocks.append(
                (
                    f"[{source.source_id}] "
                    f"{source.title}\n"
                    f"Evidence: {excerpt}"
                )
            )

        available_sources = (
            "\n\n".join(
                source_blocks
            )
        )

        problems = []

        if validation.invalid_source_ids:

            invalid = ", ".join(
                validation.invalid_source_ids
            )

            problems.append(
                "The previous answer contains "
                "source IDs that do not exist: "
                f"{invalid}."
            )

        if validation.malformed_source_ids:

            malformed = ", ".join(
                validation.malformed_source_ids
            )

            problems.append(
                "The previous answer contains "
                "malformed source IDs: "
                f"{malformed}."
            )

        if (
            sources
            and not validation.cited_source_ids
        ):

            problems.append(
                "The previous answer used web "
                "research but did not include "
                "valid citations."
            )

        problem_text = "\n".join(
            problems
        )

        return f"""
Your previous answer failed citation validation.

Previous answer:
---
{answer}
---

Problems:
{problem_text}

Available evidence:
{available_sources}

Revise the previous answer.

Rules:
1. Use only source IDs listed in the available evidence.
2. Copy source IDs exactly.
3. Never invent, shorten, or modify a source ID.
4. Place citations directly after the supported factual claim.
5. Do not invent URLs or Markdown citation links.
6. Do not perform new research.
7. Preserve useful content where it is supported.
8. Return only the revised final answer.
""".strip()