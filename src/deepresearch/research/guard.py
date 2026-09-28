from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from deepresearch.llm import (
    BaseLLM,
    Message,
)

from .citations import (
    CitationValidation,
    collect_sources,
    validate_citations,
)

from .sources import Source


if TYPE_CHECKING:
    from deepresearch.agent.types import AgentResult


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
    Validate citations in a research answer and ask the LLM
    to repair the answer when citations are missing or invalid.

    The guard never performs new research. It may only use
    sources that already exist in the agent trajectory.
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
        result: AgentResult,
    ) -> CitationGuardResult:

        sources = collect_sources(
            result.messages
        )

        answer = result.answer or ""

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

        messages = list(
            result.messages
        )

        for attempt in range(
            1,
            self.max_repair_attempts + 1,
        ):

            repair_prompt = (
                self._build_repair_prompt(
                    validation=validation,
                    sources=sources,
                )
            )

            messages.append(
                Message(
                    role="user",
                    content=repair_prompt,
                )
            )

            response = self.llm.chat(
                messages=messages,
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

            answer = response.content

            messages.append(
                Message(
                    role="assistant",
                    content=answer,
                )
            )

            validation = (
                validate_citations(
                    answer,
                    sources,
                )
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
                    repaired=True,
                    attempts=attempt,
                )

        return CitationGuardResult(
            answer=answer,
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
        validation: CitationValidation,
        sources: dict[str, Source],
    ) -> str:

        source_lines = []

        for source in sources.values():

            source_lines.append(
                f"[{source.source_id}] "
                f"{source.title}"
            )

        available_sources = "\n".join(
            source_lines
        )

        problems = []

        if validation.invalid_source_ids:

            invalid = ", ".join(
                validation.invalid_source_ids
            )

            problems.append(
                "The previous answer contains "
                "invalid source IDs: "
                f"{invalid}."
            )

        if validation.malformed_source_ids:

            malformed = ", ".join(
                validation.malformed_source_ids
            )

            problems.append(
                "The previous answer contains "
                "malformed source IDs: "
                f"{malformed}. "
                "Source IDs must exactly match "
                "one of the allowed IDs."
            )

        if (
            sources
            and not validation.cited_source_ids
        ):

            problems.append(
                "The previous answer did not "
                "include citations even though "
                "web sources were used."
            )

        problem_text = "\n".join(
            problems
        )

        return f"""
Your previous answer failed citation validation.

Problems:
{problem_text}

Allowed source IDs:
{available_sources}

Revise the previous answer.

Rules:
1. Use only source IDs listed above.
2. Never invent a source ID.
3. Place citations directly after the supported factual claim.
4. Do not invent URLs or Markdown links.
5. Do not perform new research.
6. Preserve the useful content of the answer where possible.
7. Return only the revised final answer.
""".strip()