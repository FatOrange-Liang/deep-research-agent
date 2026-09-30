from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .state import ResearchState


EvidenceAction = Literal[
    "search_more",
    "read_more",
    "synthesize",
]


@dataclass(frozen=True)
class EvidencePolicyConfig:
    """
    Thresholds used to decide whether enough evidence
    has been collected to synthesize an answer.
    """

    min_sources: int = 3

    min_full_pages: int = 1

    min_search_queries: int = 1

    max_recommended_reads: int = 3

    def __post_init__(self) -> None:

        if self.min_sources < 1:
            raise ValueError(
                "'min_sources' must be at least 1."
            )

        if self.min_full_pages < 0:
            raise ValueError(
                "'min_full_pages' cannot be negative."
            )

        if self.min_search_queries < 0:
            raise ValueError(
                "'min_search_queries' cannot be negative."
            )

        if self.max_recommended_reads < 1:
            raise ValueError(
                "'max_recommended_reads' must be at least 1."
            )


@dataclass(frozen=True)
class EvidenceDecision:
    """
    Decision returned by the evidence policy.
    """

    action: EvidenceAction

    reasons: tuple[str, ...]

    recommended_source_ids: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.action == "synthesize"


class EvidencePolicy:
    """
    Deterministic research-evidence controller.

    It decides whether the agent should:

    - search for more sources,
    - read discovered sources in more depth,
    - or synthesize the final answer.

    It does not generate queries or answers itself.
    """

    def __init__(
        self,
        config: EvidencePolicyConfig | None = None,
    ) -> None:

        self.config = (
            config
            or EvidencePolicyConfig()
        )

    def evaluate(
        self,
        state: ResearchState,
    ) -> EvidenceDecision:

        config = self.config

        query_count = len(
            state.search_queries
        )

        # -----------------------------------------
        # Stage 1:
        # Have we searched enough?
        # -----------------------------------------

        search_reasons: list[str] = []

        if (
            query_count
            < config.min_search_queries
        ):
            search_reasons.append(
                (
                    "Only "
                    f"{query_count} search queries "
                    "have been executed; "
                    f"at least "
                    f"{config.min_search_queries} "
                    "are required."
                )
            )

        if (
            state.source_count
            < config.min_sources
        ):
            search_reasons.append(
                (
                    "Only "
                    f"{state.source_count} sources "
                    "have been discovered; "
                    f"at least "
                    f"{config.min_sources} "
                    "are required."
                )
            )

        if search_reasons:

            return EvidenceDecision(
                action="search_more",
                reasons=tuple(
                    search_reasons
                ),
            )

        # -----------------------------------------
        # Stage 2:
        # Have we read enough sources deeply?
        # -----------------------------------------

        if (
            state.read_source_count
            < config.min_full_pages
        ):

            recommended = (
                self._recommend_unread_sources(
                    state
                )
            )

            # We need more full-page evidence,
            # but all current sources have already
            # been read. Discover more sources.
            if not recommended:

                return EvidenceDecision(
                    action="search_more",
                    reasons=(
                        (
                            "The full-page evidence "
                            "target has not been met, "
                            "and there are no unread "
                            "sources remaining."
                        ),
                    ),
                )

            return EvidenceDecision(
                action="read_more",
                reasons=(
                    (
                        "Only "
                        f"{state.read_source_count} "
                        "full pages have been read; "
                        f"at least "
                        f"{config.min_full_pages} "
                        "are required."
                    ),
                ),
                recommended_source_ids=(
                    recommended
                ),
            )

        # -----------------------------------------
        # Stage 3:
        # Evidence threshold satisfied.
        # -----------------------------------------

        return EvidenceDecision(
            action="synthesize",
            reasons=(
                (
                    "The configured evidence "
                    "requirements have been met."
                ),
            ),
        )

    def _recommend_unread_sources(
        self,
        state: ResearchState,
    ) -> tuple[str, ...]:

        unread_records = [
            record
            for record
            in state.sources.values()
            if not record.read_full_page
        ]

        def ranking_key(
            record,
        ) -> tuple[float, str]:

            score = (
                record.source.score
                if record.source.score
                is not None
                else -1.0
            )

            return (
                -score,
                record.source.source_id,
            )

        unread_records.sort(
            key=ranking_key
        )

        selected = unread_records[
            : self.config.max_recommended_reads
        ]

        return tuple(
            record.source.source_id
            for record in selected
        )