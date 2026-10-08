
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .authority import SourceAuthorityPolicy
from .state import ResearchState


EvidenceAction = Literal[
    "search_more",
    "read_more",
    "synthesize",
]


@dataclass(frozen=True)
class EvidencePolicyConfig:
    """
    Evidence sufficiency requirements.
    """

    min_sources: int = 3

    min_full_pages: int = 1

    min_search_queries: int = 1

    max_recommended_reads: int = 3

    # Empty means authority constraints are disabled.
    required_hosts: tuple[str, ...] = ()

    # Applied only when required_hosts is configured.
    min_required_host_full_pages: int = 1

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

        if self.min_required_host_full_pages < 1:
            raise ValueError(
                "'min_required_host_full_pages' "
                "must be at least 1."
            )

        if self.required_hosts:

            # Validate the configured hostnames.
            SourceAuthorityPolicy(
                required_hosts=self.required_hosts
            )


@dataclass(frozen=True)
class EvidenceDecision:

    action: EvidenceAction

    reasons: tuple[str, ...]

    recommended_source_ids: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:

        return self.action == "synthesize"


class EvidencePolicy:
    """
    Deterministic research evidence controller.

    Evaluation stages:

    1. Search quantity
    2. Required authoritative evidence
    3. General full-page evidence
    4. Synthesis permission
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

        # =====================================
        # Stage 1: Search quantity
        # =====================================

        search_reasons: list[str] = []

        if query_count < config.min_search_queries:

            search_reasons.append(
                (
                    f"Only {query_count} search queries "
                    "have been executed; at least "
                    f"{config.min_search_queries} "
                    "are required."
                )
            )

        if state.source_count < config.min_sources:

            search_reasons.append(
                (
                    f"Only {state.source_count} sources "
                    "have been discovered; at least "
                    f"{config.min_sources} "
                    "are required."
                )
            )

        if search_reasons:

            return EvidenceDecision(
                action="search_more",
                reasons=tuple(search_reasons),
            )

        # =====================================
        # Stage 2: Source authority requirement
        # =====================================

        if config.required_hosts:

            authority = SourceAuthorityPolicy(
                required_hosts=config.required_hosts
            )

            matching_records = [
                record
                for record in state.sources.values()
                if authority.matches(
                    record.source.url
                )
            ]

            authoritative_read_count = sum(
                1
                for record in matching_records
                if record.read_full_page
            )

            if (
                authoritative_read_count
                < config.min_required_host_full_pages
            ):

                # Only recommend matching sources that
                # have not been read in full.
                candidate_ids = {
                    record.source.source_id
                    for record in matching_records
                    if not record.read_full_page
                }

                recommended = (
                    self._recommend_unread_sources(
                        state,
                        allowed_source_ids=candidate_ids,
                    )
                )

                # No matching unread official source.
                # Search for more authoritative evidence.
                if not recommended:

                    return EvidenceDecision(
                        action="search_more",
                        reasons=(
                            (
                                "Required authoritative "
                                "full-page evidence is "
                                "insufficient. Search for "
                                "additional sources from: "
                                + ", ".join(
                                    authority.required_hosts
                                )
                            ),
                        ),
                    )

                # We have found suitable official sources,
                # but have not read enough of them.
                return EvidenceDecision(
                    action="read_more",
                    reasons=(
                        (
                            "Authoritative full-page evidence "
                            "is insufficient. "
                            f"Read {config.min_required_host_full_pages} "
                            "page(s) from the required hosts."
                        ),
                    ),
                    recommended_source_ids=recommended,
                )

        # =====================================
        # Stage 3: General full-page requirement
        # =====================================

        if (
            state.read_source_count
            < config.min_full_pages
        ):

            recommended = (
                self._recommend_unread_sources(
                    state
                )
            )

            if not recommended:

                return EvidenceDecision(
                    action="search_more",
                    reasons=(
                        (
                            "The full-page evidence target "
                            "has not been met, and there "
                            "are no unread sources remaining."
                        ),
                    ),
                )

            return EvidenceDecision(
                action="read_more",
                reasons=(
                    (
                        f"Only {state.read_source_count} "
                        "full pages have been read; "
                        f"at least {config.min_full_pages} "
                        "are required."
                    ),
                ),
                recommended_source_ids=recommended,
            )

        # =====================================
        # Stage 4: Evidence requirements satisfied
        # =====================================

        return EvidenceDecision(
            action="synthesize",
            reasons=(
                (
                    "All configured evidence "
                    "requirements have been met."
                ),
            ),
        )

    def _recommend_unread_sources(
        self,
        state: ResearchState,
        *,
        allowed_source_ids: set[str] | None = None,
    ) -> tuple[str, ...]:
        """
        Recommend unread sources ordered by search score.

        If allowed_source_ids is provided, recommendations
        are restricted to that subset.
        """

        unread_records = [
            record
            for record in state.sources.values()
            if not record.read_full_page
            and (
                allowed_source_ids is None
                or record.source.source_id
                in allowed_source_ids
            )
        ]

        def ranking_key(
            record,
        ) -> tuple[float, str]:

            score = (
                record.source.score
                if record.source.score is not None
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
            :self.config.max_recommended_reads
        ]

        return tuple(
            record.source.source_id
            for record in selected
        )
