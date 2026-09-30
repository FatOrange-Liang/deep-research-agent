from __future__ import annotations

from dataclasses import dataclass

from .policy import (
    EvidenceDecision,
    EvidencePolicy,
)

from .state import ResearchState


@dataclass(frozen=True)
class ResearchControl:
    """
    Control signal returned by the research controller.
    """

    can_finish: bool

    decision: EvidenceDecision

    instruction: str | None = None


class ResearchController:
    """
    Translate deterministic evidence-policy decisions
    into actionable instructions for the research agent.
    """

    def __init__(
        self,
        policy: EvidencePolicy | None = None,
    ) -> None:

        self.policy = (
            policy
            or EvidencePolicy()
        )

    def evaluate(
        self,
        state: ResearchState,
    ) -> ResearchControl:

        decision = self.policy.evaluate(
            state
        )

        if decision.action == "synthesize":

            return ResearchControl(
                can_finish=True,
                decision=decision,
                instruction=None,
            )

        if decision.action == "search_more":

            return ResearchControl(
                can_finish=False,
                decision=decision,
                instruction=(
                    self._build_search_instruction(
                        decision
                    )
                ),
            )

        return ResearchControl(
            can_finish=False,
            decision=decision,
            instruction=(
                self._build_read_instruction(
                    state=state,
                    decision=decision,
                )
            ),
        )

    @staticmethod
    def _build_search_instruction(
        decision: EvidenceDecision,
    ) -> str:

        reasons = " ".join(
            decision.reasons
        )

        return (
            "Research is not complete yet. "
            "Do not provide the final answer. "
            "Use web_search to gather additional "
            "relevant evidence. "
            "Prefer authoritative or primary sources. "
            f"Reason: {reasons}"
        )

    @staticmethod
    def _build_read_instruction(
        *,
        state: ResearchState,
        decision: EvidenceDecision,
    ) -> str:

        source_lines = []

        for source_id in (
            decision.recommended_source_ids
        ):

            record = state.sources.get(
                source_id
            )

            if record is None:
                continue

            source_lines.append(
                (
                    f"- [{source_id}] "
                    f"{record.source.title}: "
                    f"{record.source.url}"
                )
            )

        sources_text = "\n".join(
            source_lines
        )

        return (
            "Research is not complete yet. "
            "Do not provide the final answer. "
            "Use web_page_reader to read one or more "
            "of the recommended sources before "
            "synthesizing the answer.\n\n"
            "Recommended sources:\n"
            f"{sources_text}"
        )