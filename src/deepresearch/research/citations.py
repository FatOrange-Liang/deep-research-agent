import json
import re
from dataclasses import dataclass

from deepresearch.llm import Message

from .sources import Source


CITATION_PATTERN = re.compile(
    r"\[(S_[0-9a-fA-F]{8})\]"
)


@dataclass(frozen=True)
class CitationValidation:
    cited_source_ids: list[str]
    valid_source_ids: list[str]
    invalid_source_ids: list[str]

    @property
    def is_valid(self) -> bool:
        return len(
            self.invalid_source_ids
        ) == 0


def collect_sources(
    messages: list[Message],
) -> dict[str, Source]:
    """
    Extract structured sources from web_search tool messages.
    """

    sources: dict[str, Source] = {}

    for message in messages:

        if message.role != "tool":
            continue

        if message.name != "web_search":
            continue

        if not message.content:
            continue

        try:
            data = json.loads(
                message.content
            )
        except json.JSONDecodeError:
            continue

        results = data.get(
            "results",
            [],
        )

        if not isinstance(
            results,
            list,
        ):
            continue

        for item in results:

            if not isinstance(
                item,
                dict,
            ):
                continue

            source_id = item.get(
                "source_id"
            )

            url = item.get(
                "url",
                "",
            )

            if (
                not isinstance(
                    source_id,
                    str,
                )
                or not source_id
                or not isinstance(
                    url,
                    str,
                )
                or not url
            ):
                continue

            score = item.get(
                "score"
            )

            if not isinstance(
                score,
                (int, float),
            ):
                score = None

            sources[source_id] = Source(
                source_id=source_id,
                title=str(
                    item.get(
                        "title",
                        "",
                    )
                ),
                url=url,
                content=str(
                    item.get(
                        "content",
                        "",
                    )
                ),
                score=score,
            )

    return sources


def extract_citation_ids(
    answer: str,
) -> list[str]:
    """
    Extract citation markers such as [S_1a2b3c4d].
    """

    matches = CITATION_PATTERN.findall(
        answer
    )

    # Preserve order while removing duplicates.
    return list(
        dict.fromkeys(matches)
    )


def validate_citations(
    answer: str,
    sources: dict[str, Source],
) -> CitationValidation:
    """
    Verify that every source ID cited by the LLM
    actually exists in collected tool evidence.
    """

    cited_ids = extract_citation_ids(
        answer
    )

    valid_ids = [
        source_id
        for source_id in cited_ids
        if source_id in sources
    ]

    invalid_ids = [
        source_id
        for source_id in cited_ids
        if source_id not in sources
    ]

    return CitationValidation(
        cited_source_ids=cited_ids,
        valid_source_ids=valid_ids,
        invalid_source_ids=invalid_ids,
    )