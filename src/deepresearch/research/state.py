from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

from deepresearch.llm import Message

from .sources import Source


@dataclass
class EvidenceRecord:
    """
    One research source together with its provenance state.

    A source may first be discovered through web search and
    later enriched with full-page content.
    """

    source: Source

    discovered_by_search: bool = False

    read_full_page: bool = False

    search_queries: list[str] = field(
        default_factory=list
    )

    content_type: str | None = None

    truncated: bool | None = None


@dataclass
class ResearchState:
    """
    Structured state accumulated during a research task.

    This separates research evidence from the raw LLM
    conversation history.
    """

    task: str

    search_queries: list[str] = field(
        default_factory=list
    )

    sources: dict[str, EvidenceRecord] = field(
        default_factory=dict
    )

    tool_observation_count: int = 0

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def citation_sources(
        self,
    ) -> dict[str, Source]:
        """
        Return the current best version of every source
        for citation validation.

        Search snippets may have been enriched by
        full-page evidence.
        """

        return {
            source_id: record.source
            for source_id, record
            in self.sources.items()
    }

    @property
    def read_source_count(self) -> int:
        return sum(
            1
            for record in self.sources.values()
            if record.read_full_page
        )

    @property
    def unread_source_count(self) -> int:
        return (
            self.source_count
            - self.read_source_count
        )

    def ingest_message(
        self,
        message: Message,
    ) -> bool:
        """
        Ingest one tool observation into research state.

        Returns True when the message was a supported,
        successfully parsed research observation.
        """

        if message.role != "tool":
            return False

        if message.name not in {
            "web_search",
            "web_page_reader",
        }:
            return False

        if not message.content:
            return False

        try:
            data = json.loads(
                message.content
            )
        except json.JSONDecodeError:
            return False

        if not isinstance(data, dict):
            return False

        if message.name == "web_search":
            success = (
                self._ingest_web_search(
                    data
                )
            )

        else:
            success = (
                self._ingest_web_page(
                    data
                )
            )

        if success:
            self.tool_observation_count += 1

        return success

    def _ingest_web_search(
        self,
        data: dict,
    ) -> bool:

        query = data.get(
            "query",
            "",
        )

        if isinstance(query, str):
            query = query.strip()
        else:
            query = ""

        results = data.get(
            "results"
        )

        if not isinstance(results, list):
            return False

        if (
            query
            and query not in self.search_queries
        ):
            self.search_queries.append(
                query
            )

        for item in results:

            if not isinstance(item, dict):
                continue

            source = self._source_from_dict(
                item
            )

            if source is None:
                continue

            existing = self.sources.get(
                source.source_id
            )

            if existing is None:

                record = EvidenceRecord(
                    source=source,
                    discovered_by_search=True,
                )

                if query:
                    record.search_queries.append(
                        query
                    )

                self.sources[
                    source.source_id
                ] = record

                continue

            existing.discovered_by_search = True

            if (
                query
                and query
                not in existing.search_queries
            ):
                existing.search_queries.append(
                    query
                )

            # Do not replace full-page evidence with
            # a shorter search snippet.
            content = (
                existing.source.content
                if existing.read_full_page
                else source.content
            )

            score = self._merge_score(
                existing.source.score,
                source.score,
            )

            existing.source = Source(
                source_id=source.source_id,
                title=(
                    source.title
                    or existing.source.title
                ),
                url=(
                    source.url
                    or existing.source.url
                ),
                content=content,
                score=score,
            )

        return True

    def _ingest_web_page(
        self,
        data: dict,
    ) -> bool:

        source = self._source_from_dict(
            data
        )

        if source is None:
            return False

        existing = self.sources.get(
            source.source_id
        )

        if existing is None:

            record = EvidenceRecord(
                source=source,
                read_full_page=True,
            )

            self.sources[
                source.source_id
            ] = record

        else:

            # Preserve the search relevance score while
            # replacing snippet content with full-page text.
            existing.source = Source(
                source_id=source.source_id,
                title=(
                    source.title
                    or existing.source.title
                ),
                url=(
                    source.url
                    or existing.source.url
                ),
                content=source.content,
                score=existing.source.score,
            )

            existing.read_full_page = True

            record = existing

        content_type = data.get(
            "content_type"
        )

        if isinstance(
            content_type,
            str,
        ):
            record.content_type = (
                content_type
            )

        truncated = data.get(
            "truncated"
        )

        if isinstance(
            truncated,
            bool,
        ):
            record.truncated = truncated

        return True

    @classmethod
    def from_messages(
        cls,
        *,
        task: str,
        messages: Sequence[Message],
    ) -> ResearchState:
        """
        Reconstruct research state from an existing
        agent trajectory.
        """

        state = cls(
            task=task
        )

        for message in messages:
            state.ingest_message(
                message
            )

        return state

    @staticmethod
    def _source_from_dict(
        data: dict,
    ) -> Source | None:

        source_id = data.get(
            "source_id"
        )

        url = data.get(
            "url"
        )

        if (
            not isinstance(source_id, str)
            or not source_id
        ):
            return None

        if (
            not isinstance(url, str)
            or not url
        ):
            return None

        title = data.get(
            "title",
            "",
        )

        content = data.get(
            "content",
            "",
        )

        score = data.get(
            "score"
        )

        if not isinstance(
            title,
            str,
        ):
            title = str(title)

        if not isinstance(
            content,
            str,
        ):
            content = str(content)

        if (
            isinstance(score, bool)
            or not isinstance(
                score,
                (int, float),
            )
        ):
            score = None
        else:
            score = float(score)

        return Source(
            source_id=source_id,
            title=title,
            url=url,
            content=content,
            score=score,
        )

    @staticmethod
    def _merge_score(
        first: float | None,
        second: float | None,
    ) -> float | None:

        if first is None:
            return second

        if second is None:
            return first

        return max(
            first,
            second,
        )