
import json

import pytest

from deepresearch.llm import Message

from deepresearch.research import (
    ClaimManifestBuilder,
    EvidenceCandidateRetriever,
    ResearchState,
)

from deepresearch.research.claim_anchors import (
    normalize_text,
)

from deepresearch.research.sources import (
    make_source_id,
)


URL = "https://docs.example.com/langgraph"

OTHER_URL = "https://example.org/article"

SOURCE_ID = make_source_id(URL)

OTHER_ID = make_source_id(OTHER_URL)


CLAIM_A = (
    "LangGraph saves state through checkpointers"
)

CLAIM_B = (
    "LangGraph supports human review through interrupts"
)

QUOTE_A = (
    "A checkpointer persists graph state "
    "after each execution step."
)

QUOTE_B = (
    "Interrupts pause execution so a "
    "human can review and resume."
)

SOURCE_CONTENT = (
    "LangGraph documentation.\n"
    + QUOTE_A
    + "\n"
    + QUOTE_B
    + "\n"
    + "General guidance about agents and tools."
)

ANSWER = (
    f"{CLAIM_A} [{SOURCE_ID}].\n"
    f"{CLAIM_B} [{SOURCE_ID}]."
)


def create_state(
    *,
    read_official: bool = True,
    include_other: bool = False,
    content: str = SOURCE_CONTENT,
) -> ResearchState:

    state = ResearchState(
        task="Research LangGraph"
    )

    results = [
        {
            "source_id": SOURCE_ID,
            "title": "Official Docs",
            "url": URL,
            "content": "Search snippet",
            "score": 0.9,
        }
    ]

    if include_other:

        results.append(
            {
                "source_id": OTHER_ID,
                "title": "Other Article",
                "url": OTHER_URL,
                "content": "Another snippet",
                "score": 0.95,
            }
        )

    state.ingest_message(
        Message(
            role="tool",
            name="web_search",
            tool_call_id="search_1",
            content=json.dumps(
                {
                    "query": "LangGraph",
                    "results": results,
                }
            ),
        )
    )

    if read_official:

        state.ingest_message(
            Message(
                role="tool",
                name="web_page_reader",
                tool_call_id="reader_1",
                content=json.dumps(
                    {
                        "source_id": SOURCE_ID,
                        "title": "Official Docs",
                        "url": URL,
                        "content": content,
                        "content_type": "text/html",
                        "truncated": False,
                    }
                ),
            )
        )

    if include_other:

        state.ingest_message(
            Message(
                role="tool",
                name="web_page_reader",
                tool_call_id="reader_other",
                content=json.dumps(
                    {
                        "source_id": OTHER_ID,
                        "title": "Other Article",
                        "url": OTHER_URL,
                        "content": SOURCE_CONTENT,
                        "content_type": "text/html",
                        "truncated": False,
                    }
                ),
            )
        )

    return state


def extract_claims(
    answer: str = ANSWER,
):

    return (
        ClaimManifestBuilder()
        .extract_claims(answer)
    )


def test_retrieves_relevant_candidates() -> None:

    state = create_state()

    retriever = EvidenceCandidateRetriever()

    candidates = retriever.retrieve(
        claims=extract_claims(),
        state=state,
    )

    assert candidates["C001"]

    assert candidates["C002"]

    assert (
        "checkpointer"
        in candidates["C001"][0].quote.lower()
    )

    assert (
        "interrupts"
        in candidates["C002"][0].quote.lower()
    )

    for candidate_list in candidates.values():

        for candidate in candidate_list:

            assert candidate.source_id == SOURCE_ID

            assert (
                normalize_text(candidate.quote)
                in normalize_text(SOURCE_CONTENT)
            )


def test_retriever_integrates_with_manifest() -> None:

    state = create_state()

    builder = ClaimManifestBuilder()

    claims = builder.extract_claims(
        ANSWER
    )

    retriever = EvidenceCandidateRetriever(
        top_k_per_claim=1
    )

    candidates = retriever.retrieve(
        claims=claims,
        state=state,
    )

    proposals = [
        candidate.to_proposal()
        for candidate_list in candidates.values()
        for candidate in candidate_list
    ]

    manifest = builder.build(
        answer=ANSWER,
        state=state,
        proposals=proposals,
    )

    assert manifest.claim_count == 2

    assert manifest.citation_coverage == 1.0

    assert manifest.anchored_claim_count == 2

    assert manifest.anchor_coverage == 1.0

    assert manifest.all_claim_units_anchored


def test_unread_source_has_no_candidates() -> None:

    state = create_state(
        read_official=False
    )

    candidates = (
        EvidenceCandidateRetriever()
        .retrieve(
            claims=extract_claims(),
            state=state,
        )
    )

    assert candidates["C001"] == ()

    assert candidates["C002"] == ()


def test_uncited_claim_has_no_candidates() -> None:

    answer = (
        f"{CLAIM_A} [{SOURCE_ID}].\n"
        "A completely uncited conclusion."
    )

    candidates = (
        EvidenceCandidateRetriever()
        .retrieve(
            claims=extract_claims(answer),
            state=create_state(),
        )
    )

    assert candidates["C001"]

    assert candidates["C002"] == ()


def test_retrieval_only_uses_cited_source() -> None:

    state = create_state(
        include_other=True
    )

    candidates = (
        EvidenceCandidateRetriever()
        .retrieve(
            claims=extract_claims(),
            state=state,
        )
    )

    for candidate_list in candidates.values():

        for candidate in candidate_list:

            assert candidate.source_id == SOURCE_ID

            assert candidate.source_id != OTHER_ID


def test_no_lexical_match_returns_empty() -> None:

    answer = (
        "Quantum pineapple teleportation is enabled "
        f"[{SOURCE_ID}]."
    )

    candidates = (
        EvidenceCandidateRetriever()
        .retrieve(
            claims=extract_claims(answer),
            state=create_state(),
        )
    )

    assert candidates["C001"] == ()


def test_long_passage_is_bounded() -> None:

    long_content = (
        QUOTE_A
        + " "
        + ("Additional transaction metadata " * 30)
        + "."
    )

    state = create_state(
        content=long_content
    )

    retriever = EvidenceCandidateRetriever(
        min_quote_chars=24,
        max_quote_chars=120,
    )

    candidates = retriever.retrieve(
        claims=extract_claims(),
        state=state,
    )

    assert candidates["C001"]

    for candidate in candidates["C001"]:

        assert len(candidate.quote) <= 120

        assert (
            normalize_text(candidate.quote)
            in normalize_text(long_content)
        )


def test_invalid_retriever_config() -> None:

    with pytest.raises(ValueError):

        EvidenceCandidateRetriever(
            top_k_per_claim=0
        )

    with pytest.raises(ValueError):

        EvidenceCandidateRetriever(
            min_quote_chars=24,
            max_quote_chars=20,
        )

    with pytest.raises(ValueError):

        EvidenceCandidateRetriever(
            min_lexical_score=1.5
        )
