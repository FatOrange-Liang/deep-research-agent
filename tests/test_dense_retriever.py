
import json

import pytest

from deepresearch.llm import Message

from deepresearch.research import (
    ClaimManifestBuilder,
    DenseEvidenceRetriever,
    EvidenceCandidateRetriever,
    ResearchState,
)

from deepresearch.research.sources import (
    make_source_id,
)


URL = "https://docs.example.com/langgraph"

OTHER_URL = "https://example.org/article"

SOURCE_ID = make_source_id(URL)

OTHER_ID = make_source_id(OTHER_URL)


QUOTE_A = (
    "A checkpointer persists graph state "
    "after each execution step."
)

QUOTE_B = (
    "Interrupts pause execution so a "
    "human can review and resume."
)

SOURCE_CONTENT = (
    QUOTE_A
    + "\n"
    + QUOTE_B
)

ANSWER = (
    "LangGraph 通过检查点保存状态 "
    f"[{SOURCE_ID}]。\n"
    "LangGraph 允许通过中断进行人工审核 "
    f"[{SOURCE_ID}]。"
)


# =========================================
# Fake bilingual embedding model
# =========================================

class FakeBilingualEmbedder:

    @staticmethod
    def _vector(text: str) -> list[float]:

        text = text.casefold()

        if (
            "检查点" in text
            or "checkpointer" in text
        ):
            return [1.0, 0.0, 0.0]

        if (
            "人工审核" in text
            or "中断" in text
            or "interrupt" in text
        ):
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]

    def encode_queries(self, texts):

        return [
            self._vector(text)
            for text in texts
        ]

    def encode_passages(self, texts):

        return [
            self._vector(text)
            for text in texts
        ]


# =========================================
# ResearchState fixture
# =========================================

def create_state(
    *,
    read_page: bool = True,
    include_other: bool = False,
) -> ResearchState:

    state = ResearchState(
        task="Research LangGraph"
    )

    results = [
        {
            "source_id": SOURCE_ID,
            "title": "Official Docs",
            "url": URL,
            "content": "Official snippet",
            "score": 0.9,
        }
    ]

    if include_other:

        results.append(
            {
                "source_id": OTHER_ID,
                "title": "Third-party article",
                "url": OTHER_URL,
                "content": "Other snippet",
                "score": 0.8,
            }
        )

    assert state.ingest_message(
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

    if read_page:

        assert state.ingest_message(
            Message(
                role="tool",
                name="web_page_reader",
                tool_call_id="reader_1",
                content=json.dumps(
                    {
                        "source_id": SOURCE_ID,
                        "title": "Official Docs",
                        "url": URL,
                        "content": SOURCE_CONTENT,
                        "content_type": "text/html",
                        "truncated": False,
                    }
                ),
            )
        )

    if include_other:

        assert state.ingest_message(
            Message(
                role="tool",
                name="web_page_reader",
                tool_call_id="reader_2",
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


def extract_claims():

    return (
        ClaimManifestBuilder()
        .extract_claims(ANSWER)
    )


def make_retriever(**kwargs):

    return DenseEvidenceRetriever(
        embedder=FakeBilingualEmbedder(),
        **kwargs,
    )


# =========================================
# Test 1: Bilingual retrieval contract
# =========================================

def test_dense_recovers_chinese_to_english_candidate():

    state = create_state()

    claims = extract_claims()

    lexical = EvidenceCandidateRetriever()

    lexical_results = lexical.retrieve(
        claims=claims,
        state=state,
    )

    # Existing lexical baseline cannot bridge
    # the Chinese-English terminology gap.
    assert lexical_results["C001"] == ()

    dense_results = make_retriever().retrieve(
        claims=claims,
        state=state,
    )

    assert dense_results["C001"]

    assert (
        "checkpointer"
        in dense_results["C001"][0].quote.lower()
    )


# =========================================
# Test 2: Multiple claim retrieval
# =========================================

def test_dense_retrieves_two_distinct_claims():

    results = make_retriever().retrieve(
        claims=extract_claims(),
        state=create_state(),
    )

    assert results["C001"]

    assert results["C002"]

    assert "checkpointer" in (
        results["C001"][0].quote.lower()
    )

    assert "interrupt" in (
        results["C002"][0].quote.lower()
    )


# =========================================
# Test 3: Manifest integration
# =========================================

def test_dense_candidates_integrate_with_manifest():

    state = create_state()

    builder = ClaimManifestBuilder()

    claims = builder.extract_claims(
        ANSWER
    )

    candidates = make_retriever().retrieve(
        claims=claims,
        state=state,
    )

    proposals = [
        candidate.to_proposal()
        for items in candidates.values()
        for candidate in items
    ]

    manifest = builder.build(
        answer=ANSWER,
        state=state,
        proposals=proposals,
    )

    assert manifest.claim_count == 2

    assert manifest.citation_coverage == 1.0

    assert manifest.anchor_coverage == 1.0

    assert manifest.all_claim_units_anchored


# =========================================
# Test 4: Citation scope restriction
# =========================================

def test_dense_only_uses_cited_source():

    state = create_state(
        include_other=True
    )

    results = make_retriever().retrieve(
        claims=extract_claims(),
        state=state,
    )

    for items in results.values():

        for candidate in items:

            assert candidate.source_id == SOURCE_ID

            assert candidate.source_id != OTHER_ID


# =========================================
# Test 5: Unread source cannot contribute
# =========================================

def test_unread_source_has_no_dense_candidates():

    state = create_state(
        read_page=False
    )

    results = make_retriever().retrieve(
        claims=extract_claims(),
        state=state,
    )

    assert results["C001"] == ()

    assert results["C002"] == ()


# =========================================
# Test 6: Zero vectors
# =========================================

def test_zero_vectors_do_not_create_candidates():

    class ZeroEmbedder:

        def encode_queries(self, texts):

            return [
                [0.0, 0.0, 0.0]
                for _ in texts
            ]

        def encode_passages(self, texts):

            return [
                [0.0, 0.0, 0.0]
                for _ in texts
            ]

    retriever = DenseEvidenceRetriever(
        embedder=ZeroEmbedder(),
    )

    results = retriever.retrieve(
        claims=extract_claims(),
        state=create_state(),
    )

    assert results["C001"] == ()


# =========================================
# Test 7: Invalid embedding dimensions
# =========================================

def test_embedding_dimension_mismatch_is_rejected():

    class BadEmbedder:

        def encode_queries(self, texts):

            return [
                [1.0, 0.0]
                for _ in texts
            ]

        def encode_passages(self, texts):

            return [
                [1.0, 0.0, 0.0]
                for _ in texts
            ]

    retriever = DenseEvidenceRetriever(
        embedder=BadEmbedder(),
    )

    with pytest.raises(
        ValueError,
        match="dimension mismatch",
    ):

        retriever.retrieve(
            claims=extract_claims(),
            state=create_state(),
        )


# =========================================
# Test 8: Invalid configuration
# =========================================

def test_invalid_dense_config():

    with pytest.raises(ValueError):

        make_retriever(
            top_k_per_claim=0
        )

    with pytest.raises(ValueError):

        make_retriever(
            min_similarity=2.0
        )

    with pytest.raises(ValueError):

        make_retriever(
            min_quote_chars=100,
            max_quote_chars=50,
        )
