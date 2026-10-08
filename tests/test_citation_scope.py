import json

from deepresearch.llm import Message
from deepresearch.research.claim_anchors import (
    ClaimAnchorVerifier,
    ClaimEvidence,
)
from deepresearch.research.manifest import (
    ClaimManifestBuilder,
)
from deepresearch.research.state import (
    ResearchState,
)


SOURCE_A = "S_12345678"
SOURCE_B = "S_abcdef12"


def test_trailing_bullet_citation_is_inherited_by_all_factual_sentences():
    answer = (
        "- **核心定位。** LangGraph 是低层编排运行时。"
        "它支持长期运行、有状态的 agent。"
        f"它可以独立使用。[{SOURCE_A}]"
    )

    claims = ClaimManifestBuilder().extract_claims(
        answer
    )

    assert len(claims) == 4

    assert all(
        claim.cited_source_ids
        == (SOURCE_A,)
        for claim in claims
    )


def test_citation_scope_does_not_cross_bullet_boundaries():
    answer = (
        "- 第一条事实。第二条事实。"
        f"[{SOURCE_A}]\n"
        "- 第三条事实。第四条事实。"
        f"[{SOURCE_B}]"
    )

    claims = ClaimManifestBuilder().extract_claims(
        answer
    )

    assert [
        claim.cited_source_ids
        for claim in claims
    ] == [
        (SOURCE_A,),
        (SOURCE_A,),
        (SOURCE_B,),
        (SOURCE_B,),
    ]


def test_markdown_only_and_navigation_text_do_not_become_claims():
    answer = (
        "**\n\n"
        "[官方概览](https://example.com)\n\n"
        "我查阅了官方资料和相关文档。\n\n"
        "- 真正的事实陈述。"
        f"[{SOURCE_A}]"
    )

    claims = ClaimManifestBuilder().extract_claims(
        answer
    )

    assert len(claims) == 1
    assert claims[0].text == "真正的事实陈述"
    assert claims[0].cited_source_ids == (
        SOURCE_A,
    )


def _state_with_read_source(
    *,
    source_id: str,
    content: str,
) -> ResearchState:
    state = ResearchState(
        task="citation scope test"
    )

    url = "https://example.com/docs"

    search_message = Message(
        role="tool",
        name="web_search",
        tool_call_id="search-1",
        content=json.dumps(
            {
                "query": "test",
                "results": [
                    {
                        "source_id": source_id,
                        "title": "Example",
                        "url": url,
                        "content": content[:100],
                        "score": 1.0,
                    }
                ],
            }
        ),
    )

    assert state.ingest_message(
        search_message
    )

    page_message = Message(
        role="tool",
        name="web_page_reader",
        tool_call_id="page-1",
        content=json.dumps(
            {
                "source_id": source_id,
                "title": "Example",
                "url": url,
                "content": content,
                "content_type": "text/html",
                "truncated": False,
            }
        ),
    )

    assert state.ingest_message(
        page_message
    )

    return state


def test_anchor_accepts_citation_at_end_of_same_bullet_scope():
    evidence_quote = (
        "LangGraph is a low-level orchestration framework "
        "and runtime for long-running stateful agents."
    )

    state = _state_with_read_source(
        source_id=SOURCE_A,
        content=evidence_quote,
    )

    answer = (
        "- LangGraph 是低层编排运行时。"
        "它支持长期运行、有状态的 agent。"
        f"[{SOURCE_A}]"
    )

    report = ClaimAnchorVerifier().verify(
        answer=answer,
        state=state,
        items=[
            ClaimEvidence(
                claim="LangGraph 是低层编排运行时",
                source_id=SOURCE_A,
                evidence_quote=evidence_quote,
            )
        ],
    )

    assert report.checks[0].status == (
        "anchored"
    )


def test_detached_citation_in_plain_paragraph_is_not_inherited():
    evidence_quote = (
        "LangGraph uses a checkpointer for state persistence."
    )

    state = _state_with_read_source(
        source_id=SOURCE_A,
        content=evidence_quote,
    )

    answer = (
        "LangGraph uses a checkpointer for state persistence. "
        f"A different statement [{SOURCE_A}]."
    )

    report = ClaimAnchorVerifier().verify(
        answer=answer,
        state=state,
        items=[
            ClaimEvidence(
                claim=(
                    "LangGraph uses a checkpointer "
                    "for state persistence"
                ),
                source_id=SOURCE_A,
                evidence_quote=evidence_quote,
            )
        ],
    )

    assert report.all_anchored is False
    assert report.checks[0].status == "citation_not_attached"
