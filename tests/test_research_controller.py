import json

from deepresearch.llm import Message

from deepresearch.research import (
    EvidencePolicy,
    EvidencePolicyConfig,
    ResearchController,
    ResearchState,
)


def add_search(
    state: ResearchState,
) -> None:

    message = Message(
        role="tool",
        name="web_search",
        tool_call_id="search_001",
        content=json.dumps(
            {
                "query": "LangGraph",
                "results": [
                    {
                        "source_id":
                            "S_11111111",
                        "title":
                            "Official docs A",
                        "url":
                            "https://example.com/a",
                        "content":
                            "Snippet A",
                        "score":
                            0.9,
                    },
                    {
                        "source_id":
                            "S_22222222",
                        "title":
                            "Official docs B",
                        "url":
                            "https://example.com/b",
                        "content":
                            "Snippet B",
                        "score":
                            0.8,
                    },
                    {
                        "source_id":
                            "S_33333333",
                        "title":
                            "Official docs C",
                        "url":
                            "https://example.com/c",
                        "content":
                            "Snippet C",
                        "score":
                            0.7,
                    },
                ],
            }
        ),
    )

    state.ingest_message(
        message
    )


def add_page(
    state: ResearchState,
) -> None:

    message = Message(
        role="tool",
        name="web_page_reader",
        tool_call_id="read_001",
        content=json.dumps(
            {
                "source_id":
                    "S_11111111",
                "title":
                    "Official docs A",
                "url":
                    "https://example.com/a",
                "content":
                    "Full evidence",
                "content_type":
                    "text/html",
                "truncated":
                    False,
            }
        ),
    )

    state.ingest_message(
        message
    )


def create_controller() -> ResearchController:

    policy = EvidencePolicy(
        EvidencePolicyConfig(
            min_sources=3,
            min_full_pages=1,
        )
    )

    return ResearchController(
        policy=policy
    )


def test_controller_requires_search() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    control = (
        create_controller()
        .evaluate(state)
    )

    assert (
        control.can_finish
        is False
    )

    assert (
        control.decision.action
        == "search_more"
    )

    assert control.instruction

    assert (
        "web_search"
        in control.instruction
    )


def test_controller_requires_page_read() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    add_search(state)

    control = (
        create_controller()
        .evaluate(state)
    )

    assert (
        control.can_finish
        is False
    )

    assert (
        control.decision.action
        == "read_more"
    )

    assert control.instruction

    assert (
        "web_page_reader"
        in control.instruction
    )

    assert (
        "https://example.com/a"
        in control.instruction
    )


def test_controller_allows_finish() -> None:

    state = ResearchState(
        task="Research LangGraph"
    )

    add_search(state)

    add_page(state)

    control = (
        create_controller()
        .evaluate(state)
    )

    assert control.can_finish

    assert (
        control.decision.action
        == "synthesize"
    )

    assert (
        control.instruction
        is None
    )