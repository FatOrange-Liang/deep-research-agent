import json

import httpx
import pytest

from deepresearch.tools import (
    WebSearchTool,
)


def create_mock_transport() -> httpx.MockTransport:

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        assert (
            request.url
            == "https://api.tavily.com/search"
        )

        assert (
            request.headers[
                "Authorization"
            ]
            == "Bearer test-key"
        )

        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title":
                            "LangGraph",
                        "url":
                            "https://example.com/langgraph",
                        "content":
                            "LangGraph is a framework "
                            "for stateful agents.",
                        "score":
                            0.95,
                    },
                    {
                        "title":
                            "smolagents",
                        "url":
                            "https://example.com/smolagents",
                        "content":
                            "smolagents is a lightweight "
                            "agent framework.",
                        "score":
                            0.90,
                    },
                ]
            },
        )

    return httpx.MockTransport(
        handler
    )


def test_web_search() -> None:

    tool = WebSearchTool(
        api_key="test-key",
        transport=create_mock_transport(),
    )

    result_text = tool.execute(
        query="LangGraph vs smolagents",
    )

    result = json.loads(
        result_text
    )

    assert (
        result["query"]
        == "LangGraph vs smolagents"
    )

    assert len(
        result["results"]
    ) == 2

    assert (
        result["results"][0]["title"]
        == "LangGraph"
    )

    assert (
        result["results"][0]["source_id"]
        is not None
    )

    assert (
        result["results"][0]["source_id"]
        .startswith("S_")
    )


def test_empty_query() -> None:

    tool = WebSearchTool(
        api_key="test-key",
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        tool.execute(
            query="   "
        )


def test_non_string_query() -> None:

    tool = WebSearchTool(
        api_key="test-key",
    )

    with pytest.raises(
        ValueError,
        match="must be a string",
    ):
        tool.execute(
            query=123
        )


def test_empty_api_key() -> None:

    with pytest.raises(
        ValueError,
        match="api_key",
    ):
        WebSearchTool(
            api_key=""
        )


def test_http_error() -> None:

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        return httpx.Response(
            401,
            json={
                "error": "unauthorized",
            },
        )

    tool = WebSearchTool(
        api_key="bad-key",
        transport=httpx.MockTransport(
            handler
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="HTTP 401",
    ):
        tool.execute(
            query="test"
        )