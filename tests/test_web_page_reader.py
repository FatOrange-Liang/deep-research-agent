import json

import httpx
import pytest

from deepresearch.tools import (
    WebPageReaderTool,
)

def test_reader_prefers_main_content() -> None:

    html = """
    <html>
        <head>
            <title>Example Docs</title>
        </head>

        <body>
            <nav>
                Navigation
                Search
                Home
            </nav>

            <main>
                <h1>Important Documentation</h1>
                <p>
                    This is the actual research evidence.
                </p>
            </main>

            <footer>
                Copyright
            </footer>
        </body>
    </html>
    """

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        return httpx.Response(
            200,
            headers={
                "content-type": "text/html"
            },
            text=html,
            request=request,
        )

    tool = WebPageReaderTool(
        transport=httpx.MockTransport(
            handler
        )
    )

    result = json.loads(
        tool.execute(
            url="https://example.com/docs"
        )
    )

    assert (
        "actual research evidence"
        in result["content"]
    )

    assert (
        "Navigation"
        not in result["content"]
    )

    assert (
        "Copyright"
        not in result["content"]
    )

def create_mock_transport() -> httpx.MockTransport:

    html = """
    <html>
        <head>
            <title>LangGraph Docs</title>
            <style>
                body { color: red; }
            </style>
        </head>

        <body>
            <h1>LangGraph</h1>

            <p>
                LangGraph is a framework
                for stateful agents.
            </p>

            <script>
                console.log("ignore me");
            </script>
        </body>
    </html>
    """

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        return httpx.Response(
            200,
            headers={
                "content-type":
                    "text/html; charset=utf-8"
            },
            text=html,
            request=request,
        )

    return httpx.MockTransport(
        handler
    )


def test_web_page_reader() -> None:

    tool = WebPageReaderTool(
        transport=create_mock_transport(),
    )

    result_text = tool.execute(
        url="https://example.com/langgraph",
    )

    result = json.loads(
        result_text
    )

    assert (
        result["title"]
        == "LangGraph Docs"
    )

    assert (
        "stateful agents"
        in result["content"]
    )

    assert (
        "console.log"
        not in result["content"]
    )

    assert result[
        "source_id"
    ].startswith("S_")

    assert result[
        "truncated"
    ] is False


def test_reader_rejects_empty_url() -> None:

    tool = WebPageReaderTool()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        tool.execute(
            url="   "
        )


def test_reader_rejects_non_http_scheme() -> None:

    tool = WebPageReaderTool()

    with pytest.raises(
        ValueError,
        match="HTTP and HTTPS",
    ):
        tool.execute(
            url="file:///etc/passwd"
        )


def test_reader_rejects_localhost() -> None:

    tool = WebPageReaderTool()

    with pytest.raises(
        ValueError,
        match="Localhost",
    ):
        tool.execute(
            url="http://localhost:8000"
        )


def test_reader_rejects_private_ip() -> None:

    tool = WebPageReaderTool()

    with pytest.raises(
        ValueError,
        match="Private",
    ):
        tool.execute(
            url="http://192.168.1.1"
        )


def test_reader_rejects_non_html() -> None:

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        return httpx.Response(
            200,
            headers={
                "content-type":
                    "application/pdf"
            },
            content=b"PDF",
            request=request,
        )

    tool = WebPageReaderTool(
        transport=httpx.MockTransport(
            handler
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Unsupported",
    ):
        tool.execute(
            url="https://example.com/file.pdf"
        )


def test_reader_truncates_long_page() -> None:

    html = (
        "<html><body>"
        + ("A" * 1000)
        + "</body></html>"
    )

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        return httpx.Response(
            200,
            headers={
                "content-type":
                    "text/html"
            },
            text=html,
            request=request,
        )

    tool = WebPageReaderTool(
        max_chars=100,
        transport=httpx.MockTransport(
            handler
        ),
    )

    result = json.loads(
        tool.execute(
            url="https://example.com"
        )
    )

    assert len(
        result["content"]
    ) <= 100

    assert result[
        "truncated"
    ] is True