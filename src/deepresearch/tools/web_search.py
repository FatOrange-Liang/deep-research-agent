import json
from typing import Any

import httpx

from .base import BaseTool

from deepresearch.research.sources import (
    make_source_id,
)


class WebSearchTool(BaseTool):
    """
    Search the web through the Tavily Search API.

    The tool returns normalized search results that are easy
    for an LLM to consume.
    """

    name = "web_search"

    description = (
        "Search the web for current or external information. "
        "Use this tool when the answer depends on recent facts, "
        "web pages, documentation, news, or information that may "
        "not be contained in the model's internal knowledge."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The web search query."
                ),
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        api_key: str,
        max_results: int = 5,
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:

        api_key = api_key.strip()

        if not api_key:
            raise ValueError(
                "'api_key' cannot be empty."
            )

        if max_results <= 0:
            raise ValueError(
                "'max_results' must be greater than zero."
            )

        self.api_key = api_key
        self.max_results = max_results
        self.timeout = timeout
        self.transport = transport

    def execute(
        self,
        **kwargs: Any,
    ) -> str:

        query = kwargs.get("query")

        if not isinstance(query, str):
            raise ValueError(
                "'query' must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "'query' cannot be empty."
            )

        if len(query) > 1000:
            raise ValueError(
                "Search query is too long."
            )

        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": False,
        }

        headers = {
            "Authorization":
                f"Bearer {self.api_key}",
            "Content-Type":
                "application/json",
        }

        try:
            with httpx.Client(
                timeout=self.timeout,
                transport=self.transport,
                trust_env=False,
            ) as client:

                response = client.post(
                    "https://api.tavily.com/search",
                    json=payload,
                    headers=headers,
                )

                response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                "Web search request timed out."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "Web search API returned "
                f"HTTP {exc.response.status_code}."
            ) from exc

        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Web search request failed: {exc}"
            ) from exc

        data = response.json()

        raw_results = data.get(
            "results",
            [],
        )

        normalized_results = []

        for item in raw_results:

            url = item.get(
                "url",
                "",
            ).strip()

            normalized_results.append(
                {
                    "source_id": (
                        make_source_id(url)
                        if url
                        else None
                    ),
                    "title":
                        item.get("title", ""),
                    "url":
                        url,
                    "content":
                        item.get("content", ""),
                    "score":
                        item.get("score"),
                }
            )

        result = {
            "query": query,
            "results":
                normalized_results,
        }

        return json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )