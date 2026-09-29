from __future__ import annotations

import json
import ipaddress
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from deepresearch.research.sources import (
    make_source_id,
)

from .base import BaseTool


class WebPageReaderTool(BaseTool):
    """
    Fetch and extract readable text from a public web page.

    The tool intentionally returns structured JSON so that
    downstream research components can preserve provenance.
    """

    name = "web_page_reader"

    description = (
        "Open and read the content of a public web page. "
        "Use this tool after web_search when a search result "
        "looks important and you need more complete evidence "
        "than the search snippet provides."
    )

    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "The public HTTP or HTTPS URL to read. "
                    "Prefer URLs returned by web_search."
                ),
            }
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        max_chars: int = 20000,
        transport: httpx.BaseTransport | None = None,
        trust_env: bool = False,
    ) -> None:

        if timeout <= 0:
            raise ValueError(
                "'timeout' must be greater than zero."
            )

        if max_chars <= 0:
            raise ValueError(
                "'max_chars' must be greater than zero."
            )

        self.timeout = timeout
        self.max_chars = max_chars
        self.transport = transport
        self.trust_env = trust_env

    def execute(
        self,
        **kwargs: Any,
    ) -> str:

        url = kwargs.get("url")

        if not isinstance(url, str):
            raise ValueError(
                "'url' must be a string."
            )

        url = url.strip()

        if not url:
            raise ValueError(
                "'url' cannot be empty."
            )

        self._validate_url(url)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(compatible; DeepResearchAgent/0.1)"
            ),
            "Accept": (
                "text/html,"
                "application/xhtml+xml"
            ),
        }

        try:
            with httpx.Client(
                timeout=self.timeout,
                transport=self.transport,
                trust_env=self.trust_env,
                follow_redirects=True,
            ) as client:

                response = client.get(
                    url,
                    headers=headers,
                )

                response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                "Web page request timed out."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "Web page returned "
                f"HTTP {exc.response.status_code}."
            ) from exc

        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Web page request failed: {exc}"
            ) from exc

        content_type = (
            response.headers
            .get("content-type", "")
            .lower()
        )

        if (
            "text/html" not in content_type
            and "application/xhtml+xml"
            not in content_type
        ):
            raise RuntimeError(
                "Unsupported page content type: "
                f"{content_type or 'unknown'}"
            )

        title, text = self._extract_html(
            response.text
        )

        truncated = (
            len(text) > self.max_chars
        )

        if truncated:
            text = text[
                : self.max_chars
            ]

        final_url = str(
            response.url
        )

        result = {
            "source_id":
                make_source_id(final_url),
            "title":
                title,
            "url":
                final_url,
            "content":
                text,
            "content_type":
                content_type,
            "truncated":
                truncated,
        }

        return json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def _extract_html(
        html: str,
    ) -> tuple[str, str]:

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        # Remove non-content elements.
        for element in soup.find_all(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "nav",
                "header",
                "footer",
                "aside",
                "form",
            ]
        ):
            element.decompose()

        # Remove common navigation / UI containers.
        for element in soup.select(
            (
                '[role="navigation"], '
                '[role="banner"], '
                '[role="complementary"]'
            )
        ):
            element.decompose()

        title = ""

        if soup.title:
            title = soup.title.get_text(
                " ",
                strip=True,
            )

        # Prefer the semantic main content of the page.
        content_root = (
            soup.find("main")
            or soup.find("article")
            or soup.body
            or soup
        )

        text = content_root.get_text(
            "\n",
            strip=True,
        )

        lines = []

        for raw_line in text.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            # Remove immediate duplicate lines.
            if (
                lines
                and lines[-1] == line
            ):
                continue

            lines.append(line)

        normalized_text = "\n".join(
            lines
        )

        return title, normalized_text

    @staticmethod
    def _validate_url(
        url: str,
    ) -> None:

        parsed = urlparse(url)

        if parsed.scheme not in {
            "http",
            "https",
        }:
            raise ValueError(
                "Only HTTP and HTTPS URLs "
                "are supported."
            )

        hostname = parsed.hostname

        if not hostname:
            raise ValueError(
                "URL must contain a hostname."
            )

        if hostname.lower() in {
            "localhost",
            "localhost.localdomain",
        }:
            raise ValueError(
                "Localhost URLs are not allowed."
            )

        # Block obvious private IP targets.
        try:
            ip = ipaddress.ip_address(
                hostname
            )
        except ValueError:
            return

        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise ValueError(
                "Private or local network "
                "URLs are not allowed."
            )