from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class Source:
    """
    A source discovered during research.
    """

    source_id: str
    title: str
    url: str
    content: str
    score: float | None = None


def make_source_id(url: str) -> str:
    """
    Create a short deterministic source ID from a URL.

    Example:
        https://example.com/page
            ↓
        S_8f2ab341

    Using the URL hash keeps IDs stable across repeated searches
    and avoids S1/S2 collisions between multiple search calls.
    """

    normalized_url = url.strip()

    if not normalized_url:
        raise ValueError(
            "'url' cannot be empty."
        )

    digest = sha256(
        normalized_url.encode("utf-8")
    ).hexdigest()[:8]

    return f"S_{digest}"