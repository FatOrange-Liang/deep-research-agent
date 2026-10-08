
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class SourceAuthorityPolicy:
    """
    Check whether a source belongs to a configured
    authoritative hostname.

    This checks source origin, not whether its
    factual claims are necessarily correct.
    """

    required_hosts: tuple[str, ...]

    def __post_init__(self) -> None:

        if (
            not isinstance(self.required_hosts, tuple)
            or not self.required_hosts
        ):
            raise ValueError(
                "'required_hosts' must be a non-empty tuple."
            )

        normalized = []

        for raw_host in self.required_hosts:

            if not isinstance(raw_host, str):
                raise ValueError(
                    "Each required host must be a string."
                )

            host = (
                raw_host.strip()
                .lower()
                .rstrip(".")
            )

            # Require plain hostnames, not URLs,
            # wildcards, paths or user information.
            if (
                not host
                or "." not in host
                or host.startswith(".")
                or ".." in host
                or any(
                    char in host
                    for char in "/:@?*#\\ "
                )
            ):
                raise ValueError(
                    f"Invalid authoritative host: {raw_host!r}"
                )

            normalized.append(host)

        # Deduplicate while preserving order.
        object.__setattr__(
            self,
            "required_hosts",
            tuple(dict.fromkeys(normalized)),
        )

    def matches(
        self,
        url: str,
    ) -> bool:
        """
        Return True only when the parsed URL hostname
        exactly matches an allowed hostname.
        """

        if not isinstance(url, str):
            return False

        if not url.strip():
            return False

        try:
            parsed = urlsplit(
                url.strip()
            )

            hostname = (
                parsed.hostname or ""
            ).lower().rstrip(".")

        except ValueError:
            return False

        if parsed.scheme.lower() not in {
            "http",
            "https",
        }:
            return False

        return hostname in self.required_hosts
