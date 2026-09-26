from collections.abc import Iterable, Sequence
from typing import Any

from .base import BaseLLM
from .types import LLMResponse, Message


class MockLLM(BaseLLM):
    """
    Deterministic fake LLM used for unit tests.

    Responses are returned in the order they were provided.
    """

    def __init__(
        self,
        responses: Iterable[LLMResponse],
    ) -> None:
        self._responses = list(responses)

        self.calls: list[
            tuple[
                list[Message],
                list[dict[str, Any]] | None,
            ]
        ] = []

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:

        self.calls.append(
            (
                list(messages),
                list(tools) if tools is not None else None,
            )
        )

        if not self._responses:
            raise RuntimeError(
                "MockLLM has no responses remaining."
            )

        return self._responses.pop(0)

    @property
    def remaining_responses(self) -> int:
        return len(self._responses)