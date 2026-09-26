from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from .types import LLMResponse, Message


class BaseLLM(ABC):
    """
    Provider-independent interface for language models.

    Agent logic should depend only on BaseLLM rather than
    directly depending on a specific model provider SDK.
    """

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """
        Send messages to the language model.

        Args:
            messages:
                Conversation history.

            tools:
                Optional tool schemas exposed to the model.

        Returns:
            A normalized LLMResponse.
        """
        raise NotImplementedError