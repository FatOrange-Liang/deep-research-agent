from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """
    Base class for every tool available to the agent.

    A tool exposes:
    - name: unique tool identifier
    - description: tells the LLM when the tool should be used
    - parameters: JSON-schema-like description of arguments
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """
        Execute the tool.

        Returns:
            A textual observation that can later be sent back to the LLM.
        """
        raise NotImplementedError

    def to_schema(self) -> dict[str, Any]:
        """
        Convert the tool into a structured schema.

        This representation will later be provided to an LLM
        for tool selection and argument generation.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }