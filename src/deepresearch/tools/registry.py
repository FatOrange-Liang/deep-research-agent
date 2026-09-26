from typing import Any

from .base import BaseTool


class ToolRegistry:
    """
    Central registry for all tools available to the agent.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered."
            )

        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            available = ", ".join(self._tools.keys()) or "none"

            raise KeyError(
                f"Tool '{name}' was not found. "
                f"Available tools: {available}"
            )

        return self._tools[name]

    def execute(self, name: str, **kwargs: Any) -> str:
        tool = self.get(name)

        return tool.execute(**kwargs)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            tool.to_schema()
            for tool in self._tools.values()
        ]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)