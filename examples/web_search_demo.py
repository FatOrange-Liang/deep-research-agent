from deepresearch.config import (
    load_settings,
)

from deepresearch.tools import (
    CalculatorTool,
    ToolRegistry,
    WebSearchTool,
)


def main() -> None:

    settings = load_settings()

    if not settings.tavily_api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not configured."
        )

    tool = WebSearchTool(
        api_key=settings.tavily_api_key,
        max_results=5,
    )

    registry = ToolRegistry()

    registry.register(
        CalculatorTool()
    )

    if settings.tavily_api_key:
        registry.register(
            WebSearchTool(
                api_key=settings.tavily_api_key,
            )
        )

    result = tool.execute(
        query=(
            "LangGraph agent framework "
            "official documentation"
        )
    )

    print(result)


if __name__ == "__main__":
    main()