import pytest

from deepresearch.tools import CalculatorTool, ToolRegistry


def test_register_tool() -> None:
    registry = ToolRegistry()

    registry.register(CalculatorTool())

    assert registry.names() == ["calculator"]
    assert len(registry) == 1


def test_get_tool() -> None:
    registry = ToolRegistry()
    calculator = CalculatorTool()

    registry.register(calculator)

    assert registry.get("calculator") is calculator


def test_execute_tool() -> None:
    registry = ToolRegistry()

    registry.register(CalculatorTool())

    result = registry.execute(
        "calculator",
        expression="12 * 8",
    )

    assert result == "96"


def test_duplicate_registration() -> None:
    registry = ToolRegistry()

    registry.register(CalculatorTool())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(CalculatorTool())


def test_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(KeyError, match="was not found"):
        registry.get("unknown_tool")


def test_registry_schemas() -> None:
    registry = ToolRegistry()

    registry.register(CalculatorTool())

    schemas = registry.schemas()

    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "calculator"