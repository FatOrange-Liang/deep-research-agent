import pytest

from deepresearch.tools import CalculatorTool


def test_basic_multiplication() -> None:
    tool = CalculatorTool()

    result = tool.execute(expression="28374 * 928")

    assert result == "26331072"


def test_parentheses() -> None:
    tool = CalculatorTool()

    result = tool.execute(expression="(100 + 25) * 4")

    assert result == "500"


def test_float_division() -> None:
    tool = CalculatorTool()

    result = tool.execute(expression="10 / 4")

    assert result == "2.5"


def test_negative_number() -> None:
    tool = CalculatorTool()

    result = tool.execute(expression="-10 + 3")

    assert result == "-7"


def test_division_by_zero() -> None:
    tool = CalculatorTool()

    with pytest.raises(ValueError, match="Division by zero"):
        tool.execute(expression="10 / 0")


def test_empty_expression() -> None:
    tool = CalculatorTool()

    with pytest.raises(ValueError, match="cannot be empty"):
        tool.execute(expression="   ")


def test_non_string_expression() -> None:
    tool = CalculatorTool()

    with pytest.raises(ValueError, match="must be a string"):
        tool.execute(expression=123)


def test_function_call_is_rejected() -> None:
    tool = CalculatorTool()

    with pytest.raises(ValueError):
        tool.execute(expression="print(123)")


def test_python_code_execution_is_rejected() -> None:
    tool = CalculatorTool()

    with pytest.raises(ValueError):
        tool.execute(
            expression="__import__('os').system('echo unsafe')"
        )


def test_schema() -> None:
    tool = CalculatorTool()

    schema = tool.to_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculator"
    assert "expression" in schema["function"]["parameters"]["properties"]