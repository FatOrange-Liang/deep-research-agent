from deepresearch.tools import CalculatorTool, ToolRegistry


def main() -> None:
    registry = ToolRegistry()

    calculator = CalculatorTool()

    registry.register(calculator)

    print("Available tools:")
    print(registry.names())

    print("\nTool schema:")
    print(calculator.to_schema())

    print("\nExecuting calculator:")

    expression = "28374 * 928"

    result = registry.execute(
        "calculator",
        expression=expression,
    )

    print(f"{expression} = {result}")


if __name__ == "__main__":
    main()