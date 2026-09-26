import ast
import operator
from typing import Any

from .base import BaseTool


class CalculatorTool(BaseTool):
    name = "calculator"

    description = (
        "Evaluate a mathematical arithmetic expression. "
        "Use this tool when exact numerical calculation is required."
    )

    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, for example: 1729 * 346",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    _binary_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    _unary_operators = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def execute(self, **kwargs: Any) -> str:
        expression = kwargs.get("expression")

        if not isinstance(expression, str):
            raise ValueError("'expression' must be a string.")

        expression = expression.strip()

        if not expression:
            raise ValueError("'expression' cannot be empty.")

        if len(expression) > 200:
            raise ValueError("Expression is too long.")

        try:
            tree = ast.parse(expression, mode="eval")
            result = self._evaluate(tree.body)
        except ZeroDivisionError:
            raise ValueError("Division by zero is not allowed.")
        except (SyntaxError, TypeError):
            raise ValueError(f"Invalid arithmetic expression: {expression}")

        return str(result)

    def _evaluate(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise ValueError("Boolean values are not supported.")

            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Only numeric constants are supported.")

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)

            if operator_type not in self._binary_operators:
                raise ValueError(
                    f"Operator {operator_type.__name__} is not supported."
                )

            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large.")

            return self._binary_operators[operator_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)

            if operator_type not in self._unary_operators:
                raise ValueError(
                    f"Operator {operator_type.__name__} is not supported."
                )

            operand = self._evaluate(node.operand)

            return self._unary_operators[operator_type](operand)

        raise ValueError(
            f"Unsupported expression element: {type(node).__name__}"
        )