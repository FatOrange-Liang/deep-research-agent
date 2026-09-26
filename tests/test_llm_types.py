import pytest

from deepresearch.llm import (
    LLMResponse,
    Message,
    ToolCall,
)


def test_user_message() -> None:
    message = Message(
        role="user",
        content="Hello",
    )

    assert message.role == "user"
    assert message.content == "Hello"
    assert message.tool_calls == []


def test_tool_call() -> None:
    call = ToolCall(
        id="call_001",
        name="calculator",
        arguments={
            "expression": "12 * 8",
        },
    )

    assert call.id == "call_001"
    assert call.name == "calculator"
    assert call.arguments["expression"] == "12 * 8"


def test_tool_message_requires_tool_call_id() -> None:
    with pytest.raises(
        ValueError,
        match="tool_call_id",
    ):
        Message(
            role="tool",
            content="96",
        )


def test_valid_tool_message() -> None:
    message = Message(
        role="tool",
        content="96",
        name="calculator",
        tool_call_id="call_001",
    )

    assert message.content == "96"
    assert message.name == "calculator"
    assert message.tool_call_id == "call_001"


def test_invalid_role() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid message role",
    ):
        Message(
            role="invalid",  # type: ignore[arg-type]
            content="Hello",
        )


def test_response_with_tool_call() -> None:
    call = ToolCall(
        id="call_001",
        name="calculator",
        arguments={
            "expression": "12 * 8",
        },
    )

    response = LLMResponse(
        tool_calls=[call],
    )

    assert response.has_tool_calls is True
    assert response.is_final_answer is False


def test_final_answer_response() -> None:
    response = LLMResponse(
        content="The answer is 96.",
    )

    assert response.has_tool_calls is False
    assert response.is_final_answer is True