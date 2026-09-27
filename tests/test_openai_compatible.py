import json

from deepresearch.llm import (
    Message,
    ToolCall,
)
from deepresearch.llm.openai_compatible import (
    OpenAICompatibleLLM,
)


def test_serialize_user_message() -> None:
    message = Message(
        role="user",
        content="hello",
    )

    result = (
        OpenAICompatibleLLM
        ._serialize_message(message)
    )

    assert result == {
        "role": "user",
        "content": "hello",
    }


def test_serialize_system_message() -> None:
    message = Message(
        role="system",
        content="You are helpful.",
    )

    result = (
        OpenAICompatibleLLM
        ._serialize_message(message)
    )

    assert result == {
        "role": "system",
        "content": "You are helpful.",
    }


def test_serialize_assistant_text_message() -> None:
    message = Message(
        role="assistant",
        content="Hello!",
    )

    result = (
        OpenAICompatibleLLM
        ._serialize_message(message)
    )

    assert result == {
        "role": "assistant",
        "content": "Hello!",
    }


def test_serialize_assistant_tool_call() -> None:
    message = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_001",
                name="calculator",
                arguments={
                    "expression": "12 * 8",
                },
            )
        ],
    )

    result = (
        OpenAICompatibleLLM
        ._serialize_message(message)
    )

    assert result["role"] == "assistant"

    assert len(
        result["tool_calls"]
    ) == 1

    tool_call = (
        result["tool_calls"][0]
    )

    assert tool_call["id"] == "call_001"

    assert (
        tool_call["function"]["name"]
        == "calculator"
    )

    arguments = json.loads(
        tool_call["function"]["arguments"]
    )

    assert arguments == {
        "expression": "12 * 8",
    }


def test_serialize_tool_message() -> None:
    message = Message(
        role="tool",
        content="96",
        name="calculator",
        tool_call_id="call_001",
    )

    result = (
        OpenAICompatibleLLM
        ._serialize_message(message)
    )

    assert result == {
        "role": "tool",
        "tool_call_id": "call_001",
        "content": "96",
    }


def test_unicode_tool_arguments() -> None:
    message = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_001",
                name="example_tool",
                arguments={
                    "query": "大模型 Agent",
                },
            )
        ],
    )

    result = (
        OpenAICompatibleLLM
        ._serialize_message(message)
    )

    arguments_text = (
        result["tool_calls"][0]
        ["function"]["arguments"]
    )

    assert "大模型 Agent" in (
        arguments_text
    )