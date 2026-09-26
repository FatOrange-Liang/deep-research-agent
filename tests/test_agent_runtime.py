import pytest

from deepresearch.agent import Agent
from deepresearch.llm import (
    LLMResponse,
    MockLLM,
    ToolCall,
)
from deepresearch.tools import (
    CalculatorTool,
    ToolRegistry,
)


def create_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        CalculatorTool()
    )

    return registry


def test_agent_executes_tool_and_returns_answer() -> None:

    tool_call = ToolCall(
        id="call_001",
        name="calculator",
        arguments={
            "expression": "12 * 8",
        },
    )

    llm = MockLLM(
        responses=[
            LLMResponse(
                tool_calls=[tool_call],
            ),
            LLMResponse(
                content="12 × 8 = 96",
            ),
        ]
    )

    agent = Agent(
        llm=llm,
        tools=create_registry(),
    )

    result = agent.run(
        "Calculate 12 * 8"
    )

    assert result.completed
    assert result.answer == "12 × 8 = 96"

    assert len(result.steps) == 2

    assert (
        result.steps[0]
        .observations[0]
        .content
        == "96"
    )


def test_tool_observation_is_sent_back_to_llm() -> None:

    tool_call = ToolCall(
        id="call_001",
        name="calculator",
        arguments={
            "expression": "20 + 22",
        },
    )

    llm = MockLLM(
        responses=[
            LLMResponse(
                tool_calls=[tool_call],
            ),
            LLMResponse(
                content="The answer is 42.",
            ),
        ]
    )

    agent = Agent(
        llm=llm,
        tools=create_registry(),
    )

    agent.run(
        "Calculate 20 + 22"
    )

    assert len(llm.calls) == 2

    second_call_messages, _ = (
        llm.calls[1]
    )

    tool_messages = [
        message
        for message in second_call_messages
        if message.role == "tool"
    ]

    assert len(tool_messages) == 1

    assert tool_messages[0].content == "42"

    assert (
        tool_messages[0].tool_call_id
        == "call_001"
    )


def test_tool_error_becomes_observation() -> None:

    tool_call = ToolCall(
        id="call_error",
        name="calculator",
        arguments={
            "expression": "10 / 0",
        },
    )

    llm = MockLLM(
        responses=[
            LLMResponse(
                tool_calls=[tool_call],
            ),
            LLMResponse(
                content=(
                    "The calculation failed "
                    "because division by zero "
                    "is not allowed."
                ),
            ),
        ]
    )

    agent = Agent(
        llm=llm,
        tools=create_registry(),
    )

    result = agent.run(
        "Calculate 10 / 0"
    )

    observation = (
        result.steps[0]
        .observations[0]
    )

    assert observation.is_error is True

    assert "TOOL_ERROR" in (
        observation.content
    )

    assert "Division by zero" in (
        observation.content
    )


def test_unknown_tool_does_not_crash_agent() -> None:

    tool_call = ToolCall(
        id="call_unknown",
        name="nonexistent_tool",
        arguments={},
    )

    llm = MockLLM(
        responses=[
            LLMResponse(
                tool_calls=[tool_call],
            ),
            LLMResponse(
                content=(
                    "The requested tool "
                    "is unavailable."
                ),
            ),
        ]
    )

    agent = Agent(
        llm=llm,
        tools=create_registry(),
    )

    result = agent.run(
        "Use an unavailable tool."
    )

    observation = (
        result.steps[0]
        .observations[0]
    )

    assert observation.is_error

    assert "TOOL_ERROR" in (
        observation.content
    )


def test_agent_stops_at_max_steps() -> None:

    responses = []

    for i in range(3):

        responses.append(
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id=f"call_{i}",
                        name="calculator",
                        arguments={
                            "expression": "1 + 1",
                        },
                    )
                ]
            )
        )

    llm = MockLLM(
        responses=responses
    )

    agent = Agent(
        llm=llm,
        tools=create_registry(),
        max_steps=3,
    )

    result = agent.run(
        "Keep calculating."
    )

    assert result.completed is False

    assert (
        result.stop_reason
        == "max_steps"
    )

    assert result.answer is None

    assert len(result.steps) == 3


def test_empty_task_is_rejected() -> None:

    llm = MockLLM(
        responses=[]
    )

    agent = Agent(
        llm=llm,
        tools=create_registry(),
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        agent.run("   ")