import pytest
import json

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

from deepresearch.tools import BaseTool

from deepresearch.research import (
    EvidencePolicy,
    EvidencePolicyConfig,
    ResearchController,
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


class FakeWebSearchTool(BaseTool):

    name = "web_search"
    description = "Fake web search."

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def execute(
        self,
        **kwargs,
    ) -> str:

        return json.dumps(
            {
                "query": kwargs["query"],
                "results": [
                    {
                        "source_id":
                            "S_12345678",
                        "title":
                            "LangGraph docs",
                        "url":
                            "https://example.com/langgraph",
                        "content":
                            "Search snippet",
                        "score":
                            0.9,
                    }
                ],
            }
        )


class FakeWebPageReaderTool(BaseTool):

    name = "web_page_reader"
    description = "Fake page reader."

    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
            }
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def execute(
        self,
        **kwargs,
    ) -> str:

        return json.dumps(
            {
                "source_id":
                    "S_12345678",
                "title":
                    "LangGraph docs",
                "url":
                    kwargs["url"],
                "content":
                    "Full page evidence",
                "content_type":
                    "text/html",
                "truncated":
                    False,
            }
        )

def test_agent_updates_research_state_from_search() -> None:

    llm = MockLLM(
        responses=[
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call_search",
                        name="web_search",
                        arguments={
                            "query": "LangGraph",
                        },
                    )
                ]
            ),
            LLMResponse(
                content="Done."
            ),
        ]
    )

    registry = ToolRegistry()

    registry.register(
        FakeWebSearchTool()
    )

    agent = Agent(
        llm=llm,
        tools=registry,
    )

    result = agent.run(
        "Research LangGraph"
    )

    assert result.research_state is not None

    state = result.research_state

    assert state.source_count == 1

    assert state.search_queries == [
        "LangGraph"
    ]

    assert (
        state.tool_observation_count
        == 1
    )

    record = state.sources[
        "S_12345678"
    ]

    assert (
        record.discovered_by_search
        is True
    )

    assert (
        record.read_full_page
        is False
    )

def test_agent_enriches_research_state_from_page_reader() -> None:

    llm = MockLLM(
        responses=[
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call_search",
                        name="web_search",
                        arguments={
                            "query": "LangGraph",
                        },
                    )
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call_reader",
                        name="web_page_reader",
                        arguments={
                            "url":
                                "https://example.com/langgraph",
                        },
                    )
                ]
            ),
            LLMResponse(
                content="Done."
            ),
        ]
    )

    registry = ToolRegistry()

    registry.register(
        FakeWebSearchTool()
    )

    registry.register(
        FakeWebPageReaderTool()
    )

    agent = Agent(
        llm=llm,
        tools=registry,
    )

    result = agent.run(
        "Research LangGraph"
    )

    assert result.research_state is not None

    state = result.research_state

    assert state.source_count == 1

    assert state.read_source_count == 1

    assert state.unread_source_count == 0

    assert (
        state.tool_observation_count
        == 2
    )

    record = state.sources[
        "S_12345678"
    ]

    assert (
        record.source.content
        == "Full page evidence"
    )

    assert (
        record.source.score
        == 0.9
    )

    assert (
        record.read_full_page
        is True
    )

    assert (
        record.content_type
        == "text/html"
    )

    assert (
        record.truncated
        is False
    )


def test_research_controller_blocks_premature_answer() -> None:

    llm = MockLLM(
        responses=[
            # Step 1: LLM tries to answer without research.
            LLMResponse(
                content="Premature answer."
            ),

            # Step 2: Controller requires search.
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call_search",
                        name="web_search",
                        arguments={
                            "query": "LangGraph"
                        },
                    )
                ]
            ),

            # Step 3: LLM tries to finish again.
            LLMResponse(
                content="Still premature."
            ),

            # Step 4: Controller requires page reading.
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call_reader",
                        name="web_page_reader",
                        arguments={
                            "url":
                                "https://example.com/langgraph"
                        },
                    )
                ]
            ),

            # Step 5: Research is complete.
            LLMResponse(
                content="Final researched answer."
            ),
        ]
    )

    registry = ToolRegistry()

    registry.register(
        FakeWebSearchTool()
    )

    registry.register(
        FakeWebPageReaderTool()
    )

    policy = EvidencePolicy(
        EvidencePolicyConfig(
            min_sources=1,
            min_full_pages=1,
            min_search_queries=1,
        )
    )

    controller = ResearchController(
        policy=policy
    )

    agent = Agent(
        llm=llm,
        tools=registry,
        research_controller=controller,
        max_steps=5,
    )

    result = agent.run(
        "Research LangGraph"
    )

    # The first two premature answers must not
    # terminate the agent.
    assert result.completed

    assert (
        result.answer
        == "Final researched answer."
    )

    assert len(result.steps) == 5

    # Research evidence must have been collected.
    assert result.research_state is not None

    state = result.research_state

    assert state.source_count == 1

    assert state.read_source_count == 1

    assert state.tool_observation_count == 2

def test_agent_without_controller_can_finish_directly() -> None:

    llm = MockLLM(
        responses=[
            LLMResponse(
                content="Direct answer."
            )
        ]
    )

    agent = Agent(
        llm=llm,
        tools=ToolRegistry(),
    )

    result = agent.run(
        "Answer directly."
    )

    assert result.completed

    assert result.answer == "Direct answer."

    assert len(result.steps) == 1

    assert result.research_state is not None