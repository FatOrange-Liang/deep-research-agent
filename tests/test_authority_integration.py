
import json

from deepresearch.agent import Agent

from deepresearch.llm import (
    LLMResponse,
    MockLLM,
    ToolCall,
)

from deepresearch.tools import (
    BaseTool,
    ToolRegistry,
)

from deepresearch.research import (
    EvidencePolicy,
    EvidencePolicyConfig,
    ResearchController,
)

from deepresearch.research.sources import (
    make_source_id,
)


# =========================================
# Test URLs
# =========================================

OFFICIAL_URL = (
    "https://docs.langchain.com/"
    "oss/python/langgraph/overview"
)

BLOG_URL = (
    "https://example.com/blog"
)

ARTICLE_URL = (
    "https://example.org/article"
)

OFFICIAL_ID = make_source_id(
    OFFICIAL_URL
)

BLOG_ID = make_source_id(
    BLOG_URL
)

ARTICLE_ID = make_source_id(
    ARTICLE_URL
)


# =========================================
# Fake Web Search
# =========================================

class FakeAuthoritySearchTool(BaseTool):

    name = "web_search"

    description = "Offline research search tool."

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

    def execute(self, **kwargs) -> str:

        return json.dumps(
            {
                "query": kwargs["query"],
                "results": [
                    {
                        "source_id": BLOG_ID,
                        "title": "Community Blog",
                        "url": BLOG_URL,
                        "content": "Blog snippet",
                        "score": 0.99,
                    },
                    {
                        "source_id": ARTICLE_ID,
                        "title": "Third-party Article",
                        "url": ARTICLE_URL,
                        "content": "Article snippet",
                        "score": 0.85,
                    },
                    {
                        "source_id": OFFICIAL_ID,
                        "title": "LangGraph Official Docs",
                        "url": OFFICIAL_URL,
                        "content": "Official documentation snippet",
                        "score": 0.65,
                    },
                ],
            },
            ensure_ascii=False,
        )


# =========================================
# Fake Web Page Reader
# =========================================

class FakeAuthorityReaderTool(BaseTool):

    name = "web_page_reader"

    description = "Offline page reader."

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

    def execute(self, **kwargs) -> str:

        url = kwargs["url"]

        if url == OFFICIAL_URL:

            content = (
                "Official LangGraph documentation. "
                "LangGraph supports persistence "
                "through checkpointers and "
                "human-in-the-loop through interrupts."
            )

        elif url == BLOG_URL:

            content = (
                "A community blog discussing LangGraph."
            )

        else:

            raise ValueError(
                f"Unexpected test URL: {url}"
            )

        return json.dumps(
            {
                "source_id": make_source_id(url),
                "title": "Full Page",
                "url": url,
                "content": content,
                "content_type": "text/html",
                "truncated": False,
            },
            ensure_ascii=False,
        )


# =========================================
# Build Research Agent
# =========================================

def build_research_agent(
    *,
    responses: list[LLMResponse],
    max_steps: int,
) -> Agent:

    llm = MockLLM(
        responses=responses
    )

    registry = ToolRegistry()

    registry.register(
        FakeAuthoritySearchTool()
    )

    registry.register(
        FakeAuthorityReaderTool()
    )

    policy = EvidencePolicy(
        EvidencePolicyConfig(
            min_sources=3,
            min_full_pages=1,
            min_search_queries=1,
            required_hosts=(
                "docs.langchain.com",
            ),
            min_required_host_full_pages=1,
        )
    )

    controller = ResearchController(
        policy=policy
    )

    return Agent(
        llm=llm,
        tools=registry,
        research_controller=controller,
        max_steps=max_steps,
    )


# =========================================
# Test 1:
# Official document must be read
# =========================================

def test_agent_requires_official_full_page() -> None:

    final_answer = (
        "LangGraph supports persistence "
        f"and interrupts [{OFFICIAL_ID}]."
    )

    agent = build_research_agent(
        responses=[
            # Step 1: Search.
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="search_1",
                        name="web_search",
                        arguments={
                            "query": "LangGraph official docs"
                        },
                    )
                ]
            ),

            # Step 2: Read a blog.
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="reader_blog",
                        name="web_page_reader",
                        arguments={
                            "url": BLOG_URL
                        },
                    )
                ]
            ),

            # Step 3: Premature final answer.
            LLMResponse(
                content="I have enough evidence."
            ),

            # Step 4: Read official documentation.
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="reader_official",
                        name="web_page_reader",
                        arguments={
                            "url": OFFICIAL_URL
                        },
                    )
                ]
            ),

            # Step 5: Final answer.
            LLMResponse(
                content=final_answer
            ),
        ],
        max_steps=5,
    )

    result = agent.run(
        "Research official LangGraph documentation."
    )

    assert result.completed

    assert result.answer == final_answer

    assert len(result.steps) == 5

    assert result.research_state is not None

    state = result.research_state

    assert state.source_count == 3

    assert state.read_source_count == 2

    assert state.sources[
        BLOG_ID
    ].read_full_page

    assert state.sources[
        OFFICIAL_ID
    ].read_full_page

    # Controller must have rejected the premature
    # answer and recommended the official document.
    assert any(
        message.role == "user"
        and message.content
        and OFFICIAL_URL in message.content
        for message in result.messages
    )


# =========================================
# Test 2:
# Blogs cannot replace official evidence
# =========================================

def test_blog_only_research_cannot_finish() -> None:

    agent = build_research_agent(
        responses=[
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="search_1",
                        name="web_search",
                        arguments={
                            "query": "LangGraph docs"
                        },
                    )
                ]
            ),

            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="reader_blog",
                        name="web_page_reader",
                        arguments={
                            "url": BLOG_URL
                        },
                    )
                ]
            ),

            # LLM tries to finish without reading
            # the official document.
            LLMResponse(
                content="Final answer from blog."
            ),
        ],
        max_steps=3,
    )

    result = agent.run(
        "Research LangGraph official docs."
    )

    assert not result.completed

    assert result.stop_reason == "max_steps"

    assert result.answer is None

    assert result.research_state is not None

    state = result.research_state

    assert state.source_count == 3

    assert state.read_source_count == 1

    assert not state.sources[
        OFFICIAL_ID
    ].read_full_page

    control = (
        agent.research_controller.evaluate(state)
    )

    assert not control.can_finish

    assert control.decision.action == "read_more"

    assert (
        control.decision.recommended_source_ids
        == (OFFICIAL_ID,)
    )
