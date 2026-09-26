from deepresearch.llm import (
    LLMResponse,
    Message,
    MockLLM,
    ToolCall,
)


def main() -> None:

    tool_call = ToolCall(
        id="call_001",
        name="calculator",
        arguments={
            "expression": "28374 * 928",
        },
    )

    llm = MockLLM(
        responses=[
            LLMResponse(
                tool_calls=[tool_call],
            ),
            LLMResponse(
                content="28374 × 928 = 26331072",
            ),
        ]
    )

    messages = [
        Message(
            role="user",
            content="Calculate 28374 * 928",
        )
    ]

    first_response = llm.chat(messages)

    print("First LLM response:")
    print(first_response)

    messages.append(
        Message(
            role="assistant",
            tool_calls=first_response.tool_calls,
        )
    )

    messages.append(
        Message(
            role="tool",
            content="26331072",
            name="calculator",
            tool_call_id="call_001",
        )
    )

    second_response = llm.chat(messages)

    print("\nSecond LLM response:")
    print(second_response)


if __name__ == "__main__":
    main()