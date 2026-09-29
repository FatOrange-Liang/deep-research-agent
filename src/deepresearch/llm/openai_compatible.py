import json
from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from .base import BaseLLM
from .types import (
    LLMResponse,
    Message,
    ToolCall,
)


class OpenAICompatibleLLM(BaseLLM):
    """
    LLM adapter for OpenAI-style Chat Completions APIs.

    The rest of the agent runtime remains provider-independent.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        max_completion_tokens: int = 2048,
    ) -> None:

        model = model.strip()
        api_key = api_key.strip()

        if not model:
            raise ValueError(
                "'model' cannot be empty."
            )

        if not api_key:
            raise ValueError(
                "'api_key' cannot be empty."
            )

        if max_completion_tokens <= 0:
            raise ValueError(
                "'max_completion_tokens' "
                "must be greater than zero."
            )

        self.model = model
        self.max_completion_tokens = (
            max_completion_tokens
        )

        client_kwargs = {
            "api_key": api_key,
        }

        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(
            **client_kwargs
        )

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]]
        | None = None,
    ) -> LLMResponse:

        provider_messages = [
            self._serialize_message(message)
            for message in messages
        ]

        request = {
            "model": self.model,
            "messages": [
                self._serialize_message(message)
                for message in messages
            ],
            "max_completion_tokens":
                self.max_completion_tokens,
        }

        if self.model.startswith("gpt-6"):
            request["reasoning_effort"] = "none"

        if tools:
            request["tools"] = list(tools)
            request["tool_choice"] = "auto"


        # serialized_messages = request.get(
        #     "messages",
        #     [],
        # )

        # serialized_tools = request.get(
        #     "tools",
        #     [],
        # )

        # message_char_counts = [
        #     len(
        #         json.dumps(
        #             message,
        #             ensure_ascii=False,
        #         )
        #     )
        #     for message in serialized_messages
        # ]

        # tool_char_counts = [
        #     len(
        #         json.dumps(
        #             tool,
        #             ensure_ascii=False,
        #         )
        #     )
        #     for tool in serialized_tools
        # ]

        # print(
        #     "DEBUG LLM REQUEST:",
        #     {
        #         "model":
        #             request.get("model"),

        #         "max_completion_tokens":
        #             request.get(
        #                 "max_completion_tokens"
        #             ),

        #         "reasoning_effort":
        #             request.get(
        #                 "reasoning_effort"
        #             ),

        #         "message_count":
        #             len(serialized_messages),

        #         "message_char_counts":
        #             message_char_counts,

        #         "message_chars_total":
        #             sum(message_char_counts),

        #         "tool_count":
        #             len(serialized_tools),

        #         "tool_char_counts":
        #             tool_char_counts,

        #         "tool_chars_total":
        #             sum(tool_char_counts),
        #     },
        # )

        # raise RuntimeError(
        #     "DEBUG_STOP_BEFORE_API_CALL"
        # )


        completion = (
            self.client.chat.completions.create(
                **request
            )
        )

        message = (
            completion
            .choices[0]
            .message
        )

        tool_calls: list[ToolCall] = []

        if message.tool_calls:

            for provider_call in (
                message.tool_calls
            ):

                arguments_text = (
                    provider_call
                    .function
                    .arguments
                )

                try:
                    arguments = json.loads(
                        arguments_text
                    )
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "LLM returned invalid JSON "
                        "tool arguments: "
                        f"{arguments_text}"
                    ) from exc

                if not isinstance(
                    arguments,
                    dict,
                ):
                    raise ValueError(
                        "Tool arguments must decode "
                        "to a JSON object."
                    )

                tool_calls.append(
                    ToolCall(
                        id=provider_call.id,
                        name=(
                            provider_call
                            .function
                            .name
                        ),
                        arguments=arguments,
                    )
                )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _serialize_message(
        message: Message,
    ) -> dict[str, Any]:

        # ----------------------------------
        # System / User
        # ----------------------------------

        if message.role in {
            "system",
            "user",
        }:
            return {
                "role": message.role,
                "content": (
                    message.content or ""
                ),
            }

        # ----------------------------------
        # Assistant
        # ----------------------------------

        if message.role == "assistant":

            serialized: dict[str, Any] = {
                "role": "assistant",
                "content": message.content,
            }

            if message.tool_calls:

                serialized["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name":
                                tool_call.name,
                            "arguments":
                                json.dumps(
                                    tool_call
                                    .arguments,
                                    ensure_ascii=False,
                                ),
                        },
                    }
                    for tool_call
                    in message.tool_calls
                ]

            return serialized

        # ----------------------------------
        # Tool observation
        # ----------------------------------

        if message.role == "tool":

            return {
                "role": "tool",
                "tool_call_id":
                    message.tool_call_id,
                "content":
                    message.content or "",
            }

        raise ValueError(
            f"Unsupported message role: "
            f"{message.role}"
        )