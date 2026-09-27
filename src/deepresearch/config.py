import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_model: str
    llm_base_url: str | None


def load_settings() -> Settings:
    load_dotenv(
        override=False
    )

    api_key = os.getenv(
        "LLM_API_KEY",
        "",
    ).strip()

    model = os.getenv(
        "LLM_MODEL",
        "",
    ).strip()

    base_url = os.getenv(
        "LLM_BASE_URL",
        "",
    ).strip()

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is not configured."
        )

    if not api_key.isascii():
        raise RuntimeError(
            "LLM_API_KEY contains non-ASCII characters. "
            "Please replace placeholder text with a real API key."
        )

    if "your_api_key" in api_key.lower():
        raise RuntimeError(
            "LLM_API_KEY is still using the example placeholder."
        )

    if not model:
        raise RuntimeError(
            "LLM_MODEL is not configured."
        )

    if (
        "这里" in model
        or "your_model" in model.lower()
    ):
        raise RuntimeError(
            "LLM_MODEL is still using the example placeholder."
        )

    return Settings(
        llm_api_key=api_key,
        llm_model=model,
        llm_base_url=base_url or None,
    )