import pytest

from deepresearch.research import (
    make_source_id,
)


def test_source_id_is_stable() -> None:

    url = (
        "https://example.com/page"
    )

    first = make_source_id(url)
    second = make_source_id(url)

    assert first == second


def test_different_urls_have_different_ids() -> None:

    first = make_source_id(
        "https://example.com/a"
    )

    second = make_source_id(
        "https://example.com/b"
    )

    assert first != second


def test_source_id_format() -> None:

    source_id = make_source_id(
        "https://example.com"
    )

    assert source_id.startswith(
        "S_"
    )

    assert len(source_id) == 10


def test_empty_url_is_rejected() -> None:

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        make_source_id("   ")