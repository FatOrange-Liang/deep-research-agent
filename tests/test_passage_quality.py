from deepresearch.research.passage_quality import (
    assess_passage,
)


def test_clean_prose_is_kept():
    result = assess_passage(
        "Interrupts allow graph execution "
        "to pause and wait for external input."
    )

    assert result.keep is True
    assert result.tier == "clean"


def test_useful_technical_text_is_not_rejected():
    result = assess_passage(
        "Use Command(resume=True) to resume "
        "execution after an interrupt."
    )

    assert result.keep is True


def test_broken_code_fragment_is_rejected():
    result = assess_passage(
        '" ) age = interrupt ( "What\'s your age?'
    )

    assert result.keep is False
    assert result.tier == "reject"

    assert (
        "broken_code_fragment"
        in result.reasons
    )


def test_broken_prefix_code_is_rejected():
    result = assess_passage(
        'ART , "a" ) . add_edge ( START , "b" ) '
        '. add_edge ( "a" , END ) graph = builder .'
    )

    assert result.keep is False


def test_code_like_does_not_mean_bad():
    result = assess_passage(
        "The graph can be resumed by calling "
        "graph.invoke(Command(resume=True), config)."
    )

    assert result.keep is True