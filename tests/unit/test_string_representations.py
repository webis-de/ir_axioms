from axioms.tools.terrier import TerrierIndexContext


def test_string_representation_of_terrier_index_context_01():
    # Needed for caching
    expected = "TerrierIndexContext(index_location)"
    actual = str(TerrierIndexContext(index_location="index_location"))
    assert expected == actual


def test_string_representation_of_terrier_index_context_02():
    # Needed for caching
    expected = "TerrierIndexContext(index_location)"
    actual = str(TerrierIndexContext(index_location="ignore/absolute/path/index_location"))
    assert expected == actual


def test_string_representation_of_terrier_index_context_03():
    # Needed for caching
    expected = "TerrierIndexContext(index_location)"
    actual = str(
        TerrierIndexContext(index_location="ignore/absolute/path/index_location ignore suffix")
    )
    assert expected == actual
