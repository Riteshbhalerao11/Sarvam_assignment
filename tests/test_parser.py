import pytest
from einops_rearrange import Parser, EinopsError
from einops_rearrange.parser import AnonymousAxis

def test_anonymous_axes():
    # output side numeric tokens (except "1") become AnonymousAxis.
    p = Parser("5 (3 4)", is_input=False)
    
    assert len(p.identifiers) == 3

    group0, group1 = p.structure
    assert isinstance(group0, list)
    assert isinstance(group0[0], AnonymousAxis)
    assert group0[0].value == 5

    assert isinstance(group1, list)
    assert isinstance(group1[0], AnonymousAxis)
    assert group1[0].value == 3
    assert isinstance(group1[1], AnonymousAxis)
    assert group1[1].value == 4

    anon_values = {axis.value for axis in p.identifiers if isinstance(axis, AnonymousAxis)}
    assert anon_values == {5, 3, 4}
    
    # Nonunitary anonymous axes in input
    with pytest.raises(EinopsError):
        p = Parser("5 (3 4)", is_input=True)
    
    # negative not allowed
    with pytest.raises(EinopsError):
        p = Parser("-10 2 a", is_input=True)
    
    # "1" tokens are simply skipped.
    p = Parser("1 1 1 ()", is_input=True)
    assert p.identifiers == set()
    assert p.structure == [[], [], [], []]

    p = Parser("1 1 1 ()", is_input=False)
    assert p.identifiers == set()
    assert p.structure == [[], [], [], []]

    p = Parser("5 (3 4)", is_input=False)
    


def test_invalid_ellipsis_expressions():

    Parser("... a b", is_input=True)
    Parser("(... a)", is_input=False)

    with pytest.raises(EinopsError):
        Parser("......", is_input=True)
    with pytest.raises(EinopsError):
        Parser("...a", is_input=True)
    # Multiple ellipses and ellipses inside paranthesis on left side
    with pytest.raises(EinopsError):
        Parser("... a b c d ...", is_input=True)
    with pytest.raises(EinopsError):
        Parser("... a b c (d ...)", is_input=True)
    with pytest.raises(EinopsError):
        Parser("(... a) b c (d ...)", is_input=True)


def test_invalid_paranthesis_expressions():

    Parser("... (a b)", is_input=True)
    Parser("(s d a) (...)", is_input=False)

    # Test mismatched parentheses.
    Parser("(a) b c (d ...)", is_input=False)
    with pytest.raises(EinopsError):
        Parser("(a)) b c (d ...)", is_input=False)
    with pytest.raises(EinopsError):
        Parser("(a b c (d ...)", is_input=False)
    with pytest.raises(EinopsError):
        Parser("(a) (() b c (d ...)", is_input=False)
    with pytest.raises(EinopsError):
        Parser("(a) ((b c) (d ...))", is_input=False)
    with pytest.raises(EinopsError):
        Parser("(() () a)", is_input=False)



def test_parse_expression():
    
    # Test a simple expression on the input side.
    p = Parser("  (  a   7  )   b   ( c  9   d  )   ", is_input=False)
    p = Parser("a1  b1   c1    d1", is_input=True)
    
    assert p.identifiers == {"a1", "b1", "c1", "d1"}
    assert p.structure == [["a1"], ["b1"], ["c1"], ["d1"]]
    assert not p.has_ellipsis

    # Test invalid identifiers and duplicate dimensions.
    Parser("camelCase under_scored cApiTaLs ß ...", is_input=True)
    with pytest.raises(EinopsError):
        Parser("1a", is_input=True)
    with pytest.raises(EinopsError):
        Parser("...pre", is_input=True)
    with pytest.raises(EinopsError):
        Parser("pre...", is_input=True)
    # Duplicate non-numeric axis on input side.
    with pytest.raises(EinopsError):
        Parser("a a", is_input=True)


    # Test mixed pattern with both named axes and anonymous axes along with ellipsis.
    p = Parser("name1 ... a1 12 (name2 14)", is_input=False)

    # Expected identifiers include the string names and anonymous axes created from "12" and "14".
    id_set = set()
    for token in p.identifiers:
        if isinstance(token, AnonymousAxis):
            id_set.add(token.value)
        else:
            id_set.add(token)
    # The set should contain the names, ellipsis as string, and the anonymous axis numeric values.
    assert id_set.issuperset({"name1", "ELLIPSIS", "a1", "name2", 12, 14})

    # Test multiple ellipses (should raise an error).
    with pytest.raises(EinopsError):
        Parser("a ... b ... c", is_input=False)

