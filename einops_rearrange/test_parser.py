import pytest
from . import EinopsError
from .parser import Parser, AnonymousAxis

def test_anonymous_axes():
    # Test that on the output side numeric tokens (except "1") become AnonymousAxis.
    p = Parser("5 (3 4)", is_input=False)
    # Expected structure: [ [AnonymousAxis("5")], [AnonymousAxis("3"), AnonymousAxis("4")]]
    group0, group1 = p.structure
    assert isinstance(group0, list)
    assert isinstance(group0[0], AnonymousAxis)
    assert group0[0].value == 5

    assert isinstance(group1, list)
    assert isinstance(group1[0], AnonymousAxis)
    assert group1[0].value == 3
    assert isinstance(group1[1], AnonymousAxis)
    assert group1[1].value == 4

    # The identifiers set should contain these anonymous axis objects.
    # We extract numeric values from identifiers for comparison.
    anon_values = {axis.value for axis in p.identifiers if isinstance(axis, AnonymousAxis)}
    assert anon_values == {5, 3, 4}


def test_elementary_axis_name():
    # Test valid non-numeric axis names (input side).
    valid_names = [
        "a", "b", "h", "dx", "h1", "zz", "i9123",
        "somelongname", "Alex", "camelCase", "u_n_d_e_r_score",
        "unreasonablyLongAxisName"
    ]
    for name in valid_names:
        p = Parser(name, is_input=True)
        # Since the token is valid and not numeric, it appears as a singleton group.
        assert p.structure == [[name]]
        assert name in p.identifiers

    invalid_names = ["2b", "12", "@", "abc..."]
    for name in invalid_names:
        with pytest.raises(EinopsError):
            Parser(name, is_input=True)


def test_invalid_expressions():
    # Test multiple ellipses or misplaced dots.
    # A correct ellipsis outside any parentheses is allowed if exactly one appears.
    Parser("... a b c d", is_input=True)
    with pytest.raises(EinopsError):
        Parser("... a b c d ...", is_input=True)
    with pytest.raises(EinopsError):
        Parser("... a b c (d ...)", is_input=True)
    with pytest.raises(EinopsError):
        Parser("(... a) b c (d ...)", is_input=True)

    # Test mismatched parentheses.
    Parser("(a) b c (d ...)", is_input=False)
    with pytest.raises(EinopsError):
        Parser("(a)) b c (d ...)", is_input=False)
    with pytest.raises(EinopsError):
        Parser("(a b c (d ...)", is_input=False)
    with pytest.raises(EinopsError):
        Parser("(a) (()) b c (d ...)", is_input=False)
    with pytest.raises(EinopsError):
        Parser("(a) ((b c) (d ...))", is_input=False)

    # Test invalid identifiers and duplicate dimensions.
    # Note: the valid Python identifier 'ß' is allowed, so we do not expect an error here.
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

    # Test numeric tokens on input side (other than "1") should not be allowed.
    with pytest.raises(EinopsError):
        Parser("5", is_input=True)
    # "1" tokens are simply skipped.
    p = Parser("1 1 1 ()", is_input=True)
    assert p.identifiers == set()
    assert p.structure == [[], [], [], []]

    # Ellipsis inside parentheses on the input side is not allowed.
    with pytest.raises(EinopsError):
        Parser("( ... a) b", is_input=True)


def test_parse_expression():
    # Test a simple expression on the input side.
    p = Parser("a1  b1   c1    d1", is_input=True)
    assert p.identifiers == {"a1", "b1", "c1", "d1"}
    assert p.structure == [["a1"], ["b1"], ["c1"], ["d1"]]
    assert not p.has_ellipsis

    # Test explicit "1" tokens should be skipped.
    p = Parser("1 1 1 ()", is_input=True)
    assert p.identifiers == set()
    assert p.structure == [[], [], [], []]
    assert not p.has_ellipsis

    # Test output side expression with anonymous axes.
    p = Parser("5 (3 4)", is_input=False)
    group0, group1 = p.structure
    assert isinstance(group0[0], AnonymousAxis)
    assert group0[0].value == 5
    assert isinstance(group1[0], AnonymousAxis)
    assert group1[0].value == 3
    assert isinstance(group1[1], AnonymousAxis)
    assert group1[1].value == 4
    anon_vals = {axis.value for axis in p.identifiers if isinstance(axis, AnonymousAxis)}
    assert anon_vals == {5, 3, 4}

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
    assert id_set.issuperset({"name1", "...", "a1", "name2", 12, 14})

    # Test multiple ellipses (should raise an error).
    with pytest.raises(EinopsError):
        Parser("a ... b ... c", is_input=False)


def test_flat_axes_order():
    # Test the flat_axes_order method that flattens nested axes.
    p = Parser("(a b) c ... (d)", is_input=False)
    # Expected structure:
    #   group 0: ["a", "b"]
    #   token: "c"
    #   token: "..."
    #   group 1: ["d"]
    # The flat_axes_order should return: ["a", "b", "c", "...", "d"]
    flat = p.flat_axes_order()
    expected = ["a", "b", "c", "...", "d"]
    assert flat == expected


def test_complex_edge_cases():
    # Test extra whitespace, nested parentheses without inner ellipsis, and more.
    # Multiple valid groups with mix of numeric and named tokens on output side.
    p = Parser("  (  a   7  )   b   ( c  9   d  )   ", is_input=False)
    # Expected structure:
    #   group 0: ["a", AnonymousAxis("7")] where 7 > 1
    #   group 1: ["b"]
    #   group 2: ["c", AnonymousAxis("9"), "d"]
    group0, group1, group2 = p.structure
    assert group0[0] == "a"
    assert isinstance(group0[1], AnonymousAxis)
    assert group0[1].value == 7
    assert group1 == ["b"]
    assert group2[0] == "c"
    assert isinstance(group2[1], AnonymousAxis)
    assert group2[1].value == 9
    assert group2[2] == "d"

    # Test a case with multiple nested parentheses that are valid.
    # Although our parser does not support nested parentheses (i.e. parentheses inside parentheses),
    # we can test consecutive groups.
    p = Parser("(a) (b) (c)", is_input=True)
    assert p.structure == [["a"], ["b"], ["c"]]

    # Test a case where an explicit "1" token is provided in a nested group.
    p = Parser("(a 1 b)", is_input=True)
    # "1" is skipped, so group becomes ["a", "b"]
    assert p.structure == [["a", "b"]]

    # Test an expression with ellipsis and adjacent parentheses.
    p = Parser("x (y) ... z", is_input=False)
    # Expected structure: [["x"], ["y"], "...", ["z"]]
    assert p.structure[0] == ["x"]
    assert p.structure[1] == ["y"]
    assert p.structure[2] == "..."
    assert p.structure[3] == ["z"]

    # Test a case where numeric tokens are improperly used on the input side.
    with pytest.raises(EinopsError):
        Parser("7", is_input=True)
