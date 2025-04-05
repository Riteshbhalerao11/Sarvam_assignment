import pytest
from einops_rearrange.transform import extract_information, ExtractedInfo
from einops_rearrange import EinopsError

def test_basic_pattern():
    # Test extracting information from a basic pattern
    pattern = "a b c -> c b a"
    known_axes = {}
    ndim = 3
    
    result = extract_information(pattern, known_axes, ndim)
    assert isinstance(result,ExtractedInfo)
    assert len(result.known_unk) == 3
    assert len(result.permutation_order) == 3
    assert result.permutation_order == [2, 1, 0]
    assert list(result.axis_position_map.keys()) == ['a', 'b', 'c']
    assert result.output_structure == [['c'], ['b'], ['a']]
    assert result.added_axes == {}

def test_reshape_pattern():
    # Test extracting information from a pattern with reshaping.
    pattern = "a b c -> (a b) c"
    known_axes = {}
    ndim = 3
    
    result = extract_information(pattern, known_axes, ndim)
    
    assert len(result.output_structure) == 2
    assert result.output_structure == [['a', 'b'], ['c']]
    assert result.permutation_order == [0, 1, 2] 
    assert result.added_axes == {}

def test_with_known_axes():
    # Test with known axes provided
    pattern = "a b c -> (a d) c b"
    known_axes = {'d': 5}
    ndim = 3
    
    result = extract_information(pattern, known_axes, ndim)
    
    assert 'd' in result.axis_position_map
    assert result.added_axes != {}
    assert list(result.added_axes.values())[0] == 5  
    assert result.permutation_order == [0,2,1]

def test_ellipsis_pattern():
    # Test extracting information from a pattern with ellipsis
    pattern = "a ... c -> c ... a"
    known_axes = {}
    ndim = 5  # a, b, c, d, e
    
    result = extract_information(pattern, known_axes, ndim)
    
    assert len(result.permutation_order) == 5
    assert result.permutation_order[0] == 4  
    assert result.permutation_order[-1] == 0  

def test_complex_reshape():
    # Test extracting information from a complex reshape pattern.
    pattern = "(a b) (c d) -> a c (b d)"
    known_axes = {'a': 2, 'b': 3, 'c': 4, 'd': 5}
    ndim = 2
    
    result = extract_information(pattern, known_axes, ndim)

    assert len(result.known_unk) == 2
    assert result.output_structure == [['a'], ['c'], ['b', 'd']]
    assert -1 not in result.axis_lengths  

    pattern = "(a b) (c d e) -> (a) (c e h) (b d 2)"
    known_axes = {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'h' : 10}
    ndim = 2
    
    result = extract_information(pattern, known_axes, ndim)

    assert result.output_structure[:-1] == [['a'], ['c', 'e', 'h']]
    assert result.output_structure[-1][-1].value == 2 # Anonymous axis value
    assert result.axis_lengths == [2, 3, 4, 5, 6, 10, 2]
    assert result.added_axes == {3: 10, 6: 2}


def test_empty_pattern():
    with pytest.raises(EinopsError):
        extract_information("", {}, 3)

def test_missing_arrow():
    with pytest.raises(EinopsError):
        extract_information("a b c", {}, 3)

def test_multiple_arrows():
    with pytest.raises(EinopsError):
        extract_information("a b -> b a -> a b", {}, 2)

def test_missing_identifiers():
    with pytest.raises(EinopsError):
        extract_information("-> a b c", {}, 0)
        
    with pytest.raises(EinopsError):
        extract_information("a b c ->", {}, 3)


def test_ellipsis_on_input_only():
    with pytest.raises(EinopsError):
        extract_information("a ... c -> a b c", {}, 4)

def test_unexpected_identifiers():
    with pytest.raises(EinopsError):
        extract_information("a b c -> a b", {}, 3)

def test_unspecified_new_axes():
    with pytest.raises(EinopsError):
        extract_information("a b -> a b c", {}, 2)

def test_shape_mismatch():
    # Pattern requires 3 dims, tensor has 2
    with pytest.raises(EinopsError):
        extract_information("a b c -> c b a", {}, 2)
    
    # Pattern requires 2 dims, tensor has 3
    with pytest.raises(EinopsError):
        extract_information("a b -> b a", {}, 3)

def test_ellipsis_shape_mismatch():
    with pytest.raises(EinopsError):
        # Pattern requires at least 2 static dims, tensor has only 1
        extract_information("a ... c -> c ... a", {}, 1)

def test_repeated_axis_name():
    pattern = "a a -> a a"
    known_axes = {}
    ndim = 2
    
    with pytest.raises(EinopsError):
        extract_information(pattern, known_axes, ndim)

def test_unspecified_axis_in_kwargs():
    pattern = "a b -> b a"
    known_axes = {'c': 5}
    ndim = 2
    
    with pytest.raises(EinopsError):
        extract_information(pattern, known_axes, ndim)


def test_ellipsis_pattern():
    pattern = "a ... (c d) -> (a c) ... d"
    known_axes = {'c': 3, 'd': 4}
    ndim = 5  # a, b1, b2, c, d
    
    result = extract_information(pattern, known_axes, ndim)
    
    assert len(result.permutation_order) == 6
    assert result.output_structure[0] == ['a', 'c']
    assert result.output_structure[-1] == ['d']
    assert result.output_structure[1:-1] == [[f"ELLIPSIS_{i}"] for i in range(3)]

    pattern = "a ... (c d) -> (a c) (...) d"
    known_axes = {'c': 3, 'd': 4}
    ndim = 5  # a, b1, b2, c, d
    
    result = extract_information(pattern, known_axes, ndim)
    
    assert len(result.permutation_order) == 6
    assert result.output_structure[0] == ['a', 'c']
    assert result.output_structure[-1] == ['d']
    assert result.output_structure[1] == [f"ELLIPSIS_{i}" for i in range(3)]


def test_mismatched_parentheses():
    pattern = "((a b) c) -> a (b c)"
    known_axes = {'a': 2, 'b': 3, 'c': 4}
    ndim = 1
    with pytest.raises(EinopsError):
        extract_information(pattern, known_axes, ndim)
    
    pattern = "((a b)) -> (a) (b)"
    known_axes = {'a': 2, 'b': 3}
    ndim = 1
    with pytest.raises(EinopsError):
        extract_information(pattern, known_axes, ndim)
    
