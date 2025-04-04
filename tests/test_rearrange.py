import pytest
import numpy as np
from einops_rearrange import rearrange, EinopsError


def test_rearrange_transpose():    
    # Basic transpose operation.
    x = np.random.rand(3, 4)
    result = rearrange(x, "h w -> w h")
    expected = x.T 
    np.testing.assert_allclose(result, expected)


def test_rearrange_flatten():    
    # Collapsing multiple dimensions into one.
    x = np.random.rand(2, 3, 4)
    result = rearrange(x, "a b c -> (a b) c")
    expected = x.reshape(6, 4)
    np.testing.assert_allclose(result, expected)


def test_rearrange_expand():
    # Test expanding dimensions (introducing a new axis)
    x = np.random.rand(3, 4)
    result = rearrange(x, "h w -> h w 1")
    assert result.shape == (3, 4, 1)


def test_rearrange_ellipsis_identity():
    # Identity transformation with ellipsis
    x = np.random.rand(2, 3, 4, 5)
    result = rearrange(x, "... -> ...")
    print(result.shape)
    np.testing.assert_allclose(result, x)


def test_rearrange_ellipsis_transpose():
    # Test transposing with ellipsis
    x = np.random.rand(2, 3, 4, 5)
    result = rearrange(x, "b ... d -> d ... b")
    assert result.shape == (5,3,4,2)


def test_rearrange_invalid_pattern_missing_arrow():
    # Invalid pattern with missing '->' separator
    x = np.random.rand(3, 4)
    with pytest.raises(EinopsError):
        rearrange(x, "h w")  


def test_rearrange_invalid_extra_axes():
    # Test rearrange with an unspecified new axis.
    x = np.random.rand(3, 4)
    with pytest.raises(EinopsError):
        rearrange(x, "h w -> h c")  

def test_singular_grouping():
    # Singular group
    x = np.random.rand(2,3,4,5)
    result = rearrange(x, "(a) (b) (c) (d) -> (a) (b) (c) (d)")
    assert result.shape == x.shape
    np.testing.assert_allclose(result, x)
    result = rearrange(x, "(a) (b) (c) (d) -> a b c d")
    assert result.shape == x.shape
    np.testing.assert_allclose(result, x)


def test_ellipsis_in_paranthesis_left():
    x = np.random.rand(2,3,4,5)
    with pytest.raises(EinopsError):
        rearrange(x,"(...) b c d -> d c b ...")
    result = rearrange(x,"... b c d -> d c b ...")
    assert result.shape == (5,4,3,2)

def test_singular_dims():
    # Check addition of singleton dims using brackets
    x = np.random.rand(2,3,4,5)
    result = rearrange(x, "a b c d -> a b c d () ()")
    assert result.shape == (2, 3, 4, 5, 1, 1)
    result = rearrange(x, "a b c d -> a () b () c d () ()")
    assert result.shape == (2, 1, 3, 1, 4, 5, 1, 1)


def test_rearrange_shape_mismatch():
    # Output shape doesn't match input shape.
    x = np.random.rand(3, 4)
    with pytest.raises(EinopsError):
        rearrange(x, "a b c -> a c b")  # Input has only 2 dimensions


def test_rearrange_merge_and_split():
    # Test merging and then splitting dimensions
    x = np.random.rand(2, 3, 4)
    result = rearrange(x, "a b c -> (a b) c")
    expected = x.reshape(6, 4)
    np.testing.assert_allclose(result, expected)
    
    result2 = rearrange(result, "(a b) c -> a b c", a=2, b=3)
    np.testing.assert_allclose(result2, x)


def test_rearrange_broadcasting():
    # Test broadcasting new dimensions by inserting singleton axes
    x = np.random.rand(3, 4)
    result = rearrange(x, "h w -> h w 1 1")
    assert result.shape == (3, 4, 1, 1)


def test_rearrange_multiple_ellipsis():
    # Test multiple ellipses should raise an error
    x = np.random.rand(2, 3, 4)
    with pytest.raises(EinopsError):
        rearrange(x, "... ... -> ...")


def test_rearrange_axis_swapping():
    # Test arbitrary swapping of axes
    x = np.random.rand(2, 3, 4, 5)
    result = rearrange(x, "a b c d -> d c b a")
    expected = np.transpose(x, (3, 2, 1, 0))
    np.testing.assert_allclose(result, expected)


def test_complex_pattern():
    # Test more complex rearrange patterns
    x = np.arange(120).reshape(2, 3, 4, 5)
    result = rearrange(x, "a b c d -> b (a c) d")
    expected = x.transpose(1, 0, 2, 3).reshape(3, 2 * 4, 5)
    np.testing.assert_array_equal(result, expected)


def test_axes_lengths_parameter():
    # Test specifying axes lengths explicitly
    x = np.arange(24).reshape(6, 4)
    result = rearrange(x, "(a b) c -> a b c", a=2, b=3)
    expected = x.reshape(2, 3, 4)
    np.testing.assert_array_equal(result, expected)


def test_multiple_dimensions_in_parentheses():
    # Test handling multiple dimensions in parentheses
    x = np.arange(120).reshape(2, 3, 4, 5)
    result = rearrange(x, "a b c d -> (a b) (c d)")
    expected = x.reshape(2 * 3, 4 * 5)
    np.testing.assert_array_equal(result, expected)


def test_single_dimension_tensor():
    # Test rearranging a 1D tensor
    x = np.arange(5)
    result = rearrange(x, "a -> a")
    np.testing.assert_array_equal(result, x)
    
    # Expand dimension
    result = rearrange(x, "a -> 1 a")
    expected = x.reshape(1, 5)
    np.testing.assert_array_equal(result, expected)



def test_empty_tensor():
    # Test rearranging an empty tensor
    x = np.array([])
    result = rearrange(x, "a -> a")
    np.testing.assert_array_equal(result, x)

    result = rearrange(x, "a -> () a")
    assert result.shape == (1,0)

    result = rearrange(x, "a -> () a b",b=34)
    assert result.shape == (1,0,34)


def test_inconsistent_dimensions():
    # Test attempting to rearrange with inconsistent dimensions
    x = np.arange(24).reshape(2, 3, 4)
    with pytest.raises(EinopsError):
        rearrange(x, "a b c d -> a b c d")  # Wrong number of dimensions

def test_ellipsis_reshape():
    # Test reshaping with ellipsis
    x = np.arange(120).reshape(2, 3, 4, 5)
    result = rearrange(x, "a ... c -> a (...) c")
    expected = x.reshape(2, 3 * 4, 5)
    np.testing.assert_array_equal(result, expected)


def test_ellipsis_one_side_only():
    # Test having ellipsis on only one side of the pattern
    x = np.arange(120).reshape(2, 3, 4, 5)
    with pytest.raises(EinopsError):
        rearrange(x, "a ... -> a b c")  # Ellipsis on input but not output


def test_too_many_arrows():
    # Test pattern with too many arrows
    x = np.arange(24).reshape(2, 3, 4)
    with pytest.raises(EinopsError):
        rearrange(x, "a b c -> c b a -> a c b") 


def test_missing_identifiers():
    # Test pattern with missing identifiers
    x = np.arange(24).reshape(2, 3, 4)
    with pytest.raises(EinopsError):
        rearrange(x, "-> a b c") 

def test_undefined_dimension():
    # Test using an undefined dimension
    x = np.arange(24).reshape(2, 3, 4)
    with pytest.raises(EinopsError):
        rearrange(x, "a b c -> a b c d") 


def test_multiple_unknown_dimensions():
    # Test having multiple unknown dimensions in a composite axis
    x = np.arange(24).reshape(2 * 3, 4)
    with pytest.raises(EinopsError):
        rearrange(x, "(a b) c -> a b c")  

def test_redundant_ellipsis():
    x = np.random.rand(3,4,5)
    result = rearrange(x, "... a b c -> a b c ...")
    np.testing.assert_array_equal(result, x)
    # Add singleton dim 
    result = rearrange(x, "... a b c -> a b c (...)")
    assert result.shape == (3,4,5,1)

def test_miscellaneous_cases():
    # Miscellaneous test cases
    x = np.random.rand(3,4,5)

    # Invalid cases
    invalid_cases = [
        "abc->abc", # No space separation
        "a (b1 b2) c -> a b1 b2 c", # Implicit shape assignment
        "a b c () () () -> a b c", # Extra dims in pattern
        "() () () -> () () ()",
        "(a b c -> a) b c", # Incorrect paranthesis placement
        "( a b ( c -> a b c", 
        ") a b c -> a b c",
    ]

    for case in invalid_cases:
        with pytest.raises(EinopsError):
            print(case)
            rearrange(x,case)