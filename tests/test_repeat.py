import pytest
import numpy as np
from einops_rearrange import rearrange, EinopsError


def test_basic_repeat():
    x = np.arange(24).reshape(2, 3, 4)
    result = rearrange(x, "a b c -> a b c d", d=5)    
    assert result.shape == (2,3,4,5)

def test_multiple_repeat_dimensions():
    x = np.arange(24).reshape(2, 3, 4)
    result = rearrange(x, "a b c -> a d b c e", d=5, e=2) 
    assert result.shape == (2,5,3,4,2)

def test_repeat_with_empty_tensor():
    # Test repeating with an empty tensor
    x = np.array([]).reshape(0, 3)

    result = rearrange(x, "a b -> a b c", c=5)

    assert result.size == 0
    assert result.shape == (0,3,5)

def test_repeat_with_anonymous():
    # Test repeating with a scalar value

    x = np.array([42])
    result = rearrange(x, "a -> a 5 2")
    # Shapes and values should match
    assert result.shape == (1, 5, 2)
    

def test_repeat_with_composite_axes():
    # Test repeating with composite axes in the pattern.
    x = np.arange(24).reshape(2, 3, 4)  
    result = rearrange(x, "a b c -> a (b d) c", d=2)
    # Expected shape
    assert result.shape == (2, 6, 4)
    assert np.array_equal(result , np.repeat(x,repeats=2,axis=1).reshape(2,6,4))


def test_repeat_with_ellipsis():
    # Test repeating with ellipsis in the pattern
    x = np.arange(120).reshape(2, 3, 4, 5) 
    result = rearrange(x, "a ... -> a b ...", b=2) 
    x_expanded = np.expand_dims(x, axis=1)
    x_expanded = np.repeat(x_expanded, repeats=2, axis=1)  
    assert result.shape == (2, 2, 3, 4, 5)
    np.testing.assert_array_equal(result, x_expanded)


def test_repeat_with_zero_size():
    # Test repeating with zero size 
    x = np.arange(24).reshape(2, 3, 4)
    with pytest.raises(ValueError):
        rearrange(x, "a b c -> a b c d", d=0)
        rearrange(x, "a b c -> a b c 0")


def test_repeat_with_missing_sizes():
    # Test behavior when missing size specifications
    x = np.arange(24).reshape(2, 3, 4)
    with pytest.raises(EinopsError):
        rearrange(x, "a b c -> a b c d")


def test_repeat_multiple_new_axes_in_composite():
    # Test multiple new axes in a composite expression.
    
    x = np.arange(24).reshape(2, 3, 4)  
    result = rearrange(x, "a b c -> a (b d e) c", d=2, e=3)  
    assert result.shape == (2, 18, 4)
    x = np.repeat(x, repeats=2, axis=1)  
    x = np.repeat(x, repeats=3, axis=1) 
    np.testing.assert_array_equal(result,x)


def test_repeat_with_multi_dimensional_reshaping():
    # Test a case that requires multi-dimensional reshaping during repeating
    x = np.arange(24).reshape(2, 3, 4)  
    rearrange_result = rearrange(x, "a b c -> (a d) (b e) c", d=2, e=3)

    x_expanded = np.repeat(x, repeats=2, axis=0)  
    x_expanded = np.repeat(x_expanded, repeats=3, axis=1) 

    assert rearrange_result.shape == x_expanded.shape

    np.testing.assert_array_equal(rearrange_result, x_expanded)


def test_repeat_on_scalar_broadcast():
    # Broadcasting a scalar value along multiple new axes
    x = np.array(7)
    result = rearrange(x, "  -> b c d", b=2, c=3, d=4)

    expected = np.full((2, 3, 4), 7)
    np.testing.assert_array_equal(result, expected)


def test_repeat_with_named_and_computed_axis():
    x = np.arange(6).reshape(2, 3)
    result = rearrange(x, "a b -> (a a2) b", a2=2)
    assert result.shape == (4, 3)
    expected = np.repeat(x, repeats=2, axis=0)
    np.testing.assert_array_equal(result, expected)


def test_repeat_and_group_mixed():
    x = np.arange(24).reshape(2, 3, 4)
    result = rearrange(x, "a b c -> a (b d) (c e)", d=2, e=3)
    assert result.shape == (2, 6, 12)
    x_rep = np.repeat(x, 2, axis=1)
    x_rep = np.repeat(x_rep, 3, axis=2)
    np.testing.assert_array_equal(result, x_rep)


def test_repeat_in_middle_of_tensor():
    x = np.arange(120).reshape(2, 3, 4, 5)
    result = rearrange(x, "a b c d -> a b r c d", r=2)
    x_exp = np.expand_dims(x, 2)
    x_exp = np.repeat(x_exp, repeats=2, axis=2)
    np.testing.assert_array_equal(result, x_exp)


def test_repeat_with_multiple_ellipses_locations():
    x = np.arange(2*3*4*5).reshape(2, 3, 4, 5)
    result = rearrange(x, "... c d -> ... c d r", r=2)
    x_exp = np.expand_dims(x, axis=-1)
    x_exp = np.repeat(x_exp, repeats=2, axis=-1)
    np.testing.assert_array_equal(result, x_exp)
