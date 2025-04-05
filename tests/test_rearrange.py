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


# TESTING ON CASES FROM EINOPS LIBRARY (https://github.com/arogozhnikov/einops/blob/main/einops/tests/)

def test_rearrange_examples():
    def test1(x):
        y = rearrange(x, "b c h w -> b h w c")
        assert y.shape == (10, 30, 40, 20)
        return y

    def test2(x):
        y = rearrange(x, "b c h w -> b (c h w)")
        assert y.shape == (10, 20 * 30 * 40)
        return y

    def test3(x):
        y = rearrange(x, "b (c h1 w1) h w -> b c (h h1) (w w1)", h1=2, w1=2)
        assert y.shape == (10, 5, 60, 80)
        return y

    def test4(x):
        y = rearrange(x, "b c (h h1) (w w1) -> b (h1 w1 c) h w", h1=2, w1=2)
        assert y.shape == (10, 20 * 4, 15, 20)
        return y

    def test5(x):
        y = rearrange(x, "b1 sound b2 letter -> b1 b2 sound letter")
        assert y.shape == (10, 30, 20, 40)
        return y

    def test6(x):
        t = rearrange(x, "b c h w -> (b h w) c")
        t = t[:, ::2]
        assert t.shape == (10 * 30 * 40, 10)
        return t

    def test7(x):
        y = rearrange(x, "b (c g) h w -> g b c h w", g=2)
        assert y.shape == (2, 10, 10, 30, 40)
        return y[0] + y[1]

    def test8(x):
        tensors = rearrange(x, "b c h w -> b h w c")
        assert tensors.shape == (10, 30, 40, 20)
        return tensors

    def test9(x):
        tensors = rearrange(x, "b c h w -> h (b w) c")
        assert tensors.shape == (30, 10 * 40, 20)
        return tensors

    def shufflenet(x, convolve, c1, c2):
        x = convolve(x)
        x = rearrange(x, "b (c1 c2) h w -> b (c2 c1) h w", c1=c1, c2=c2)
        x = convolve(x)
        return x

    def convolve_strided_1d(x, stride, usual_convolution):
        x = rearrange(x, "b c t1 t2 -> b c (t1 t2)")
        x = rearrange(x, "b c (t stride) -> (stride b) c t", stride=stride)
        x = usual_convolution(x)
        x = rearrange(x, "(stride b) c t -> b c (t stride)", stride=stride)
        return x

    def convolve_strided_2d(x, h_stride, w_stride, usual_convolution):
        x = rearrange(x, "b c (h hs) (w ws) -> (hs ws b) c h w", hs=h_stride, ws=w_stride)
        x = usual_convolution(x)
        x = rearrange(x, "(hs ws b) c h w -> b c (h hs) (w ws)", hs=h_stride, ws=w_stride)
        return x

    def unet_like_1d(x, usual_convolution):
        x = rearrange(x, "b c t1 t2 -> b c (t1 t2)")
        y = rearrange(x, "b c (t dt) -> b (dt c) t", dt=2)
        y = usual_convolution(y)
        x = x + rearrange(y, "b (dt c) t -> b c (t dt)", dt=2)
        return x

    def convolve_mock(x):
        return x

    tests = [
        test1,
        test2,
        test3,
        test4,
        test5,
        test6,
        test7,
        test8,
        test9,
        lambda x: shufflenet(x, convolve=convolve_mock, c1=4, c2=5),
        lambda x: convolve_strided_1d(x, stride=2, usual_convolution=convolve_mock),
        lambda x: convolve_strided_2d(x, h_stride=2, w_stride=2, usual_convolution=convolve_mock),
        lambda x: unet_like_1d(x, usual_convolution=convolve_mock),
    ]

    for test in tests:
        x = np.arange(10 * 20 * 30 * 40).reshape([10, 20, 30, 40])
        result1 = test(x.copy())

        # Now test on sliced version
        x = np.arange(10 * 2 * 20 * 3 * 30 * 1 * 40).reshape([20, 60, 30, 40])
        result2 = test(x[::2, ::3, ::1, ::-1])  


equivalent_rearrange_patterns = [
    ("a b c d e -> (a b) c d e", "a b ... -> (a b) ... "),
    ("a b c d e -> a b (c d) e", "... c d e -> ... (c d) e"),
    ("a b c d e -> a b c d e", "... -> ... "),
    ("a b c d e -> (a b c d e)", "... ->  (...)"),
    ("a b c d e -> b (c d e) a", "a b ... -> b (...) a"),
    ("a b c d e -> b (a c d) e", "a b ... e -> b (a ...) e"),
]

identity_patterns = [
    "...->...",
    "a b c d e-> a b c d e",
    "a b c d e ...-> ... a b c d e",
    "a b c d e ...-> a ... b c d e",
    "... a b c d e -> ... a b c d e",
    "a ... e-> a ... e",
    "a ... -> a ... ",
    "a ... c d e -> a (...) c d e",
]


def test_equivalent_rearrange_patterns():

    x = np.random.randn(2, 3, 4, 5, 6)

    # Test each pair of equivalent patterns
    for concrete, abstract in equivalent_rearrange_patterns:
        y1 = rearrange(x, concrete)
        y2 = rearrange(x, abstract)
        assert y1.shape == y2.shape, f"Shape mismatch: {concrete} vs {abstract}"
        assert np.array_equal(y1, y2), f"Value mismatch for: {concrete} vs {abstract}"


def test_identity_patterns():
    x = np.random.randn(2, 3, 4, 5, 6)

    # Test each pair of equivalent patterns
    for pattern in identity_patterns:
        result = rearrange(x, pattern)
        assert x.shape == result.shape, f"Shape mismatch: for {pattern}"
        assert np.array_equal(x, result), f"Value mismatch for: {pattern}"