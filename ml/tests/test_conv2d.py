import numpy as np
from foundations.conv2d import conv2d, conv2d_multi, relu

IMAGE = np.array([
    [0, 0, 9, 9, 9],
    [0, 0, 9, 9, 9],
    [0, 0, 9, 9, 9],
    [0, 0, 9, 9, 9],
    [0, 0, 9, 9, 9],
])
VERTICAL = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
HORIZONTAL = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

def test_conv2d_plain():
    result = conv2d(IMAGE, VERTICAL)
    expected = np.array([[27, 27, 0],
                         [27, 27, 0],
                         [27, 27, 0]])
    assert np.array_equal(result, expected)

def test_conv2d_padding():
    result = conv2d(IMAGE, VERTICAL, padding=1, stride=1)
    expected = np.array([[0, 18, 18, 0, -18],
                         [0, 27, 27, 0, -27],
                         [0, 27, 27, 0, -27],
                         [0, 27, 27, 0, -27],
                         [0, 18, 18, 0, -18]])
    assert result.shape == (5, 5)
    assert np.array_equal(result, expected)


def test_conv2d_stride():
    result = conv2d(IMAGE, VERTICAL, padding=0, stride=2)
    expected = np.array([[27, 0],
                         [27, 0]])
    assert np.array_equal(result, expected)


def test_conv2d_padding_and_stride():
    result = conv2d(IMAGE, VERTICAL, padding=1, stride=2)
    expected = np.array([[0, 18, -18],
                         [0, 27, -27],
                         [0, 18, -18]])
    assert np.array_equal(result, expected)


def test_edge_direction():
    horizontal_image = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [9, 9, 9, 9, 9],
        [9, 9, 9, 9, 9],
        [9, 9, 9, 9, 9],
    ])
    # the vertical filter is blind to a horizontal edge
    assert np.all(conv2d(horizontal_image, VERTICAL) == 0)
    # the horizontal filter finds it
    expected = np.array([[27, 27, 27],
                         [27, 27, 27],
                         [0, 0, 0]])
    assert np.array_equal(conv2d(horizontal_image, HORIZONTAL), expected)

def test_conv2d_multi():
    image = np.stack([IMAGE, IMAGE, IMAGE])
    kernels = np.stack([
        np.stack([VERTICAL, VERTICAL, VERTICAL]),
        np.stack([HORIZONTAL, HORIZONTAL, HORIZONTAL]),
    ])
    result = conv2d_multi(image, kernels)
    assert result.shape == (2, 3, 3)
    assert np.array_equal(result[0], np.array([[81, 81, 0]] * 3))
    assert np.all(result[1] == 0)

def test_relu():
    x = np.array([[-2, 5], [0, -7]])
    assert np.array_equal(relu(x), np.array([[0, 5], [0, 0]]))

def test_relu_on_flipped_edge():
    flipped = np.fliplr(IMAGE)
    image = np.stack([flipped, flipped, flipped])
    kernels = np.stack([np.stack([VERTICAL, VERTICAL, VERTICAL])])
    result = conv2d_multi(image, kernels)
    assert np.all(result <= 0)
    assert np.all(relu(result) == 0)

if __name__ == "__main__":
    all_tests = [
        test_conv2d_plain,
        test_conv2d_padding,
        test_conv2d_stride,
        test_conv2d_padding_and_stride,
        test_edge_direction,
        test_conv2d_multi,
        test_relu,
        test_relu_on_flipped_edge,
    ]
    for t in all_tests:
        t()
        print("PASS", t.__name__)