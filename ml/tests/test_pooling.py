import numpy as np
from foundations.pooling import max_pool

picture = np.array([
    [1,3,2,1],
    [4,6,5,2],
    [7,2,8,3],
    [1,0,4,9]
])

def test_max_pool():
    expected = np.array([
        [6,5],
        [7,9]
    ])
    assert np.array_equal(max_pool(picture), expected), "Max pooling test failed"

def test_max_pool_stride1():
    result=max_pool(picture, 2, 1)
    expected = np.array([
        [6, 6, 5],
        [7, 8, 8],
        [7, 8, 9]
    ])
    assert result.shape == (3, 3), "Max pooling stride 1 test failed shape"
    assert np.array_equal(result, expected), "Max pooling stride 1 test failed values"
    
if __name__ == "__main__":
    test_max_pool()
    test_max_pool_stride1()
    print("Passes both tests")