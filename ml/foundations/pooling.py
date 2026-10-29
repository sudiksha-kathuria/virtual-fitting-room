import numpy as np

def max_pool(image, size=2, stride=2):
    img_h , img_w = image.shape
    output_h = (img_h - size)//stride + 1
    output_w = (img_w - size)//stride + 1
    output = np.zeros((output_h,output_w))
    for row in range(output_h):
        for col in range(output_w):
            start_row = stride * row
            start_col = stride * col
            patch = image[ start_row : start_row + size, start_col : start_col + size]
            output[row,col] = np.max(patch)
    return output